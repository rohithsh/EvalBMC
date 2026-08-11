#!/usr/bin/env python3
"""
predict_bounds.py

Runner. Reads the dataset, asks context_builder for the program text, asks
build_prompts for the prompt, asks llm_client for an answer, writes the result.


TWO CONFIGURATIONS  (--informed)
    blind (default)  the expected verdict is withheld. The model is asked for
                     the smallest depth at which the checker reaches a
                     conclusive result -- the same quantity whether or not the
                     program has a bug.
    informed         the verdict is supplied.


Usage:
  # inspect prompts for free
  python3 predict_bounds.py --json datasets/cleaned/loops.json \
      --dataset datasets/cleaned/svcomp_clean --out /tmp/echo.jsonl \
      --provider echo --mode full --limit 5 --restart

  # hosted endpoint, blind
  python3 predict_bounds.py --json datasets/cleaned/loops.json \
      --dataset datasets/cleaned/svcomp_clean \
      --out results/<filename>.jsonl \
      --provider openai --base-url <url> \
      --model <model> --api-key-env LLM_API_KEY \
      --mode full --max-chars 400000 [--jobs] [--limit] [--informed]

"""

import argparse
import csv
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import build_prompts
import context_builder
import llm_client

PROP_KEYS = {
    "reach": "unreach-call",
    "memsafety": "valid-memsafety",
    "overflow": "no-overflow",
}

# context fields that are useful to keep per row; the long lists produced by the
# slicer are dropped so the JSONL stays readable
CONTEXT_KEEP = ("mode", "source_lines", "source_chars", "kept_lines",
                "kept_chars", "dropped_lines", "reduction", "rounds",
                "note", "error")


def program_category(rec):
    cats = {l.get("category") for l in rec.get("loops", []) if l.get("category")}
    if not cats:
        return "none"
    return next(iter(cats)) if len(cats) == 1 else "mixed"


def run_one(job):
    (rec, prop, verdict, dataset, mode, informed, max_chars, provider,
     samples, retries) = job

    out = {
        "task": rec.get("task"), "dir": rec.get("dir"),
        "c_file": rec.get("c_file"),
        "property": prop, "verdict": verdict,
        "config": "informed" if informed else "blind",
        "category": program_category(rec),
        "n_loops": len(rec.get("loops", [])),
        "mode": mode,
        "model": provider.describe(),
        "predictions": [], "errors": [],
    }

    # --- program text ---------------------------------------------------
    try:
        source, info = context_builder.build_context(rec, dataset, mode)
    except Exception:
        out["errors"].append("context: " + traceback.format_exc(limit=2)[-300:])
        return out

    info = info or {}
    out["context"] = {k: info[k] for k in CONTEXT_KEEP if k in info}
    if source is None:
        out["errors"].append(info.get("error", "context failed"))
        return out
    if informed is None:
        informed = False
    # --- prompt -----------------------------------------------------------
    try:
        system, user = build_prompts.build(rec, prop, verdict, source,
                                           informed=informed)
    except Exception:
        out["errors"].append("prompt: " + traceback.format_exc(limit=2)[-300:])
        return out

    out["prompt_chars"] = len(system) + len(user)
    if len(user) > max_chars:
        out["errors"].append("too_large")
        return out

    # --- model ------------------------------------------------------------
    for s in range(samples):
        try:
            t0 = time.time()
            raw = provider.complete(system, user, retries=retries)
            dt = round(time.time() - t0, 2)
            parsed, err = build_prompts.parse(raw)
            out["predictions"].append({
                "sample": s, "latency_s": dt,
                "parsed": parsed, "parse_error": err, "raw": raw,
            })
        except Exception as e:
            out["errors"].append("{}: {}".format(type(e).__name__, str(e)[:160]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="loops.json")
    ap.add_argument("--dataset", required=True, help="svcomp_clean")
    ap.add_argument("--out", required=True, help="JSONL output")

    ap.add_argument("--mode", choices=["full", "slice"], default="full")
    ap.add_argument("--informed", action="store_true",
                    help="supply the expected verdict to the model "
                         "(capability upper bound, not a fair baseline)")
    ap.add_argument("--max-chars", type=int, default=400000,
                    help="skip a program whose prompt exceeds this")
    ap.add_argument("--samples", type=int, default=1,
                    help=">1 with --temperature>0 measures prediction stability")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=4)

    ap.add_argument("--properties", nargs="+", default=list(PROP_KEYS),
                    choices=list(PROP_KEYS))
    ap.add_argument("--verdicts", nargs="+", default=["unsafe", "safe"],
                    choices=["unsafe", "safe"])
    ap.add_argument("--only-tasks", default=None,
                    help="CSV with task and property columns; predict only "
                         "those pairs (e.g. the k* results, so predictions "
                         "exist only where there is ground truth)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--restart", action="store_true")

    llm_client.add_provider_args(ap)
    a = ap.parse_args()

    recs = json.load(open(a.json))

    allow = None
    if a.only_tasks:
        allow = set()
        with open(a.only_tasks) as fh:
            for r in csv.DictReader(fh):
                allow.add((r["task"], r["property"]))
        print("restricted to {} task/property pairs".format(len(allow)))

    jobs_spec = []
    for r in recs:
        v = r.get("verdicts", {}) or {}
        for prop in a.properties:
            got = v.get(PROP_KEYS[prop])
            if got not in a.verdicts:
                continue
            if allow is not None and (r["task"], prop) not in allow:
                continue
            jobs_spec.append((r, prop, got))

    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if a.restart and outp.exists():
        outp.unlink()
    elif outp.exists():
        done = set()
        with open(outp) as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                    done.add((d.get("task"), d.get("property")))
                except Exception:
                    pass
        before = len(jobs_spec)
        jobs_spec = [j for j in jobs_spec if (j[0]["task"], j[1]) not in done]
        print("resuming: {} done, {} to go".format(before - len(jobs_spec),
                                                   len(jobs_spec)))

    if a.limit:
        jobs_spec = jobs_spec[:a.limit]
    if not jobs_spec:
        print("nothing to do")
        return 0

    provider = llm_client.build_provider(a)
    print("provider : {}".format(provider.describe()))
    print("config   : {}   mode: {}   items: {}   samples: {}".format(
        "informed" if a.informed else "blind", a.mode, len(jobs_spec), a.samples))

    jobs = [(rec, prop, verdict, a.dataset, a.mode, a.informed, a.max_chars,
             provider, a.samples, a.retries)
            for (rec, prop, verdict) in jobs_spec]

    n = ok = bad = skipped = 0
    kinds = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(run_one, j): (j[0]["task"], j[1]) for j in jobs}
        for f in as_completed(futs):
            task, prop = futs[f]
            try:
                res = f.result()
            except Exception:
                res = {"task": task, "property": prop, "predictions": [],
                       "errors": ["worker: " + traceback.format_exc(limit=3)[-400:]]}
            with open(outp, "a") as fh:
                fh.write(json.dumps(res) + "\n")

            n += 1
            preds = res.get("predictions") or []
            errs = res.get("errors") or []
            if "too_large" in errs:
                skipped += 1
                tag = "too_large"
            elif preds and preds[0].get("parsed"):
                ok += 1
                p = preds[0]["parsed"]
                kinds[p.get("kind")] = kinds.get(p.get("kind"), 0) + 1
                tag = p.get("kind")
                if p.get("kind") == "uniform":
                    tag += " k={}".format(p.get("k"))
                elif p.get("kind") == "schedule":
                    tag += " ({} loops)".format(len(p.get("schedule", {})))
            else:
                bad += 1
                tag = (preds[0].get("parse_error") if preds
                       else (errs[0] if errs else "?"))
                tag = str(tag).replace("\n", " ")[:40]
            print("[{}/{}] {:<38} {:<10} {:<42} ({:.0f}m)".format(
                n, len(jobs), str(res.get("task", task))[:38],
                res.get("property", prop), tag, (time.time() - t0) / 60.0),
                flush=True)

    print("\nparsed ok {}   unusable {}   skipped {}".format(ok, bad, skipped))
    if kinds:
        print("answer kinds: {}".format(kinds))
    if bad:
        print("\nfirst few failures:")
        shown = 0
        with open(outp) as fh:
            for line in fh:
                d = json.loads(line)
                preds = d.get("predictions") or []
                if not (preds and preds[0].get("parsed")):
                    print("  {}  {}".format(
                        d.get("task"),
                        (preds[0].get("parse_error") if preds
                         else (d.get("errors") or ["?"])[0])))
                    shown += 1
                    if shown >= 5:
                        break
    print("predictions -> {}".format(outp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())