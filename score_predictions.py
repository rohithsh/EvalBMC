#!/usr/bin/env python3
"""
score_predictions.py

Score predicted unrolling bounds two ways.

  SCORE A -- agreement with the ground truth.
      The predicted bound and the ground-truth bound are both converted to a
      COST (total loop-body unrollings: each loop's bound multiplied by the
      bounds of every enclosing loop, summed over loops), and compared:
      exact / over / under, plus a tightness ratio pred_cost / true_cost.
      Note: the cost model is a static estimate. It agrees with CBMC for
      ordinary loops but overstates loops CBMC never reaches, so report the
      measured columns (steps, variables, clauses, seconds) for any claim
      about cost.

  SCORE B -- does the predicted bound actually work?
      CBMC is run at the predicted bound and the result is checked under the
      soundness rule used for the ground truth: it works if a property-matched
      violation is reported (unsafe) or the property is proved (safe), in
      both cases with NO unwinding assertion failing.

      Outcomes: works / insufficient (a loop was cut short) / no_violation
      (sound, but the bug was not reached) / other_property / inconclusive /
      timeout / impractical (a bound above MAX_PRACTICAL_K, not run, since
      CBMC cannot build that formula) / not_run (an "unbounded" answer).

  TIMING AND SIZE, for every row CBMC was run on
      verify_seconds          the one CBMC run at the predicted bound
      steps, variables,       CBMC's own size of the unrolled program and of
      clauses                 the SAT formula; machine-independent
      baseline_seconds        time iterative deepening took (kstar CSV)
      predict_seconds         the model's API latency for this answer
      speedup                 baseline / verify
      speedup_with_prediction baseline / (verify + predict)
      solved_where_baseline_failed
                              "yes" when the bound works and iterative
                              deepening timed out or hit its cap
      Speedups are reported only where iterative deepening found k* > 0 and
      the predicted bound works.

  VERIFICATION CACHE (bmc_cache.py)
      A CBMC run is keyed by task, property, program-file hash, the bound,
      the CBMC version and the CBMC flags -- not by who proposed the bound.
      So a bound that any approach or model has been scored on before is
      not run again. --retry-timeouts re-runs cached timeouts.

  ABSTENTION
      An "unbounded" reply is warranted when no finite k* exists and
      unwarranted when one does.

The CBMC flags per property and the soundness analysis are taken from
derive_kstar_all.py, so they have one definition.

Usage:
  python3 score_predictions.py \
      --pred    results/pred_qwen397b_full.clean.jsonl \
      --kstar   results/kstar_final.csv \
      --json    datasets/cleaned/loops.json \
      --dataset datasets/cleaned/svcomp_clean \
      --out     results/score_qwen397b_full.csv \
      [--jobs 8] [--verify-timeout 600] [--no-verify] [--skip-exact]
      [--no-cache] [--retry-timeouts]
"""

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from bmc_cache import VerifyCache, VERIFY_DB, sha_file, sha_text, bound_key, \
    cbmc_version

# The CBMC flags and the soundness analysis live in the derivation script;
# importing them keeps one definition rather than a copy that can drift.
try:
    import derive_kstar as kstar          # the k* script in the project
except ImportError:
    try:
        import derive_kstar_all as kstar  # older name for the same script
    except ImportError:
        print("cannot import derive_kstar.py -- put it beside this script",
              file=sys.stderr)
        raise

MAX_PRACTICAL_K = 10000      # beyond this CBMC cannot build the formula
VERBOSITY = "8"              # needed for the size and runtime statistics

STEPS_RE = re.compile(r"size of program expression:\s*(\d+)\s*steps")
SAT_RE = re.compile(r"(\d+)\s+variables,\s+(\d+)\s+clauses")


# ------------------------------------------------------------------- helpers

def num(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def fnum(x):
    """float or None. Never int(): int('2.81') fails and a truncated 2.81 -> 2
    once halved a reported speedup."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def loop_structure(rec):
    ids = [l["id"] for l in rec.get("loops", []) if l.get("id")]
    parents = {l["id"]: l.get("parent") for l in rec.get("loops", []) if l.get("id")}
    return ids, parents


def pred_bound(parsed):
    """(kind, uniform_k_or_None, schedule_or_None)"""
    kind = parsed.get("kind")
    if kind == "uniform":
        return "uniform", parsed.get("k"), None
    if kind == "schedule":
        return "schedule", None, {k: v for k, v in (parsed.get("schedule") or {}).items()}
    return "unbounded", None, None


def cost_of(kind, k, sched, loop_ids, parents):
    if kind == "uniform" and k is not None:
        return kstar.uniform_cost(k, loop_ids, parents)
    if kind == "schedule" and sched:
        full = {lid: sched.get(lid, 0) for lid in loop_ids}
        for lid, v in sched.items():
            full.setdefault(lid, v)
        return kstar.loop_cost(full, parents)
    return None


def model_name(p):
    """predict_bounds.py writes a string; predict_syntactic.py a dict."""
    m = p.get("model")
    if isinstance(m, dict):
        return str(m.get("model") or "")
    return str(m or "")


def provider_name(p):
    m = p.get("model")
    if isinstance(m, dict) and m.get("provider"):
        return str(m["provider"])
    return str(p.get("provider") or "")


def cbmc_flags(data_model, prop):
    """Everything on the CBMC command line except the file and the bound."""
    return ["--32" if data_model == "ILP32" else "--64",
            "--unwinding-assertions", *kstar.PROFILE[prop],
            "--json-ui", "--verbosity", VERBOSITY]


# ------------------------------------------------------------ CBMC verify

def verify(job):
    """
    One CBMC run at the predicted bound. Returns a dict with the status under
    the soundness rule, plus the measured time and formula size.
    """
    (cfile, data_model, prop, verdict, kind, k, sched, timeout, rule) = job
    cmd = [kstar.CBMC, str(cfile), *cbmc_flags(data_model, prop)]
    if kind == "uniform":
        cmd += ["--unwind", str(k)]
    elif kind == "schedule":
        cmd += ["--unwind", "0", "--unwindset",
                ",".join("{}:{}".format(a, b) for a, b in sorted(sched.items()))]
    else:
        return {"status": "not_applicable"}

    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "seconds": float(timeout)}
    except Exception as e:
        return {"status": "error", "detail": str(e)[:80]}
    secs = round(time.time() - t0, 2)
    out = p.stdout or ""

    sm = STEPS_RE.findall(out)
    vm = SAT_RE.findall(out)
    size = {"steps": int(sm[-1]) if sm else None,
            "variables": int(vm[-1][0]) if vm else None,
            "clauses": int(vm[-1][1]) if vm else None}

    try:
        res, failing, _, other = kstar.analyse(out, prop, verdict, rule)
    except TypeError:
        if rule != "complete":
            raise SystemExit("derive_kstar.py beside this script has no "
                             "--unsafe-rule support; use the updated one")
        res, failing, _, other = kstar.analyse(out, prop)
    if kstar.is_done(verdict, res):
        status, detail = "works", failing
    elif res == "unwind":
        status, detail = "insufficient", ""     # a loop was cut short
    elif res == "success" and verdict == "unsafe":
        status, detail = "no_violation", ""     # sound, bug not reached
    elif res == "other":
        status, detail = "other_property", other
    else:
        status, detail = "inconclusive", ""
    return {"status": status, "detail": detail, "seconds": secs, **size}


# ------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="predictions JSONL")
    ap.add_argument("--kstar", required=True, help="derive_kstar_all CSV")
    ap.add_argument("--json", required=True, help="loops.json")
    ap.add_argument("--dataset", required=True, help="svcomp_clean")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--verify-timeout", type=int, default=600)
    ap.add_argument("--no-verify", action="store_true",
                    help="Score A only; do not run CBMC")
    ap.add_argument("--skip-exact", action="store_true",
                    help="do not run CBMC when the prediction equals k* "
                         "(saves time, but those rows then have no timing)")
    ap.add_argument("--verify-db", default=str(VERIFY_DB))
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--unsafe-rule", choices=["exposure", "complete"],
                    default="exposure",
                    help="exposure (default): on an unsafe pair a property-"
                         "matched violation counts even if another unwinding "
                         "assertion fails; complete: the earlier, stricter rule")
    ap.add_argument("--retry-timeouts", action="store_true",
                    help="re-run bounds whose cached result is a timeout")
    a = ap.parse_args()

    if not hasattr(kstar, "PROFILE"):
        print("derive_kstar_all.py has no PROFILE table; cannot build the CBMC "
              "flags", file=sys.stderr)
        return 2

    recs = {r["task"]: r for r in json.load(open(a.json))}
    truth = {(r["task"], r["property"]): r for r in csv.DictReader(open(a.kstar))}

    preds = []
    with open(a.pred) as fh:
        for line in fh:
            try:
                preds.append(json.loads(line))
            except Exception:
                pass
    print("predictions: {}   ground-truth rows: {}".format(len(preds), len(truth)))

    models = Counter(model_name(p) or "(unrecorded)" for p in preds)
    if len(models) == 1:
        print("model: {}".format(next(iter(models))))
    else:
        print("WARNING: this file holds rows from more than one model, so the")
        print("         headline numbers below mix them. Split it first.")
        for m, n in models.most_common():
            print("           {:<28} {} rows".format(m, n))

    version = cbmc_version(kstar.CBMC)
    cache = None if a.no_cache else VerifyCache(a.verify_db)
    print("cbmc: {}   unsafe rule: {}".format(version, a.unsafe_rule))
    if cache:
        print("verification cache: {}  ({} runs stored)".format(
            a.verify_db, cache.count()))

    rows, verify_jobs = [], []
    file_shas = {}
    shown_cmd = False

    for p in preds:
        task, prop = p.get("task"), p.get("property")
        rec = recs.get(task)
        gt = truth.get((task, prop))
        preds_list = p.get("predictions") or []
        parsed = preds_list[0].get("parsed") if preds_list else None

        row = {
            "task": task, "property": prop,
            "verdict": p.get("verdict"), "category": p.get("category"),
            "n_loops": p.get("n_loops"),
            "mode": p.get("mode", "") or p.get("input_mode", ""),
            "model": model_name(p), "provider": provider_name(p),
            "pred_kind": "", "pred_k": "", "pred_schedule": "", "pred_cost": "",
            "true_outcome": "", "true_kstar": "", "true_cost": "",
            "score_a": "", "tightness": "",
            "score_b": "", "verify_detail": "", "cached": "",
            "verify_seconds": "", "steps": "", "variables": "", "clauses": "",
            "baseline_seconds": "", "predict_seconds": "",
            "speedup": "", "speedup_with_prediction": "",
            "solved_where_baseline_failed": "",
        }
        if preds_list:
            lat = fnum(preds_list[0].get("latency_s"))
            if lat is not None:
                row["predict_seconds"] = lat

        if gt:
            row["true_outcome"] = gt.get("uniform_outcome", "")
            row["true_kstar"] = gt.get("uniform_kstar", "")
            row["true_cost"] = gt.get("schedule_cost") or gt.get("uniform_cost") or ""
            bs = fnum(gt.get("uniform_time"))
            if bs is not None:
                row["baseline_seconds"] = bs

        if not parsed:
            row["score_a"] = "no_prediction"
            rows.append(row)
            continue

        kind, k, sched = pred_bound(parsed)
        row["pred_kind"] = kind
        if kind == "uniform":
            row["pred_k"] = k
        elif kind == "schedule":
            row["pred_schedule"] = ";".join(
                "{}:{}".format(x, y) for x, y in sorted(sched.items()))

        if rec is None:
            row["score_a"] = "unknown_task"
            rows.append(row)
            continue

        loop_ids, parents = loop_structure(rec)
        pcost = cost_of(kind, k, sched, loop_ids, parents)
        if pcost is not None:
            row["pred_cost"] = pcost

        true_k = num(gt.get("uniform_kstar")) if gt else None
        has_truth = (gt is not None and gt.get("uniform_outcome") == "found"
                     and true_k is not None and true_k > 0)

        # ---- Score A ----
        if kind == "unbounded":
            row["score_a"] = "abstain_unwarranted" if has_truth else "abstain_warranted"
        elif not has_truth:
            row["score_a"] = "no_ground_truth"
        else:
            tcost = num(row["true_cost"])
            if tcost is None:
                tcost = kstar.uniform_cost(true_k, loop_ids, parents)
                row["true_cost"] = tcost
            if pcost is None or tcost in (None, 0):
                row["score_a"] = "incomparable"
            else:
                row["tightness"] = round(pcost / float(tcost), 3)
                row["score_a"] = ("exact" if pcost == tcost else
                                  "over" if pcost > tcost else "under")

        # ---- Score B ----
        full_sched = ({lid: sched.get(lid, 0) for lid in loop_ids}
                      if kind == "schedule" else None)
        if full_sched is not None:
            for lid, v in sched.items():
                full_sched.setdefault(lid, v)
        biggest = (k if kind == "uniform" else
                   max(full_sched.values()) if full_sched else 0) or 0

        if a.no_verify or kind == "unbounded":
            row["score_b"] = "not_run"
        elif biggest > MAX_PRACTICAL_K:
            row["score_b"] = "impractical"
            row["verify_detail"] = "bound {} > {}".format(biggest, MAX_PRACTICAL_K)
        elif row["score_a"] == "exact" and a.skip_exact:
            row["score_b"] = "works"
            row["verify_detail"] = "equals k*, not re-run"
        else:
            cfile = Path(a.dataset) / rec["dir"] / rec["c_file"]
            if not cfile.exists():
                row["score_b"] = "missing_file"
            else:
                dm = rec.get("data_model", "ILP32")
                job = (cfile, dm, prop, p.get("verdict"), kind, k,
                       full_sched, a.verify_timeout, a.unsafe_rule)
                if not shown_cmd:
                    print("first CBMC command, check it matches your k* run:")
                    print("   {} {} {} --unwind ...".format(
                        kstar.CBMC, cfile, " ".join(cbmc_flags(dm, prop))))
                    shown_cmd = True
                fkey = str(cfile)
                if fkey not in file_shas:
                    file_shas[fkey] = sha_file(cfile)
                base = (task, prop, file_shas[fkey],
                        bound_key(kind, k, full_sched), version)
                old_key = base + (sha_text(*cbmc_flags(dm, prop)),)
                ckey = old_key
                hit = None
                if a.unsafe_rule == "exposure" and p.get("verdict") == "unsafe":
                    # results under the exposure rule get their own key. An
                    # older result is reused unless it is one the rule can
                    # change: a too-small run may have exposed the violation
                    # while another loop was cut short.
                    ckey = base + (sha_text(*cbmc_flags(dm, prop),
                                            "unsafe-rule=exposure"),)
                    hit = (cache.get(*ckey, a.verify_timeout, a.retry_timeouts)
                           if cache else None)
                    if hit is None and cache:
                        old = cache.get(*old_key, a.verify_timeout, a.retry_timeouts)
                        if old and old.get("status") not in ("insufficient",
                                                             "inconclusive"):
                            hit = old
                elif cache:
                    hit = cache.get(*ckey, a.verify_timeout, a.retry_timeouts)
                if hit:
                    row["_res"] = hit
                    row["cached"] = "yes"
                else:
                    verify_jobs.append((len(rows), job, ckey))
        rows.append(row)

    # ---- run what the cache could not answer ----
    if verify_jobs:
        print("verifying {} predictions with CBMC (--jobs {}){}".format(
            len(verify_jobs), a.jobs,
            ", {} more answered from the cache".format(cache.hits) if cache else ""))
        t0 = time.time()
        done = 0
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            futs = {ex.submit(verify, job): (idx, ckey)
                    for idx, job, ckey in verify_jobs}
            for f in as_completed(futs):
                idx, ckey = futs[f]
                try:
                    res = f.result()
                except Exception as e:
                    res = {"status": "error", "detail": str(e)[:80]}
                rows[idx]["_res"] = res
                rows[idx]["cached"] = "no"
                if cache:
                    cache.put(*ckey, res["status"], detail=res.get("detail", ""),
                              seconds=res.get("seconds"),
                              timeout=a.verify_timeout, steps=res.get("steps"),
                              variables=res.get("variables"),
                              clauses=res.get("clauses"))
                done += 1
                if done % 25 == 0 or done == len(verify_jobs):
                    print("   {}/{}  ({:.0f}m)".format(
                        done, len(verify_jobs), (time.time() - t0) / 60.0),
                        flush=True)
    elif cache and cache.hits:
        print("all {} CBMC runs answered from the cache".format(cache.hits))

    # ---- fill results, timing, speedups ----
    for row in rows:
        res = row.pop("_res", None)
        if res is None:
            continue
        status = res.get("status", "")
        if row["score_a"] == "exact":
            # k* is known to work; a re-run that does not confirm it (most
            # likely a timeout on a loaded machine) is noted, not counted
            row["score_b"] = "works"
            if status != "works":
                row["verify_detail"] = "equals k*; re-run gave {}".format(status)
        else:
            row["score_b"] = status
            row["verify_detail"] = res.get("detail", "") or ""
        if status in ("works", "insufficient", "no_violation", "other_property",
                      "inconclusive"):
            row["verify_seconds"] = res.get("seconds", "")
        for col in ("steps", "variables", "clauses"):
            if res.get(col) is not None:
                row[col] = res[col]

        works = row["score_b"] == "works"
        if works and row["true_outcome"] in ("timeout", "cap_reached"):
            row["solved_where_baseline_failed"] = "yes"
        vs, bs = fnum(row["verify_seconds"]), fnum(row["baseline_seconds"])
        tk = num(row["true_kstar"])
        if (works and status == "works" and row["true_outcome"] == "found"
                and (tk or 0) > 0 and vs and bs):
            row["speedup"] = round(bs / vs, 3)
            ps = fnum(row["predict_seconds"])
            if ps is not None:
                row["speedup_with_prediction"] = round(bs / (vs + ps), 3)

    # ---- write ----
    cols = list(rows[0].keys()) if rows else []
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # ---- summary ----
    print("\n" + "=" * 66)
    print("SCORE A -- agreement with k* (by cost)")
    print("=" * 66)
    for k_, v in Counter(r["score_a"] for r in rows).most_common():
        print("   {:<22} {:>5}  ({:.1f}%)".format(k_, v, 100.0 * v / len(rows)))
    tight = sorted(float(r["tightness"]) for r in rows if r["tightness"] != "")
    if tight:
        print("\n   tightness (pred cost / true cost), n={}".format(len(tight)))
        print("      median={:.2f}  p25={:.2f}  p75={:.2f}  max={:.1f}".format(
            tight[len(tight) // 2], tight[len(tight) // 4],
            tight[3 * len(tight) // 4], tight[-1]))

    print("\n" + "=" * 66)
    print("SCORE B -- does the predicted bound work?")
    print("=" * 66)
    cb = Counter(r["score_b"] for r in rows if r["score_b"])
    tot_b = sum(cb.values()) or 1
    for k_, v in cb.most_common():
        print("   {:<22} {:>5}  ({:.1f}%)".format(k_, v, 100.0 * v / tot_b))

    covered = [r for r in rows if r["score_b"] in
               ("works", "insufficient", "no_violation", "inconclusive",
                "other_property", "timeout", "impractical")]
    if covered:
        w_ = sum(1 for r in covered if r["score_b"] == "works")
        print("\n   usable rate: {}/{} ({:.1f}%)".format(
            w_, len(covered), 100.0 * w_ / len(covered)))
    solved = sum(1 for r in rows if r["solved_where_baseline_failed"] == "yes")
    failed_base = sum(1 for r in rows if r["true_outcome"] in ("timeout", "cap_reached"))
    print("   solved where iterative deepening failed: {} (of {} rows where it "
          "failed)".format(solved, failed_base))

    sp = [float(r["speedup"]) for r in rows if r["speedup"] != ""]
    spp = [float(r["speedup_with_prediction"]) for r in rows
           if r["speedup_with_prediction"] != ""]
    if sp:
        print("\n   speedup over iterative deepening, where both work (n={})".format(len(sp)))
        print("      verification only    median {:.2f}x".format(statistics.median(sp)))
    if spp:
        print("      including prediction median {:.2f}x".format(statistics.median(spp)))

    print("\nby loop category (Score B):")
    for cat in ("constant", "input-bounded", "data-dependent", "mixed"):
        sub = [r for r in rows if r["category"] == cat and r["score_b"]
               and r["score_b"] != "not_run"]
        if sub:
            w_ = sum(1 for r in sub if r["score_b"] == "works")
            print("   {:<16} {}/{} worked ({:.1f}%)".format(
                cat, w_, len(sub), 100.0 * w_ / len(sub)))

    if cache:
        print("\nverification cache: {} hits, {} CBMC runs, {} stored".format(
            cache.hits, len(verify_jobs), cache.writes))
    print("\nscores -> {}".format(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())