#!/usr/bin/env python3
"""
score_predictions.py

Score predicted unrolling bounds two ways.

  SCORE A -- agreement with the ground truth.
      The predicted bound and the ground-truth bound are both converted to a
      COST (total loop-body unrollings: each loop's bound multiplied by the
      bounds of every enclosing loop, summed over loops). Comparing costs lets a
      per-loop schedule be measured against a uniform bound on the same scale,
      which a raw integer comparison cannot do.
      Reported as exact / over / under, plus a tightness ratio
      pred_cost / true_cost, where 1.0 is ideal.

  SCORE B -- does the predicted bound actually work?
      CBMC is run at the predicted bound and the result is checked under the
      same soundness rule used to derive the ground truth: the bound works if a
      property-matched violation is reported (unsafe program) or the property is
      proved (safe program), in both cases with NO unwinding assertion failing.

      CBMC is run only when it can tell us something new:
        prediction == k*        -> skipped, it is known to work
        prediction != k*        -> run, a different bound may still work
        no finite k* was found  -> run by default, since Score A is undefined
                                   and this is the only measurement available

      This matters because k* is undefined for the ~1700 program-property pairs
      whose search timed out. Score B covers them; Score A cannot.

  ABSTENTION
      An "unbounded" reply is warranted when no finite k* exists and unwarranted
      when one does. Scoring them together would flatter a predictor that never
      answers.

Ground truth comes from derive_kstar_all.py's CSV, and the CBMC invocation is
imported from that script so the soundness rule has a single definition.

Usage:
  python3 score_predictions.py \
      --pred    results/pred_blind.jsonl \
      --kstar   results/kstar_all_v2.csv \
      --json    datasets/cleaned/loops.json \
      --dataset datasets/cleaned/svcomp_clean \
      --out     results/scores_blind.csv \
      [--jobs 8] [--verify-timeout 600] [--no-verify]
"""

import argparse
import csv
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# The CBMC call and the soundness rule live in the derivation script; importing
# them keeps one definition rather than a copy that can drift.
try:
    import derive_kstar as kstar
except ImportError:
    print("cannot import derive_kstar_all.py -- put it beside this script "
          "(rename derive_kstar_all_v2.py if needed)", file=sys.stderr)
    raise


# ------------------------------------------------------------------- helpers

def num(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def parse_schedule_field(s):
    """'main.0:3;main.1:200' -> {'main.0':3,'main.1':200}"""
    out = {}
    for part in (s or "").split(";"):
        if ":" in part:
            k, _, v = part.partition(":")
            n = num(v)
            if n is not None:
                out[k.strip()] = n
    return out


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
        # a schedule may name loops that do not exist, or omit some; unnamed
        # loops fall back to 0 under --unwindset with --unwind 0
        full = {lid: sched.get(lid, 0) for lid in loop_ids}
        for lid, v in sched.items():
            full.setdefault(lid, v)
        return kstar.loop_cost(full, parents)
    return None


# ------------------------------------------------------------ CBMC verify

def verify(job):
    """
    Run CBMC once at the predicted bound and report whether it is conclusive
    under the soundness rule.
    """
    (cfile, data_model, prop, verdict, kind, k, sched, timeout) = job
    try:
        if kind == "uniform":
            st, out = kstar.cbmc_run(cfile, data_model, prop,
                                     unwind=k, timeout=timeout)
        elif kind == "schedule":
            st, out = kstar.cbmc_run(cfile, data_model, prop,
                                     unwind=0, unwindset=sched, timeout=timeout)
        else:
            return "not_applicable", ""
    except Exception as e:
        return "error", str(e)[:80]

    if st != "ok":
        return "timeout", ""

    res, failing, _, other = kstar.analyse(out, prop)
    if kstar.is_done(verdict, res):
        return "works", failing
    if res == "unwind":
        return "insufficient", ""        # bound too small: loop cut short
    if res == "success" and verdict == "unsafe":
        return "no_violation", ""        # sound, but the bug was not reached
    if res == "other":
        return "other_property", other
    return "inconclusive", ""


# ------------------------------------------------------------------- main


def model_name(p):
    """
    The model behind a prediction row. predict_bounds.py writes a string;
    predict_syntactic.py writes {"provider": ..., "model": ...}.
    """
    m = p.get("model")
    if isinstance(m, dict):
        return str(m.get("model") or "")
    return str(m or "")


def provider_name(p):
    m = p.get("model")
    if isinstance(m, dict) and m.get("provider"):
        return str(m["provider"])
    return str(p.get("provider") or "")

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
    a = ap.parse_args()

    recs = {r["task"]: r for r in json.load(open(a.json))}
    truth = {}
    for r in csv.DictReader(open(a.kstar)):
        truth[(r["task"], r["property"])] = r

    preds = []
    with open(a.pred) as fh:
        for line in fh:
            try:
                preds.append(json.loads(line))
            except Exception:
                pass
    print("predictions: {}   ground-truth rows: {}".format(len(preds), len(truth)))

    models = {}
    for p in preds:
        m = model_name(p) or "(unrecorded)"
        models[m] = models.get(m, 0) + 1
    if len(models) == 1:
        print("model: {}".format(next(iter(models))))
    else:
        print("WARNING: this file holds rows from more than one model, so the")
        print("         headline numbers below mix them. Split it first.")
        for m, n in sorted(models.items(), key=lambda kv: -kv[1]):
            print("           {:<28} {} rows".format(m, n))

    rows, verify_jobs = [], []

    for p in preds:
        task, prop = p.get("task"), p.get("property")
        rec = recs.get(task)
        gt = truth.get((task, prop))
        preds_list = p.get("predictions") or []
        parsed = preds_list[0].get("parsed") if preds_list else None

        row = {
            "task": task, "property": prop,
            "verdict": p.get("verdict"), "category": p.get("category"),
            "n_loops": p.get("n_loops"), "config": p.get("config", ""),
            "mode": p.get("mode", "") or p.get("input_mode", ""),
            # carried from the prediction row so a score CSV can always be
            # attributed to the model that produced it
            "model": model_name(p), "provider": provider_name(p),
            "pred_kind": "", "pred_k": "", "pred_schedule": "", "pred_cost": "",
            "true_outcome": "", "true_kstar": "", "true_cost": "",
            "score_a": "", "tightness": "",
            "score_b": "", "verify_detail": "", "verify_time": "",
        }

        if gt:
            row["true_outcome"] = gt.get("uniform_outcome", "")
            row["true_kstar"] = gt.get("uniform_kstar", "")
            row["true_cost"] = gt.get("schedule_cost") or gt.get("uniform_cost") or ""

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
                if pcost == tcost:
                    row["score_a"] = "exact"
                elif pcost > tcost:
                    row["score_a"] = "over"
                else:
                    row["score_a"] = "under"

        # ---- Score B: run CBMC unless the answer is already known ----
        if a.no_verify or kind == "unbounded":
            row["score_b"] = "not_run"
        elif row["score_a"] == "exact":
            row["score_b"] = "works"          # equals k*, known to work
            row["verify_detail"] = "equals k*, not re-run"
        else:
            cfile = Path(a.dataset) / rec["dir"] / rec["c_file"]
            if not cfile.exists():
                row["score_b"] = "missing_file"
            else:
                verify_jobs.append((len(rows), (
                    cfile, rec.get("data_model", "ILP32"), prop,
                    p.get("verdict"), kind, k,
                    {lid: sched.get(lid, 0) for lid in loop_ids} if sched else None,
                    a.verify_timeout)))
        rows.append(row)

    # ---- run the verifications ----
    if verify_jobs:
        print("verifying {} predictions with CBMC (--jobs {})".format(
            len(verify_jobs), a.jobs))
        t0 = time.time()
        done = 0
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            futs = {ex.submit(verify, job): idx for idx, job in verify_jobs}
            for f in as_completed(futs):
                idx = futs[f]
                t1 = time.time()
                try:
                    status, detail = f.result()
                except Exception as e:
                    status, detail = "error", str(e)[:80]
                rows[idx]["score_b"] = status
                rows[idx]["verify_detail"] = detail
                done += 1
                if done % 25 == 0 or done == len(verify_jobs):
                    print("   {}/{}  ({:.0f}m)".format(
                        done, len(verify_jobs), (time.time() - t0) / 60.0),
                        flush=True)

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
    ca = Counter(r["score_a"] for r in rows)
    for k_, v in ca.most_common():
        print("   {:<22} {:>5}  ({:.1f}%)".format(k_, v, 100.0 * v / len(rows)))

    tight = sorted(float(r["tightness"]) for r in rows if r["tightness"] != "")
    if tight:
        print("\n   tightness (pred cost / true cost), n={}".format(len(tight)))
        print("      median={:.2f}  p25={:.2f}  p75={:.2f}  max={:.1f}".format(
            tight[len(tight) // 2], tight[len(tight) // 4],
            tight[3 * len(tight) // 4], tight[-1]))
        within = lambda f: sum(1 for x in tight if 1 / f <= x <= f)
        for f in (1.5, 2.0, 10.0):
            print("      within {:>4}x of k*: {:>4} ({:.1f}%)".format(
                f, within(f), 100.0 * within(f) / len(tight)))

    print("\n" + "=" * 66)
    print("SCORE B -- does the predicted bound work?")
    print("=" * 66)
    cb = Counter(r["score_b"] for r in rows if r["score_b"])
    tot_b = sum(cb.values()) or 1
    for k_, v in cb.most_common():
        print("   {:<22} {:>5}  ({:.1f}%)".format(k_, v, 100.0 * v / tot_b))

    covered = [r for r in rows if r["score_b"] in
               ("works", "insufficient", "no_violation", "inconclusive",
                "other_property", "timeout")]
    if covered:
        works = sum(1 for r in covered if r["score_b"] == "works")
        print("\n   usable rate: {}/{} ({:.1f}%)".format(
            works, len(covered), 100.0 * works / len(covered)))
        no_gt = [r for r in covered if r["score_a"] == "no_ground_truth"]
        if no_gt:
            w2 = sum(1 for r in no_gt if r["score_b"] == "works")
            print("   of which where k* was never derived: {}/{} worked "
                  "({:.1f}%)".format(w2, len(no_gt), 100.0 * w2 / len(no_gt)))

    print("\nby loop category (Score B):")
    for cat in ("constant", "input-bounded", "data-dependent", "mixed"):
        sub = [r for r in rows if r["category"] == cat and r["score_b"]
               and r["score_b"] != "not_run"]
        if sub:
            w = sum(1 for r in sub if r["score_b"] == "works")
            print("   {:<16} {}/{} worked ({:.1f}%)".format(
                cat, w, len(sub), 100.0 * w / len(sub)))

    print("\nscores -> {}".format(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())