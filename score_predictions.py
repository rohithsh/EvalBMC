#!/usr/bin/env python3
"""
score_predictions.py  (v2)

Score predicted unrolling bounds. Verification comes first and gates everything
else, because in bounded model checking a bound either suffices or it does not
-- there is no partial credit. A prediction of 11 against a true bound of 12 is
not "92% right", it simply fails.

  STEP 1 -- DOES THE BOUND WORK?
      CBMC is run at the predicted bound and judged under the same soundness
      rule that produced the ground truth: the bound works if a property-matched
      violation is reported (unsafe program) or the property is proved (safe
      program), in both cases with NO unwinding assertion failing.

      CBMC is skipped only when the prediction is IDENTICAL to the ground-truth
      bound, where the answer is already known. Equal COST is not enough: two
      different schedules can cost the same while allocating depth differently.

  STEP 2a -- TIGHTNESS, for predictions that worked.
      tightness = predicted cost / true cost, where cost is the total number of
      loop-body unrollings (each loop's bound multiplied by the bounds of every
      enclosing loop, summed over loops). Cost puts a per-loop schedule and a
      uniform bound on the same scale. 1.0 is ideal; above 1.0 the prediction
      overshoots and the solver does more work than necessary.

      Predictions that did not work have no tightness. Reporting one would
      reward a near-miss that is useless in practice.

  STEP 2b -- MARGIN, for predictions that did not work.
      A bound that fails is almost always slightly too small. The margin asks
      what multiplier would have been enough:

        analytic   when a finite k* is known: true cost / predicted cost
        empirical  the prediction is scaled by each factor in --margins and
                   re-run through CBMC, ascending, stopping at the first that
                   works. This covers the programs where k* was never derived,
                   where the analytic margin is undefined.

      The result answers a practical question: does multiplying a predicted
      bound by a small factor turn a failing predictor into a usable one, and
      what does that cost in extra unrolling?

Ground truth comes from derive_kstar_all.py's CSV, and the CBMC invocation and
soundness rule are imported from that script so there is a single definition.

Usage:
  python3 score_predictions.py \
      --pred    results/pred_blind.jsonl \
      --kstar   results/kstar_all_v2.csv \
      --json    datasets/cleaned/loops.json \
      --dataset datasets/cleaned/svcomp_clean \
      --out     results/scores_blind.csv \
      [--jobs 8] [--verify-timeout 600] [--margins 1.5 2 4] [--no-verify]
"""

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import derive_kstar as kstar


# ------------------------------------------------------------------- helpers

def num(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def parse_schedule_field(s):
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
    kind = parsed.get("kind")
    if kind == "uniform":
        return "uniform", parsed.get("k"), None
    if kind == "schedule":
        return "schedule", None, dict(parsed.get("schedule") or {})
    return "unbounded", None, None


def normalise_schedule(sched, loop_ids):
    """Every loop named; loops the model omitted default to 0."""
    full = {lid: int(sched.get(lid, 0)) for lid in loop_ids}
    for lid, v in (sched or {}).items():
        full.setdefault(lid, int(v))
    return full


def cost_of(kind, k, sched, loop_ids, parents):
    if kind == "uniform" and k is not None:
        return kstar.uniform_cost(k, loop_ids, parents)
    if kind == "schedule" and sched:
        return kstar.loop_cost(sched, parents)
    return None


def scale(kind, k, sched, factor):
    """Scale a bound by a factor, rounding up; a zero bound becomes one."""
    if kind == "uniform":
        return max(1, int(math.ceil((k or 0) * factor))) if k is not None else None, None
    if kind == "schedule":
        return None, {lid: max(1, int(math.ceil(v * factor)))
                      for lid, v in sched.items()}
    return None, None


# ------------------------------------------------------------ CBMC verify

def run_bound(cfile, data_model, prop, verdict, kind, k, sched, timeout):
    """One CBMC run at a given bound. Returns (status, detail)."""
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
        return "insufficient", ""       # a loop was cut short
    if res == "success" and verdict == "unsafe":
        return "no_violation", ""       # sound, but the bug was not reached
    if res == "other":
        return "other_property", other
    return "inconclusive", ""


def verify_base(job):
    idx, args = job
    return idx, run_bound(*args)


def verify_margins(job):
    """
    Scale the prediction by each factor in ascending order and stop at the
    first that works. Sequential inside one worker so the early exit is real.
    """
    idx, (cfile, dm, prop, verdict, kind, k, sched, timeout, factors) = job
    tried = []
    for f in factors:
        sk, ss = scale(kind, k, sched, f)
        st, _ = run_bound(cfile, dm, prop, verdict, kind, sk, ss, timeout)
        tried.append("{}x:{}".format(f, st))
        if st == "works":
            return idx, f, st, ";".join(tried)
    return idx, None, "none_worked", ";".join(tried)


# ------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--kstar", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--verify-timeout", type=int, default=600)
    ap.add_argument("--margins", nargs="*", type=float, default=[1.5, 2.0, 4.0],
                    help="multipliers tried on predictions that did not work")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip all CBMC runs; only structural scores")
    a = ap.parse_args()

    recs = {r["task"]: r for r in json.load(open(a.json))}
    truth = {(r["task"], r["property"]): r
             for r in csv.DictReader(open(a.kstar))}

    preds = []
    with open(a.pred) as fh:
        for line in fh:
            try:
                preds.append(json.loads(line))
            except Exception:
                pass
    print("predictions: {}   ground-truth rows: {}".format(len(preds), len(truth)))

    rows, base_jobs = [], []

    for p in preds:
        task, prop = p.get("task"), p.get("property")
        rec = recs.get(task)
        gt = truth.get((task, prop))
        plist = p.get("predictions") or []
        parsed = plist[0].get("parsed") if plist else None

        row = {
            "task": task, "property": prop, "verdict": p.get("verdict"),
            "category": p.get("category"), "n_loops": p.get("n_loops"),
            "config": p.get("config", ""), "mode": p.get("mode", ""),
            "pred_kind": "", "pred_k": "", "pred_schedule": "", "pred_cost": "",
            "true_outcome": "", "true_kstar": "", "true_cost": "",
            "verified": "", "verify_detail": "",
            "tightness": "", "margin_analytic": "", "margin_empirical": "",
            "margin_detail": "", "outcome": "",
        }
        if gt:
            row["true_outcome"] = gt.get("uniform_outcome", "")
            row["true_kstar"] = gt.get("uniform_kstar", "")

        if not parsed:
            row["outcome"] = "no_prediction"
            rows.append(row)
            continue
        if rec is None:
            row["outcome"] = "unknown_task"
            rows.append(row)
            continue

        kind, k, sched = pred_bound(parsed)
        loop_ids, parents = loop_structure(rec)
        if kind == "schedule":
            sched = normalise_schedule(sched, loop_ids)

        row["pred_kind"] = kind
        if kind == "uniform":
            row["pred_k"] = k
        elif kind == "schedule":
            row["pred_schedule"] = ";".join("{}:{}".format(x, y)
                                            for x, y in sorted(sched.items()))
        pcost = cost_of(kind, k, sched, loop_ids, parents)
        if pcost is not None:
            row["pred_cost"] = pcost

        true_k = num(gt.get("uniform_kstar")) if gt else None
        has_truth = (gt is not None and gt.get("uniform_outcome") == "found"
                     and true_k is not None and true_k > 0)
        if has_truth:
            tcost = num(gt.get("schedule_cost")) or num(gt.get("uniform_cost"))
            if tcost is None:
                tcost = kstar.uniform_cost(true_k, loop_ids, parents)
            row["true_cost"] = tcost

        # abstention needs no verification
        if kind == "unbounded":
            row["outcome"] = ("abstain_unwarranted" if has_truth
                              else "abstain_warranted")
            row["verified"] = "not_run"
            rows.append(row)
            continue

        # skip CBMC only when the bound is IDENTICAL to the ground truth --
        # equal cost is not enough, since different schedules can cost the same
        identical = False
        if has_truth:
            if kind == "uniform" and k == true_k:
                identical = True
            elif kind == "schedule":
                gts = parse_schedule_field(gt.get("schedule_kstar", ""))
                if gts and sched == normalise_schedule(gts, loop_ids):
                    identical = True

        if a.no_verify:
            row["verified"] = "not_run"
        elif identical:
            row["verified"] = "works"
            row["verify_detail"] = "identical to k*, not re-run"
        else:
            cfile = Path(a.dataset) / rec["dir"] / rec["c_file"]
            if not cfile.exists():
                row["verified"] = "missing_file"
            else:
                base_jobs.append((len(rows), (
                    cfile, rec.get("data_model", "ILP32"), prop,
                    p.get("verdict"), kind, k, sched, a.verify_timeout)))
        rows.append(row)

    # ---- step 1: verify the predicted bounds -------------------------------
    if base_jobs:
        print("\nstep 1: verifying {} predicted bounds".format(len(base_jobs)))
        t0, done = time.time(), 0
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            futs = [ex.submit(verify_base, j) for j in base_jobs]
            for f in as_completed(futs):
                try:
                    idx, (status, detail) = f.result()
                except Exception as e:
                    continue
                rows[idx]["verified"] = status
                rows[idx]["verify_detail"] = detail
                done += 1
                if done % 25 == 0 or done == len(base_jobs):
                    print("   {}/{}  ({:.0f}m)".format(
                        done, len(base_jobs), (time.time() - t0) / 60.0), flush=True)

    # ---- step 2a: tightness, only where the bound worked -------------------
    margin_jobs = []
    for idx, row in enumerate(rows):
        if row["outcome"]:            # abstention or error, already settled
            continue
        v = row["verified"]
        if v == "works":
            row["outcome"] = "works"
            pc, tc = num(row["pred_cost"]), num(row["true_cost"])
            if pc is not None and tc:
                row["tightness"] = round(pc / float(tc), 3)
        elif v in ("insufficient", "no_violation", "inconclusive"):
            row["outcome"] = "failed"
            # analytic margin, when the true cost is known
            pc, tc = num(row["pred_cost"]), num(row["true_cost"])
            if pc and tc and pc < tc:
                row["margin_analytic"] = round(tc / float(pc), 3)
            # empirical margin: scale and re-run
            if not a.no_verify and a.margins:
                task, prop = row["task"], row["property"]
                rec = recs.get(task)
                if rec:
                    loop_ids, _ = loop_structure(rec)
                    kind = row["pred_kind"]
                    k = num(row["pred_k"])
                    sched = parse_schedule_field(row["pred_schedule"]) or None
                    cfile = Path(a.dataset) / rec["dir"] / rec["c_file"]
                    if cfile.exists():
                        margin_jobs.append((idx, (
                            cfile, rec.get("data_model", "ILP32"), prop,
                            row["verdict"], kind, k, sched,
                            a.verify_timeout, sorted(a.margins))))
        elif v == "timeout":
            row["outcome"] = "verify_timeout"
        elif v == "not_run":
            row["outcome"] = "not_verified"
        else:
            row["outcome"] = v or "unknown"

    # ---- step 2b: empirical margin ----------------------------------------
    if margin_jobs:
        print("\nstep 2: scaling {} failed predictions by {}".format(
            len(margin_jobs), sorted(a.margins)))
        t0, done = time.time(), 0
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            futs = [ex.submit(verify_margins, j) for j in margin_jobs]
            for f in as_completed(futs):
                try:
                    idx, factor, status, detail = f.result()
                except Exception:
                    continue
                rows[idx]["margin_empirical"] = factor if factor else ""
                rows[idx]["margin_detail"] = detail
                done += 1
                if done % 25 == 0 or done == len(margin_jobs):
                    print("   {}/{}  ({:.0f}m)".format(
                        done, len(margin_jobs), (time.time() - t0) / 60.0), flush=True)

    # ---- write ------------------------------------------------------------
    cols = list(rows[0].keys()) if rows else []
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    def summarise(rows, margins, no_verify):
        """
        Everything is reported per verdict. A safe program and an unsafe one are
        different prediction tasks: for a safe program k* is the loop's full trip
        count and predicting it means counting iterations, whereas for an unsafe
        one k* is the depth at which the violation first becomes reachable, usually
        far below the trip count. Pooling them would average two unrelated numbers.
        """
        print("\n" + "=" * 68)
        print("PREDICTIONS: {} total".format(len(rows)))
        print("=" * 68)
        for v in ("unsafe", "safe"):
            n = sum(1 for r in rows if r["verdict"] == v)
            print("   {:<8} {}".format(v, n))
        other = [r for r in rows if r["verdict"] not in ("unsafe", "safe")]
        if other:
            print("   {:<8} {}".format("(other)", len(other)))

        for v in ("unsafe", "safe"):
            sub = [r for r in rows if r["verdict"] == v]
            if not sub:
                continue

            print("\n" + "#" * 68)
            print("# {} PROGRAMS  (n={})".format(v.upper(), len(sub)))
            if v == "unsafe":
                print("#   k* is the depth at which the violation first appears")
            else:
                print("#   k* is the depth at which every loop is exhausted "
                      "and the property is proved")
            print("#" * 68)

            # ---- outcomes ----
            print("\nOUTCOME")
            for k_, c in Counter(r["outcome"] for r in sub).most_common():
                print("   {:<22} {:>5}  ({:.1f}%)".format(k_, c, 100.0 * c / len(sub)))

            scored = [r for r in sub if r["outcome"] in ("works", "failed")]
            if scored:
                w = sum(1 for r in scored if r["outcome"] == "works")
                print("   {:<22} {}/{}  ({:.1f}%)".format(
                    "usable rate", w, len(scored), 100.0 * w / len(scored)))

            # ---- abstention ----
            aw = sum(1 for r in sub if r["outcome"] == "abstain_warranted")
            au = sum(1 for r in sub if r["outcome"] == "abstain_unwarranted")
            if aw or au:
                print("\nABSTENTION")
                print("   warranted (no finite k* exists) : {}".format(aw))
                print("   unwarranted (a bound existed)   : {}".format(au))

            # ---- tightness, working predictions only ----
            tight = sorted(float(r["tightness"]) for r in sub if r["tightness"] != "")
            print("\nTIGHTNESS  (working predictions only, n={})".format(len(tight)))
            if tight:
                print("   median={:.2f}  p25={:.2f}  p75={:.2f}  max={:.1f}".format(
                    tight[len(tight) // 2], tight[len(tight) // 4],
                    tight[3 * len(tight) // 4], tight[-1]))
                for f in (1.0, 1.5, 2.0, 10.0):
                    c = sum(1 for x in tight if x <= f)
                    print("   within {:>4}x of k*: {:>4} ({:.1f}%)".format(
                        f, c, 100.0 * c / len(tight)))
            else:
                print("   no working predictions to measure")

            # ---- margin, failed predictions only ----
            failed = [r for r in sub if r["outcome"] == "failed"]
            print("\nMARGIN  (failed predictions only, n={})".format(len(failed)))
            if failed:
                ana = sorted(float(r["margin_analytic"]) for r in failed
                             if r["margin_analytic"] != "")
                if ana:
                    print("   needed cost multiplier, where k* is known (n={}):"
                          .format(len(ana)))
                    print("      median={:.2f}  p75={:.2f}  p90={:.2f}  max={:.1f}".format(
                        ana[len(ana) // 2], ana[3 * len(ana) // 4],
                        ana[min(int(0.9 * len(ana)), len(ana) - 1)], ana[-1]))
                if not no_verify and margins:
                    emp = [r for r in failed if r["margin_empirical"] != ""]
                    print("   measured by scaling the BOUND and re-running CBMC:")
                    for f in sorted(margins):
                        c = sum(1 for r in emp if float(r["margin_empirical"]) <= f)
                        print("      {}x worked for {:>4}/{}  ({:.1f}%)".format(
                            f, c, len(failed), 100.0 * c / len(failed)))
                    none = sum(1 for r in failed
                               if r["margin_detail"] and r["margin_empirical"] == "")
                    if none:
                        print("      no factor tried worked: {}".format(none))
                    print("   (scaling each bound by f raises cost by about f^depth,"
                          " so the empirical rate exceeds what the cost ratio implies)")
            else:
                print("   none")

            # ---- by loop category ----
            print("\nBY LOOP CATEGORY")
            for cat in ("constant", "input-bounded", "data-dependent", "mixed"):
                cs = [r for r in sub if r["category"] == cat
                      and r["outcome"] in ("works", "failed")]
                if cs:
                    w = sum(1 for r in cs if r["outcome"] == "works")
                    print("   {:<16} {}/{} worked ({:.1f}%)".format(
                        cat, w, len(cs), 100.0 * w / len(cs)))

            # ---- by loop count ----
            print("\nBY LOOP COUNT")
            for label, pred in (("single", lambda n: n == 1),
                                ("double", lambda n: n == 2),
                                ("multi", lambda n: n >= 3)):
                cs = [r for r in sub if r["outcome"] in ("works", "failed")
                      and str(r["n_loops"]).isdigit() and pred(int(r["n_loops"]))]
                if cs:
                    w = sum(1 for r in cs if r["outcome"] == "works")
                    print("   {:<8} {}/{} worked ({:.1f}%)".format(
                        label, w, len(cs), 100.0 * w / len(cs)))

    summarise(rows, a.margins, a.no_verify)
    print("\nscores -> {}".format(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())