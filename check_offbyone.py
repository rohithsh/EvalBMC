#!/usr/bin/env python3
"""
check_offbyone.py

Go/no-go gate on a scoring CSV, aimed at the failure that cost us the last
full run: the model answering the iteration count instead of the unwinding
depth, so every bound came back one too small.

Reads the CSV written by score_predictions.py and reports

  * counts by score_b (works / insufficient / no_violation / ...)
  * counts by score_a (exact / over / under / abstain_* / ...)
  * the usable rate, i.e. works over the rows that had both a parseable
    prediction and a ground truth to compare against
  * among the insufficient rows, the gap  true_kstar - pred_k  for uniform
    predictions.  A spike at exactly 1 means the +1 rule in the system
    prompt is not landing and the run should be stopped.
  * for schedule predictions the per-loop gap is not in this CSV, so the
    cost ratio pred_cost / true_cost is reported instead.  A cluster just
    under 1.0 is the same symptom.

Usage
-----
  python3 check_offbyone.py results/score_qwen_probe.csv
  python3 check_offbyone.py results/score_full.csv --by property --by verdict
"""

import argparse
import csv
from collections import Counter, defaultdict


def fnum(x):
    """float or None.  Never int(): a truncated 2.81 -> 2 is how the
    speedup numbers went wrong once already."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def pct(n, d):
    return "{:>5.1f}%".format(100.0 * n / d) if d else "    - "


def table(title, counter, total=None):
    total = total if total is not None else sum(counter.values())
    print("\n  {}  (n = {})".format(title, total))
    if not counter:
        print("    -")
        return
    w = max(len(str(k)) for k in counter)
    for k, v in sorted(counter.items(), key=lambda kv: -kv[1]):
        print("    {:<{w}}  {:>6}  {}".format(str(k), v, pct(v, total), w=w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvfile")
    ap.add_argument("--by", action="append", default=[],
                    help="also break the usable rate down by this column "
                         "(repeatable: property, verdict, category, n_loops)")
    a = ap.parse_args()

    with open(a.csvfile, newline="") as fh:
        rows = list(csv.DictReader(fh))

    print("check_offbyone")
    print("  {}   {} rows".format(a.csvfile, len(rows)))

    table("score_b", Counter(r.get("score_b") or "(blank)" for r in rows), len(rows))
    table("score_a", Counter(r.get("score_a") or "(blank)" for r in rows), len(rows))

    # ---- usable rate -------------------------------------------------
    # rows that were actually put to CBMC: a prediction existed and there
    # was a ground truth to compare it with
    attempted = [r for r in rows
                 if (r.get("score_b") or "") not in ("", "not_run")]
    works = [r for r in attempted if r["score_b"] == "works"]
    print("\n  usable rate   {} / {}   {}".format(
        len(works), len(attempted), pct(len(works), len(attempted))))

    for col in a.by:
        groups = defaultdict(lambda: [0, 0])
        for r in attempted:
            g = groups[r.get(col) or "(blank)"]
            g[1] += 1
            if r["score_b"] == "works":
                g[0] += 1
        print("\n  usable rate by {}".format(col))
        w = max((len(str(k)) for k in groups), default=4)
        for k, (ok, n) in sorted(groups.items(), key=lambda kv: -kv[1][1]):
            print("    {:<{w}}  {:>5} / {:<5}  {}".format(
                str(k), ok, n, pct(ok, n), w=w))

    # ---- the off-by-one probe ----------------------------------------
    insuff = [r for r in rows if r.get("score_b") == "insufficient"]
    print("\n  insufficient rows: {}".format(len(insuff)))
    if not insuff:
        print("    nothing truncated, the +1 rule is holding")
        return 0

    gaps = Counter()
    unmeasured = 0
    ratios = []
    for r in insuff:
        pk, tk = fnum(r.get("pred_k")), fnum(r.get("true_kstar"))
        if pk is not None and tk is not None and pk > 0:
            gaps[int(tk - pk)] += 1
            continue
        pc, tc = fnum(r.get("pred_cost")), fnum(r.get("true_cost"))
        if pc is not None and tc and tc > 0:
            ratios.append(pc / tc)
        else:
            unmeasured += 1

    if gaps:
        n = sum(gaps.values())
        print("\n    uniform predictions, gap = true_kstar - pred_k  (n = {})".format(n))
        for g, c in sorted(gaps.items()):
            mark = "   <-- off by one" if g == 1 else ""
            print("      {:>+6}   {:>6}  {}{}".format(g, c, pct(c, n), mark))
        one = gaps.get(1, 0)
        print("\n    off by exactly one: {} of {}  {}".format(one, n, pct(one, n)))
        if one >= 0.5 * n:
            print("    STOP: the +1 rule is not reaching the model.  Check that")
            print("    build_prompts.py SYSTEM carries the 'k is ONE MORE than the")
            print("    number of iterations' paragraph and that the prediction run")
            print("    is using the current build_prompts.py.")
        else:
            print("    the +1 rule is holding; these are genuine underestimates")

    if ratios:
        ratios.sort()
        mid = ratios[len(ratios) // 2]
        near = sum(1 for x in ratios if 0.5 <= x < 1.0)
        print("\n    schedule predictions, pred_cost / true_cost  (n = {})".format(len(ratios)))
        print("      median {:.3f}   in [0.5, 1.0): {}  {}".format(
            mid, near, pct(near, len(ratios))))
        print("      (a cluster just under 1.0 is the same off-by-one, one loop at a time)")

    if unmeasured:
        print("\n    {} insufficient rows had no comparable numbers".format(unmeasured))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())