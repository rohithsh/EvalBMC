#!/usr/bin/env python3
"""
analyse_kstar_all.py

Follow-up analysis of derive_kstar_all.py output. Answers the questions the
built-in summary leaves open:

  1. How did the SCHEDULE search fare? (the summary reports uniform only)
  2. Does scheduling succeed where the uniform bound times out?
  3. Cost ratio restricted to multi-loop programs, where it is meaningful --
     single-loop rows are 1.0 by construction and flatten the median.
  4. Where do the k*=0 programs come from, safe and unsafe?
  5. Timeout shape: stuck at low k (hard formula) vs still climbing (deep bug).

Usage:
  python3 analyse_kstar_all.py --csv results/kstar_all.csv \
      [--json datasets/cleaned/loops.json]   # for k*=0 loop detail
"""

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict


def num(x, default=None):
    try:
        return int(x)
    except (TypeError, ValueError):
        try:
            return float(x)
        except (TypeError, ValueError):
            return default


def bucket(n):
    if n is None:
        return "?"
    if n == 1:
        return "single"
    if n == 2:
        return "double"
    return "multi"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv)))
    print("rows: {}".format(len(rows)))

    multi = [r for r in rows if (num(r["n_loops"]) or 0) > 1]
    print("multi-loop rows: {}".format(len(multi)))

    # ---- 1 & 2. schedule outcomes, and schedule vs uniform ----------------
    print("\n" + "=" * 68)
    print("1. SCHEDULE SEARCH OUTCOMES  (multi-loop programs only)")
    print("=" * 68)
    oc = Counter(r["schedule_outcome"] for r in multi)
    for k, v in oc.most_common():
        print("   {:<24} {:>5}  ({:.1f}%)".format(k or "(none)", v,
                                                  100.0 * v / max(1, len(multi))))

    print("\n" + "=" * 68)
    print("2. SCHEDULE vs UNIFORM  (multi-loop programs)")
    print("=" * 68)
    cross = Counter((r["uniform_outcome"], r["schedule_outcome"]) for r in multi)
    print("   {:<18}{:<18}{:>7}".format("uniform", "schedule", "n"))
    print("   " + "-" * 43)
    for (u, s), n in cross.most_common():
        print("   {:<18}{:<18}{:>7}".format(u or "-", s or "-", n))

    rescued = [r for r in multi
               if r["uniform_outcome"] != "found" and r["schedule_outcome"] == "found"]
    lost = [r for r in multi
            if r["uniform_outcome"] == "found" and r["schedule_outcome"] != "found"]
    print("\n   scheduling succeeded where uniform failed : {}".format(len(rescued)))
    print("   uniform succeeded where scheduling failed : {}".format(len(lost)))
    if rescued:
        print("\n   examples rescued by scheduling:")
        for r in rescued[:10]:
            print("      {:<40} {:<10} loops={:<3} sched={}".format(
                r["task"][:40], r["property"], r["n_loops"],
                (r["schedule_kstar"] or "")[:50]))

    # ---- 3. cost ratio, multi-loop only ----------------------------------
    print("\n" + "=" * 68)
    print("3. COST RATIO  uniform / schedule   (multi-loop, both searches found)")
    print("=" * 68)
    both = [r for r in multi
            if r["uniform_outcome"] == "found" and r["schedule_outcome"] == "found"
            and num(r["cost_ratio"]) is not None]
    ratios = sorted(num(r["cost_ratio"]) for r in both)
    if ratios:
        print("   n={}  min={:.2f}  median={:.2f}  mean={:.1f}  max={:.1f}".format(
            len(ratios), ratios[0], statistics.median(ratios),
            statistics.mean(ratios), ratios[-1]))
        for thr in (1.0, 2.0, 10.0, 100.0):
            n = sum(1 for x in ratios if x > thr)
            print("      ratio > {:<6} : {:>4}  ({:.1f}%)".format(
                thr, n, 100.0 * n / len(ratios)))
        print("\n   largest savings:")
        for r in sorted(both, key=lambda x: -num(x["cost_ratio"]))[:10]:
            print("      {:<38} loops={:<3} uniform={:<10} sched={:<10} x{:.0f}".format(
                r["task"][:38], r["n_loops"], r["uniform_cost"],
                r["schedule_cost"], num(r["cost_ratio"])))
    else:
        print("   no rows where both searches found a bound")

    # ---- 4. k*=0 ----------------------------------------------------------
    print("\n" + "=" * 68)
    print("4. k*=0  (violation/proof needs no loop iteration)")
    print("=" * 68)
    zero = [r for r in rows
            if r["uniform_outcome"] == "found" and num(r["uniform_kstar"]) == 0]
    print("   total: {}".format(len(zero)))
    for kind in ("unsafe", "safe"):
        sub = [r for r in zero if r["verdict"] == kind]
        print("   {:<8} {}".format(kind, len(sub)))
        if sub:
            print("      by property : {}".format(
                dict(Counter(r["property"] for r in sub))))
            print("      by category : {}".format(
                dict(Counter(r["category"] for r in sub))))
            print("      by loops    : {}".format(
                dict(Counter(bucket(num(r["n_loops"])) for r in sub))))
    if zero:
        print("\n   top directories:")
        for d, n in Counter(r["dir"] for r in zero).most_common(8):
            print("      {:<28} {}".format(d, n))

    # ---- 5. timeout shape --------------------------------------------------
    print("\n" + "=" * 68)
    print("5. TIMEOUT SHAPE  (uniform search)")
    print("=" * 68)
    to = [r for r in rows if r["uniform_outcome"] == "timeout"]
    print("   timeouts: {}".format(len(to)))
    depth = Counter()
    for r in to:
        k = num(r["uniform_k_reached"]) or 0
        if k <= 2:
            depth["stuck at k<=2 (hard formula)"] += 1
        elif k <= 20:
            depth["k 3-20"] += 1
        elif k <= 100:
            depth["k 21-100"] += 1
        elif k <= 1000:
            depth["k 101-1000"] += 1
        else:
            depth["k >1000 (deep, still climbing)"] += 1
    for k, v in depth.most_common():
        print("   {:<34} {:>5}  ({:.1f}%)".format(k, v, 100.0 * v / max(1, len(to))))

    print("\n   by loop count:")
    for lb in ("single", "double", "multi"):
        sub = [r for r in rows if bucket(num(r["n_loops"])) == lb]
        t = sum(1 for r in sub if r["uniform_outcome"] == "timeout")
        if sub:
            print("      {:<8} {}/{} timed out ({:.1f}%)".format(
                lb, t, len(sub), 100.0 * t / len(sub)))

    # ---- usable population -------------------------------------------------
    print("\n" + "=" * 68)
    print("USABLE POPULATION FOR RQ1  (finite k* > 0)")
    print("=" * 68)
    usable = [r for r in rows
              if r["uniform_outcome"] == "found" and (num(r["uniform_kstar"]) or 0) > 0]
    print("   total: {}".format(len(usable)))
    print("   by verdict  : {}".format(dict(Counter(r["verdict"] for r in usable))))
    print("   by property : {}".format(dict(Counter(r["property"] for r in usable))))
    print("   by category : {}".format(dict(Counter(r["category"] for r in usable))))
    print("   by loops    : {}".format(
        dict(Counter(bucket(num(r["n_loops"])) for r in usable))))
    ks = sorted(num(r["uniform_kstar"]) for r in usable)
    if ks:
        qs = [ks[int(len(ks) * p)] for p in (0.25, 0.5, 0.75, 0.9)]
        print("   k* quartiles: 25%={} 50%={} 75%={} 90%={}  max={}".format(
            qs[0], qs[1], qs[2], qs[3], ks[-1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())