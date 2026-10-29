#!/usr/bin/env python3
"""
conformal_planning.py

Detailed breakdown of the ground-truth population, laid out for deciding how
conformal calibration can be applied.

WHAT CONFORMAL NEEDS HERE
  The loss is one-sided: a bound below k* fails outright, a bound above it only
  costs solver time. So the natural construction is a one-sided multiplier.
  Calibrate lambda on held-out programs such that

      k_pred * lambda  >=  k*

  holds for at least (1 - alpha) of them, and the same guarantee then transfers
  to new programs. The nonconformity score is the ratio k* / k_pred, and lambda
  is its (1 - alpha) quantile.

  Three things decide whether this is feasible, and this script reports all
  three.

  1. CALIBRATION SIZE. Split conformal at level alpha needs at least
         n >= 1/alpha - 1
     calibration points, because lambda is the ceil((n+1)(1-alpha))-th smallest
     score and that index must exist. 95% coverage needs n >= 19, 90% needs
     n >= 9. Below that the quantile is undefined and no guarantee is available.

     Note this counts PREDICTIONS, not programs: a stratum with 300 programs is
     no use until the strategy has been run on enough of them.

  2. CONDITIONAL vs MARGINAL COVERAGE. Conformal gives marginal coverage: 95%
     overall does not mean 95% within each category. Since the categories here
     behave very differently, a single lambda would over-cover the easy strata
     and under-cover the hard ones. Per-stratum calibration fixes that but
     needs the size above in EVERY stratum, which is the real constraint.

  3. EXCHANGEABILITY. Calibration and test points must be exchangeable. This
     dataset is heavily clustered: several directories hold many near-identical
     programs, so a random split puts variants of the same program on both
     sides and leaks. The report therefore includes, per stratum, how many
     distinct directories and base program families it draws on, and what share
     the largest family holds. Splitting must be done by family, not by row.

Usage:
  python3 conformal_planning.py --kstar results/kstar_all_v2.csv \
      [--alphas 0.1 0.05] [--pred results/pred_llm_v2.jsonl]
"""

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict

PROPS = ["reach", "memsafety", "overflow"]
CATS = ["constant", "input-bounded", "data-dependent", "mixed"]
VERDICTS = ["unsafe", "safe"]

TRAIL_NUM = re.compile(r'(?:[-_])?\d+$')


def num(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def family(task):
    """
    Programs that are variants of one another share a family. Used to check
    clustering: a random calibration/test split that separates rows rather than
    families would leak, because variants are near-identical.
    """
    d, _, f = task.partition("/")
    stem = f[:-4] if f.endswith(".yml") else f
    base = TRAIL_NUM.sub("", stem).rstrip("-_")
    return "{}/{}".format(d, base or stem)


def min_n_for(alpha):
    """Smallest calibration set that admits a (1-alpha) quantile."""
    return int(math.ceil(1.0 / alpha)) - 1


def quantile(sorted_vals, q):
    if not sorted_vals:
        return None
    i = min(int(math.ceil(q * len(sorted_vals))) - 1, len(sorted_vals) - 1)
    return sorted_vals[max(i, 0)]


def describe(vals):
    if not vals:
        return "-"
    v = sorted(vals)
    return "min={} p25={} med={} p75={} p90={} max={}".format(
        v[0], quantile(v, .25), quantile(v, .5),
        quantile(v, .75), quantile(v, .9), v[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kstar", required=True)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.10, 0.05])
    ap.add_argument("--pred", default=None,
                    help="predictions JSONL; adds the observed ratio "
                         "distribution where predictions exist")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.kstar)))

    usable, zero, timeout = [], [], []
    for r in rows:
        k = num(r.get("uniform_kstar"))
        if r.get("uniform_outcome") == "found" and k is not None:
            (usable if k > 0 else zero).append(r)
        else:
            timeout.append(r)

    print("=" * 74)
    print("POPULATION")
    print("=" * 74)
    print("   total program-property pairs : {}".format(len(rows)))
    print("   finite k* > 0  (calibratable): {}".format(len(usable)))
    print("   k* = 0  (no bound to predict): {}".format(len(zero)))
    print("   no k* (timed out / capped)   : {}".format(len(timeout)))
    print("\n   Only the finite k* > 0 rows can serve as calibration points:")
    print("   a conformal ratio k*/k_pred needs a k*.")

    # ---------------------------------------------------- verdict x property
    print("\n" + "=" * 74)
    print("VERDICT x PROPERTY   (finite k* > 0)")
    print("=" * 74)
    print("   {:<10}{:>12}{:>12}{:>12}{:>10}".format(
        "", *PROPS, "total"))
    for v in VERDICTS:
        cells = [sum(1 for r in usable
                     if r["verdict"] == v and r["property"] == p) for p in PROPS]
        print("   {:<10}{:>12}{:>12}{:>12}{:>10}".format(v, *cells, sum(cells)))
    cells = [sum(1 for r in usable if r["property"] == p) for p in PROPS]
    print("   {:<10}{:>12}{:>12}{:>12}{:>10}".format("total", *cells, sum(cells)))

    # ---------------------------------------------------- verdict x category
    print("\n" + "=" * 74)
    print("VERDICT x LOOP CATEGORY   (finite k* > 0)")
    print("=" * 74)
    print("   {:<10}" + "")
    hdr = "   {:<10}" + "".join("{:>17}" for _ in CATS) + "{:>9}"
    print(hdr.format("", *CATS, "total"))
    for v in VERDICTS:
        cells = [sum(1 for r in usable
                     if r["verdict"] == v and (r["category"] or "mixed") == c)
                 for c in CATS]
        print(hdr.format(v, *cells, sum(cells)))
    cells = [sum(1 for r in usable if (r["category"] or "mixed") == c) for c in CATS]
    print(hdr.format("total", *cells, sum(cells)))

    # ---------------------------------------- full stratum table with k* spread
    print("\n" + "=" * 74)
    print("STRATA IN FULL   (verdict x property x category)")
    print("=" * 74)
    print("   {:<8}{:<11}{:<16}{:>6}{:>7}{:>8}   {}".format(
        "verdict", "property", "category", "n", "fams", "maxfam", "k* spread"))
    print("   " + "-" * 88)

    strata = defaultdict(list)
    for r in usable:
        strata[(r["verdict"], r["property"], r["category"] or "mixed")].append(r)

    for v in VERDICTS:
        for p in PROPS:
            for c in CATS:
                sub = strata.get((v, p, c), [])
                if not sub:
                    continue
                ks = [num(r["uniform_kstar"]) for r in sub]
                fams = Counter(family(r["task"]) for r in sub)
                maxshare = 100.0 * max(fams.values()) / len(sub)
                print("   {:<8}{:<11}{:<16}{:>6}{:>7}{:>7.0f}%   {}".format(
                    v, p, c, len(sub), len(fams), maxshare, describe(ks)))

    # ------------------------------------------------ calibration feasibility
    print("\n" + "=" * 74)
    print("CALIBRATION FEASIBILITY")
    print("=" * 74)
    for alpha in sorted(a.alphas, reverse=True):
        need = min_n_for(alpha)
        cov = int(round((1 - alpha) * 100))
        print("\n   {}% coverage needs n >= {} calibration points per stratum"
              .format(cov, need))

        # a calibration/test split needs enough on BOTH sides
        for label, factor in (("calibrate on all rows", 1.0),
                              ("50/50 calibrate/test split", 0.5)):
            ok_v, ok_vp, ok_vpc = [], [], []
            for v in VERDICTS:
                n = sum(1 for r in usable if r["verdict"] == v)
                if n * factor >= need:
                    ok_v.append(v)
                for p in PROPS:
                    n = sum(1 for r in usable
                            if r["verdict"] == v and r["property"] == p)
                    if n * factor >= need:
                        ok_vp.append((v, p))
                    for c in CATS:
                        n = len(strata.get((v, p, c), []))
                        if n * factor >= need:
                            ok_vpc.append((v, p, c))
            total_vp = len([1 for v in VERDICTS for p in PROPS
                            if any(r["verdict"] == v and r["property"] == p
                                   for r in usable)])
            total_vpc = len([k for k, s in strata.items() if s])
            print("      {}:".format(label))
            print("         by verdict              : {}/2 strata".format(len(ok_v)))
            print("         by verdict x property   : {}/{} strata".format(
                len(ok_vp), total_vp))
            print("         by verdict x prop x cat : {}/{} strata".format(
                len(ok_vpc), total_vpc))

    print("\n   These counts are PROGRAMS available. A calibration point also")
    print("   requires a prediction for that program, so the strategy must be")
    print("   run on at least this many in every stratum before a per-stratum")
    print("   lambda can be computed.")

    # ------------------------------------------------------------- clustering
    print("\n" + "=" * 74)
    print("CLUSTERING   (exchangeability check)")
    print("=" * 74)
    fams = Counter(family(r["task"]) for r in usable)
    dirs = Counter(r["dir"] for r in usable)
    print("   distinct families   : {} across {} rows".format(len(fams), len(usable)))
    print("   distinct directories: {}".format(len(dirs)))
    big = [(f, n) for f, n in fams.most_common(8)]
    print("\n   largest families:")
    for f, n in big:
        print("      {:<46} {:>4}  ({:.1f}%)".format(f, n, 100.0 * n / len(usable)))
    print("\n   largest directories:")
    for d, n in dirs.most_common(8):
        print("      {:<46} {:>4}  ({:.1f}%)".format(d, n, 100.0 * n / len(usable)))
    multi = sum(n for f, n in fams.items() if n > 1)
    print("\n   rows belonging to a family with siblings: {} ({:.1f}%)".format(
        multi, 100.0 * multi / len(usable)))
    print("   A random row-level split would place siblings on both sides of")
    print("   the calibration/test divide and inflate coverage. Split by family.")

    # --------------------------------------------- observed ratios, if we have
    if a.pred:
        preds = {}
        for line in open(a.pred):
            try:
                d = json.loads(line)
            except Exception:
                continue
            pl = d.get("predictions") or []
            if pl and pl[0].get("parsed"):
                preds[(d["task"], d["property"])] = pl[0]["parsed"]

        truth = {(r["task"], r["property"]): r for r in usable}
        ratios = defaultdict(list)
        for key, p in preds.items():
            gt = truth.get(key)
            if not gt or p.get("kind") != "uniform":
                continue
            k = p.get("k")
            ks = num(gt["uniform_kstar"])
            if k and ks:
                ratios[(gt["verdict"], gt["category"] or "mixed")].append(ks / float(k))

        if ratios:
            print("\n" + "=" * 74)
            print("OBSERVED NONCONFORMITY   k* / k_pred, uniform answers only")
            print("=" * 74)
            print("   A lambda at the (1-alpha) quantile of these ratios is the")
            print("   multiplier that would have covered that fraction.\n")
            print("   {:<8}{:<16}{:>6}{:>9}{:>9}{:>9}{:>9}".format(
                "verdict", "category", "n", "med", "p75", "p90", "p95"))
            for (v, c), vals in sorted(ratios.items()):
                s = sorted(vals)
                print("   {:<8}{:<16}{:>6}{:>9.2f}{:>9.2f}{:>9.2f}{:>9.2f}".format(
                    v, c, len(s), quantile(s, .5), quantile(s, .75),
                    quantile(s, .9), quantile(s, .95)))
            allv = sorted(x for vs in ratios.values() for x in vs)
            print("   {:<8}{:<16}{:>6}{:>9.2f}{:>9.2f}{:>9.2f}{:>9.2f}".format(
                "ALL", "", len(allv), quantile(allv, .5), quantile(allv, .75),
                quantile(allv, .9), quantile(allv, .95)))
            under = sum(1 for x in allv if x > 1.0)
            print("\n   predictions below k* (lambda > 1 needed): {}/{} ({:.1f}%)"
                  .format(under, len(allv), 100.0 * under / len(allv)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())