#!/usr/bin/env python3
"""
results_summary.py -- every number the Results section reports, per strategy,
from the final score CSVs (score_predictions.py output) and the k* table.

  python3 results_summary.py --kstar results/kstar_v2.csv \
      --score "Syntactic=results/score_syntactic_v2.csv" \
      --score "Qwen3.5=results/score_qwen397b_v2.csv"

Prints:
  ground truth        pairs, k* found (k* = 0 / > 0), no k*, by verdict
  Table I  (RQ1)      proposed, abstained, solved, solved / proposed on the
                      k* > 0 pairs, exact / over / under / mixed there
  RQ2                 on the no-k* pairs: proposed, solved, by verdict; overlap
                      between strategies
  Table III           outcome of every proposed bound
  RQ3                 pairs where both ID and the strategy solve (k* > 0):
                      ID mean / median s, strategy CBMC run mean / median s,
                      bound time mean / median s, faster (CBMC only) and
                      faster (incl. bound time); formula size of solved runs
  Table II (RQ4)      solved / proposed by loop-bound category, and solved on
                      no-k* pairs by category
"""

import argparse
import csv
import statistics
from collections import Counter

CATS = ("constant", "input-bounded", "data-dependent", "mixed")
OUTCOMES = ("works", "insufficient", "impractical", "timeout", "other_property",
            "inconclusive", "no_violation")


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def med(v):
    return statistics.median(v) if v else float("nan")


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kstar", required=True)
    ap.add_argument("--score", action="append", required=True, help="label=path")
    a = ap.parse_args()

    K = {(r["task"], r["property"]): r for r in csv.DictReader(open(a.kstar))}
    found = {k for k, r in K.items() if r["uniform_outcome"] == "found"}
    kpos = {k for k in found if (f(K[k]["uniform_kstar"]) or 0) > 0}
    nok = {k for k, r in K.items() if r["uniform_outcome"] in ("timeout", "cap_reached")}

    print("GROUND TRUTH  pairs {}  k* found {} (k*=0 {}, k*>0 {})  no k* {}  other {}".format(
        len(K), len(found), len(found) - len(kpos), len(kpos), len(nok),
        len(K) - len(found) - len(nok)))
    print("   by verdict:", dict(Counter((r["verdict"], r["uniform_outcome"]) for r in K.values())))

    solved_nok = {}
    for spec in a.score:
        label, _, path = spec.partition("=")
        S = {(r["task"], r["property"]): r for r in csv.DictReader(open(path))}
        prop = {k for k, r in S.items() if r.get("score_b") not in ("", "not_run", None)}
        works = {k for k in prop if S[k]["score_b"] == "works"}
        print("\n=== {} ===".format(label))
        print("TABLE I   proposed {}  abstained {}  solved {} ({:.1f}% of proposed)".format(
            len(prop), len(S) - len(prop), len(works), 100.0 * len(works) / max(1, len(prop))))
        p_pos = prop & kpos
        sa = Counter(S[k].get("score_a") for k in p_pos)
        print("   on k*>0 ({}): proposed {}  solved {}  exact {}  over {}  under {}  mixed {}".format(
            len(kpos), len(p_pos), len(works & kpos), sa["exact"], sa["over"], sa["under"],
            sa["mixed"]))
        print("   solved by verdict:", dict(Counter(K[k]["verdict"] for k in works if k in K)),
              " proposed by verdict:", dict(Counter(K[k]["verdict"] for k in prop if k in K)))

        p_nok, s_nok = prop & nok, works & nok
        solved_nok[label] = s_nok
        print("RQ2       no k* ({}): proposed {}  solved {}  by verdict {}".format(
            len(nok), len(p_nok), len(s_nok), dict(Counter(K[k]["verdict"] for k in s_nok))))

        oc = Counter(S[k]["score_b"] for k in prop)
        print("TABLE III", "  ".join("{} {}".format(o, oc.get(o, 0)) for o in OUTCOMES),
              "  (k* found / no k*: too small {} / {}, impractical {} / {})".format(
                  sum(1 for k in prop & found if S[k]["score_b"] == "insufficient"),
                  sum(1 for k in prop & nok if S[k]["score_b"] == "insufficient"),
                  sum(1 for k in prop & found if S[k]["score_b"] == "impractical"),
                  sum(1 for k in prop & nok if S[k]["score_b"] == "impractical")))

        both = [k for k in works & kpos
                if f(S[k].get("verify_seconds")) is not None and f(K[k].get("uniform_time")) is not None]
        idt = [f(K[k]["uniform_time"]) for k in both]
        vt = [f(S[k]["verify_seconds"]) for k in both]
        pt = [f(S[k].get("predict_seconds")) or 0.0 for k in both]
        tot = [v + p for v, p in zip(vt, pt)]
        print("RQ3       pairs {}  ID mean {:.1f} s / median {:.1f} s   CBMC run mean {:.1f} / median {:.1f}"
              "   bound mean {:.1f} / median {:.1f}   faster: CBMC {} , incl. bound {}".format(
                  len(both), mean(idt), med(idt), mean(vt), med(vt), mean(pt), med(pt),
                  sum(v < i for v, i in zip(vt, idt)), sum(t < i for t, i in zip(tot, idt))))
        for col in ("steps", "variables", "clauses"):
            v = sorted(x for x in (f(S[k].get(col)) for k in works) if x is not None)
            if v:
                print("          {:<9} n={:<5} median {:>12,.0f}  max {:>14,.0f}".format(col, len(v), med(v), v[-1]))

        print("TABLE II ", "  ".join("{} {}/{}".format(
            c, sum(1 for k in works if S[k].get("category") == c),
            sum(1 for k in prop if S[k].get("category") == c)) for c in CATS),
            "  | no-k* solved:", dict(Counter(S[k].get("category") for k in s_nok)))

    labels = list(solved_nok)
    if len(labels) >= 2:
        print("\nRQ2 overlap on no-k* pairs:")
        for i, x in enumerate(labels):
            for y in labels[i + 1:]:
                a_, b_ = solved_nok[x], solved_nok[y]
                print("   {} & {}: both {}  only {} {}  only {} {}  union {}".format(
                    x, y, len(a_ & b_), x, len(a_ - b_), y, len(b_ - a_), len(a_ | b_)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())