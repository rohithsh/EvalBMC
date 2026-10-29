#!/usr/bin/env python3
"""
inspect_abstentions.py

Look at the cases where the model declined to give a bound.

Two kinds:

  warranted    no finite k* was derived, so declining may be right -- but read
               the caveat below.
  unwarranted  a finite k* exists and the model declined anyway. These are
               plain misses and the interesting group.

IMPORTANT CAVEAT ON "WARRANTED"
  A missing k* means one of two different things, and the scorer currently
  treats them alike:

    (a) no finite bound exists -- a loop's trip count is an unconstrained
        input, so no depth ever discharges the unwinding assertion. Declining
        is correct.
    (b) the search ran out of time. A finite bound may well exist; CBMC simply
        did not reach it within the budget. Declining is UNVERIFIED, not right.

  Only (a) makes an abstention genuinely warranted. This script separates them
  using the k* outcome and the depth reached at timeout: a search that timed out
  while still climbing is evidence a bound exists further out, whereas an
  input-bounded loop that never discharges its unwinding assertion at any depth
  is evidence none does. The loop category from the classifier is reported
  alongside, since input-bounded is the category where no finite bound is
  expected.

For each case the script prints the model's own reasoning, the loop guards, and
the ground truth, so the two can be compared directly.

Usage:
  python3 inspect_abstentions.py \
      --pred   results/pred_llm_v2.jsonl \
      --scores results/scores_llm_v2.csv \
      --kstar  results/kstar_all_v2.csv \
      --json   datasets/cleaned/loops.json \
      [--kind unwarranted|warranted|both] [--show 15] [--full]
"""

import argparse
import csv
import json
from collections import Counter


def num(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--kstar", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--kind", choices=["unwarranted", "warranted", "both"],
                    default="both")
    ap.add_argument("--show", type=int, default=15)
    ap.add_argument("--full", action="store_true",
                    help="print full reasoning rather than a single line")
    a = ap.parse_args()

    recs = {r["task"]: r for r in json.load(open(a.json))}
    truth = {(r["task"], r["property"]): r
             for r in csv.DictReader(open(a.kstar))}
    scores = {(r["task"], r["property"]): r
              for r in csv.DictReader(open(a.scores))}

    reasoning = {}
    for line in open(a.pred):
        try:
            d = json.loads(line)
        except Exception:
            continue
        preds = d.get("predictions") or []
        if preds and preds[0].get("parsed"):
            reasoning[(d["task"], d["property"])] = preds[0]["parsed"]

    groups = {"unwarranted": [], "warranted": []}
    for key, s in scores.items():
        oc = s.get("outcome", "")
        if oc == "abstain_unwarranted":
            groups["unwarranted"].append(key)
        elif oc == "abstain_warranted":
            groups["warranted"].append(key)

    print("abstentions: {} unwarranted, {} warranted".format(
        len(groups["unwarranted"]), len(groups["warranted"])))

    # ---- how warranted are the "warranted" ones? --------------------------
    if groups["warranted"]:
        print("\n" + "=" * 70)
        print("WARRANTED abstentions -- is declining actually correct?")
        print("=" * 70)
        cats, outcomes, reached = Counter(), Counter(), []
        for key in groups["warranted"]:
            gt = truth.get(key, {})
            s = scores.get(key, {})
            outcomes[gt.get("uniform_outcome", "?")] += 1
            cats[s.get("category", "?")] += 1
            kr = num(gt.get("uniform_k_reached"))
            if kr is not None:
                reached.append(kr)

        print("\n   k* outcome:")
        for k, v in outcomes.most_common():
            print("      {:<16} {}".format(k, v))
        print("\n   loop category (input-bounded is where no finite bound is "
              "expected):")
        for k, v in cats.most_common():
            print("      {:<16} {}".format(k, v))
        if reached:
            reached.sort()
            print("\n   depth reached before the search stopped:")
            print("      min={} median={} max={}".format(
                reached[0], reached[len(reached) // 2], reached[-1]))
            deep = sum(1 for x in reached if x > 20)
            print("      still climbing past k=20: {}/{}  -- for these a finite "
                  "bound probably does exist,".format(deep, len(reached)))
            print("      so the abstention is unverified rather than correct")

    # ---- the detail ------------------------------------------------------
    order = (["unwarranted", "warranted"] if a.kind == "both" else [a.kind])
    for kind in order:
        keys = groups[kind]
        if not keys:
            continue
        print("\n" + "=" * 70)
        print("{} ({} cases){}".format(
            kind.upper(), len(keys),
            "" if len(keys) <= a.show else "  -- showing {}".format(a.show)))
        print("=" * 70)

        # unwarranted: show the largest k* first, since a big bound is the
        # easiest kind of program to mistake for unbounded
        keys.sort(key=lambda k: -(num(truth.get(k, {}).get("uniform_kstar")) or 0))

        for key in keys[:a.show]:
            task, prop = key
            s = scores.get(key, {})
            gt = truth.get(key, {})
            rec = recs.get(task, {})
            p = reasoning.get(key, {})

            print("\n{}   [{}]  {}".format(task, prop, s.get("verdict", "?")))
            print("   category   : {}   loops: {}".format(
                s.get("category", "?"), s.get("n_loops", "?")))
            if kind == "unwarranted":
                print("   true k*    : {}   (cost {})".format(
                    gt.get("uniform_kstar", "?"), gt.get("uniform_cost", "?")))
            else:
                print("   k* outcome : {}   reached k={}".format(
                    gt.get("uniform_outcome", "?"),
                    gt.get("uniform_k_reached", "?")))

            for l in rec.get("loops", [])[:4]:
                con = ""
                if l.get("bound_constrained"):
                    con = "  [capped: {}]".format((l.get("constraint") or "")[:40])
                print("   loop {:<12} line {:<5} {:<30} bound={}{}".format(
                    l.get("id", "?"), l.get("line", "?"),
                    (l.get("guard") or "")[:30],
                    l.get("bound_expr") or "-", con))
            if len(rec.get("loops", [])) > 4:
                print("   ... {} more loops".format(len(rec["loops"]) - 4))

            r = p.get("reasoning") or "(none)"
            if not a.full:
                r = r[:200] + ("..." if len(r) > 200 else "")
            print("   model says : {}".format(r))
            if p.get("confidence"):
                print("   confidence : {}".format(p["confidence"]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())