#!/usr/bin/env python3
"""
filter_goto_loops.py

Remove goto-based loops from the parser's JSON.

CBMC reports a loop wherever its GOTO program has a backward jump, including
hand-written goto spins such as:

    void myexit(int s) { _EXIT: goto _EXIT; }

libclang has no AST node for these (there is no for/while/do statement), so the
parser recorded them as category "unmatched". They cannot be classified with an
AST-based analysis, so this script strips them out.

NOTE: CBMC still knows about these loops. If you later pass per-loop bounds via
--unwindset, the ids removed here will not appear in the JSON but CBMC may still
require a bound for them. The removed ids are written to a side file so they are
not lost.

Usage:
  python3 filter_goto_loops.py \
      --in  results/loops_full.json \
      --out results/loops_clean.json \
      [--removed results/goto_loops.json] \
      [--drop-empty]        # also drop programs left with zero loops
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--removed", default=None,
                    help="write the removed goto loops here")
    ap.add_argument("--drop-empty", action="store_true",
                    help="drop programs that have no loops left")
    a = ap.parse_args()

    data = json.load(open(a.inp))

    removed_programs = []
    kept_records = []

    for rec in data:
        loops = rec.get("loops", [])
        goto = [l for l in loops if l.get("category") == "unmatched"]
        if goto:
            removed_programs.append({
                "task": rec.get("task"), "dir": rec.get("dir"),
                "n_goto_loops": len(goto),
                "n_total_loops": len(loops),
                "goto_ids": [l.get("id") for l in goto],
            })
            continue  # drop the whole program
        kept_records.append(rec)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(kept_records, fh, indent=1)

    if a.removed:
        with open(a.removed, "w") as fh:
            json.dump(removed_programs, fh, indent=1)

    print("programs in  : {}".format(len(data)))
    print("programs out : {}".format(len(kept_records)))
    print("programs dropped (contain a goto loop): {}".format(len(removed_programs)))

    if removed_programs:
        print("\ndropped per directory:")
        for k, v in Counter(r["dir"] for r in removed_programs).most_common(12):
            print("  {:<26} {}".format(k, v))

    cats = Counter()
    n = 0
    for r in kept_records:
        for l in r["loops"]:
            cats[l["category"]] += 1
            n += 1
    print("\nloops remaining: {}".format(n))
    for k, v in cats.most_common():
        print("  {:<16} {}".format(k, v))
    print("\nJSON -> {}".format(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())