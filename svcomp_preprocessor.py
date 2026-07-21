#!/usr/bin/env python3
"""
svcomp_preprocessor.py

Run the entire SV-COMP preprocessing pipeline end to end.

Given the original sv-benchmarks/c directory and a list of .set files, this
produces, in the destination:

    <dest>/svcomp_clean/     final programs used for k* derivation
    <dest>/loops.json        final classified loops (three categories)
    <dest>/stats.csv         per-directory + per-set loop/verdict statistics

All intermediate artifacts are kept under <dest>/temp/ (not deleted), so any
stage can be inspected.

Pipeline (each stage calls the corresponding standalone script as a subprocess,
exactly as run by hand during development):

    1. dedupe scaled variants        svcomp_dedupe.py
    2. remove loop-free programs      svcomp_remove_loopfree.py
    3. parse + classify loops         svcomp_loop_parser.py   (runs CBMC)
    4. drop goto-loop programs        filter_goto_loops.py
    5. build final clean dataset      build_clean_dataset.py
    6. final statistics               svcomp_stats.py

The loop-parsing stage runs CBMC on every program and is the slow part
(minutes, not seconds).

Usage:
  python3 svcomp_preprocessor.py \
      --root  datasets/sv-benchmarks/c \
      --sets  Loops.set Arrays.set Heap.set LinkedLists.set ControlFlow.set BitVectors.set \
      --dest  datasets/run1 \
      [--scripts .]        # dir holding the stage scripts (default: this dir)
      [--jobs 8] [--threshold 2]
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd, title):
    print("\n" + "=" * 72)
    print("STAGE: {}".format(title))
    print("  $ {}".format(" ".join(str(c) for c in cmd)))
    print("=" * 72, flush=True)
    r = subprocess.run([str(c) for c in cmd])
    if r.returncode != 0:
        print("\n[abort] stage '{}' failed (exit {})".format(title, r.returncode),
              file=sys.stderr)
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="original sv-benchmarks/c (holds .set files and program dirs)")
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--dest", required=True,
                    help="destination; will hold svcomp_clean/, loops.json, stats.csv")
    ap.add_argument("--scripts", default=None,
                    help="directory containing the stage scripts (default: this script's dir)")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--threshold", type=int, default=2,
                    help="dedupe: max changed lines to treat as a scaled variant")
    ap.add_argument("--python", default=sys.executable,
                    help="python interpreter to use for the stages")
    a = ap.parse_args()

    root = Path(a.root).resolve()
    dest = Path(a.dest).resolve()
    scripts = Path(a.scripts).resolve() if a.scripts else Path(__file__).resolve().parent
    py = a.python

    temp = dest / "temp"
    temp.mkdir(parents=True, exist_ok=True)

    # intermediate + final locations
    deduped_dir = temp / "svcomp_deduped"
    loopfree_marker = deduped_dir            # remove-loopfree edits this in place
    loops_full = temp / "loops_full.json"    # parser output (goto loops = 'unmatched')
    loops_clean = dest / "loops.json"        # final classified loops
    dropped_goto = temp / "dropped_goto_programs.json"
    clean_dir = dest / "svcomp_clean"        # final dataset
    stats_csv = dest / "stats.csv"

    def script(name):
        p = scripts / name
        if not p.exists():
            print("[error] stage script not found: {}".format(p), file=sys.stderr)
            sys.exit(2)
        return p

    # ---- 1. dedupe scaled variants (root -> temp/svcomp_deduped) ----
    run([py, script("svcomp_dedupe.py"),
         "--root", root, "--sets", *a.sets,
         "--dest", deduped_dir, "--threshold", a.threshold],
        "1/6  dedupe scaled variants")

    # ---- 2. remove loop-free programs (in place on the deduped copy) ----
    run([py, script("svcomp_remove_loopfree.py"),
         "--path", deduped_dir],
        "2/6  remove loop-free programs")

    # ---- 3. parse + classify loops (runs CBMC; slow) ----
    run([py, script("svcomp_loop_parser.py"),
         "--dataset", deduped_dir, "--out", loops_full, "--jobs", a.jobs],
        "3/6  parse and classify loops (runs CBMC)")

    # ---- 4. drop programs containing goto loops (loops_full -> loops.json) ----
    run([py, script("filter_goto_loops.py"),
         "--in", loops_full, "--out", loops_clean, "--removed", dropped_goto],
        "4/6  drop goto-loop programs")

    # ---- 5. build the final clean dataset from the cleaned JSON ----
    run([py, script("build_clean_dataset.py"),
         "--json", loops_clean, "--source", deduped_dir, "--dest", clean_dir],
        "5/6  build final clean dataset")

    # ---- 6. final statistics ----
    run([py, script("svcomp_stats.py"),
         "--root", root, "--dataset", clean_dir, "--sets", *a.sets,
         "--level", "both", "--out", stats_csv],
        "6/6  final statistics")

    # ---- final loop-category summary from loops.json ----
    import json
    from collections import Counter
    recs = json.load(open(loops_clean))
    cats = Counter()
    nloops = 0
    for r in recs:
        for l in r.get("loops", []):
            cats[l["category"]] += 1
            nloops += 1

    print("\n" + "#" * 72)
    print("PREPROCESSING COMPLETE")
    print("#" * 72)
    print("  final programs : {}".format(len(recs)))
    print("  final loops    : {}".format(nloops))
    print("  loop categories:")
    for k, v in cats.most_common():
        pct = (100.0 * v / nloops) if nloops else 0.0
        print("     {:<16} {:>5}  {:.1f}%".format(k, v, pct))
    print("\n  outputs:")
    print("     dataset : {}".format(clean_dir))
    print("     loops   : {}".format(loops_clean))
    print("     stats   : {}".format(stats_csv))
    print("     (intermediates kept under {})".format(temp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())