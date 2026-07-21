#!/usr/bin/env python3
"""
svcomp_dedupe.py

Build a deduplicated SV-COMP dataset by removing  VARIANTS (the same
program saved at different sizes) while keeping genuinely different programs.

Method:
  1. Take one or more .set files; resolve their globs to the .yml and .c files.
  2. Group .c files by stripped base name.
  3. Within each group, cluster files that are near-identical: two files are
     linked if they differ by <= THRESHOLD changed lines (default 2), clustered
     transitively. Files not linked to anything are singletons.
  4. Keep rule:
       - singleton (not a variant of anything)        -> keep
       - cluster of 2 near-identical files            -> keep both
       - cluster of 3+ near-identical files           -> keep smallest, middle,
                                                         and largest by the number
                                                         embedded in the filename
  5. Copy every KEPT task (both .c and .yml, plus any extra input files) to
     <dest>, preserving the directory name (dest/<dir>/<file>).

Usage:
  python3 svcomp_dedupe.py \
      --root datasets/sv-benchmarks/c \
      --sets Loops.set Arrays.set Heap.set LinkedLists.set ControlFlow.set BitVectors.set \
      --dest datasets/svcomp_dedup \
      [--threshold 2] [--dry-run]
"""

import argparse
import csv
import difflib
import re
import shutil
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

try:
    import yaml
except ImportError:
    print("need pyyaml:  pip install pyyaml", file=sys.stderr)
    raise

TRAIL_NUM = re.compile(r'(?:[-_])?\d+$')
LAST_INT = re.compile(r'(\d+)(?!.*\d)')   # the last integer anywhere in a string


def base_name(filename):
    stem = filename[:-2] if filename.endswith(".c") else filename
    base = TRAIL_NUM.sub("", stem)
    return base.rstrip("-_") or stem


def sort_key_number(filename):
    """Last integer in the stem; 0 if none (used to order variants by size)."""
    stem = filename[:-2] if filename.endswith(".c") else filename
    m = LAST_INT.search(stem)
    return int(m.group(1)) if m else 0


def changed_lines(a_lines, b_lines):
    diff = difflib.unified_diff(a_lines, b_lines, lineterm="")
    add = dele = 0
    for ln in diff:
        if ln.startswith("+") and not ln.startswith("+++"):
            add += 1
        elif ln.startswith("-") and not ln.startswith("---"):
            dele += 1
    return add + dele


def read_set_file(path):
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def input_files_of(yml_path):
    """Return the .c file names a .yml points at (list)."""
    try:
        d = yaml.safe_load(yml_path.read_text())
    except Exception:
        return []
    if not isinstance(d, dict):
        return []
    inp = d.get("input_files", "")
    names = inp if isinstance(inp, list) else [inp]
    return [str(n) for n in names if n]


def cluster(files_lines, threshold):
    """
    files_lines: list of (name, lines). Return list of clusters (each a list of
    names) where two files are in the same cluster if they are within
    'threshold' changed lines, transitively.
    """
    names = [n for n, _ in files_lines]
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for (n1, l1), (n2, l2) in combinations(files_lines, 2):
        if changed_lines(l1, l2) <= threshold:
            union(n1, n2)

    clusters = defaultdict(list)
    for n in names:
        clusters[find(n)].append(n)
    return list(clusters.values())


def choose_keep(cluster_names):
    """Apply the keep rule to one cluster of near-identical files."""
    if len(cluster_names) <= 2:
        return list(cluster_names)                # keep 1 or 2
    ordered = sorted(cluster_names, key=sort_key_number)
    lo = ordered[0]
    hi = ordered[-1]
    mid = ordered[len(ordered) // 2]
    return list(dict.fromkeys([lo, mid, hi]))     # smallest / middle / largest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--threshold", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be kept/dropped, copy nothing")
    a = ap.parse_args()

    root = Path(a.root)
    dest = Path(a.dest)

    # resolve tasks: dir -> list of (yml_path, primary_c_name)
    dir_files = defaultdict(list)          # dir -> [(yml_path, c_name)]
    for sname in a.sets:
        sp = root / sname
        if not sp.exists():
            print("[warn] missing set: {}".format(sp), file=sys.stderr)
            continue
        for pat in read_set_file(sp):
            for y in sorted(root.glob(pat)):
                inputs = input_files_of(y)
                primary = inputs[0] if inputs else ""
                dir_files[y.parent.name].append((y, primary, inputs))

    kept, dropped = [], []

    for dirname, entries in sorted(dir_files.items()):
        # group by base name
        groups = defaultdict(list)         # base -> [(yml, primary, inputs)]
        for e in entries:
            groups[base_name(e[1])].append(e)

        for base, members in groups.items():
            if len(members) < 2:
                kept.extend(members)
                continue
            # load primary .c contents for clustering
            loaded = []
            path_of = {}
            for (yml, primary, inputs) in members:
                cpath = yml.parent / primary
                path_of[primary] = (yml, primary, inputs)
                if cpath.exists():
                    try:
                        loaded.append((primary, cpath.read_text(errors="replace").splitlines()))
                    except Exception:
                        loaded.append((primary, []))
                else:
                    loaded.append((primary, []))

            clusters = cluster(loaded, a.threshold)
            for cl in clusters:
                if len(cl) == 1:
                    kept.append(path_of[cl[0]])
                    continue
                keepset = set(choose_keep(cl))
                for name in cl:
                    if name in keepset:
                        kept.append(path_of[name])
                    else:
                        dropped.append(path_of[name])

    # ---- report ----
    print("resolved tasks: {}".format(sum(len(v) for v in dir_files.values())))
    print("kept:    {}".format(len(kept)))
    print("dropped: {}".format(len(dropped)))

    # per-dir drop summary
    ddrop = defaultdict(int)
    for (yml, _, _) in dropped:
        ddrop[yml.parent.name] += 1
    if ddrop:
        print("\ndropped per directory:")
        for d in sorted(ddrop, key=lambda x: -ddrop[x]):
            print("  {:<26} {}".format(d, ddrop[d]))

    if a.dry_run:
        print("\n[dry-run] nothing copied.")
        return 0

    # ---- copy kept tasks ----
    n_copied = 0
    for (yml, primary, inputs) in kept:
        out_dir = dest / yml.parent.name
        out_dir.mkdir(parents=True, exist_ok=True)
        # copy the .yml
        shutil.copy2(yml, out_dir / yml.name)
        # copy each input .c (usually one)
        for nm in (inputs or [primary]):
            src = yml.parent / nm
            if src.exists():
                shutil.copy2(src, out_dir / nm)
        n_copied += 1

    print("\ncopied {} tasks -> {}".format(n_copied, dest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())