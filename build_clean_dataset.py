#!/usr/bin/env python3
"""
build_clean_dataset.py

Copy only the programs listed in the cleaned loops JSON into a new directory,
so the dataset on disk matches the JSON exactly.

The JSON's "task" field is "<dir>/<yml>", and each record names its .c file.
Both are copied, preserving the directory name:  <dest>/<dir>/<file>

Nothing in the source is modified or deleted.

Usage:
  python3 build_clean_dataset.py \
      --json   results/loops_clean.json \
      --source datasets/svcomp_deduped \
      --dest   datasets/svcomp_clean \
      [--dry-run]
"""

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

try:
    import yaml
except ImportError:
    print("need pyyaml: pip install pyyaml", file=sys.stderr)
    raise


def input_files_of(yml_path):
    """Every .c the task points at (usually one)."""
    try:
        d = yaml.safe_load(yml_path.read_text())
    except Exception:
        return []
    if not isinstance(d, dict):
        return []
    inp = d.get("input_files", "")
    names = inp if isinstance(inp, list) else [inp]
    return [str(n) for n in names if n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="cleaned loops JSON")
    ap.add_argument("--source", required=True, help="dataset to copy from")
    ap.add_argument("--dest", required=True, help="new dataset directory")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    src = Path(a.source)
    dest = Path(a.dest)
    records = json.load(open(a.json))

    copied, missing = [], []
    per_dir = Counter()

    for rec in records:
        task = rec.get("task", "")
        if "/" not in task:
            missing.append((task, "malformed task"))
            continue
        dirname, ymlname = task.split("/", 1)
        yml_src = src / dirname / ymlname
        if not yml_src.exists():
            missing.append((task, "yml not found"))
            continue

        # the .c files come from the yml, not the record, so extras are covered
        inputs = input_files_of(yml_src)
        if not inputs:
            inputs = [rec.get("c_file", "")]

        files = [yml_src] + [src / dirname / n for n in inputs if n]
        absent = [str(f) for f in files if not f.exists()]
        if absent:
            missing.append((task, "missing: {}".format(", ".join(absent))))
            continue

        if not a.dry_run:
            out_dir = dest / dirname
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, out_dir / f.name)

        copied.append(task)
        per_dir[dirname] += 1

    print("records in JSON : {}".format(len(records)))
    print("programs copied : {}".format(len(copied)))
    if missing:
        print("problems        : {}".format(len(missing)))
        for t, why in missing[:10]:
            print("   {:<44} {}".format(t, why))

    print("\nper directory:")
    for d in sorted(per_dir):
        print("  {:<28} {}".format(d, per_dir[d]))

    if a.dry_run:
        print("\n[dry-run] nothing copied.")
    else:
        print("\ncopied -> {}".format(dest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())