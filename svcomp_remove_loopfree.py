#!/usr/bin/env python3
"""
svcomp_remove_loopfree.py

Remove every program that contains no loop from a dataset directory.
A program with no loop has no unrolling bound to predict, so it is irrelevant
to this study.

Works IN PLACE on the given path. For each task it deletes both the .yml and
its input .c file(s) when the source contains no for/while/do loop.

Pipeline order:  svcomp_dedupe.py  ->  svcomp_remove_loopfree.py  ->  svcomp_stats.py

Usage:
  python3 svcomp_remove_loopfree.py --path datasets/svcomp_deduped [--dry-run]
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

try:
    import yaml
except ImportError:
    print("need pyyaml:  pip install pyyaml", file=sys.stderr)
    raise

_CLEAN_RE = re.compile(
    r'/\*.*?\*/'            # block comment
    r'|//[^\n]*'            # line comment
    r'|"(?:\\.|[^"\\])*"'   # string literal
    r"|'(?:\\.|[^'\\])*'",  # char literal
    re.DOTALL,
)


def count_loops(src):
    """for + genuine while + do, correcting do-while double counting."""
    clean = _CLEAN_RE.sub(" ", src)
    n_for = len(re.findall(r"\bfor\b", clean))
    n_do = len(re.findall(r"\bdo\b", clean))
    n_while = len(re.findall(r"\bwhile\b", clean))
    return n_for + max(0, n_while - n_do) + n_do


def input_files_of(yml_path):
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
    ap.add_argument("--path", required=True,
                    help="dataset directory to clean, e.g. datasets/svcomp_deduped")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be removed, delete nothing")
    a = ap.parse_args()

    root = Path(a.path)
    if not root.is_dir():
        print("no such directory: {}".format(root), file=sys.stderr)
        return 1

    ymls = sorted(root.glob("*/*.yml"))
    if not ymls:
        # maybe the tasks sit directly under root
        ymls = sorted(root.glob("*.yml"))
    if not ymls:
        print("no .yml tasks found under {}".format(root), file=sys.stderr)
        return 1

    kept = 0
    to_remove = []          # list of (yml_path, [c_paths])
    unreadable = 0

    for y in ymls:
        inputs = input_files_of(y)
        if not inputs:
            unreadable += 1
            continue
        src = ""
        cpaths = []
        for nm in inputs:
            cp = y.parent / nm
            cpaths.append(cp)
            if cp.exists():
                try:
                    src += cp.read_text(errors="replace") + "\n"
                except Exception:
                    pass
        if not src:
            unreadable += 1
            continue

        if count_loops(src) == 0:
            to_remove.append((y, cpaths))
        else:
            kept += 1

    print("tasks scanned : {}".format(len(ymls)))
    print("with a loop   : {}  (kept)".format(kept))
    print("loop-free     : {}  (to remove)".format(len(to_remove)))
    if unreadable:
        print("unreadable    : {}".format(unreadable))

    if to_remove:
        per_dir = Counter(y.parent.name for y, _ in to_remove)
        print("\nloop-free per directory:")
        for d in sorted(per_dir, key=lambda x: -per_dir[x]):
            print("  {:<34} {}".format(d, per_dir[d]))

    if a.dry_run:
        print("\n[dry-run] nothing deleted.")
        return 0

    n_files = 0
    for y, cpaths in to_remove:
        for p in [y] + cpaths:
            if p.exists():
                try:
                    p.unlink()
                    n_files += 1
                except Exception as e:
                    print("[warn] could not delete {}: {}".format(p, e), file=sys.stderr)

    # clean up any directory left empty
    empties = []
    for d in sorted({y.parent for y, _ in to_remove}):
        if d.is_dir() and not any(d.iterdir()):
            try:
                d.rmdir()
                empties.append(d.name)
            except Exception:
                pass

    print("\nremoved {} tasks ({} files).".format(len(to_remove), n_files))
    if empties:
        print("removed {} now-empty directories: {}".format(len(empties), ", ".join(empties)))
    print("remaining tasks: {}".format(kept))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())