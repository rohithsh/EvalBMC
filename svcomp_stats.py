#!/usr/bin/env python3
"""
svcomp_stats.py

Descriptive statistics for a (deduped) SV-COMP dataset.

Reports, at directory level and/or set level:
  - number of programs, and how many contain a loop
  - loop counts: single (1), double (2), multiple (3+)
  - safe / unsafe counts per property (unreach-call, valid-memsafety,
    no-overflow) and an overall "any bug" count
  - average lines of code (non-blank, non-comment)

Set membership is read from the .set manifests in --root; the actual programs
are read from --dataset (e.g. the deduped tree). A directory not listed in any
given set is skipped.

Usage:
  python3 svcomp_stats.py \
      --root    datasets/sv-benchmarks/c \
      --dataset datasets/svcomp_deduped \
      --sets    Loops.set Arrays.set Heap.set LinkedLists.set ControlFlow.set BitVectors.set \
      --level   both \
      [--out results/svcomp_stats.csv]
"""

import argparse
import csv
import re
import statistics
import sys
from collections import defaultdict, Counter
from pathlib import Path

try:
    import yaml
except ImportError:
    print("need pyyaml:  pip install pyyaml", file=sys.stderr)
    raise

PROPS = ["unreach-call", "valid-memsafety", "no-overflow"]

_CLEAN_RE = re.compile(
    r'/\*.*?\*/'            # block comment
    r'|//[^\n]*'            # line comment
    r'|"(?:\\.|[^"\\])*"'   # string literal
    r"|'(?:\\.|[^'\\])*'",  # char literal
    re.DOTALL,
)


def strip_comments_and_strings(src):
    return _CLEAN_RE.sub(" ", src)


def count_loops(src):
    """for + genuine while + do, correcting do-while double counting."""
    clean = strip_comments_and_strings(src)
    n_for = len(re.findall(r"\bfor\b", clean))
    n_do = len(re.findall(r"\bdo\b", clean))
    n_while = len(re.findall(r"\bwhile\b", clean))
    n_while_only = max(0, n_while - n_do)
    return n_for + n_while_only + n_do


def count_loc(src):
    """Non-blank, non-comment lines."""
    clean = strip_comments_and_strings(src)
    return sum(1 for ln in clean.splitlines() if ln.strip())


def read_set_file(path):
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def dirs_of_set(root, setname):
    """Directory names covered by a .set manifest's glob patterns."""
    sp = root / setname
    if not sp.exists():
        print("[warn] missing set: {}".format(sp), file=sys.stderr)
        return set()
    dirs = set()
    for pat in read_set_file(sp):
        # pattern looks like 'loops/*.yml' -> take the leading directory
        dirs.add(pat.split("/")[0])
    return dirs


def parse_yml(p):
    try:
        d = yaml.safe_load(p.read_text())
    except Exception:
        return None
    if not isinstance(d, dict):
        return None
    inp = d.get("input_files", "")
    names = inp if isinstance(inp, list) else [inp]
    verdicts = {}
    for entry in (d.get("properties") or []):
        name = Path(entry.get("property_file", "")).stem
        if name in PROPS and "expected_verdict" in entry:
            v = entry["expected_verdict"]
            verdicts[name] = "safe" if v is True else ("unsafe" if v is False else "")
    return {"inputs": [str(n) for n in names if n], "verdicts": verdicts}


def blank_stats():
    d = {
        "programs": 0, "with_loop": 0,
        "single": 0, "double": 0, "multiple": 0, "loop_free": 0,
        "loc_list": [],
        "any_bug": 0,
    }
    for p in PROPS:
        d[p + "_safe"] = 0
        d[p + "_unsafe"] = 0
    return d


def finish(d):
    out = {k: v for k, v in d.items() if k != "loc_list"}
    locs = d["loc_list"]
    out["avg_loc"] = round(statistics.mean(locs), 1) if locs else 0
    out["median_loc"] = int(statistics.median(locs)) if locs else 0
    return out


def print_table(title, stats_by_key):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    hdr = ("{:<26}{:>6}{:>6}{:>7}{:>7}{:>9}{:>8}{:>8}{:>8}{:>8}{:>8}".format(
        "name", "prog", "loop", "single", "double", "multiple",
        "reach✗", "mem✗", "ovf✗", "anyBug", "avgLOC"))
    print(hdr)
    print("-" * 100)
    tot = blank_stats()
    for k in sorted(stats_by_key):
        s = stats_by_key[k]
        print("{:<26}{:>6}{:>6}{:>7}{:>7}{:>9}{:>8}{:>8}{:>8}{:>8}{:>8}".format(
            k, s["programs"], s["with_loop"], s["single"], s["double"],
            s["multiple"], s["unreach-call_unsafe"], s["valid-memsafety_unsafe"],
            s["no-overflow_unsafe"], s["any_bug"], s["avg_loc"]))
        for f in ["programs", "with_loop", "single", "double", "multiple",
                  "loop_free", "any_bug"] + [p + "_safe" for p in PROPS] + \
                 [p + "_unsafe" for p in PROPS]:
            tot[f] += s[f]
    print("-" * 100)
    print("{:<26}{:>6}{:>6}{:>7}{:>7}{:>9}{:>8}{:>8}{:>8}{:>8}".format(
        "TOTAL", tot["programs"], tot["with_loop"], tot["single"], tot["double"],
        tot["multiple"], tot["unreach-call_unsafe"], tot["valid-memsafety_unsafe"],
        tot["no-overflow_unsafe"], tot["any_bug"]))

    print("\nsafe counts (per property):")
    for p in PROPS:
        print("  {:<18} safe={:<6} unsafe={:<6}".format(
            p, tot[p + "_safe"], tot[p + "_unsafe"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="original sv-benchmarks/c (for .set files)")
    ap.add_argument("--dataset", required=True, help="dataset tree to measure (e.g. deduped)")
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--level", choices=["dir", "set", "both"], default="both")
    ap.add_argument("--out", default=None, help="optional per-directory CSV")
    a = ap.parse_args()

    root = Path(a.root)
    data = Path(a.dataset)

    # dir -> set name
    dir2set = {}
    for s in a.sets:
        for d in dirs_of_set(root, s):
            dir2set[d] = s

    per_dir = defaultdict(blank_stats)
    per_set = defaultdict(blank_stats)

    for ymlp in sorted(data.glob("*/*.yml")):
        dirname = ymlp.parent.name
        setname = dir2set.get(dirname)
        if setname is None:
            continue                      # directory not in the requested sets

        info = parse_yml(ymlp)
        if not info:
            continue

        # read the program source (concatenate if multiple input files)
        src = ""
        for nm in info["inputs"]:
            cp = ymlp.parent / nm
            if cp.exists():
                try:
                    src += cp.read_text(errors="replace") + "\n"
                except Exception:
                    pass
        if not src:
            continue

        nloops = count_loops(src)
        loc = count_loc(src)

        for bucket in (per_dir[dirname], per_set[setname]):
            bucket["programs"] += 1
            bucket["loc_list"].append(loc)
            if nloops == 0:
                bucket["loop_free"] += 1
            else:
                bucket["with_loop"] += 1
                if nloops == 1:
                    bucket["single"] += 1
                elif nloops == 2:
                    bucket["double"] += 1
                else:
                    bucket["multiple"] += 1
            any_bug = False
            for p in PROPS:
                v = info["verdicts"].get(p, "")
                if v == "safe":
                    bucket[p + "_safe"] += 1
                elif v == "unsafe":
                    bucket[p + "_unsafe"] += 1
                    any_bug = True
            if any_bug:
                bucket["any_bug"] += 1

    dir_stats = {k: finish(v) for k, v in per_dir.items()}
    set_stats = {k.replace(".set", ""): finish(v) for k, v in per_set.items()}

    if a.level in ("set", "both"):
        print_table("SET LEVEL", set_stats)
    if a.level in ("dir", "both"):
        print_table("DIRECTORY LEVEL", dir_stats)

    if a.out:
        cols = ["name", "programs", "with_loop", "loop_free", "single", "double",
                "multiple", "any_bug", "avg_loc", "median_loc"] + \
               [p + "_safe" for p in PROPS] + [p + "_unsafe" for p in PROPS]
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for k in sorted(dir_stats):
                row = dict(dir_stats[k]); row["name"] = k
                w.writerow(row)
        print("\nper-directory CSV -> {}".format(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())