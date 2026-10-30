#!/usr/bin/env python3
"""
import_static_bounds.py

Turn the static-analysis bounds sent back in bounds_template.csv (the file
package_tdx_set.py ships) into the predictions JSONL that score_predictions.py
reads, so static analysis is scored exactly like syntactic counting and
Qwen3.5.

PARTIAL ANSWERS -- the paper's rule, the same for every strategy
  A program on which the analysis bounds at least one loop is run. Every loop
  it leaves open (empty cell, or a loop missing from the CSV) gets bound 0,
  which is also what score_predictions.py does for any loop a schedule does
  not list (--unwind 0).
  The program counts as an ABSTENTION only when the analysis bounds none of
  its non-trivial loops. The do{}while(0) wrappers were pre-filled with 1 in
  the template by us, so they do not count as the analysis answering --
  except on tasks that have no other loop, where 1 is the whole answer.
  A cell reading "unbounded" means the analysis claims no finite bound
  exists, so the whole program is answered "unbounded", the same answer
  kind Qwen3.5 gives, and is not run.

CELL VALUES
  33            the unwinding depth (iterations + 1), as asked in the README
  unbounded     no finite bound
  (empty)       the analysis could not tell -> 0 if the program is run

Usage
  python3 import_static_bounds.py \
      --bounds  static/bounds_filled.csv \
      --json    results/tdx_cprover_loops.json \
      --out     results/pred_static_tdx.jsonl \
      --name    <tool name and version> \
      [--seconds static/times.csv]      # optional: path,seconds per program
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

PROP_KEYS = {
    "reach": "unreach-call",
    "memsafety": "valid-memsafety",
    "overflow": "no-overflow",
}


TRIVIAL_GUARDS = {"0", "0u", "0l", "0ul", "false", "(0)", "(false)"}


def is_trivial(loop):
    """The same definition filter_trivial_loops.py and predict_syntactic.py use."""
    if "always-false guard" in (loop.get("reason") or "").lower():
        return True
    return (loop.get("guard") or "").strip().lower().replace(" ", "") in TRIVIAL_GUARDS


def key_of(path):
    """(dir, stem) from a template path such as programs/<dir>/<file>.i"""
    p = PurePosixPath(path)
    parts = list(p.parts)
    if parts and parts[0] == "programs":
        parts = parts[1:]
    d = "/".join(parts[:-1])
    return d, PurePosixPath(parts[-1]).stem if parts else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bounds", required=True, help="filled bounds_template.csv")
    ap.add_argument("--json", required=True, help="loops JSON of the same set")
    ap.add_argument("--out", required=True, help="predictions JSONL")
    ap.add_argument("--name", required=True, help="tool name/version for the rows")
    ap.add_argument("--seconds", default=None,
                    help="optional CSV with columns path,seconds (analysis time)")
    a = ap.parse_args()

    recs = json.load(open(a.json))
    by_key = {}
    for r in recs:
        by_key[(r.get("dir", ""), Path(r.get("c_file", "")).stem)] = r

    cells = defaultdict(dict)        # key -> {loop_id: (value, trivial)}
    unknown_paths = set()
    for row in csv.DictReader(open(a.bounds)):
        k = key_of(row["path"])
        if k not in by_key:
            unknown_paths.add(row["path"])
            continue
        cells[k][row["loop_id"]] = ((row.get("bound") or "").strip(),
                                    (row.get("trivial") or "") == "yes")

    secs = {}
    if a.seconds:
        for row in csv.DictReader(open(a.seconds)):
            try:
                secs[key_of(row["path"])] = float(row["seconds"])
            except (KeyError, ValueError):
                pass

    stats = Counter()
    bad_cells = []
    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as fh:
        for k, rec in by_key.items():
            loops = [l for l in rec.get("loops", []) if l.get("id")]
            given = cells.get(k, {})
            sched, unbounded, answered_nontrivial = {}, False, 0
            n_nontrivial = 0
            for l in loops:
                val, triv = given.get(l["id"], ("", False))
                triv = triv or is_trivial(l)
                if not triv:
                    n_nontrivial += 1
                low = val.lower()
                if low == "unbounded":
                    unbounded = True
                elif val:
                    try:
                        b = int(float(val))
                        if b < 0:
                            raise ValueError
                        sched[l["id"]] = b
                        if not triv:
                            answered_nontrivial += 1
                    except ValueError:
                        bad_cells.append((rec["task"], l["id"], val))
                        sched[l["id"]] = 0
                else:
                    sched[l["id"]] = 0          # left open -> 0
            open_loops = n_nontrivial - answered_nontrivial

            if unbounded:
                parsed, kind = {"kind": "unbounded",
                                "reasoning": "a loop is marked unbounded"}, "unbounded"
            elif n_nontrivial and not answered_nontrivial:
                parsed, kind = None, "abstain"
            else:
                parsed = {"kind": "schedule", "schedule": sched,
                          "reasoning": "static analysis; {} open loop(s) at 0"
                          .format(open_loops)}
                kind = "partial" if open_loops else "complete"
            stats[kind] += 1
            stats["nontrivial_loops"] += n_nontrivial
            stats["nontrivial_bounded"] += answered_nontrivial

            v = rec.get("verdicts", {}) or {}
            cats = {l.get("category") for l in loops if not is_trivial(l)}
            cats.discard(None)
            for prop, pkey in PROP_KEYS.items():
                verdict = v.get(pkey)
                if verdict not in ("unsafe", "safe"):
                    continue
                out = {
                    "task": rec["task"], "dir": rec.get("dir", ""),
                    "c_file": rec.get("c_file", ""), "property": prop,
                    "verdict": verdict,
                    "category": (next(iter(cats)) if len(cats) == 1
                                 else "mixed" if cats else "none"),
                    "n_loops": len(loops), "config": "static", "mode": "static",
                    "model": {"provider": "static-analysis", "model": a.name},
                    "open_loops": open_loops,
                    "predictions": [], "errors": [],
                }
                if parsed is None:
                    out["abstained"] = True
                    out["predictions"].append({
                        "sample": 0, "latency_s": secs.get(k), "parsed": None,
                        "parse_error": "abstain: no non-trivial loop bounded",
                        "raw": ""})
                else:
                    out["predictions"].append({
                        "sample": 0, "latency_s": secs.get(k), "parsed": parsed,
                        "parse_error": "", "raw": json.dumps(parsed)})
                fh.write(json.dumps(out) + "\n")

    print("programs: {}".format(len(by_key)))
    print("  complete (every non-trivial loop bounded): {}".format(stats["complete"]))
    print("  partial  (open loops run at 0):            {}".format(stats["partial"]))
    print("  abstain  (no non-trivial loop bounded):    {}".format(stats["abstain"]))
    print("  unbounded (a loop marked unbounded):       {}".format(stats["unbounded"]))
    print("non-trivial loops bounded: {} of {}".format(
        stats["nontrivial_bounded"], stats["nontrivial_loops"]))
    if unknown_paths:
        print("WARNING: {} CSV paths match no program in --json, e.g. {}".format(
            len(unknown_paths), sorted(unknown_paths)[0]), file=sys.stderr)
    if bad_cells:
        print("WARNING: {} cells are neither a number nor 'unbounded' (run as 0), "
              "e.g. {}".format(len(bad_cells), bad_cells[0]), file=sys.stderr)
    print("predictions -> {}".format(outp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())