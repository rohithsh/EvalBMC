#!/usr/bin/env python3
"""
check_slice_kstar.py

Does slicing preserve the ground truth?

A slice is a different program, so its k* need not match the original's. If the
slice keeps everything that determines when the property is decided, the two
should agree; if they differ, the slice dropped something relevant and any
prediction measured on it would be measured against the wrong program.

This runs both and compares, per program:

  1. slice with frama-c, criterion -slice-calls reach_error,__assert_fail
     (the property for reachability tasks is whether reach_error is called,
     so slicing on that call and on the assertion body preserves both the
     path to the violation and the violation itself)
  2. strip the frama-c artefacts so CBMC can compile the result
  3. derive k* on the original and on the slice, using the same soundness rule
     as the main pipeline: a result counts only when no unwinding assertion
     fails
  4. report agreement, and size

WHY THE CLEANING STEP
  Frama-C's output includes "__fc_builtin.h" and calls Frama_C_interval, which
  CBMC cannot compile, and renames the nondet function to *_slice_N. The
  cleaner removes the include, drops the generated model body, and restores the
  original extern declaration.

Usage:
  python3 check_slice_kstar.py \
      --kstar   results/kstar_all.csv \
      --dataset datasets/cleaned/svcomp_clean \
      --json    results/loops_reclassified.json \
      --model   m.c \
      --out     results/slice_kstar_check.csv \
      [--n 20] [--max-k 60] [--timeout 300] [--keep-dir /tmp/slices]
"""

import argparse
import csv
import json
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

# frama-c artefacts that CBMC cannot compile
FC_INCLUDE = re.compile(r'#include\s+"__fc_[a-z_]*\.h"\s*\n')
FC_MODEL_BODY = re.compile(
    r'\w[\w \*]*\s+__VERIFIER_nondet_(\w+)_slice_\d+\s*\(void\)\s*\{.*?\n\}\n',
    re.S)
FC_SLICE_SUFFIX = re.compile(r'(__VERIFIER_nondet_\w+)_slice_\d+')

NONDET_TYPE = {
    "int": "int", "uint": "unsigned int", "char": "char", "uchar": "unsigned char",
    "short": "short", "ushort": "unsigned short", "long": "long",
    "ulong": "unsigned long", "ulonglong": "unsigned long long",
    "bool": "_Bool", "float": "float", "double": "double",
}


def clean_slice(text):
    """Make a frama-c slice compilable by CBMC."""
    text = FC_INCLUDE.sub("", text)

    # which nondet variants does the slice use?
    used = set(FC_SLICE_SUFFIX.findall(text))
    used |= {m for m in re.findall(r'__VERIFIER_nondet_(\w+?)_slice_\d+', text)}

    # remove the generated model bodies and restore the plain names
    text = FC_MODEL_BODY.sub("", text)
    text = FC_SLICE_SUFFIX.sub(r"\1", text)

    # declare whatever the slice still calls
    decls = []
    for name in sorted(set(re.findall(r'__VERIFIER_nondet_(\w+)\s*\(', text))):
        ty = NONDET_TYPE.get(name)
        if ty:
            decls.append("extern {} __VERIFIER_nondet_{}(void);".format(ty, name))
    if decls:
        text = "\n".join(decls) + "\n" + text
    return text


def slice_program(framac, src, model, out_path, machdep, timeout):
    cmd = [framac, "-machdep", machdep,
           "-cpp-extra-args=-include stdlib.h",
           str(src), str(model),
           "-slice-calls", "reach_error,__assert_fail",
           "-eva-slevel", "100",
           "-then-on", "Slicing export", "-print", "-ocode", str(out_path)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "timeout", "frama-c exceeded {}s".format(timeout)
    except FileNotFoundError:
        return "failed", "frama-c not found"
    if not out_path.exists() or out_path.stat().st_size == 0:
        err = (p.stderr or p.stdout or "").strip().splitlines()
        return "failed", (err[-1] if err else "no output")[:150]
    return "ok", ""


def cbmc_at(cfile, k, data_model, timeout=120):
    """One CBMC run. Returns bug | safe | deeper | timeout | error."""
    cmd = ["cbmc", str(cfile), "--32" if data_model == "ILP32" else "--64",
           "--unwind", str(k), "--unwinding-assertions",
           "--no-standard-checks", "--json-ui"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "timeout"
    except Exception:
        return "error"
    try:
        data = json.loads(p.stdout)
    except Exception:
        return "error"

    failures = []
    for it in data:
        if isinstance(it, dict) and "result" in it:
            for r in it["result"]:
                if r.get("status") == "FAILURE":
                    failures.append(r.get("property", ""))
    # same soundness rule as the main pipeline
    if any(".unwind." in f for f in failures):
        return "deeper"
    if failures:
        return "bug"
    return "safe"


def derive_kstar(cfile, verdict, data_model, max_k, budget):
    """Incremental search, stopping at the verdict the benchmark expects."""
    want = "bug" if verdict == "unsafe" else "safe"
    start = time.time()
    for k in range(0, max_k + 1):
        if time.time() - start > budget:
            return None, "timeout", k
        left = budget - (time.time() - start)
        res = cbmc_at(cfile, k, data_model, timeout=int(min(120, max(5, left))))
        if res in ("timeout", "error"):
            return None, res, k
        if res == want:
            return k, "found", k
        if res != "deeper":
            # decided, but not the expected verdict
            return None, "unexpected_" + res, k
    return None, "cap_reached", max_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kstar", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--model", required=True, help="nondet model file (m.c)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--framac", default="frama-c")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-k", type=int, default=60)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--keep-dir", default="sliced_programs/")
    a = ap.parse_args()

    models = {r["task"]: r.get("data_model", "ILP32")
              for r in json.load(open(a.json))}

    # reachability tasks with a small finite k*, where the comparison is cheap
    cand = [r for r in csv.DictReader(open(a.kstar))
            if r["property"] == "reach" and r["uniform_outcome"] == "found"
            and r["uniform_kstar"].isdigit() and 0 < int(r["uniform_kstar"]) <= a.max_k]
    cand.sort(key=lambda r: int(r["uniform_kstar"]))
    sample = cand[:a.n]
    print("candidates with a finite k* <= {}: {}   testing {}".format(
        a.max_k, len(cand), len(sample)))

    keep = Path(a.keep_dir)
    keep.mkdir(parents=True, exist_ok=True)
    rows = []

    for i, r in enumerate(sample, 1):
        task = r["task"]
        src = Path(a.dataset) / r["dir"] / r["c_file"]
        dm = models.get(task, "ILP32")
        machdep = "x86_32" if dm == "ILP32" else "x86_64"
        stem = Path(r["c_file"]).stem

        row = {"task": task, "c_file": r["c_file"], "verdict": r["verdict"],
               "category": r.get("category", ""), "data_model": dm,
               "kstar_recorded": r["uniform_kstar"],
               "slice_status": "", "slice_error": "",
               "orig_lines": 0, "slice_lines": 0, "reduction": "",
               "kstar_orig": "", "orig_status": "",
               "kstar_slice": "", "slice_kstar_status": "",
               "agree": ""}

        if not src.exists():
            row["slice_status"] = "missing_source"
            rows.append(row)
            continue
        row["orig_lines"] = src.read_text(errors="replace").count("\n") + 1

        raw = keep / "{}.sliced.c".format(stem)
        st, err = slice_program(a.framac, src, a.model, raw, machdep, a.timeout)
        row["slice_status"] = st
        row["slice_error"] = err
        if st != "ok":
            rows.append(row)
            print("[{}/{}] {:<40} slice {}".format(i, len(sample), task[:40], st))
            continue

        cleaned = keep / "{}.clean.c".format(stem)
        cleaned.write_text(clean_slice(raw.read_text(errors="replace")))
        row["slice_lines"] = cleaned.read_text().count("\n") + 1
        if row["orig_lines"]:
            row["reduction"] = round(
                1.0 - row["slice_lines"] / float(row["orig_lines"]), 3)

        ko, sto, _ = derive_kstar(src, r["verdict"], dm, a.max_k, a.timeout)
        row["kstar_orig"] = "" if ko is None else ko
        row["orig_status"] = sto

        ks, sts, _ = derive_kstar(cleaned, r["verdict"], dm, a.max_k, a.timeout)
        row["kstar_slice"] = "" if ks is None else ks
        row["slice_kstar_status"] = sts

        if sto == "found" and sts == "found":
            row["agree"] = "yes" if ko == ks else "NO"
        rows.append(row)

        print("[{}/{}] {:<38} orig k*={:<5}({}) slice k*={:<5}({}) {:<4} {}".format(
            i, len(sample), task[:38],
            row["kstar_orig"] or "-", sto,
            row["kstar_slice"] or "-", sts,
            row["agree"],
            "-{:.0%}".format(row["reduction"]) if row["reduction"] != "" else ""))

    cols = list(rows[0].keys())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 62)
    print("SLICE STATUS")
    print("=" * 62)
    for k, v in Counter(r["slice_status"] for r in rows).most_common():
        print("   {:<16} {}".format(k, v))

    comparable = [r for r in rows if r["agree"] in ("yes", "NO")]
    print("\n" + "=" * 62)
    print("k* AGREEMENT  ({} comparable)".format(len(comparable)))
    print("=" * 62)
    if comparable:
        ok = sum(1 for r in comparable if r["agree"] == "yes")
        print("   agree    : {}/{}  ({:.0f}%)".format(
            ok, len(comparable), 100.0 * ok / len(comparable)))
        bad = [r for r in comparable if r["agree"] == "NO"]
        if bad:
            print("   DISAGREE : {}".format(len(bad)))
            for r in bad[:10]:
                print("      {:<40} orig={} slice={}".format(
                    r["task"][:40], r["kstar_orig"], r["kstar_slice"]))
            print("\n   A mismatch means the slice dropped something that "
                  "affects when the")
            print("   property is decided, so predictions made on it would "
                  "not transfer.")

    red = sorted(r["reduction"] for r in rows if r["reduction"] != "")
    if red:
        print("\nSIZE REDUCTION")
        print("   median={:.1%}  min={:.1%}  max={:.1%}".format(
            red[len(red) // 2], red[0], red[-1]))
        gained = sum(1 for x in red if x <= 0)
        print("   no smaller than the original: {}/{}".format(gained, len(red)))

    print("\nresults -> {}".format(a.out))
    print("slices  -> {}".format(keep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())