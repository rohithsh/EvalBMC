#!/usr/bin/env python3
"""
slicer.py

Produce backward slices of each program on its loop control variables, so the
LLM sees only the code that determines a loop's trip count.

WHAT IT SLICES ON
  Per loop, the criterion is the union of:

    bound_vars       from loops.json -- the variables in the loop guard that
                     are not the counter
    induction_vars   from loops.json -- the counter itself
    exit vars        found by an AST pass here: for a loop with break, return,
                     continue or goto leaving it, the variables in the enclosing
                     if conditions. A while(1) loop has no guard variables at
                     all, so without this its criterion would be empty and the
                     slice would keep nothing. The same gap exists for any loop
                     whose real bound is a break condition rather than the
                     header.

  Two kinds of slice are written per program:

    <program>__all.c      one slice on the union of every loop's criterion
    <program>__<loop>.c   one slice per loop, on that loop's criterion alone

OUTPUT LAYOUT
    <dest>/sliced_programs/<program_name>/<program>__all.c
    <dest>/sliced_programs/<program_name>/<program>__main.0.c
    <dest>/slice_stats.csv
    <dest>/slice_failures.csv

ON FAILURE
  Frama-C does not handle every C construct. When it fails or times out the
  full source is copied instead, the row is marked fallback=yes, and the
  program is named in slice_failures.csv. A silent fallback would look like a
  slice that simply did not reduce anything.

KNOWN LIMITATION
  -slice-value keeps the statements affecting a variable's value at the END of
  the program. Our criterion variables are usually locals, which are out of
  scope there, so Frama-C may keep more than the minimum or reject the
  criterion. If that shows up in the first run, the precise alternative is
  -slice-pragma with a pragma inserted at the loop header, which needs the
  source rewritten first.

Usage:
  python3 slicer.py \
      --json    datasets/cleaned/loops.json \
      --dataset datasets/cleaned/svcomp_clean \
      --dest    datasets/sliced \
      [--programs a.yml b.yml] [--jobs 4] [--timeout 300]
      [--frama-c frama-c] [--extra-flags "-no-frama-c-stdlib"]
"""

import argparse
import csv
import json
import shutil
import subprocess
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import clang.cindex as ci


# ------------------------------------------------------- AST: exit conditions

JUMPS = (ci.CursorKind.BREAK_STMT, ci.CursorKind.RETURN_STMT,
         ci.CursorKind.CONTINUE_STMT, ci.CursorKind.GOTO_STMT)

LOOPS = (ci.CursorKind.FOR_STMT, ci.CursorKind.WHILE_STMT, ci.CursorKind.DO_STMT)


def idents(n):
    out = []

    def w(x):
        if x.kind == ci.CursorKind.DECL_REF_EXPR:
            out.append(x.spelling)
        for c in x.get_children():
            w(c)

    if n is not None:
        w(n)
    return out


def exit_condition_vars(loop_node):
    """
    Variables in the conditions guarding any jump that can leave this loop.

    A loop's trip count is decided by its header guard AND by whatever makes it
    break out early. `while (1) { if (state == DONE) break; }` has no guard
    variables, but `state` decides how long it runs, so `state` belongs in the
    criterion.

    Nested loops are not descended into: a break inside an inner loop leaves
    that loop, not this one.
    """
    found = set()

    def walk(n, guards, depth):
        # do not follow into a nested loop: its jumps belong to it
        if depth > 0 and n.kind in LOOPS:
            return

        g = guards
        if n.kind == ci.CursorKind.IF_STMT:
            kids = list(n.get_children())
            if kids:
                g = guards + [kids[0]]
        elif n.kind == ci.CursorKind.SWITCH_STMT:
            kids = list(n.get_children())
            if kids:
                g = guards + [kids[0]]

        if n.kind in JUMPS:
            for cond in g:
                found.update(idents(cond))
            # `return x;` -- x itself may carry the exit decision
            if n.kind == ci.CursorKind.RETURN_STMT:
                found.update(idents(n))

        for c in n.get_children():
            walk(c, g, depth + 1)

    kids = list(loop_node.get_children())
    body = kids[-1] if kids else None
    if body is not None:
        walk(body, [], 0)
    return found


def loop_criteria_from_ast(cfile, data_model, loops_meta):
    """
    {line: set(exit vars)} for each loop in loops.json, matched by source line.
    Returns {} when the file cannot be parsed -- the JSON criteria still apply.
    """
    args = ["-std=gnu99", "-ferror-limit=0", "-w",
            "-m32" if data_model == "ILP32" else "-m64"]
    try:
        tu = ci.Index.create().parse(str(cfile), args=args)
    except Exception:
        return {}

    want = {l.get("line") for l in loops_meta if l.get("line")}
    out = {}

    def walk(n):
        loc = n.location
        if (n.kind in LOOPS and loc.file is not None
                and Path(str(loc.file)).name == cfile.name
                and loc.line in want):
            out.setdefault(loc.line, set()).update(exit_condition_vars(n))
        for c in n.get_children():
            walk(c)

    walk(tu.cursor)
    return out


# ------------------------------------------------------------------ Frama-C

def run_framac(framac, src, criteria, out_path, timeout, extra_flags):
    """
    One Frama-C slice. Returns (status, message) with status ok|failed|timeout.

    -slice-value takes a comma-separated list of locations and keeps the
    statements that affect their values. The sliced project is then printed
    with -then-on 'Slicing export'.
    """
    if not criteria:
        return "failed", "empty criterion"

    cmd = [framac, str(src),
           "-slice-value", ",".join(sorted(criteria)),
           "-then-on", "Slicing export", "-print", "-ocode", str(out_path)]
    if extra_flags:
        cmd[1:1] = extra_flags.split()

    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "timeout", "frama-c exceeded {}s".format(timeout)
    except FileNotFoundError:
        return "failed", "frama-c not found: {}".format(framac)

    if p.returncode != 0:
        err = (p.stderr or p.stdout or "").strip().splitlines()
        return "failed", (err[-1] if err else "exit {}".format(p.returncode))[:200]
    if not out_path.exists() or out_path.stat().st_size == 0:
        return "failed", "no output produced"
    return "ok", ""


def measure(path):
    try:
        t = path.read_text(errors="replace")
    except Exception:
        return 0, 0
    return t.count("\n") + 1, len(t)


# ------------------------------------------------------------------- worker

def slice_program(job):
    (rec, dataset, dest, framac, timeout, extra_flags) = job

    task = rec["task"]
    src = Path(dataset) / rec["dir"] / rec["c_file"]
    name = Path(rec["c_file"]).stem
    outdir = Path(dest) / "sliced_programs" / name
    outdir.mkdir(parents=True, exist_ok=True)

    row = {
        "task": task, "dir": rec["dir"], "c_file": rec["c_file"],
        "n_loops": len(rec.get("loops", [])),
        "orig_lines": 0, "orig_chars": 0,
        "all_status": "", "all_lines": 0, "all_chars": 0, "all_reduction": "",
        "all_criteria": "", "n_loop_slices_ok": 0, "n_loop_slices_failed": 0,
        "fallback": "no", "error": "",
    }
    failures = []

    if not src.exists():
        row["all_status"] = "missing"
        row["error"] = "source not found"
        failures.append({**row})
        return row, failures, []

    row["orig_lines"], row["orig_chars"] = measure(src)

    # --- build the criterion for each loop ------------------------------
    loops_meta = rec.get("loops", [])
    ast_vars = loop_criteria_from_ast(src, rec.get("data_model", "ILP32"),
                                      loops_meta)

    per_loop = {}
    for l in loops_meta:
        crit = set(l.get("bound_vars") or []) | set(l.get("induction_vars") or [])
        crit |= ast_vars.get(l.get("line"), set())
        # drop things that are not plausible C identifiers
        crit = {c for c in crit if c and c.replace("_", "a").isalnum()}
        if crit:
            per_loop[l.get("id") or "loop@{}".format(l.get("line"))] = crit

    combined = set()
    for c in per_loop.values():
        combined |= c
    row["all_criteria"] = ";".join(sorted(combined))

    # --- combined slice --------------------------------------------------
    all_out = outdir / "{}__all.c".format(name)
    status, msg = run_framac(framac, src, combined, all_out, timeout, extra_flags)
    row["all_status"] = status
    if status != "ok":
        # copy the full source so downstream steps always have a file, and
        # record it so a non-reduction is never mistaken for a small slice
        shutil.copy2(src, all_out)
        row["fallback"] = "yes"
        row["error"] = msg
        failures.append({"task": task, "which": "all", "status": status,
                         "error": msg, "criteria": row["all_criteria"]})
    row["all_lines"], row["all_chars"] = measure(all_out)
    if row["orig_chars"]:
        row["all_reduction"] = round(
            1.0 - row["all_chars"] / float(row["orig_chars"]), 3)

    # --- one slice per loop ---------------------------------------------
    loop_rows = []
    for lid, crit in sorted(per_loop.items()):
        lout = outdir / "{}__{}.c".format(name, lid.replace("/", "_"))
        st, m = run_framac(framac, src, crit, lout, timeout, extra_flags)
        if st != "ok":
            shutil.copy2(src, lout)
            row["n_loop_slices_failed"] += 1
            failures.append({"task": task, "which": lid, "status": st,
                             "error": m, "criteria": ";".join(sorted(crit))})
        else:
            row["n_loop_slices_ok"] += 1
        ln, ch = measure(lout)
        loop_rows.append({
            "task": task, "loop_id": lid, "status": st,
            "criteria": ";".join(sorted(crit)),
            "lines": ln, "chars": ch,
            "reduction": round(1.0 - ch / float(row["orig_chars"]), 3)
            if row["orig_chars"] else "",
        })

    return row, failures, loop_rows


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="loops.json")
    ap.add_argument("--dataset", required=True, help="directory holding the programs")
    ap.add_argument("--dest", required=True, help="where slices and stats are written")
    ap.add_argument("--programs", nargs="*", default=None,
                    help="optional: only these tasks (matched by substring)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=300,
                    help="seconds per Frama-C invocation")
    ap.add_argument("--frama-c", default="frama-c")
    ap.add_argument("--extra-flags", default="",
                    help="extra Frama-C flags, e.g. '-no-frama-c-stdlib -machdep x86_32'")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    recs = json.load(open(a.json))
    if a.programs:
        keep = []
        for r in recs:
            if any(p in r["task"] for p in a.programs):
                keep.append(r)
        recs = keep
    if a.limit:
        recs = recs[:a.limit]

    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    print("slicing {} programs, {} at a time, {}s timeout".format(
        len(recs), a.jobs, a.timeout))

    jobs = [(r, a.dataset, a.dest, a.frama_c, a.timeout, a.extra_flags)
            for r in recs]

    rows, failures, loop_rows = [], [], []
    done = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(slice_program, j) for j in jobs]
        for f in as_completed(futs):
            try:
                row, fails, lrows = f.result()
            except Exception as e:
                rows.append({"task": "?", "all_status": "worker_error",
                             "error": str(e)[:120]})
                done += 1
                continue
            rows.append(row)
            failures.extend(fails)
            loop_rows.extend(lrows)
            done += 1
            print("[{}/{}] {:<44} {:<8} {}  ({:.0f}m)".format(
                done, len(jobs), row["task"][:44], row["all_status"],
                "-{:.0%}".format(row["all_reduction"])
                if row["all_reduction"] != "" else "",
                (time.time() - t0) / 60.0), flush=True)

    # ---- write ----------------------------------------------------------
    stats_path = dest / "slice_stats.csv"
    cols = ["task", "dir", "c_file", "n_loops", "orig_lines", "orig_chars",
            "all_status", "all_lines", "all_chars", "all_reduction",
            "all_criteria", "n_loop_slices_ok", "n_loop_slices_failed",
            "fallback", "error"]
    with open(stats_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    if loop_rows:
        with open(dest / "slice_stats_per_loop.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(loop_rows[0].keys()))
            w.writeheader()
            w.writerows(loop_rows)

    fail_path = dest / "slice_failures.csv"
    if failures:
        with open(fail_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["task", "which", "status",
                                               "error", "criteria"])
            w.writeheader()
            w.writerows(failures)

    # ---- summary --------------------------------------------------------
    ok = [r for r in rows if r.get("all_status") == "ok"]
    print("\n" + "=" * 66)
    print("SLICING SUMMARY  ({} programs)".format(len(rows)))
    print("=" * 66)
    for k, v in Counter(r.get("all_status", "?") for r in rows).most_common():
        print("   combined slice {:<12} {:>5}".format(k, v))
    fb = sum(1 for r in rows if r.get("fallback") == "yes")
    print("   fell back to full source: {}".format(fb))

    red = sorted(r["all_reduction"] for r in ok if r.get("all_reduction") != "")
    if red:
        print("\n   size reduction of the combined slice:")
        print("      median={:.1%}  p25={:.1%}  p75={:.1%}  min={:.1%}  max={:.1%}".format(
            red[len(red) // 2], red[len(red) // 4], red[3 * len(red) // 4],
            red[0], red[-1]))
        big = sum(1 for x in red if x > 0.5)
        print("      more than half removed: {}/{}".format(big, len(red)))

    lo = sum(r.get("n_loop_slices_ok", 0) for r in rows)
    lf = sum(r.get("n_loop_slices_failed", 0) for r in rows)
    print("\n   per-loop slices: {} ok, {} failed".format(lo, lf))

    if failures:
        print("\n   failing programs (first 15):")
        seen = set()
        for f in failures:
            if f["task"] in seen:
                continue
            seen.add(f["task"])
            print("      {:<44} {:<8} {}".format(
                f["task"][:44], f["status"], f["error"][:60]))
            if len(seen) >= 15:
                break
        print("\n   full list -> {}".format(fail_path))

    print("\n   slices -> {}/sliced_programs/".format(dest))
    print("   stats  -> {}".format(stats_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())