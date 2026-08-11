#!/usr/bin/env python3
"""
derive_kstar_all.py  (v2)

Derive ground-truth unrolling bounds for the whole preprocessed SV-COMP dataset.

PROPERTY-MATCHED ACCEPTANCE. A violation counts only if the failing check belongs to the property under test.


uniform k*   one depth for every loop (--unwind k)
schedule k*  one depth per loop (--unwindset L1:k1,...), multi-loop only
             (on a single-loop program the two are the same measurement)

THE SOUNDNESS RULE
  A result counts only when NO unwinding assertion fails. A failing unwinding
  assertion means some loop was cut short, so the model is a truncated program
  and anything it reports may be spurious.

    unsafe: k* = smallest bound with a property-matched violation and no
            unwinding assertion failing.
    safe:   k* = smallest bound at which CBMC reports success (which already
            implies no unwinding assertion failed) -- the completeness
            threshold.

  A loop whose trip count is an unconstrained input satisfies neither at any
  finite depth and is reported as having no usable finite k*.


COST MODEL
  A loop's cost is its own bound times the bounds of every enclosing loop;
  a program's cost is the sum over loops. Nested loops multiply, siblings add,
  and the total is the number of loop-body unrollings.

Usage:
  python3 derive_kstar_all.py \
      --json     datasets/cleaned/loops.json \
      --dataset  datasets/cleaned/svcomp_clean \
      --out      results/kstar_all.csv \
      --trace-dir results/kstar_traces_all \
      --jobs 8 [--mode both] [--budget 600] [--per-k-timeout 180]
      [--properties reach memsafety overflow] [--verdicts unsafe safe]
      [--restart] [--summary-only]
"""

import argparse
import csv
import json
import re
import subprocess
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

CBMC = "cbmc"

PROP_KEYS = {
    "reach": "unreach-call",
    "memsafety": "valid-memsafety",
    "overflow": "no-overflow",
}

# CBMC checks enabled per property.
#   reach     : the property is reachability of reach_error(); no value checks.
#   overflow  : no-overflow == signed integer overflow only.
#   memsafety : valid-deref + valid-free (--pointer-check, --bounds-check) and
#               valid-memtrack (--memory-leak-check, an over-approximation).

PROFILE = {
    "reach":     ["--no-standard-checks"],
    "overflow":  ["--signed-overflow-check", "--no-standard-checks"],
    "memsafety": ["--no-standard-checks", "--bounds-check", "--pointer-check",
                  "--memory-leak-check"],
}

# A violation counts only if the failing check belongs to the property under test.
ACCEPT = {
    "reach":     [".assertion", "reach_error"],
    "overflow":  [".overflow", ".signed-overflow", ".signed_overflow"],
    "memsafety": [".pointer", ".bounds", ".array_bounds", ".memory-leak",
                  ".memory_leak", ".deallocated", ".dead_object", ".free",
                  ".invalid"],
}

# main.unwind.0  ->  loop id main.0
UNWIND_RE = re.compile(r"^(.*)\.unwind\.(\d+)$")

COLS = [
    "task", "dir", "c_file", "property", "verdict", "category", "data_model",
    "n_loops", "max_nesting",
    "uniform_outcome", "uniform_kstar", "uniform_k_reached", "uniform_cost",
    "uniform_time", "uniform_failing_property",
    "schedule_outcome", "schedule_kstar", "schedule_cost", "schedule_rounds",
    "schedule_time", "schedule_failing_property", "schedule_path",
    "cost_ratio", "loop_independent", "other_property_violation", "trace_file",
]


# ----------------------------------------------------------------- CBMC calls

def cbmc_run(cfile, data_model, prop, unwind=None, unwindset=None, timeout=180):
    cmd = [CBMC, str(cfile), "--32" if data_model == "ILP32" else "--64",
           "--unwinding-assertions", *PROFILE[prop], "--json-ui"]
    if unwind is not None:
        cmd += ["--unwind", str(unwind)]
    if unwindset:
        cmd += ["--unwindset",
                ",".join("{}:{}".format(k, v) for k, v in sorted(unwindset.items()))]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "timeout", None
    except Exception as e:
        return "error", str(e)
    return "ok", p.stdout


def analyse(stdout, prop):
    """
    Return (verdict, failing_property, unwind_loop_ids, other_failure).

    verdict:
      'violation'  a PROPERTY-MATCHED failure, with no unwinding assertion failing
      'success'    nothing failed
      'unwind'     an unwinding assertion failed -> model truncated, go deeper
      'other'      something failed soundly, but it belongs to a different
                   property (e.g. reach_error while checking no-overflow).
                   Not the bug under test; recorded and treated as "keep going".
      'unknown'    could not parse
    """
    if stdout is None:
        return "unknown", "", [], ""
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        low = (stdout or "").lower()
        if "unwinding assertion" in low and "fail" in low:
            return "unwind", "", [], ""
        if "verification successful" in low:
            return "success", "", [], ""
        return "unknown", "", [], ""

    status = None
    failures = []
    for item in data:
        if isinstance(item, dict) and "cProverStatus" in item:
            status = item["cProverStatus"]
        if isinstance(item, dict) and "result" in item:
            for res in item["result"]:
                if res.get("status") == "FAILURE":
                    failures.append(res.get("property", "") or
                                    res.get("description", ""))

    unwind_ids, others = [], []
    for f in failures:
        m = UNWIND_RE.match(f)
        if m:
            unwind_ids.append("{}.{}".format(m.group(1), m.group(2)))
        elif ".unwind." in f.lower() or "unwinding" in f.lower():
            unwind_ids.append("")
        else:
            others.append(f)

    # soundness veto first
    if unwind_ids:
        return "unwind", "", [i for i in unwind_ids if i], ""

    if not others:
        if status in ("success", None):
            return "success", "", [], ""
        return "unknown", "", [], ""

    # property matching: only a failure of the property under test is the bug
    accept = ACCEPT.get(prop, [])
    for f in others:
        if any(h in f.lower() for h in accept):
            return "violation", f, [], ""
    return "other", "", [], others[0]


def is_done(verdict_kind, res):
    return (res == "violation") if verdict_kind == "unsafe" else (res == "success")


# ------------------------------------------------------------------ cost model

def loop_cost(schedule, parents):
    total = 0
    for lid, k in schedule.items():
        factor = k
        seen, p = set(), parents.get(lid)
        while p and p not in seen:
            seen.add(p)
            factor *= schedule.get(p, 1) or 1
            p = parents.get(p)
        total += factor
    return total


def uniform_cost(k, loop_ids, parents):
    return loop_cost({lid: k for lid in loop_ids}, parents)


# ------------------------------------------------------------------- searches

def search_uniform(cfile, data_model, prop, verdict_kind, budget, max_k, per_k):
    start = time.time()
    k = 0
    other_seen = ""
    while True:
        elapsed = time.time() - start
        if elapsed > budget:
            return {"outcome": "timeout", "kstar": "", "k_reached": max(k - 1, 0),
                    "failing": "", "other": other_seen,
                    "time": round(elapsed, 1), "trace": None}
        if k > max_k:
            return {"outcome": "cap_reached", "kstar": "", "k_reached": max_k,
                    "failing": "", "other": other_seen,
                    "time": round(elapsed, 1), "trace": None}

        st, out = cbmc_run(cfile, data_model, prop, unwind=k,
                           timeout=min(per_k, max(1, int(budget - elapsed))))
        if st != "ok":
            return {"outcome": "timeout", "kstar": "", "k_reached": k,
                    "failing": "", "other": other_seen,
                    "time": round(time.time() - start, 1), "trace": None}

        res, failing, _, other = analyse(out, prop)
        if other and not other_seen:
            other_seen = other
        if is_done(verdict_kind, res):
            return {"outcome": "found", "kstar": k, "k_reached": k,
                    "failing": failing, "other": other_seen,
                    "time": round(time.time() - start, 1), "trace": out}
        k += 1


def search_schedule(cfile, data_model, prop, verdict_kind, loop_ids,
                    budget, max_k, per_k):
    start = time.time()
    schedule = {lid: 0 for lid in loop_ids}
    path, rounds, other_seen = [], 0, ""

    while True:
        elapsed = time.time() - start
        if elapsed > budget:
            return {"outcome": "timeout", "schedule": schedule, "rounds": rounds,
                    "failing": "", "other": other_seen, "time": round(elapsed, 1),
                    "path": path, "trace": None}
        if schedule and max(schedule.values()) > max_k:
            return {"outcome": "cap_reached", "schedule": schedule,
                    "rounds": rounds, "failing": "", "other": other_seen,
                    "time": round(elapsed, 1), "path": path, "trace": None}

        st, out = cbmc_run(cfile, data_model, prop, unwind=0, unwindset=schedule,
                           timeout=min(per_k, max(1, int(budget - elapsed))))
        rounds += 1
        if st != "ok":
            return {"outcome": "timeout", "schedule": schedule, "rounds": rounds,
                    "failing": "", "other": other_seen,
                    "time": round(time.time() - start, 1),
                    "path": path, "trace": None}

        res, failing, unwind_ids, other = analyse(out, prop)
        if other and not other_seen:
            other_seen = other

        if is_done(verdict_kind, res):
            return {"outcome": "found", "schedule": schedule, "rounds": rounds,
                    "failing": failing, "other": other_seen,
                    "time": round(time.time() - start, 1),
                    "path": path, "trace": out}

        if res == "unwind":
            if not unwind_ids:
                for lid in schedule:
                    schedule[lid] += 1
                path.append("all+1")
            else:
                for lid in unwind_ids:
                    schedule[lid] = schedule.get(lid, 0) + 1
                path.append("+".join(sorted(unwind_ids)))
            continue

        # a safe program showing a property-matched violation contradicts the
        # benchmark verdict; record rather than loop forever
        if verdict_kind == "safe" and res == "violation":
            return {"outcome": "unexpected_violation", "schedule": schedule,
                    "rounds": rounds, "failing": failing, "other": other_seen,
                    "time": round(time.time() - start, 1),
                    "path": path, "trace": out}

        for lid in schedule:
            schedule[lid] += 1
        path.append("all+1")


# -------------------------------------------------------------------- worker

def process(job):
    (rec, prop, verdict_kind, dataset, trace_dir, mode,
     budget, max_k, per_k) = job

    task, dirname = rec["task"], rec["dir"]
    cfile = Path(dataset) / dirname / rec["c_file"]

    loops = rec.get("loops", [])
    loop_ids = [l["id"] for l in loops if l.get("id")]
    parents = {l["id"]: l.get("parent") for l in loops if l.get("id")}
    depths = [l.get("nesting_depth", 0) for l in loops] or [0]
    cats = {l["category"] for l in loops}
    category = next(iter(cats)) if len(cats) == 1 else "mixed"
    n_loops = len(loop_ids)

    row = {c: "" for c in COLS}
    row.update({
        "task": task, "dir": dirname, "c_file": rec["c_file"],
        "property": prop, "verdict": verdict_kind, "category": category,
        "data_model": rec.get("data_model", "ILP32"),
        "n_loops": n_loops, "max_nesting": max(depths),
    })

    if not cfile.exists():
        row["uniform_outcome"] = row["schedule_outcome"] = "missing_file"
        return row

    dm = rec.get("data_model", "ILP32")
    trace = None

    # single-loop programs: --unwind and --unwindset are the same measurement
    if mode == "uniform":
        do_uniform, do_schedule = True, False
    elif mode == "schedule":
        do_uniform, do_schedule = (n_loops <= 1), (n_loops > 1)
    else:
        do_uniform, do_schedule = True, (n_loops > 1)

    if do_uniform:
        u = search_uniform(cfile, dm, prop, verdict_kind, budget, max_k, per_k)
        row["uniform_outcome"] = u["outcome"]
        row["uniform_kstar"] = u["kstar"]
        row["uniform_k_reached"] = u["k_reached"]
        row["uniform_time"] = u["time"]
        row["uniform_failing_property"] = u["failing"]
        row["other_property_violation"] = u["other"]
        if u["outcome"] == "found":
            row["uniform_cost"] = uniform_cost(u["kstar"], loop_ids, parents)
            row["loop_independent"] = "yes" if u["kstar"] == 0 else "no"
            trace = trace or u.get("trace")

    if do_schedule:
        s = search_schedule(cfile, dm, prop, verdict_kind, loop_ids,
                            budget, max_k, per_k)
        row["schedule_outcome"] = s["outcome"]
        row["schedule_rounds"] = s["rounds"]
        row["schedule_time"] = s["time"]
        row["schedule_failing_property"] = s["failing"]
        row["schedule_kstar"] = ";".join(
            "{}:{}".format(k, v) for k, v in sorted(s["schedule"].items()))
        row["schedule_path"] = " | ".join(s["path"][:40])
        if not row["other_property_violation"]:
            row["other_property_violation"] = s["other"]
        if s["outcome"] == "found":
            row["schedule_cost"] = loop_cost(s["schedule"], parents)
            trace = trace or s.get("trace")
    elif n_loops <= 1 and row["uniform_outcome"] == "found":
        row["schedule_outcome"] = "same_as_uniform"
        row["schedule_kstar"] = ("{}:{}".format(loop_ids[0], row["uniform_kstar"])
                                 if loop_ids else "")
        row["schedule_cost"] = row["uniform_cost"]
        row["cost_ratio"] = 1.0

    if (row["cost_ratio"] == "" and row["uniform_cost"] != ""
            and row["schedule_cost"] not in ("", 0)):
        try:
            row["cost_ratio"] = round(
                float(row["uniform_cost"]) / float(row["schedule_cost"]), 2)
        except (ValueError, ZeroDivisionError):
            pass

    if trace:
        try:
            tp = Path(trace_dir) / "{}__{}.txt".format(task.replace("/", "__"), prop)
            tp.write_text(trace)
            row["trace_file"] = str(tp)
        except Exception:
            pass

    return row


# ---------------------------------------------------------------------- main

def build_jobs(recs, properties, verdicts):
    jobs = []
    for r in recs:
        v = r.get("verdicts", {}) or {}
        for pname in properties:
            got = v.get(PROP_KEYS[pname])
            if got in verdicts:
                jobs.append((r, pname, got))
    return jobs


def num(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def summarise(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return
    print("\n" + "=" * 70)
    print("SUMMARY  ({} program-property pairs)".format(len(rows)))
    print("=" * 70)

    for kind in ("unsafe", "safe"):
        sub = [r for r in rows if r["verdict"] == kind]
        if not sub:
            continue
        print("\n{} programs: {}".format(kind, len(sub)))
        for k, v in Counter(r["uniform_outcome"] for r in sub).most_common():
            print("   uniform {:<14} {:>5}  ({:.1f}%)".format(
                k, v, 100.0 * v / len(sub)))
        ks = sorted(num(r["uniform_kstar"]) for r in sub
                    if r["uniform_outcome"] == "found" and num(r["uniform_kstar"]) is not None)
        if ks:
            zero = sum(1 for x in ks if x == 0)
            nz = [x for x in ks if x > 0]
            print("   k*=0 (loop-independent, excluded): {}".format(zero))
            if nz:
                print("   k*>0: n={} min={} median={} max={}".format(
                    len(nz), nz[0], nz[len(nz) // 2], nz[-1]))

    other = [r for r in rows if r["other_property_violation"]]
    if other:
        print("\nprograms where a DIFFERENT property was violated "
              "(recorded, not counted): {}".format(len(other)))

    unexp = [r for r in rows if r["schedule_outcome"] == "unexpected_violation"]
    if unexp:
        print("unexpected violations on safe programs: {}  "
              "(investigate: profile or benchmark verdict)".format(len(unexp)))

    multi = [r for r in rows if (num(r["n_loops"]) or 0) > 1]
    both = [r for r in multi if r["uniform_outcome"] == "found"
            and r["schedule_outcome"] == "found" and r["cost_ratio"] != ""]
    if both:
        ratios = sorted(float(r["cost_ratio"]) for r in both)
        print("\nschedule vs uniform cost, multi-loop only: n={}".format(len(ratios)))
        print("   median={:.2f}  max={:.1f}   >2x in {} ({:.1f}%)".format(
            ratios[len(ratios) // 2], ratios[-1],
            sum(1 for x in ratios if x > 2),
            100.0 * sum(1 for x in ratios if x > 2) / len(ratios)))

    print("\nby loop category (uniform, k*>0):")
    for cat in ("constant", "input-bounded", "data-dependent", "mixed"):
        sub = [r for r in rows if r["category"] == cat]
        if not sub:
            continue
        nz = sorted(x for x in
                    (num(r["uniform_kstar"]) for r in sub
                     if r["uniform_outcome"] == "found")
                    if x is not None and x > 0)
        if nz:
            print("   {:<16} found={}/{}  median={} range={}-{}".format(
                cat, len(nz), len(sub), nz[len(nz) // 2], nz[0], nz[-1]))
        else:
            print("   {:<16} found=0/{}".format(cat, len(sub)))

    usable = [r for r in rows if r["uniform_outcome"] == "found"
              and (num(r["uniform_kstar"]) or 0) > 0]
    print("\nUSABLE FOR RQ1 (finite k*>0): {}".format(len(usable)))
    print("   by verdict : {}".format(dict(Counter(r["verdict"] for r in usable))))
    print("   by property: {}".format(dict(Counter(r["property"] for r in usable))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--mode", choices=["uniform", "schedule", "both"], default="both")
    ap.add_argument("--budget", type=int, default=600)
    ap.add_argument("--per-k-timeout", type=int, default=180)
    ap.add_argument("--max-k", type=int, default=100000)
    ap.add_argument("--properties", nargs="+", default=list(PROP_KEYS),
                    choices=list(PROP_KEYS))
    ap.add_argument("--verdicts", nargs="+", default=["unsafe", "safe"],
                    choices=["unsafe", "safe"])
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()

    outp = Path(a.out)
    if a.summary_only:
        summarise(outp)
        return 0

    recs = json.load(open(a.json))
    jobs_spec = build_jobs(recs, a.properties, set(a.verdicts))
    Path(a.trace_dir).mkdir(parents=True, exist_ok=True)
    outp.parent.mkdir(parents=True, exist_ok=True)

    if outp.exists() and not a.restart:
        done = set()
        with open(outp) as fh:
            for r in csv.DictReader(fh):
                done.add((r["task"], r["property"]))
        before = len(jobs_spec)
        jobs_spec = [j for j in jobs_spec if (j[0]["task"], j[1]) not in done]
        print("resuming: {} done, {} to go".format(before - len(jobs_spec),
                                                   len(jobs_spec)))
    else:
        with open(outp, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=COLS).writeheader()

    if not jobs_spec:
        print("nothing to do")
        summarise(outp)
        return 0

    jobs_spec.sort(key=lambda j: len(j[0].get("loops", [])))
    jobs = [(rec, prop, verdict, a.dataset, a.trace_dir, a.mode,
             a.budget, a.max_k, a.per_k_timeout)
            for (rec, prop, verdict) in jobs_spec]

    print("work items: {}   mode={}".format(len(jobs), a.mode))
    print("worst case ~{:.1f} h at --jobs {}".format(
        len(jobs) * a.budget * 2 / 3600.0 / max(1, a.jobs), a.jobs))

    t0, n_done = time.time(), 0
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(process, j): (j[0]["task"], j[1]) for j in jobs}
        for f in as_completed(futs):
            task, prop = futs[f]
            try:
                row = f.result()
            except Exception as e:
                row = {c: "" for c in COLS}
                row.update({"task": task, "property": prop,
                            "uniform_outcome": "worker_error",
                            "uniform_failing_property": str(e)[:60]})
            with open(outp, "a", newline="") as fh:
                csv.DictWriter(fh, fieldnames=COLS).writerow(row)
            n_done += 1
            print("[{}/{}] {:<38} {:<10} {:<7} {:<9} k*={:<6} ({:.0f}m)".format(
                n_done, len(jobs), row["task"][:38], row["property"],
                row["verdict"], row["uniform_outcome"] or "-",
                row["uniform_kstar"] if row["uniform_kstar"] != "" else "-",
                (time.time() - t0) / 60.0), flush=True)

    summarise(outp)
    print("\nresults -> {}".format(outp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())