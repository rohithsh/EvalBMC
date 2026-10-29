#!/usr/bin/env python3
"""
probe_kinduction.py

Feasibility probe. Before building anything on k-induction, find out whether it
can decide the programs that plain BMC could not.

WHY THIS MATTERS
  In the k* derivation, roughly two thirds of program-property pairs never
  reached a conclusive result. For a safe program k* is the completeness
  threshold -- every loop exhausted -- and a loop whose trip count is an
  unconstrained input has no such threshold at any finite depth. Plain BMC can
  never prove those safe.

  k-induction does not need a completeness threshold. It proves two things at
  depth k: no counterexample within k steps (base case), and that the property
  holding for k consecutive steps implies it holds for the next (inductive
  step). Together these give a proof for all steps, so a loop running a million
  times can be proved at k=2 if the invariant is 2-inductive.

  The open question is whether these particular programs are k-inductive at
  all. k-induction commonly fails for want of auxiliary invariants, whatever k
  is used. If that is what happens here, predicting the induction depth is
  pointless and the idea should be dropped now rather than after a long run.

WHAT IT MEASURES
  For each sampled program the script runs a k-induction verifier at increasing
  depths and records the smallest k that decides it, or that none did. Three
  outcomes matter:

    proved / refuted  k-induction decided it -- BMC could not
    inconclusive      ran but reached no verdict, usually a missing invariant
    timeout           no verdict within the budget

  The headline number is the share decided. If it is near zero, the approach
  fails for reasons no depth prediction can fix.

VERIFIERS
  cpachecker  -kInduction
  esbmc       --k-induction --k-step 1 --max-k-step N

  Both are given the same depth ladder so their k values are comparable.

Usage:
  python3 probe_kinduction.py \
      --kstar   results/kstar_all.csv \
      --dataset datasets/cleaned/svcomp_clean \
      --json    datasets/cleaned/loops.json \
      --out     results/kinduction_probe.csv \
      --verifier cpachecker --tool-path /opt/cpachecker/bin/cpachecker \
      [--n 20] [--max-k 20] [--timeout 300] [--jobs 4]
"""

import argparse
import csv
import json
import random
import re
import subprocess
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROP_FILE = {
    "reach": "unreach-call.prp",
    "memsafety": "valid-memsafety.prp",
    "overflow": "no-overflow.prp",
}

# CPAchecker and ESBMC report verdicts differently; both are normalised to
# proved / refuted / inconclusive
CPA_TRUE = re.compile(r"Verification result:\s*TRUE", re.I)
CPA_FALSE = re.compile(r"Verification result:\s*FALSE", re.I)
CPA_UNKNOWN = re.compile(r"Verification result:\s*UNKNOWN", re.I)

ESBMC_TRUE = re.compile(r"VERIFICATION SUCCESSFUL", re.I)
ESBMC_FALSE = re.compile(r"VERIFICATION FAILED", re.I)


def run_cpachecker(tool, cfile, prop, k, data_model, timeout, spec_dir):
    """
    CPAchecker with k-induction. -setprop overrides the induction depth so the
    ladder is under our control rather than the tool's own escalation.
    """
    spec = str(Path(spec_dir) / PROP_FILE[prop]) if spec_dir else None
    cmd = [tool, "-kInduction",
           "-setprop", "cpa.loopbound.maxLoopIterations={}".format(k),
           "-setprop", "cpa.loopbound.maxLoopIterationsUpperBound={}".format(k),
           "-timelimit", "{}s".format(timeout),
           "-nooutput"]
    if spec:
        cmd += ["-spec", spec]
    cmd += ["-64" if data_model == "LP64" else "-32", str(cfile)]

    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout + 30)
    except subprocess.TimeoutExpired:
        return "timeout", ""
    except FileNotFoundError:
        return "error", "cpachecker not found at {}".format(tool)

    out = (p.stdout or "") + (p.stderr or "")
    if CPA_TRUE.search(out):
        return "proved", ""
    if CPA_FALSE.search(out):
        return "refuted", ""
    if CPA_UNKNOWN.search(out):
        return "inconclusive", ""
    return "inconclusive", (out.strip().splitlines() or [""])[-1][:120]


def run_esbmc(tool, cfile, prop, k, data_model, timeout, spec_dir):
    """ESBMC with k-induction, capped at the given depth."""
    cmd = [tool, str(cfile), "--k-induction",
           "--max-k-step", str(k), "--k-step", "1",
           "--timeout", "{}s".format(timeout),
           "--32" if data_model == "ILP32" else "--64"]
    if prop == "overflow":
        cmd.append("--overflow-check")
    elif prop == "memsafety":
        cmd += ["--memory-leak-check", "--pointer-check", "--bounds-check"]
    else:
        cmd.append("--no-pointer-check")
        cmd.append("--no-bounds-check")

    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout + 30)
    except subprocess.TimeoutExpired:
        return "timeout", ""
    except FileNotFoundError:
        return "error", "esbmc not found at {}".format(tool)

    out = (p.stdout or "") + (p.stderr or "")
    if ESBMC_TRUE.search(out):
        return "proved", ""
    if ESBMC_FALSE.search(out):
        return "refuted", ""
    return "inconclusive", (out.strip().splitlines() or [""])[-1][:120]


RUNNERS = {"cpachecker": run_cpachecker, "esbmc": run_esbmc}


def probe(job):
    (row, dataset, verifier, tool, max_k, timeout, spec_dir) = job
    cfile = Path(dataset) / row["dir"] / row["c_file"]
    prop = row["property"]
    dm = row.get("data_model", "ILP32")

    out = {
        "task": row["task"], "property": prop, "verdict": row["verdict"],
        "category": row.get("category", ""), "n_loops": row.get("n_loops", ""),
        "bmc_outcome": row.get("uniform_outcome", ""),
        "bmc_k_reached": row.get("uniform_k_reached", ""),
        "ki_outcome": "", "ki_k": "", "ki_time": "", "ki_detail": "",
        "agrees_with_expected": "",
    }
    if not cfile.exists():
        out["ki_outcome"] = "missing_file"
        return out

    runner = RUNNERS[verifier]
    start = time.time()
    per_k = max(30, timeout // max(1, max_k))

    # climb the depth ladder: the smallest k that decides it is what a
    # predictor would need to guess
    for k in range(1, max_k + 1):
        left = timeout - (time.time() - start)
        if left <= 5:
            out["ki_outcome"] = "timeout"
            out["ki_k"] = k - 1
            break
        status, detail = runner(tool, cfile, prop, k, dm,
                                int(min(per_k, left)), spec_dir)
        if status in ("proved", "refuted"):
            out["ki_outcome"] = status
            out["ki_k"] = k
            out["ki_detail"] = detail
            break
        if status == "error":
            out["ki_outcome"] = "error"
            out["ki_detail"] = detail
            break
        out["ki_detail"] = detail
    else:
        out["ki_outcome"] = "inconclusive"
        out["ki_k"] = max_k

    out["ki_time"] = round(time.time() - start, 1)

    # does the verdict match what the benchmark says?
    if out["ki_outcome"] == "proved":
        out["agrees_with_expected"] = "yes" if row["verdict"] == "safe" else "NO"
    elif out["ki_outcome"] == "refuted":
        out["agrees_with_expected"] = "yes" if row["verdict"] == "unsafe" else "NO"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kstar", required=True,
                    help="k* results; the programs with no finite bound are "
                         "the ones worth probing")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--json", required=True, help="loops.json, for data_model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--verifier", choices=list(RUNNERS), default="cpachecker")
    ap.add_argument("--tool-path", required=True,
                    help="path to the cpachecker or esbmc executable")
    ap.add_argument("--spec-dir", default=None,
                    help="directory holding SV-COMP .prp files (CPAchecker)")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-k", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=300,
                    help="seconds for the whole depth ladder on one program")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-solved", action="store_true",
                    help="also probe programs BMC did solve, as a control")
    a = ap.parse_args()

    models = {r["task"]: r.get("data_model", "ILP32")
              for r in json.load(open(a.json))}

    rows = list(csv.DictReader(open(a.kstar)))
    unsolved = [r for r in rows
                if r.get("uniform_outcome") in ("timeout", "cap_reached")]
    solved = [r for r in rows if r.get("uniform_outcome") == "found"]

    print("population")
    print("   BMC found no finite bound : {}".format(len(unsolved)))
    print("   BMC found a bound          : {}".format(len(solved)))

    rng = random.Random(a.seed)

    # spread the sample over verdict and category rather than taking the first N
    def spread(pool, n):
        buckets = {}
        for r in pool:
            buckets.setdefault((r["verdict"], r.get("category") or "mixed"),
                               []).append(r)
        picked, keys = [], sorted(buckets)
        i = 0
        while len(picked) < n and keys:
            k = keys[i % len(keys)]
            if buckets[k]:
                picked.append(buckets[k].pop(rng.randrange(len(buckets[k]))))
            else:
                keys.remove(k)
                continue
            i += 1
        return picked

    sample = spread(unsolved, a.n)
    if a.include_solved:
        sample += spread(solved, max(5, a.n // 4))

    for r in sample:
        r["data_model"] = models.get(r["task"], "ILP32")

    print("\nprobing {} programs with {} (max k={}, {}s each)".format(
        len(sample), a.verifier, a.max_k, a.timeout))

    jobs = [(r, a.dataset, a.verifier, a.tool_path, a.max_k, a.timeout,
             a.spec_dir) for r in sample]

    results = []
    done = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(probe, j) for j in jobs]
        for f in as_completed(futs):
            try:
                res = f.result()
            except Exception as e:
                res = {"task": "?", "ki_outcome": "worker_error",
                       "ki_detail": str(e)[:100]}
            results.append(res)
            done += 1
            print("[{}/{}] {:<42} {:<10} {:<14} k={:<4} {}s".format(
                done, len(jobs), str(res.get("task"))[:42],
                res.get("property", ""), res.get("ki_outcome", ""),
                res.get("ki_k", "-"), res.get("ki_time", "")), flush=True)

    cols = ["task", "property", "verdict", "category", "n_loops",
            "bmc_outcome", "bmc_k_reached", "ki_outcome", "ki_k", "ki_time",
            "agrees_with_expected", "ki_detail"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    # ---- verdict ---------------------------------------------------------
    print("\n" + "=" * 62)
    print("RESULT")
    print("=" * 62)
    oc = Counter(r.get("ki_outcome") for r in results)
    for k, v in oc.most_common():
        print("   {:<16} {:>4}  ({:.0f}%)".format(k, v, 100.0 * v / len(results)))

    decided = [r for r in results if r.get("ki_outcome") in ("proved", "refuted")]
    print("\n   decided by k-induction: {}/{}  ({:.0f}%)".format(
        len(decided), len(results), 100.0 * len(decided) / max(1, len(results))))

    wrong = [r for r in decided if r.get("agrees_with_expected") == "NO"]
    if wrong:
        print("   DISAGREES with the benchmark verdict: {} -- check these "
              "before trusting the rest".format(len(wrong)))
        for r in wrong[:5]:
            print("      {} [{}] expected {} got {}".format(
                r["task"], r["property"], r["verdict"], r["ki_outcome"]))

    if decided:
        ks = sorted(int(r["ki_k"]) for r in decided if str(r["ki_k"]).isdigit())
        if ks:
            print("\n   induction depth needed: min={} median={} max={}".format(
                ks[0], ks[len(ks) // 2], ks[-1]))
            small = sum(1 for x in ks if x <= 3)
            print("   decided at k<=3: {}/{} -- a small depth suffices, so the "
                  "value of predicting it is limited".format(small, len(ks)))
        print("\n   by category:")
        for cat in ("constant", "input-bounded", "data-dependent", "mixed"):
            sub = [r for r in results if r.get("category") == cat]
            if sub:
                d = sum(1 for r in sub
                        if r.get("ki_outcome") in ("proved", "refuted"))
                print("      {:<16} {}/{} decided".format(cat, d, len(sub)))

    print("\n   Read this as: a high decided rate means k-induction reaches "
          "programs BMC cannot,")
    print("   and predicting the depth is worth pursuing. A rate near zero "
          "means these programs")
    print("   are not k-inductive and no depth prediction would help.")
    print("\nresults -> {}".format(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())