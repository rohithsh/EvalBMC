#!/usr/bin/env python3
"""
analysis.py

Evidence for the Discussion: which bound-choosing strategy suits which kind
of program, and what simple combinations of strategies would achieve.
Everything is computed offline from results that already exist -- no CBMC
run and no model call.

INPUTS
  --json     loops JSON (program structure, loop categories, body features)
  --kstar    k* CSV (iterative deepening: outcome and time per pair)
  --score    label=path to a score CSV from score_predictions.py; repeat per
             strategy (Syntactic, Static, Qwen3.5, ...)
  --kinds    optional CSV dir,kind overriding the built-in SV-COMP mapping
             (leave out for TDX: --group-by-task-name groups by task name)

PART 1 -- WHO SOLVES WHAT, BY PROGRAM KIND AND STRUCTURE
  For each grouping, per group: pairs, pairs iterative deepening solves, and
  per strategy: bounds proposed, pairs solved, and pairs solved where
  iterative deepening found no k*.
  Also per strategy: median CBMC clauses and steps of its SOLVED runs in the
  group (from the score CSV), to explain why a group is cheap or expensive.
  Formula size is known only after a CBMC run, so it is not a routing signal.
  Groupings:
    kind       SV-COMP directory kind (mapping below), or the TDX task-name
               parts with --group-by-task-name
    category   loop-bound category of the pair (constant / input / data / mixed)
    loops      number of loops: 1, 2-3, 4+
    nesting    flat (no nested loop) / nested
    arrays     some loop body indexes an array: yes / no
    pointers   some loop body dereferences a pointer: yes / no
    nondet     some loop body calls a nondet function: yes / no
    property   reach / memsafety / overflow
    verdict    unsafe / safe

PART 2 -- SIMULATED ROUTES (portfolios), from per-pair results
  Time of a strategy on a pair = time to produce the bound + its CBMC run
  (a timeout costs the 600 s limit, an impractical bound or an abstention no
  CBMC time). Iterative deepening's time is its search time (up to its
  budget when it fails).
    ID alone
    <S> alone                      for every strategy S
    <S> then ID                    S first; ID only if S does not solve
    ID + <S> after t s             ID runs; S is started at t s alongside it
                                   (t = 0 is a parallel portfolio); solved if
                                   either solves, time = the first success
    route by category              constant -> Syntactic then ID;
                                   other    -> ID + Qwen3.5 after --route-t
    oracle                         best strategy per pair (upper bound)
  Reported: pairs solved, of them where ID found no k*, total and mean time.

PART 3 -- TIGHTNESS, on the pairs where iterative deepening found k* > 0
  How close each strategy's bound is to its reference (score_predictions.py
  columns score_a, tightness, reference): a single bound is compared with
  the uniform k*, a schedule loop by loop with the per-loop k* schedule.
  Tightness = the bound's loop-body unrollings / the reference's (1 = exact).
  Per strategy, overall and per group (category, kind / TDX parts, loops,
  nesting): bounds compared, exact / over / under / mixed, the median and
  largest tightness of the over-bounds, and how many over-bounds stay
  within 2x of the reference.

OUTPUT
  <out-dir>/groups.csv, routes.csv, tightness.csv, and a printed summary.

Usage
  python3 analysis.py \
      --json datasets/cleaned/loops.json --kstar results/kstar_final.csv \
      --score "Syntactic=results/score_syntactic_partial.csv" \
      --score "Qwen3.5=results/score_qwen397b.csv" \
      --out-dir results/analysis_svcomp

  TDX:
  python3 analysis.py \
      --json results/tdx_cprover_loops.json --kstar results/tdx_kstar.csv \
      --score "Syntactic=results/score_tdx_syntactic.csv" \
      --score "Qwen3.5=results/score_tdx_qwen397b.csv" \
      --group-by-task-name --out-dir results/analysis_tdx
"""

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

PROP_KEYS = {"reach": "unreach-call", "memsafety": "valid-memsafety",
             "overflow": "no-overflow"}
VERIFY_LIMIT = 600.0
TRIVIAL_GUARDS = {"0", "0u", "0l", "0ul", "false", "(0)", "(false)"}

# SV-COMP directory -> program kind. First matching pattern wins. Illustration
# for the Discussion only; print the assignment (--show-kinds) and confirm it.
KIND_RULES = [
    (r"^nla-digbench", "non-linear arithmetic"),
    (r"^(loop-|loops$|loops-crafted|loopsCPP|loop_|loopsuite)", "integer loops"),
    (r"^(array-|reducercommutativity)", "arrays"),
    (r"^(list-|forester-heap|heap-|memsafety|ldv-memsafety|ldv-sets|"
     r"ldv-linux-3\.4-simple|memory-model|memory-alloca)", "heap / data structures"),
    (r"^bitvector", "bit-level"),
    (r"^(ntdrivers|openssl|ldv-|memsafety-cve|busybox|ssh)", "systems code"),
    (r"^(locks|eca-|psyco|product-lines|systemc|control-flow|"
     r"infeasible-control-flow)", "control flow"),
]


def is_trivial(loop):
    if "always-false guard" in (loop.get("reason") or "").lower():
        return True
    return (loop.get("guard") or "").strip().lower().replace(" ", "") in TRIVIAL_GUARDS


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def kind_of(d, override):
    if d in override:
        return override[d]
    for pat, kind in KIND_RULES:
        if re.search(pat, d or ""):
            return kind
    return "other"


def tdx_kind(task):
    """host/guest interface and harness type from a TDX task name such as
    cprover_havoc/tdg_mr_report__expected__cover_entry_cprover_havoc.yml"""
    name = Path(task).name
    side = "host (tdh_*)" if name.startswith("tdh") else \
           "guest (tdg_*)" if name.startswith("tdg") else "other"
    harness = next((h for h in ("cover_entry", "cover_exit", "cover_proof",
                                "safety_requirement") if h in name), "other")
    scen = next((s for s in ("unexpected", "expected", "invalid_input")
                 if "__{}".format(s) in name), "other")
    return side, harness, scen


def features(rec):
    loops = [l for l in rec.get("loops", []) if not is_trivial(l)]
    body = [l.get("body") or {} for l in loops]
    n = len(loops)
    return {
        "loops": "0 (wrappers only)" if n == 0 else "1" if n == 1
                 else "2-3" if n <= 3 else "4+",
        "nesting": "nested" if any((l.get("nesting_depth") or 0) > 0
                                   for l in loops) else "flat",
        "arrays": "yes" if any(b.get("array_accesses") for b in body) else "no",
        "pointers": "yes" if any(b.get("pointer_derefs") for b in body) else "no",
        "nondet": "yes" if any(b.get("nondet_calls") for b in body) else "no",
    }


def strat_time(r):
    """(solved, seconds) for one strategy row; None row = abstention."""
    if r is None:
        return False, 0.0
    sb = r.get("score_b", "")
    pred = fnum(r.get("predict_seconds")) or 0.0
    if sb == "works":
        return True, pred + (fnum(r.get("verify_seconds")) or 0.0)
    if sb == "timeout":
        return False, pred + VERIFY_LIMIT
    if sb in ("impractical", "not_run", ""):
        return False, pred
    return False, pred + (fnum(r.get("verify_seconds")) or 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--kstar", required=True)
    ap.add_argument("--score", action="append", required=True,
                    help="label=path, repeat per strategy")
    ap.add_argument("--kinds", default=None, help="CSV dir,kind override")
    ap.add_argument("--group-by-task-name", action="store_true",
                    help="TDX: group by interface side / harness / scenario")
    ap.add_argument("--route-t", type=float, default=60.0,
                    help="t for the category route (s)")
    ap.add_argument("--ts", default="0,10,30,60,120,300",
                    help="t values for 'ID + S after t s'")
    ap.add_argument("--show-kinds", action="store_true",
                    help="print which directories fall into which kind")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    override = {}
    if a.kinds:
        for r in csv.DictReader(open(a.kinds)):
            override[r["dir"]] = r["kind"]

    recs = {r["task"]: r for r in json.load(open(a.json))}
    truth = {(r["task"], r["property"]): r for r in csv.DictReader(open(a.kstar))}
    strategies = []
    for spec in a.score:
        label, _, path = spec.partition("=")
        strategies.append((label, {(r["task"], r["property"]): r
                                   for r in csv.DictReader(open(path))}))
    labels = [s[0] for s in strategies]

    # ---------------------------------------------------------- per pair
    pairs = []
    for (task, prop), t in truth.items():
        rec = recs.get(task)
        if rec is None:
            continue
        found = t.get("uniform_outcome") == "found"
        id_time = fnum(t.get("uniform_time")) or 0.0
        p = {"task": task, "property": prop, "verdict": t.get("verdict", ""),
             "category": t.get("category", ""), "id_solved": found,
             "id_time": id_time,
             "no_kstar": t.get("uniform_outcome") in ("timeout", "cap_reached")}
        if a.group_by_task_name:
            p["side"], p["harness"], p["scenario"] = tdx_kind(task)
        else:
            p["kind"] = kind_of(rec.get("dir", ""), override)
        p.update(features(rec))
        for label, rows in strategies:
            r = rows.get((task, prop))
            ok, secs = strat_time(r)
            p[label] = {"row": r, "solved": ok, "time": secs,
                        "proposed": bool(r) and r.get("score_b", "")
                        not in ("", "not_run")}
        pairs.append(p)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if a.show_kinds and not a.group_by_task_name:
        by = defaultdict(set)
        for rec in recs.values():
            by[kind_of(rec.get("dir", ""), override)].add(rec.get("dir", ""))
        for k in sorted(by):
            print("{:<24} {}".format(k, ", ".join(sorted(by[k]))))
        print()

    # ---------------------------------------------------------- part 1
    dims = (["side", "harness", "scenario"] if a.group_by_task_name else ["kind"]) + \
        ["category", "loops", "nesting", "arrays", "pointers", "nondet",
         "property", "verdict"]
    grows = []
    for dim in dims:
        groups = defaultdict(list)
        for p in pairs:
            groups[p.get(dim, "")].append(p)
        for g, ps in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            row = {"dimension": dim, "group": g, "pairs": len(ps),
                   "id_solved": sum(p["id_solved"] for p in ps),
                   "id_no_kstar": sum(p["no_kstar"] for p in ps)}
            for label in labels:
                row[label + "_proposed"] = sum(p[label]["proposed"] for p in ps)
                row[label + "_solved"] = sum(p[label]["solved"] for p in ps)
                row[label + "_solved_no_kstar"] = sum(
                    p[label]["solved"] and p["no_kstar"] for p in ps)
            for label in labels:
                for col in ("clauses", "steps"):
                    vals = [fnum(p[label]["row"].get(col)) for p in ps
                            if p[label]["solved"] and p[label]["row"]]
                    vals = [v for v in vals if v is not None]
                    row["{}_median_{}".format(label, col)] = (
                        round(statistics.median(vals)) if vals else "")
            row["any_solved"] = sum(p["id_solved"] or any(p[l]["solved"] for l in labels)
                                    for p in ps)
            grows.append(row)
    with open(out / "groups.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(grows[0].keys()))
        w.writeheader()
        w.writerows(grows)

    # ---------------------------------------------------------- part 2
    def route(name, fn):
        solved = nk = 0
        t_ok, t_fail = [], []
        for p in pairs:
            ok, secs = fn(p)
            solved += ok
            nk += ok and p["no_kstar"]
            (t_ok if ok else t_fail).append(secs)
        med = lambda v: round(statistics.median(v), 1) if v else ""
        mean = lambda v: round(sum(v) / len(v), 1) if v else ""
        # per-pair figures: time to a verdict on the pairs the route solves,
        # and time spent before giving up on the pairs it does not
        return {"route": name, "solved": solved, "solved_no_kstar": nk,
                "solved_median_s": med(t_ok), "solved_mean_s": mean(t_ok),
                "unsolved": len(t_fail), "giveup_mean_s": mean(t_fail),
                "total_s": round(sum(t_ok) + sum(t_fail))}

    def id_only(p):
        return p["id_solved"], p["id_time"]

    def alone(label):
        return lambda p: (p[label]["solved"], p[label]["time"])

    def then_id(label):
        def f(p):
            s = p[label]
            if s["solved"]:
                return True, s["time"]
            return p["id_solved"], s["time"] + p["id_time"]
        return f

    def with_after(label, t):
        """ID starts at 0. If it has not solved the pair by t s, S is started
        alongside it (or as soon as ID gives up, if that is earlier). Wall
        time: the first success, or when both have given up."""
        def f(p):
            s = p[label]
            if p["id_solved"] and p["id_time"] <= t:
                return True, p["id_time"]
            start = min(t, p["id_time"])
            cands = []
            if p["id_solved"]:
                cands.append(p["id_time"])
            if s["solved"]:
                cands.append(start + s["time"])
            if cands:
                return True, min(cands)
            return False, max(p["id_time"], start + s["time"])
        return f

    def by_category(p):
        if p["category"] == "constant" and "Syntactic" in labels:
            return then_id("Syntactic")(p)
        if "Qwen3.5" in labels:
            return with_after("Qwen3.5", a.route_t)(p)
        return id_only(p)

    def oracle(p):
        cands = []
        if p["id_solved"]:
            cands.append(p["id_time"])
        cands += [p[l]["time"] for l in labels if p[l]["solved"]]
        if cands:
            return True, min(cands)
        return False, p["id_time"]

    routes = [route("ID alone", id_only)]
    for label in labels:
        routes.append(route("{} alone".format(label), alone(label)))
    for label in labels:
        routes.append(route("{} then ID".format(label), then_id(label)))
    ts = [float(x) for x in a.ts.split(",") if x.strip()]
    for label in labels:
        for t in ts:
            routes.append(route("ID + {} after {:g} s".format(label, t),
                                with_after(label, t)))
    routes.append(route("route by category (t={:g} s)".format(a.route_t), by_category))
    routes.append(route("oracle (best per pair)", oracle))
    with open(out / "routes.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(routes[0].keys()))
        w.writeheader()
        w.writerows(routes)

    # ---------------------------------------------------------- part 3
    tdims = ["all"] + (["side", "harness"] if a.group_by_task_name else ["kind"]) + \
        ["category", "loops", "nesting"]
    trows = []
    for label in labels:
        for dim in tdims:
            groups = defaultdict(list)
            for p in pairs:
                r = p[label]["row"]
                if not r or r.get("score_a") not in ("exact", "over", "under", "mixed"):
                    continue
                groups["all" if dim == "all" else p.get(dim, "")].append(r)
            for g, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
                c = defaultdict(int)
                for r in rs:
                    c[r["score_a"]] += 1
                over = sorted(t for t in (fnum(r.get("tightness")) for r in rs
                                          if r["score_a"] == "over") if t is not None)
                trows.append({
                    "strategy": label, "dimension": dim, "group": g,
                    "compared": len(rs), "exact": c["exact"], "over": c["over"],
                    "under": c["under"], "mixed": c["mixed"],
                    "over_median_tightness": round(statistics.median(over), 2) if over else "",
                    "over_max_tightness": round(over[-1], 2) if over else "",
                    "over_within_2x": sum(t <= 2.0 for t in over),
                    "schedules": sum(r.get("reference", "").startswith("per-loop")
                                     or r.get("pred_kind") == "schedule" for r in rs),
                })
    if trows:
        with open(out / "tightness.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(trows[0].keys()))
            w.writeheader()
            w.writerows(trows)

    # ---------------------------------------------------------- print
    print("pairs: {}   strategies: {}".format(len(pairs), ", ".join(labels)))
    hdr = "{:<10} {:<26} {:>6} {:>6}".format("dim", "group", "pairs", "ID")
    for l in labels:
        hdr += " {:>16}".format(l[:16])
    print("\nsolved / proposed per strategy (in brackets: solved where ID found no k*)")
    print(hdr)
    for r in grows:
        line = "{:<10} {:<26} {:>6} {:>6}".format(
            r["dimension"], str(r["group"])[:26], r["pairs"], r["id_solved"])
        for l in labels:
            line += " {:>16}".format("{}/{} ({})".format(
                r[l + "_solved"], r[l + "_proposed"], r[l + "_solved_no_kstar"]))
        print(line)
    print("\nroutes")
    print("{:<34} {:>7} {:>6} {:>14} {:>12} {:>15}".format(
        "route", "solved", "no-k*", "median s/solved", "mean s/solved",
        "mean s/give-up"))
    for r in routes:
        print("{:<34} {:>7} {:>6} {:>14} {:>12} {:>15}".format(
            r["route"], r["solved"], r["solved_no_kstar"], r["solved_median_s"],
            r["solved_mean_s"], r["giveup_mean_s"]))
    if trows:
        print("\ntightness on pairs with k* > 0 (bound vs its reference)")
        print("{:<10} {:<10} {:<24} {:>8} {:>6} {:>5} {:>6} {:>6} {:>11} {:>8} {:>7}".format(
            "strategy", "dim", "group", "compared", "exact", "over", "under",
            "mixed", "over: med.", "max", "<=2x"))
        for r in trows:
            if r["dimension"] not in ("all", "category"):
                continue
            print("{:<10} {:<10} {:<24} {:>8} {:>6} {:>5} {:>6} {:>6} {:>11} {:>8} {:>7}".format(
                r["strategy"][:10], r["dimension"], str(r["group"])[:24], r["compared"],
                r["exact"], r["over"], r["under"], r["mixed"],
                r["over_median_tightness"], r["over_max_tightness"], r["over_within_2x"]))
        print("(all groupings in tightness.csv)")
    print("\n-> {}/groups.csv, routes.csv, tightness.csv".format(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())