#!/usr/bin/env python3
"""
predict_syntactic.py

The syntactic-counting baseline. A single pass over the AST reads each loop's
bound straight out of its header, with no reasoning about the property, the
data, or whether a violation is reachable.

WHAT IT DOES
  For a loop of the shape

      for (i = INIT; i OP BOUND; i += STEP)

  the trip count follows arithmetically:

      i <  B, step s   ->  ceil((B - INIT) / s)
      i <= B, step s   ->  ceil((B + 1 - INIT) / s)
      i >  B, step -s  ->  ceil((INIT - B) / s)
      i >= B, step -s  ->  ceil((INIT - B + 1) / s)

  The emitted bound is the trip count PLUS ONE, because --unwind k makes k
  copies of the body and then asserts the loop has finished.

  INIT, BOUND and STEP must all be compile-time constants. sizeof expressions
  are folded, since sizeof(a)/sizeof(a[0]) is the array-length idiom. Anything
  else -- a bound that is a variable, a step that is not constant, a guard that
  reads memory or calls a function, a while loop with no recognisable counter --
  makes the loop UNREADABLE. By default (--partial zero) an unreadable loop is
  given bound 0 and the program is still run, as for every other strategy;
  the strategy abstains only when it can read no loop of the program at all.

  This abstention is the point of the baseline. Syntactic counting is cheap and
  exact where it applies and silent everywhere else, which is what the LLM has
  to beat: not accuracy alone, but accuracy on the programs where cheap methods
  give nothing.

WHAT IT CANNOT DO
  It always returns the loop's full trip count. For a safe program that is the
  right answer. For an unsafe one the correct bound is the depth at which the
  violation first becomes reachable, usually far smaller, so the baseline is
  expected to overshoot badly on unsafe programs. That contrast is a result, not
  a defect: it separates "counting iterations" from "reasoning about when a bug
  appears".

OUTPUT
  The same JSONL as predict_bounds.py, so score_predictions.py scores both
  without modification and the comparison is like for like.

Usage:
  python3 predict_syntactic.py \
      --json    datasets/cleaned/loops.json \
      --dataset datasets/cleaned/svcomp_clean \
      --out     results/pred_syntactic.jsonl \
      --only-tasks results/eval_sample.csv \
      [--emit auto|uniform|schedule] [--jobs 8]
"""

import argparse
import csv
import json
import math
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import clang.cindex as ci

PROP_KEYS = {
    "reach": "unreach-call",
    "memsafety": "valid-memsafety",
    "overflow": "no-overflow",
}

COMPARE_OPS = ["<=", ">=", "!=", "==", "<", ">"]   # longest first when matching


# --------------------------------------------------------------- AST helpers

TRIVIAL_GUARDS = {"0", "0u", "0l", "0ul", "false", "(0)", "(false)"}


def is_trivial(loop):
    """The same definition filter_trivial_loops.py uses."""
    if "always-false guard" in (loop.get("reason") or "").lower():
        return True
    return (loop.get("guard") or "").strip().lower().replace(" ", "") in TRIVIAL_GUARDS


def toks(n):
    return [t.spelling for t in n.get_tokens()]


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


def for_parts(loop):
    """(init, cond, inc, body) from one get_children() call, located by
    position around the condition."""
    ch = list(loop.get_children())
    if not ch:
        return None, None, None, None
    body, rest = ch[-1], ch[:-1]
    ci_idx = None
    for i, c in enumerate(rest):
        if c.kind == ci.CursorKind.BINARY_OPERATOR and \
           any(op in toks(c) for op in COMPARE_OPS):
            ci_idx = i
            break
    if ci_idx is None:
        return (rest[0] if rest else None), None, \
               (rest[-1] if len(rest) > 1 else None), body
    return (rest[ci_idx - 1] if ci_idx >= 1 else None), \
           rest[ci_idx], \
           (rest[ci_idx + 1] if ci_idx + 1 < len(rest) else None), body


def const_value(node, globals_map):
    """
    Evaluate a compile-time constant expression, or None.
    Handles integer literals, simple arithmetic, and sizeof (folded via
    clang's type layout, so sizeof(a)/sizeof(a[0]) works).
    """
    if node is None:
        return None

    # clang can often fold it directly
    try:
        if node.kind == ci.CursorKind.INTEGER_LITERAL:
            t = toks(node)
            if t:
                return int(t[0].rstrip("uUlL"), 0)
    except (ValueError, IndexError):
        pass

    text = "".join(toks(node))
    if not text:
        return None

    # a bare name that is a known constant global
    if text in globals_map:
        return globals_map[text]

    # fold sizeof(X) using clang's type sizes
    def fold_sizeof(expr_node):
        out = []

        def w(x):
            t = toks(x)
            if t[:1] == ["sizeof"]:
                try:
                    ty = x.type
                    sz = ty.get_size() if ty is not None else -1
                except Exception:
                    sz = -1
                out.append(sz)
                return
            for c in x.get_children():
                w(c)

        w(expr_node)
        return out

    # arithmetic over literals only
    safe = re.sub(r"\b([0-9]+)[uUlL]+\b", r"\1", text)
    if re.fullmatch(r"[0-9+\-*/() ]+", safe):
        try:
            v = eval(safe, {"__builtins__": {}}, {})   # digits and operators only
            return int(v)
        except Exception:
            return None

    # sizeof(a)/sizeof(a[0]) and similar
    if "sizeof" in text:
        sizes = fold_sizeof(node)
        if len(sizes) == 2 and all(s > 0 for s in sizes):
            if "/" in text:
                return sizes[0] // sizes[1]
        if len(sizes) == 1 and sizes[0] > 0:
            m = re.fullmatch(r"sizeof\(.*\)\s*\*\s*([0-9]+)", text)
            if m:
                return sizes[0] * int(m.group(1))
            if re.fullmatch(r"sizeof\(.*\)", text):
                return sizes[0]
    return None


def scan_const_globals(tu, cfile_name):
    """File-scope variables initialised to a compile-time constant."""
    out = {}
    for c in tu.cursor.get_children():
        if c.kind != ci.CursorKind.VAR_DECL:
            continue
        loc = c.location
        if loc.file is None or Path(str(loc.file)).name != cfile_name:
            continue
        inits = [x for x in c.get_children() if x.kind != ci.CursorKind.TYPE_REF]
        if inits:
            v = const_value(inits[-1], {})
            if v is not None:
                out[c.spelling] = v
    return out


def local_const_before(fn, var, offset, globals_map):
    """Value of `var` assigned before `offset`, if it is a constant."""
    val = globals_map.get(var)

    def w(n):
        nonlocal val
        if n.extent.start.offset >= offset:
            return
        if n.kind == ci.CursorKind.VAR_DECL and n.spelling == var:
            inits = [x for x in n.get_children() if x.kind != ci.CursorKind.TYPE_REF]
            if inits:
                v = const_value(inits[-1], globals_map)
                val = v
        elif n.kind == ci.CursorKind.BINARY_OPERATOR and "=" in toks(n) \
                and not any(o in toks(n) for o in ("==", "!=", "<=", ">=")):
            kids = list(n.get_children())
            if kids and idents(kids[0])[:1] == [var] and len(kids) > 1:
                val = const_value(kids[1], globals_map)
        for c in n.get_children():
            w(c)

    if fn is not None:
        w(fn)
    return val


def step_of(inc, var):
    """Constant step of `var` in an increment expression, or None."""
    if inc is None:
        return None
    t = toks(inc)
    if not t:
        return None
    if "++" in t:
        return 1
    if "--" in t:
        return -1
    joined = "".join(t)
    m = re.fullmatch(re.escape(var) + r"\+=([0-9]+)[uUlL]*", joined)
    if m:
        return int(m.group(1))
    m = re.fullmatch(re.escape(var) + r"-=([0-9]+)[uUlL]*", joined)
    if m:
        return -int(m.group(1))
    m = re.fullmatch(re.escape(var) + r"=" + re.escape(var) + r"\+([0-9]+)[uUlL]*", joined)
    if m:
        return int(m.group(1))
    m = re.fullmatch(re.escape(var) + r"=" + re.escape(var) + r"-([0-9]+)[uUlL]*", joined)
    if m:
        return -int(m.group(1))
    return None


def trip_count(init, op, bound, step):
    """Iterations of a counted loop, or None when it cannot be determined."""
    if None in (init, bound, step) or step == 0:
        return None
    if op in ("<", "<=") and step > 0:
        limit = bound + (1 if op == "<=" else 0)
        return max(0, int(math.ceil((limit - init) / float(step))))
    if op in (">", ">=") and step < 0:
        limit = bound - (1 if op == ">=" else 0)
        return max(0, int(math.ceil((init - limit) / float(-step))))
    if op == "!=" and step > 0 and bound >= init:
        d = bound - init
        return d // step if d % step == 0 else None
    return None


# ------------------------------------------------------------------ analysis

def analyse_program(rec, dataset):
    """Return {loop_id: trip_count or None} plus a reason per unreadable loop."""
    cpath = Path(dataset) / rec["dir"] / rec["c_file"]
    result, reasons = {}, {}
    if not cpath.exists():
        return None, {"error": "missing file"}

    args = ["-std=gnu99", "-ferror-limit=0", "-w",
            "-m32" if rec.get("data_model") == "ILP32" else "-m64"]
    try:
        tu = ci.Index.create().parse(str(cpath), args=args)
    except Exception as e:
        return None, {"error": "libclang: {}".format(e)[:80]}

    gmap = scan_const_globals(tu, cpath.name)

    # index AST loops by line, matching the ids CBMC gave us
    by_line = {}

    def walk(n, fn):
        if n.kind == ci.CursorKind.FUNCTION_DECL and n.is_definition():
            fn = n
        loc = n.location
        if n.kind in (ci.CursorKind.FOR_STMT, ci.CursorKind.WHILE_STMT,
                      ci.CursorKind.DO_STMT) and loc.file is not None \
                and Path(str(loc.file)).name == cpath.name:
            by_line.setdefault(loc.line, []).append((n, fn))
        for c in n.get_children():
            walk(c, fn)

    walk(tu.cursor, None)

    for l in rec.get("loops", []):
        lid, line = l.get("id"), l.get("line")
        if is_trivial(l):
            # do { ... } while (0), as flagged by the loop classifier. Checked
            # first because a wrapper that comes from a macro shows the
            # macro's tokens to libclang, not the literal 0.
            result[lid] = 0
            reasons[lid] = "do-while(0): body runs once, k = 1"
            continue
        cands = by_line.get(line, [])
        if len(cands) != 1:
            result[lid] = None
            reasons[lid] = "no unique AST loop at line {}".format(line)
            continue
        node, fn = cands[0]

        if node.kind == ci.CursorKind.DO_STMT:
            # do { ... } while (0): the macro-wrapper idiom. The body runs
            # once and the guard is false, so CBMC needs one copy of the body:
            # k = 1. Recorded as trip count 0, which the +1 at emission turns
            # into 1. Without this, one such wrapper anywhere in a program made
            # the whole program unreadable, and on the TDX set -- where about
            # 30% of loops are these wrappers -- the baseline answered nothing.
            kids = list(node.get_children())
            cond = kids[-1] if kids else None
            ct = ("".join(toks(cond)).replace("(", "").replace(")", "")
                  .strip().lower()) if cond is not None else ""
            if ct in ("0", "0u", "0l", "0ul", "false"):
                result[lid] = 0
                reasons[lid] = "do-while(0): body runs once, k = 1"
                continue

        if node.kind != ci.CursorKind.FOR_STMT:
            result[lid] = None
            reasons[lid] = "not a counted for-loop"
            continue

        init, cond, inc, _ = for_parts(node)
        if cond is None:
            result[lid] = None
            reasons[lid] = "no comparison in the loop condition"
            continue

        ct = toks(cond)
        op = next((o for o in COMPARE_OPS if o in ct), None)
        kids = list(cond.get_children())
        if op is None or len(kids) != 2:
            result[lid] = None
            reasons[lid] = "unrecognised loop condition"
            continue

        lhs_names = idents(kids[0])
        var = lhs_names[0] if lhs_names else None
        if var is None:
            result[lid] = None
            reasons[lid] = "no counter on the left of the comparison"
            continue

        bound = const_value(kids[1], gmap)
        if bound is None:
            result[lid] = None
            reasons[lid] = "bound is not a compile-time constant"
            continue

        step = step_of(inc, var)
        if step is None:
            result[lid] = None
            reasons[lid] = "step is not a constant"
            continue

        init_v = None
        if init is not None and init.kind == ci.CursorKind.DECL_STMT:
            # for (int i = 0; ...): the counter is declared in the init clause.
            # Its text is "inti=0", which the pattern below never matched, so
            # every C99-style counted loop used to be reported unreadable.
            for vd in init.get_children():
                if vd.kind == ci.CursorKind.VAR_DECL and vd.spelling == var:
                    vals = [x for x in vd.get_children()
                            if x.kind != ci.CursorKind.TYPE_REF]
                    if vals:
                        init_v = const_value(vals[-1], gmap)
        elif init is not None:
            it = "".join(toks(init))
            m = re.fullmatch(re.escape(var) + r"=(.+)", it)
            if m:
                init_v = const_value(list(init.get_children())[-1], gmap) \
                    if list(init.get_children()) else None
        if init_v is None:
            init_v = local_const_before(fn, var, node.extent.start.offset, gmap)
        if init_v is None:
            result[lid] = None
            reasons[lid] = "initial value is not a compile-time constant"
            continue

        n = trip_count(init_v, op, bound, step)
        if n is None:
            result[lid] = None
            reasons[lid] = "trip count not derivable from init/op/bound/step"
        else:
            result[lid] = n

    return result, reasons


def predict_one(job):
    rec, prop, verdict, dataset, emit, partial = job
    out = {
        "task": rec["task"], "dir": rec["dir"], "c_file": rec["c_file"],
        "property": prop, "verdict": verdict,
        "config": "syntactic", "mode": "ast",
        "category": (lambda c: next(iter(c)) if len(c) == 1 else "mixed")(
            {l["category"] for l in rec.get("loops", [])} or {"none"}),
        "n_loops": len(rec.get("loops", [])),
        "model": {"provider": "syntactic-counting", "model": "ast-pass"},
        "predictions": [], "errors": [],
    }

    t0 = time.time()
    counts, reasons = analyse_program(rec, dataset)
    dt = round(time.time() - t0, 3)

    if counts is None:
        out["errors"].append(reasons.get("error", "analysis failed"))
        return out

    unreadable = [lid for lid, v in counts.items() if v is None]
    readable = {lid: v for lid, v in counts.items() if v is not None}

    # PARTIAL ANSWERS (the paper's rule, the same for every strategy): a
    # strategy that bounds at least one loop is run, and every loop it leaves
    # open is given bound 0. Only a program on which it bounds no loop at all
    # counts as an abstention. --partial abstain restores the old behaviour
    # (abstain as soon as any loop is unreadable).
    if unreadable and readable and partial == "zero":
        sched = {k: int(v) + 1 for k, v in readable.items()}
        sched.update({lid: 0 for lid in unreadable})
        parsed = {"kind": "schedule", "schedule": sched,
                  "confidence": "high",
                  "reasoning": "trip counts read from the loop headers, plus "
                               "one; {} unreadable loop(s) given 0".format(
                                   len(unreadable))}
        out["partial"] = len(unreadable)
        out["predictions"].append({
            "sample": 0, "latency_s": dt, "parsed": parsed,
            "parse_error": "", "raw": json.dumps(
                {"parsed": parsed, "unreadable": unreadable,
                 "reasons": {k: reasons.get(k, "?") for k in unreadable}})})
        return out

    if unreadable:
        # the strategy abstains rather than guessing: a bound it cannot read is
        # recorded as an abstention, not as a wrong number
        out["predictions"].append({
            "sample": 0, "latency_s": dt, "parsed": None,
            "parse_error": "abstain: {} of {} loops unreadable ({})".format(
                len(unreadable), len(counts),
                "; ".join("{}: {}".format(k, reasons.get(k, "?"))
                          for k in unreadable[:3])),
            "raw": json.dumps({"unreadable": unreadable, "reasons": reasons}),
        })
        out["abstained"] = True
        return out

    if not counts:
        out["errors"].append("no loops")
        return out

    # counts holds TRIP COUNTS; CBMC's --unwind k makes k copies of the body
    # and then asserts the loop has finished, so the depth is one more than
    # the trip count. Emitting the trip count itself makes every bound fail
    # its unwinding assertion, which would make this baseline a strawman.
    use_schedule = (emit == "schedule") or (emit == "auto" and len(counts) > 1)
    if use_schedule:
        parsed = {"kind": "schedule",
                  "schedule": {k: int(v) + 1 for k, v in counts.items()},
                  "confidence": "high",
                  "reasoning": "trip counts read from the loop headers, plus one"}
    else:
        parsed = {"kind": "uniform", "k": int(max(counts.values())) + 1,
                  "confidence": "high",
                  "reasoning": "largest trip count read from the loop headers, plus one"}

    out["predictions"].append({
        "sample": 0, "latency_s": dt, "parsed": parsed,
        "parse_error": "", "raw": json.dumps(parsed)})
    return out


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only-tasks", default=None)
    ap.add_argument("--emit", choices=["auto", "uniform", "schedule"],
                    default="auto",
                    help="auto: a schedule when the program has several loops")
    ap.add_argument("--properties", nargs="+", default=list(PROP_KEYS),
                    choices=list(PROP_KEYS))
    ap.add_argument("--verdicts", nargs="+", default=["unsafe", "safe"],
                    choices=["unsafe", "safe"])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--partial", choices=["zero", "abstain"], default="zero",
                    help="zero (default, the paper's rule): run a program on "
                         "which at least one loop is readable, giving the "
                         "unreadable loops bound 0; abstain: abstain as soon "
                         "as any loop is unreadable (the earlier behaviour)")
    a = ap.parse_args()

    recs = json.load(open(a.json))

    allow = None
    if a.only_tasks:
        allow = set()
        with open(a.only_tasks) as fh:
            for r in csv.DictReader(fh):
                allow.add((r["task"], r["property"]))
        print("restricted to {} task/property pairs".format(len(allow)))

    jobs = []
    for r in recs:
        v = r.get("verdicts", {}) or {}
        for prop in a.properties:
            got = v.get(PROP_KEYS[prop])
            if got not in a.verdicts:
                continue
            if allow is not None and (r["task"], prop) not in allow:
                continue
            jobs.append((r, prop, got, a.dataset, a.emit, a.partial))
    if a.limit:
        jobs = jobs[:a.limit]
    print("items: {}".format(len(jobs)))

    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if outp.exists():
        outp.unlink()

    n = predicted = abstained = errored = partial_n = 0
    kinds = {}
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(predict_one, j) for j in jobs]
        for f in as_completed(futs):
            try:
                res = f.result()
            except Exception as e:
                res = {"task": "?", "predictions": [],
                       "errors": ["worker: {}".format(str(e)[:100])]}
            with open(outp, "a") as fh:
                fh.write(json.dumps(res) + "\n")
            n += 1
            preds = res.get("predictions") or []
            if preds and preds[0].get("parsed"):
                predicted += 1
                k = preds[0]["parsed"]["kind"]
                kinds[k] = kinds.get(k, 0) + 1
                partial_n += bool(res.get("partial"))
            elif res.get("abstained"):
                abstained += 1
            else:
                errored += 1

    print("\npredicted {} (of which partial, open loops at 0: {})   "
          "abstained {}   error {}".format(predicted, partial_n, abstained, errored))
    print("coverage: {:.1f}%  (the share of programs where a cheap syntactic "
          "pass reads the bound at all)".format(100.0 * predicted / max(1, n)))
    if kinds:
        print("answer kinds: {}".format(kinds))
    print("predictions -> {}".format(outp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())