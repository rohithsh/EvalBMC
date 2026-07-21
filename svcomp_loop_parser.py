#!/usr/bin/env python3
"""
svcomp_loop_parser.py  (v7)

Per-program JSON description of every loop in the SV-COMP dataset.

TAXONOMY (three categories):
  constant        iteration count fixed at compile time (literal, sizeof(...),
                  constant-started counter).
  input-bounded   bound fixed on ENTRY, unknown at compile time (parameter,
                  nondet drawn once, uninitialised var), not changed in the loop.
                  'bound_constrained' records whether a precondition caps it.
  data-dependent  bound can CHANGE during execution: a bound-relevant variable
                  is modified in the loop or by an enclosing loop; the guard
                  re-draws nondet each iteration; the guard reads memory; or the
                  guard calls an (impure) function whose value drives the loop.

Fixes over v6:
  * sizeof(...) is invisible to bound extraction: a variable appearing only
    inside sizeof() is a compile-time size query, not a runtime read, so it
    neither counts as a bound variable nor triggers data-dependent when the
    array's CONTENTS are modified. (e.g. `for(i=0;i<sizeof(array);i++) array[i]=...`
    is constant.)
  * a call in the guard defaults to data-dependent (its return value is
    re-evaluated each iteration and drives the loop); only a small allowlist of
    pure functions (strlen/strnlen) can be input-bounded, and then only if their
    arguments are not modified in the loop. sizeof is excluded from call detection.

Loop ids come from `cbmc --show-loops` (authoritative). Loops from
builtins/headers are dropped; loops are matched to the AST by source line.

Usage:
  python3 svcomp_loop_parser.py --dataset datasets/svcomp_clean \
      --out results/loops.json [--jobs 8] [--limit 50] [--dir loops]
"""

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import clang.cindex as ci

try:
    import yaml
except ImportError:
    print("need pyyaml: pip install pyyaml", file=sys.stderr)
    raise

CBMC = "cbmc"
PROPS = ["unreach-call", "valid-memsafety", "no-overflow"]
COMPARE_OPS = {"<", "<=", ">", ">=", "!=", "=="}
TRUE_LITERALS = {"1", "true", ""}
FALSE_LITERALS = {"0", "false"}
PURE_FUNCS = {"strlen", "strnlen"}          # pure calls allowed to be input-bounded


# ------------------------- cbmc --show-loops -------------------------

SHOW_LOOP_RE = re.compile(r"^Loop\s+([A-Za-z_][\w.$]*\.\d+):\s*$")
LOC_RE = re.compile(r"^\s*file\s+(.+?)\s+line\s+(\d+)\s+function\s+(\S+)\s*$")

def is_int_literal_token(t):
    """True for 1, 1U, 1L, 1UL, 0x1F, -1, etc."""
    s = t.lstrip("-")
    if not s:
        return False
    # strip a trailing integer-suffix (u/l combinations)
    core = s.rstrip("uUlL")
    if not core:
        return False
    try:
        int(core, 0)          # base 0 handles 0x.., 0.., decimal
        return True
    except ValueError:
        return False


def cbmc_loops(cfile, data_model, timeout=120):
    flag = "--32" if data_model == "ILP32" else "--64"
    cmd = [CBMC, "--show-loops", str(cfile), flag]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "cbmc --show-loops timeout"
    except FileNotFoundError:
        return None, "cbmc not found"

    out, pending = [], None
    for line in p.stdout.splitlines():
        m = SHOW_LOOP_RE.match(line)
        if m:
            pending = m.group(1)
            continue
        if pending:
            m2 = LOC_RE.match(line)
            if m2:
                out.append({"id": pending, "file": m2.group(1),
                            "line": int(m2.group(2)), "function": m2.group(3)})
                pending = None
    if not out and p.returncode != 0:
        return None, (p.stderr or "")[:200]
    return out, ""


# ------------------------- libclang basics -------------------------

def tok(n):
    return [t.spelling for t in n.get_tokens()]

def tok_text(n):
    return " ".join(tok(n))

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

def idents_no_sizeof(n):
    """Identifiers READ in an expression, skipping anything inside sizeof().
    sizeof(a) is a compile-time size query, not a read of a's value."""
    out = []
    def w(x, in_sizeof):
        is_sizeof = in_sizeof or (tok(x)[:1] == ["sizeof"])
        if x.kind == ci.CursorKind.DECL_REF_EXPR and not is_sizeof:
            out.append(x.spelling)
        for c in x.get_children():
            w(c, is_sizeof)
    if n is not None:
        w(n, False)
    return out

def is_loop(n):
    return n.kind in (ci.CursorKind.WHILE_STMT,
                      ci.CursorKind.FOR_STMT,
                      ci.CursorKind.DO_STMT)

def loop_kind(n):
    return {ci.CursorKind.WHILE_STMT: "while",
            ci.CursorKind.FOR_STMT: "for",
            ci.CursorKind.DO_STMT: "do"}[n.kind]


def for_parts(loop):
    ch = list(loop.get_children())
    if not ch:
        return None, None, None, None
    body, rest = ch[-1], ch[:-1]
    cond_idx = None
    for i, c in enumerate(rest):
        if c.kind == ci.CursorKind.BINARY_OPERATOR and \
           any(op in tok(c) for op in COMPARE_OPS):
            cond_idx = i
            break
    if cond_idx is None:
        return (rest[0] if rest else None), None, \
               (rest[-1] if len(rest) > 1 else None), body
    return (rest[cond_idx - 1] if cond_idx >= 1 else None), \
           rest[cond_idx], \
           (rest[cond_idx + 1] if cond_idx + 1 < len(rest) else None), body


def guard_and_body(loop):
    if loop.kind == ci.CursorKind.FOR_STMT:
        _, cond, _, body = for_parts(loop)
        return cond, body
    ch = list(loop.get_children())
    body = None
    for c in reversed(ch):
        if c.kind == ci.CursorKind.COMPOUND_STMT:
            body = c
            break
    if body is None and ch:
        body = ch[-1]
    guard = None
    for c in ch:
        if body is None or c != body:
            guard = c
            break
    return guard, body


def for_increment(loop):
    if loop.kind != ci.CursorKind.FOR_STMT:
        return None
    return for_parts(loop)[2]


def expr_has_nondet(n):
    if n is None:
        return False
    t = "".join(tok(n))
    return ("__VERIFIER_nondet" in t) or ("nondet_" in t) or ("unknown(" in t)

def _bare_literal(g):
    if g is None or idents(g):
        return None
    return "".join(t for t in tok(g) if t not in ("(", ")"))

def guard_is_always_true(g):
    if g is None:
        return True
    lit = _bare_literal(g)
    return lit is not None and lit in TRUE_LITERALS

def guard_is_always_false(g):
    lit = _bare_literal(g)
    return lit is not None and lit in FALSE_LITERALS

def guard_memory(g):
    """Array subscript or pointer deref in the guard -- but NOT inside sizeof()."""
    def w(n, in_sizeof):
        sz = in_sizeof or (tok(n)[:1] == ["sizeof"])
        if not sz:
            if n.kind == ci.CursorKind.ARRAY_SUBSCRIPT_EXPR:
                return True
            if n.kind == ci.CursorKind.UNARY_OPERATOR and tok(n)[:1] == ["*"]:
                return True
            if n.kind == ci.CursorKind.MEMBER_REF_EXPR:
                return True
        return any(w(c, sz) for c in n.get_children())
    return w(g, False) if g is not None else False

def guard_literals(g):
    out = []
    def w(n):
        if n.kind == ci.CursorKind.INTEGER_LITERAL:
            tt = tok(n)
            if tt:
                out.append(tt[0])
        for c in n.get_children():
            w(c)
    if g is not None:
        w(g)
    return out

def guard_calls_nonsizeof(g):
    """Call expressions in the guard, excluding sizeof (an operator)."""
    if g is None:
        return []
    return [c for c in g.walk_preorder()
            if c.kind == ci.CursorKind.CALL_EXPR and c.spelling not in ("", "sizeof")]


def rhs_is_constant(n):
    """Compile-time constant: reads no variable and calls no function.
    Variables inside sizeof() are skipped."""
    if n is None:
        return False
    def scan(x, in_sizeof):
        is_sizeof = in_sizeof or (tok(x)[:1] == ["sizeof"])
        if x.kind == ci.CursorKind.CALL_EXPR and not is_sizeof:
            return False
        if x.kind == ci.CursorKind.DECL_REF_EXPR and not is_sizeof:
            return False
        return all(scan(c, is_sizeof) for c in x.get_children())
    return bool(tok(n)) and scan(n, False)


def ASSIGN_ONLY(tt):
    return ("=" in tt) and not any(o in tt for o in ("==", "!=", "<=", ">="))


def is_const_step(tgt, rhs):
    if rhs is None:
        return False
    tt = tok(rhs)
    if not any(op in tt for op in ("+", "-")):
        return False
    names = idents(rhs)
    lits = [t for t in tt if is_int_literal_token(t)]
    return names == [tgt] and len(lits) >= 1


def writes_in(node, exclude=None):
    written, impure = set(), set()

    def note(v, step):
        written.add(v)
        if not step:
            impure.add(v)

    def w(n):
        if exclude is not None and n == exclude:
            return
        if n.kind == ci.CursorKind.BINARY_OPERATOR and ASSIGN_ONLY(tok(n)):
            k = list(n.get_children())
            if k:
                nm = idents(k[0])
                if nm:
                    note(nm[0], is_const_step(nm[0], k[1] if len(k) > 1 else None))
        elif n.kind == ci.CursorKind.UNARY_OPERATOR and ("++" in tok(n) or "--" in tok(n)):
            nm = idents(n)
            if nm:
                note(nm[0], True)
        elif n.kind == ci.CursorKind.COMPOUND_ASSIGNMENT_OPERATOR:
            k = list(n.get_children())
            if k:
                nm = idents(k[0])
                if nm:
                    tt = tok(n)
                    step = (("+=" in tt) or ("-=" in tt)) and \
                           any(is_int_literal_token(t) for t in tt)
                    note(nm[0], step)
        for c in n.get_children():
            w(c)

    if node is not None:
        w(node)
    return written, {v for v in written if v not in impure}


def loop_writes(loop, exclude=None):
    _, body = guard_and_body(loop)
    written, impure = set(), set()
    for part in (body, for_increment(loop)):
        if part is None:
            continue
        w, cs = writes_in(part, exclude=exclude)
        written |= w
        impure |= (w - cs)
    return written, written - impure


def nondet_controlled(body):
    ctrl = set()

    def record(n):
        if n.kind == ci.CursorKind.BINARY_OPERATOR and ASSIGN_ONLY(tok(n)):
            k = list(n.get_children())
            if k:
                nm = idents(k[0])
                if nm:
                    ctrl.add(nm[0])
        elif n.kind == ci.CursorKind.UNARY_OPERATOR and ("++" in tok(n) or "--" in tok(n)):
            nm = idents(n)
            if nm:
                ctrl.add(nm[0])

    def w(n, under):
        if n.kind == ci.CursorKind.IF_STMT:
            k = list(n.get_children())
            cond = k[0] if k else None
            nu = under or expr_has_nondet(cond)
            if cond is not None:
                w(cond, under)
            for b in k[1:]:
                w(b, nu)
            return
        if under:
            record(n)
        for c in n.get_children():
            w(c, under)

    if body is not None:
        w(body, False)
    return ctrl


def body_facts(body):
    arrays, derefs, nondets, calls = set(), 0, 0, set()

    def w(n):
        nonlocal derefs, nondets
        if n.kind == ci.CursorKind.ARRAY_SUBSCRIPT_EXPR:
            nm = idents(n)
            if nm:
                arrays.add(nm[0])
        if n.kind == ci.CursorKind.UNARY_OPERATOR and tok(n)[:1] == ["*"]:
            derefs += 1
        if n.kind == ci.CursorKind.CALL_EXPR:
            nm = n.spelling or ""
            if "nondet" in nm or nm == "unknown":
                nondets += 1
            elif nm and nm != "sizeof":
                calls.add(nm)
        for c in n.get_children():
            w(c)

    if body is not None:
        w(body)
    return sorted(arrays), derefs, nondets, sorted(calls)


# ------------------------- scanning -------------------------

def scan_globals(tu_cursor, cfile_name):
    g = {"declared": set(), "events": []}
    for c in tu_cursor.get_children():
        if c.kind != ci.CursorKind.VAR_DECL:
            continue
        g["declared"].add(c.spelling)
        inits = [x for x in c.get_children() if x.kind != ci.CursorKind.TYPE_REF]
        if inits:
            g["events"].append((c.extent.start.offset, c.spelling, inits[-1]))
    return g


def scan_function(fn, glob):
    params, locals_declared = set(), set()
    constrained = {}
    events = list(glob["events"])

    if fn is None:
        return {"params": params, "locals": locals_declared,
                "globals": set(glob["declared"]),
                "events": sorted(events, key=lambda e: e[0]),
                "constrained": constrained}

    for c in fn.get_children():
        if c.kind == ci.CursorKind.PARM_DECL:
            params.add(c.spelling)

    def w(n):
        if n.kind == ci.CursorKind.VAR_DECL:
            locals_declared.add(n.spelling)
            inits = [c for c in n.get_children() if c.kind != ci.CursorKind.TYPE_REF]
            if inits:
                events.append((n.extent.start.offset, n.spelling, inits[-1]))
        if n.kind == ci.CursorKind.BINARY_OPERATOR and ASSIGN_ONLY(tok(n)):
            k = list(n.get_children())
            if k:
                nm = idents(k[0])
                if nm:
                    events.append((n.extent.start.offset, nm[0],
                                   k[1] if len(k) > 1 else None))
        if n.kind == ci.CursorKind.CALL_EXPR and \
           n.spelling in ("__VERIFIER_assume", "assume", "__CPROVER_assume"):
            txt = tok_text(n)
            for v in idents(n):
                constrained.setdefault(v, txt)
        if n.kind == ci.CursorKind.IF_STMT:
            k = list(n.get_children())
            if k and any(op in tok(k[0]) for op in COMPARE_OPS) and len(k) > 1:
                try:
                    terminates = any(
                        c.kind == ci.CursorKind.RETURN_STMT or
                        (c.kind == ci.CursorKind.CALL_EXPR and
                         c.spelling in ("abort", "exit", "__assert_fail"))
                        for c in k[1].walk_preorder())
                except Exception:
                    terminates = False
                if terminates:
                    txt = tok_text(k[0])
                    for v in idents(k[0]):
                        constrained.setdefault(v, txt)
        for c in n.get_children():
            w(c)

    w(fn)
    events.sort(key=lambda e: e[0])
    return {"params": params, "locals": locals_declared,
            "globals": set(glob["declared"]), "events": events,
            "constrained": constrained}


def origins_before(scan, cutoff_offset):
    init_kind = {}
    for v in scan["globals"]:
        init_kind[v] = "literal"
    for v in scan["locals"]:
        init_kind[v] = "none"
    for v in scan["params"]:
        init_kind[v] = "param"

    nondet_set = set()
    assigned_before = set()

    for off, tgt, rhs in scan["events"]:
        if off >= cutoff_offset:
            break
        assigned_before.add(tgt)
        if expr_has_nondet(rhs):
            init_kind[tgt] = "nondet"; nondet_set.add(tgt)
        elif rhs_is_constant(rhs):
            init_kind[tgt] = "literal"; nondet_set.discard(tgt)
        else:
            rids = idents(rhs) if rhs is not None else []
            if any(x in nondet_set or init_kind.get(x) == "param" for x in rids):
                init_kind[tgt] = "nondet"; nondet_set.add(tgt)
            elif rids and all(init_kind.get(x) == "literal" for x in rids):
                init_kind[tgt] = "literal"; nondet_set.discard(tgt)
            else:
                init_kind[tgt] = "other"; nondet_set.discard(tgt)

    for v in scan["locals"]:
        if v not in assigned_before:
            init_kind[v] = "nondet"
            nondet_set.add(v)

    return {"params": scan["params"], "nondet": nondet_set,
            "init_kind": init_kind, "constrained": scan["constrained"]}


# ------------------------- classification -------------------------

def classify(loop, fn_info, enclosing_written):
    guard, body = guard_and_body(loop)
    extras = {"guard": tok_text(guard) if guard is not None else "",
              "induction_vars": [], "bound_expr": "", "bound_vars": [],
              "bound_constrained": None, "constraint": ""}

    if guard_is_always_false(guard):
        return "constant", "always-false guard (body runs at most once)", extras
    if guard_is_always_true(guard):
        return "data-dependent", "always-true guard (exit decided at runtime)", extras

    written, conststep = loop_writes(loop)

    # a CALL in the guard (sizeof excluded)
    gcalls = guard_calls_nonsizeof(guard)
    if gcalls:
        names = [c.spelling for c in gcalls]
        if any("nondet" in n or n == "unknown" for n in names):
            return "data-dependent", "guard re-draws nondet each iteration", extras
        impure = [n for n in names if n not in PURE_FUNCS]
        if impure:
            return "data-dependent", "guard calls {} (runtime-driven bound)".format(
                ",".join(impure)), extras
        # only pure calls (strlen/strnlen): input-bounded unless args modified
        call_vars = set()
        for c in gcalls:
            call_vars |= set(idents(c))
        extras["bound_expr"] = extras["guard"]
        extras["bound_vars"] = sorted(call_vars)
        if call_vars & (written | enclosing_written):
            return "data-dependent", "pure-call argument modified in loop", extras
        return "input-bounded", "guard bound is a pure call over unmodified input", extras

    # array / pointer deref in guard (sizeof excluded) -> value from memory
    if guard_memory(guard):
        extras["bound_expr"] = extras["guard"]
        return "data-dependent", "array/pointer access in guard", extras

    # bound variables ignore anything inside sizeof()
    refs = list(dict.fromkeys(idents_no_sizeof(guard)))
    lits = guard_literals(guard)
    nc = nondet_controlled(body)

    gc = [r for r in refs if r in nc]
    if gc:
        return "data-dependent", "guard var(s) {} nondet-controlled".format(",".join(gc)), extras

    bad = [r for r in refs if r in written and r not in conststep]
    if bad:
        extras["induction_vars"] = bad
        return "data-dependent", "guard var(s) {} non-constant update".format(",".join(bad)), extras

    induction = [r for r in refs if r in conststep]
    bound_ids = [r for r in refs if r not in induction]
    extras["induction_vars"] = induction
    extras["bound_vars"] = bound_ids
    extras["bound_expr"] = ",".join(bound_ids) if bound_ids else (",".join(lits) if lits else "")

    ik = fn_info["init_kind"]
    con = fn_info["constrained"]

    if bound_ids:
        if any(b in written for b in bound_ids):
            b = next(b for b in bound_ids if b in written)
            return "data-dependent", "bound {} modified in loop".format(b), extras
        if any(b in enclosing_written for b in bound_ids):
            b = next(b for b in bound_ids if b in enclosing_written)
            return "data-dependent", "bound {} modified by enclosing loop".format(b), extras
        if all(ik.get(b) == "literal" for b in bound_ids):
            return "constant", "bound var(s) hold literal constants", extras
        for b in bound_ids:
            if ik.get(b) in ("param", "nondet", "other") or b in fn_info["params"]:
                extras["bound_constrained"] = b in con
                extras["constraint"] = con.get(b, "")
                break
        return "input-bounded", "bound fixed on entry, from input", extras

    if induction:
        starts = [ik.get(v) for v in induction]
        if all(s == "literal" for s in starts):
            return "constant", "literal/sizeof bound; induction starts from constant", extras
        for v in induction:
            if ik.get(v) in ("nondet", "param", "other"):
                extras["bound_constrained"] = v in con
                extras["constraint"] = con.get(v, "")
                break
        return "input-bounded", "literal/sizeof bound; induction starts from input", extras

    if refs:
        ks = [ik.get(v) for v in refs]
        if all(k == "literal" for k in ks):
            return "constant", "guard var literal, unmodified", extras
        return "input-bounded", "guard var from input, unmodified", extras

    # guard has only sizeof / literals and no counter -> constant
    return "constant", "constant guard (sizeof/literal only)", extras


# ------------------------- per-program driver -------------------------

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
    for e in (d.get("properties") or []):
        nm = Path(e.get("property_file", "")).stem
        if nm in PROPS and "expected_verdict" in e:
            v = e["expected_verdict"]
            verdicts[nm] = "safe" if v is True else ("unsafe" if v is False else None)
    opts = d.get("options", {}) or {}
    return {"inputs": [str(n) for n in names if n],
            "data_model": opts.get("data_model", "ILP32"),
            "verdicts": {p_: verdicts.get(p_) for p_ in PROPS}}


def collect_ast_loops(tu_cursor, cfile_name):
    found = []

    def walk(n, stack, fn):
        if n.kind == ci.CursorKind.FUNCTION_DECL and n.is_definition():
            fn = n
        loc = n.location
        in_file = loc.file is not None and Path(str(loc.file)).name == cfile_name
        if is_loop(n) and in_file:
            idx = len(found)
            found.append({"cursor": n, "fn": fn, "line": loc.line,
                          "function": fn.spelling if fn else "?",
                          "kind": loop_kind(n), "depth": len(stack),
                          "parent_idx": stack[-1] if stack else None,
                          "children_idx": []})
            if stack:
                found[stack[-1]]["children_idx"].append(idx)
            stack = stack + [idx]
        for c in n.get_children():
            walk(c, stack, fn)

    walk(tu_cursor, [], None)
    return found


def guard_cutoff(loop):
    if loop.kind == ci.CursorKind.DO_STMT:
        return loop.extent.start.offset
    g, _ = guard_and_body(loop)
    return g.extent.start.offset if g is not None else loop.extent.start.offset


def process(args):
    ymlp, dataset_root = args
    info = parse_yml(ymlp)
    if not info or not info["inputs"]:
        return {"task": "{}/{}".format(ymlp.parent.name, ymlp.name),
                "parse_ok": False, "error": "bad yml"}

    cfile = ymlp.parent / info["inputs"][0]
    rec = {"task": "{}/{}".format(ymlp.parent.name, ymlp.name),
           "dir": ymlp.parent.name, "c_file": cfile.name,
           "data_model": info["data_model"], "verdicts": info["verdicts"],
           "parse_ok": False, "flags": [], "loops": []}
    if not cfile.exists():
        rec["error"] = "missing c file"
        return rec

    cl, err = cbmc_loops(cfile, info["data_model"])
    if cl is None:
        rec["error"] = "cbmc: {}".format(err)
        return rec
    cl = [l for l in cl if Path(l["file"]).name == cfile.name]
    rec["n_loops_cbmc"] = len(cl)

    args_clang = ["-std=gnu99", "-ferror-limit=0", "-w",
                  "-m32" if info["data_model"] == "ILP32" else "-m64"]
    try:
        index = ci.Index.create()
        tu = index.parse(str(cfile), args=args_clang)
    except Exception as e:
        rec["error"] = "libclang: {}".format(e)
        return rec

    glob = scan_globals(tu.cursor, cfile.name)
    ast = collect_ast_loops(tu.cursor, cfile.name)

    by_line = defaultdict(list)
    for i, a in enumerate(ast):
        by_line[a["line"]].append(i)

    scan_cache = {}

    def cbmc_id_at(idx):
        ln = ast[idx]["line"]
        for x in cl:
            if x["line"] == ln:
                return x["id"]
        return None

    def info_for(node):
        fnj = node["fn"]
        key = fnj.spelling if fnj else "?"
        if key not in scan_cache:
            scan_cache[key] = scan_function(fnj, glob)
        return origins_before(scan_cache[key], guard_cutoff(node["cursor"]))

    def enclosing_writes(node):
        out = set()
        p = node["parent_idx"]
        child = node["cursor"]
        while p is not None:
            w, _ = loop_writes(ast[p]["cursor"], exclude=child)
            out |= w
            child = ast[p]["cursor"]
            p = ast[p]["parent_idx"]
        return out

    for l in cl:
        cand = by_line.get(l["line"], [])
        if len(cand) == 0:
            rec["loops"].append({"id": l["id"], "line": l["line"],
                                 "function": l["function"], "category": "unmatched",
                                 "reason": "no AST loop at this line"})
            rec["flags"].append("unmatched:{}".format(l["id"]))
            continue

        if len(cand) == 1:
            a = ast[cand[0]]
            cat, reason, extras = classify(a["cursor"], info_for(a), enclosing_writes(a))
        else:
            results = [classify(ast[j]["cursor"], info_for(ast[j]), enclosing_writes(ast[j]))
                       for j in cand]
            cats_here = {r[0] for r in results}
            if len(cats_here) > 1:
                rec["flags"].append("multiple-loops-on-line:{}:{}".format(l["line"], l["id"]))
                rec["loops"].append({"id": l["id"], "line": l["line"],
                                     "function": l["function"], "category": "ambiguous",
                                     "reason": "{} AST loops on line {} classify differently: {}".format(
                                         len(cand), l["line"], sorted(cats_here))})
                continue
            a = ast[cand[0]]
            cat, reason, extras = results[0]
            reason += " (line shared by {} identically-classified loops)".format(len(cand))

        arrays, derefs, nondets, calls = body_facts(guard_and_body(a["cursor"])[1])
        rec["loops"].append({
            "id": l["id"], "function": l["function"], "line": l["line"],
            "kind": a["kind"], "nesting_depth": a["depth"],
            "parent": cbmc_id_at(a["parent_idx"]) if a["parent_idx"] is not None else None,
            "children": [cbmc_id_at(c) for c in a["children_idx"]],
            "guard": extras["guard"], "induction_vars": extras["induction_vars"],
            "bound_expr": extras["bound_expr"], "bound_vars": extras["bound_vars"],
            "bound_constrained": extras["bound_constrained"], "constraint": extras["constraint"],
            "category": cat, "reason": reason,
            "body": {"array_accesses": arrays, "pointer_derefs": derefs,
                     "nondet_calls": nondets, "calls": calls},
        })

    rec["parse_ok"] = True
    rec["n_loops"] = len(rec["loops"])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dir", default=None)
    a = ap.parse_args()

    root = Path(a.dataset)
    pattern = "{}/*.yml".format(a.dir) if a.dir else "*/*.yml"
    ymls = sorted(root.glob(pattern))
    if a.limit:
        ymls = ymls[:a.limit]
    if not ymls:
        print("no tasks found", file=sys.stderr)
        return 1

    print("parsing {} programs (jobs={})".format(len(ymls), a.jobs))
    records = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(process, (y, root)): y for y in ymls}
        done = 0
        for f in as_completed(futs):
            try:
                records.append(f.result())
            except Exception as e:
                records.append({"task": str(futs[f]), "parse_ok": False,
                                "error": "EXC {}".format(e)})
            done += 1
            if done % 100 == 0:
                print("  {}/{}".format(done, len(ymls)))

    records.sort(key=lambda r: r.get("task", ""))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(records, fh, indent=1)

    ok = [r for r in records if r.get("parse_ok")]
    print("\nparsed ok : {} / {}".format(len(ok), len(records)))
    errs = [r for r in records if not r.get("parse_ok")]
    if errs:
        print("errors    : {}".format(len(errs)))
        for k, v in Counter(r.get("error", "?")[:40] for r in errs).most_common(8):
            print("   {:<44} {}".format(k, v))

    cats = Counter()
    nloops = flags = 0
    for r in ok:
        flags += len(r.get("flags", []))
        for l in r["loops"]:
            cats[l["category"]] += 1
            nloops += 1
    print("\ntotal loops: {}".format(nloops))
    for k, v in cats.most_common():
        print("  {:<16} {}".format(k, v))
    if flags:
        print("\nflags raised: {} (see 'flags' in JSON)".format(flags))
    print("\nJSON -> {}".format(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())