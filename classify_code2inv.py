#!/usr/bin/env python3
"""
probe_classify_code2inv_v3.py  (READ-ONLY — writes only a CSV, never edits .c files)

Corrected loop-bound classifier for single-loop Code2Inv programs.

Decision order (first match wins):
  0a. nondet-oracle guard (while(unknown()))                       -> unbounded
  0b. array/pointer deref in guard                                 -> data-dependent
  2.  guard var assigned under nondet-controlled branch            -> unbounded   [87-90]
  3.  guard var written by a non-constant step                     -> data-dependent [83]
  4.  bound is a VARIABLE:
        - bound var modified in loop                               -> data-dependent
        - bound var is nondet/input                                -> parametric  [i<n]
        - bound var holds literal constant                         -> constant
  5.  bound is purely LITERAL (guard like x<100); induction start decides:
        - all induction vars start at a literal                    -> constant    [103, 23/24]
        - some induction var starts at nondet/input                -> parametric  [26,28,99]
  6.  otherwise                                                    -> unclassified

Usage:
  pip install libclang
"""

import sys
import csv
from pathlib import Path
from collections import Counter

import clang.cindex as ci

# ci.Config.set_library_file(r"C:\Program Files\LLVM\bin\libclang.dll")  # if needed

VERBOSE = True
PARSE_ARGS = ["-std=c99", "-ferror-limit=0", "-w"]
COMPARE_OPS = {"<", "<=", ">", ">=", "!=", "=="}

def ASSIGN_ONLY(tt):
    return ("=" in tt) and not any(o in tt for o in ("==", "!=", "<=", ">="))


# ----------------------------- AST helpers -----------------------------

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

def find_loops(node, out):
    if node.kind in (ci.CursorKind.WHILE_STMT, ci.CursorKind.FOR_STMT, ci.CursorKind.DO_STMT):
        out.append(node)
    for c in node.get_children():
        find_loops(c, out)

def guard_and_body(loop):
    ch = list(loop.get_children())
    body = None
    for c in reversed(ch):
        if c.kind == ci.CursorKind.COMPOUND_STMT:
            body = c
            break
    guard = None
    if loop.kind == ci.CursorKind.FOR_STMT:
        exprs = [c for c in ch if c.kind != ci.CursorKind.COMPOUND_STMT]
        for c in exprs:
            if c.kind == ci.CursorKind.BINARY_OPERATOR and any(op in tok(c) for op in COMPARE_OPS):
                guard = c
                break
        if guard is None and exprs:
            guard = exprs[len(exprs) // 2] if len(exprs) >= 2 else exprs[0]
    else:
        for c in ch:
            if c.kind != ci.CursorKind.COMPOUND_STMT:
                guard = c
                break
    return guard, body

def expr_has_nondet(node):
    if node is None:
        return False
    t = "".join(tok(node))
    return ("unknown(" in t) or ("__VERIFIER_nondet" in t) or ("nondet_" in t)

def guard_is_nondet_oracle(guard):
    if guard is None:
        return False
    has_cmp = any(op in tok(guard) for op in COMPARE_OPS)
    return expr_has_nondet(guard) and not has_cmp

def guard_has_memory(guard):
    def w(n):
        if n.kind == ci.CursorKind.ARRAY_SUBSCRIPT_EXPR:
            return True
        if n.kind == ci.CursorKind.UNARY_OPERATOR and tok(n)[:1] == ["*"]:
            return True
        return any(w(c) for c in n.get_children())
    return w(guard) if guard is not None else False

def guard_literals(guard):
    out = []
    def w(n):
        if n.kind == ci.CursorKind.INTEGER_LITERAL:
            tt = tok(n)
            if tt:
                out.append(tt[0])
        for c in n.get_children():
            w(c)
    if guard is not None:
        w(guard)
    return out

def rhs_is_int_literal(node):
    """True if node is an integer literal, incl. negative like -5000."""
    if node is None:
        return False
    tt = [t for t in tok(node) if t not in ("(", ")")]
    s = "".join(tt)
    try:
        int(s)
        return True
    except ValueError:
        return node.kind == ci.CursorKind.INTEGER_LITERAL


# ----------------------- declared vars & never-assigned -----------------------

def declared_and_assigned(fn):
    declared = set()
    assigned = set()
    def w(n):
        if n.kind == ci.CursorKind.VAR_DECL:
            declared.add(n.spelling)
            inits = [c for c in n.get_children() if c.kind != ci.CursorKind.TYPE_REF]
            if inits:
                assigned.add(n.spelling)
        if n.kind == ci.CursorKind.BINARY_OPERATOR and ASSIGN_ONLY(tok(n)):
            kids = list(n.get_children())
            if kids:
                nm = idents(kids[0])
                if nm:
                    assigned.add(nm[0])
        if n.kind == ci.CursorKind.UNARY_OPERATOR and ("++" in tok(n) or "--" in tok(n)):
            nm = idents(n)
            if nm:
                assigned.add(nm[0])
        if n.kind == ci.CursorKind.COMPOUND_ASSIGNMENT_OPERATOR:
            kids = list(n.get_children())
            if kids:
                nm = idents(kids[0])
                if nm:
                    assigned.add(nm[0])
        for c in n.get_children():
            w(c)
    if fn is not None:
        w(fn)
    never = declared - assigned
    return declared, never


# ----------------------- pre-loop init origin -----------------------

def compute_init_kind(fn, loop, declared, never):
    """
    init_kind[v] from PRE-LOOP initial values, covering BOTH:
      * declarator initializers:      int x = 0;
      * statement assignments:        (x = 0);
    processed together in source order (by offset) with nondet taint.
    Never-assigned declared vars are nondet inputs.
    """
    loop_start = loop.extent.start.offset
    events = []  # (offset, target, rhs_node_or_None, is_literal_hint)

    def w(n):
        # declarator initializer:  int x = <init>;
        if n.kind == ci.CursorKind.VAR_DECL:
            off = n.extent.start.offset
            if off < loop_start:
                inits = [c for c in n.get_children() if c.kind != ci.CursorKind.TYPE_REF]
                if inits:
                    events.append((off, n.spelling, inits[-1]))
        # statement assignment:  (x = <rhs>);
        if n.kind == ci.CursorKind.BINARY_OPERATOR and ASSIGN_ONLY(tok(n)):
            off = n.extent.start.offset
            if off < loop_start:
                kids = list(n.get_children())
                if kids:
                    nm = idents(kids[0])
                    if nm:
                        rhs = kids[1] if len(kids) > 1 else None
                        events.append((off, nm[0], rhs))
        for c in n.get_children():
            w(c)
    if fn is not None:
        w(fn)
    events.sort(key=lambda t: t[0])

    init_kind = {v: "none" for v in declared}
    nondet_set = set(never)
    for v in declared:
        if init_kind[v] == "none":
            init_kind[v] = "nondet"
            nondet_set.add(v)

    for _, tgt, rhs in events:
        if expr_has_nondet(rhs):
            init_kind[tgt] = "nondet"; nondet_set.add(tgt)
        elif rhs_is_int_literal(rhs):
            init_kind[tgt] = "literal"; nondet_set.discard(tgt)
        else:
            r_ids = idents(rhs) if rhs is not None else []
            if any(x in nondet_set for x in r_ids):
                init_kind[tgt] = "nondet"; nondet_set.add(tgt)
            elif r_ids and all(init_kind.get(x) == "literal" for x in r_ids):
                init_kind[tgt] = "literal"; nondet_set.discard(tgt)
            else:
                init_kind[tgt] = "other"; nondet_set.discard(tgt)
    return init_kind


# ----------------------- loop-body analyses -----------------------

def is_const_step_assign(tgt, rhs):
    if rhs is None:
        return False
    tt = tok(rhs)
    if not any(op in tt for op in ("+", "-")):
        return False
    names = idents(rhs)
    lits = [t for t in tt if t.lstrip("-").isdigit()]
    return names == [tgt] and len(lits) >= 1

def written_and_conststep(body):
    written, impure = set(), set()
    def note(v, is_step):
        written.add(v)
        if not is_step:
            impure.add(v)
    def w(n):
        if n.kind == ci.CursorKind.BINARY_OPERATOR and ASSIGN_ONLY(tok(n)):
            kids = list(n.get_children())
            if kids:
                nm = idents(kids[0])
                if nm:
                    rhs = kids[1] if len(kids) > 1 else None
                    note(nm[0], is_const_step_assign(nm[0], rhs))
        elif n.kind == ci.CursorKind.UNARY_OPERATOR:
            tt = tok(n)
            if "++" in tt or "--" in tt:
                nm = idents(n)
                if nm:
                    note(nm[0], True)
        elif n.kind == ci.CursorKind.COMPOUND_ASSIGNMENT_OPERATOR:
            kids = list(n.get_children())
            if kids:
                nm = idents(kids[0])
                if nm:
                    tt = tok(n)
                    is_step = (("+=" in tt) or ("-=" in tt)) and \
                              any(t.lstrip("-").isdigit() for t in tt)
                    note(nm[0], is_step)
        for c in n.get_children():
            w(c)
    if body is not None:
        w(body)
    conststep = {v for v in written if v not in impure}
    return written, conststep

def nondet_controlled_vars(body):
    controlled = set()
    def record(n):
        if n.kind == ci.CursorKind.BINARY_OPERATOR and ASSIGN_ONLY(tok(n)):
            kids = list(n.get_children())
            if kids:
                nm = idents(kids[0])
                if nm:
                    controlled.add(nm[0])
        elif n.kind == ci.CursorKind.UNARY_OPERATOR and ("++" in tok(n) or "--" in tok(n)):
            nm = idents(n)
            if nm:
                controlled.add(nm[0])
        elif n.kind == ci.CursorKind.COMPOUND_ASSIGNMENT_OPERATOR:
            kids = list(n.get_children())
            if kids:
                nm = idents(kids[0])
                if nm:
                    controlled.add(nm[0])
    def w(n, under):
        if n.kind == ci.CursorKind.IF_STMT:
            kids = list(n.get_children())
            cond = kids[0] if kids else None
            branch_under = under or expr_has_nondet(cond)
            if cond is not None:
                w(cond, under)
            for b in kids[1:]:
                w(b, branch_under)
            return
        if under:
            record(n)
        for c in n.get_children():
            w(c, under)
    if body is not None:
        w(body, False)
    return controlled


# ----------------------------- classifier -----------------------------

def classify(loop, init_kind):
    guard, body = guard_and_body(loop)
    trace = []

    if guard_is_nondet_oracle(guard):
        return "unbounded", "nondet-oracle guard", []
    if guard_has_memory(guard):
        return "data-dependent", "array/pointer access in guard", []

    refs = list(dict.fromkeys(idents(guard)))
    lits = guard_literals(guard)
    written, conststep = written_and_conststep(body)
    nc = nondet_controlled_vars(body)

    init_str = ", ".join("{}:{}".format(v, init_kind.get(v, "?")) for v in refs)
    trace.append("refs={} lits={} written={} conststep={} nondet_ctrl={} init={{{}}}".format(
        refs, lits, sorted(written), sorted(conststep), sorted(nc), init_str))

    # rule 2: nondet-controlled guard variable
    gc = [r for r in refs if r in nc]
    if gc:
        return "unbounded", "guard var(s) {} nondet-controlled".format(",".join(gc)), trace

    # rule 3: guard variable updated by a non-constant step
    bad = [r for r in refs if r in written and r not in conststep]
    if bad:
        return "data-dependent", "guard var(s) {} non-constant update".format(",".join(bad)), trace

    induction = [r for r in refs if r in conststep]        # const-stepped guard vars
    bound_ids = [r for r in refs if r not in induction]    # non-induction guard vars = bound

    # rule 4: bound is a variable
    if bound_ids:
        verdicts = []
        for b in bound_ids:
            if b in written:
                verdicts.append(("data-dependent", "{} modified in loop (mutating bound)".format(b)))
            elif init_kind.get(b) == "nondet":
                verdicts.append(("parametric", "{} nondet/input bound".format(b)))
            elif init_kind.get(b) == "literal":
                verdicts.append(("constant", "{} literal-const bound".format(b)))
            else:
                verdicts.append(("unclassified", "{} bound origin unresolved ({})".format(b, init_kind.get(b))))
        order = {"data-dependent": 3, "parametric": 2, "unclassified": 1, "constant": 0}
        verdicts.sort(key=lambda v: order[v[0]], reverse=True)
        return verdicts[0][0], verdicts[0][1], trace

    # rule 5: bound is purely literal; induction start decides
    if induction:
        starts = [init_kind.get(v) for v in induction]
        if all(s == "literal" for s in starts):
            return "constant", "literal bound; induction starts literal", trace
        if any(s == "nondet" for s in starts):
            return "parametric", "literal bound; induction starts nondet/input", trace
        return "unclassified", "induction start unresolved {}".format(list(zip(induction, starts))), trace

    # rule 6: no induction and no bound var — classify unmodified guard vars by init
    if refs:
        ks = [init_kind.get(v) for v in refs]
        if any(k == "nondet" for k in ks):
            return "parametric", "guard var nondet, unmodified", trace
        if all(k == "literal" for k in ks):
            return "constant", "guard var literal, unmodified", trace
    return "unclassified", "no resolvable structure", trace


# ------------------------------- driver -------------------------------

def probe(path, index):
    tu = index.parse(str(path), args=PARSE_ARGS)
    root = tu.cursor
    loops = []
    find_loops(root, loops)
    fn = next((c for c in root.get_children()
               if c.kind == ci.CursorKind.FUNCTION_DECL and c.is_definition()), None)

    if len(loops) != 1 or fn is None:
        return {"file": path.name, "n_loops": len(loops),
                "category": "SKIP(non-single-loop)", "reason": "", "guard": ""}, []

    declared, never = declared_and_assigned(fn)
    init_kind = compute_init_kind(fn, loops[0], declared, never)
    guard, _ = guard_and_body(loops[0])
    cat, reason, trace = classify(loops[0], init_kind)
    return {"file": path.name, "n_loops": 1, "category": cat,
            "reason": reason, "guard": tok_text(guard) if guard else ""}, trace


def main():
    root_dir = Path("datasets/Code2Inv")
    files = sorted(root_dir.glob("*.c"))
    if not files:
        print(f"[error] no .c files under {root_dir}", file=sys.stderr)
        return 1

    index = ci.Index.create()
    rows = []
    for f in files:
        try:
            row, trace = probe(f, index)
        except Exception as e:
            row, trace = {"file": f.name, "n_loops": -1, "category": "PARSE_ERROR",
                          "reason": str(e), "guard": ""}, []
        rows.append(row)
        if VERBOSE:
            print("{:<8} {:<15} {}".format(row["file"], row["category"], row["reason"]))
            for t in trace:
                print("         " + t)

    out = root_dir.parent / "code2inv_classified.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\nClassified {} programs. CSV -> {}\n".format(len(rows), out))
    for k, v in Counter(r["category"] for r in rows).most_common():
        print("  {:<26} {}".format(k, v))

    unc = [r for r in rows if r["category"] == "unclassified"]
    if unc:
        print("\nUnclassified ({}):".format(len(unc)))
        for r in unc:
            print("  {:<8} guard: {}".format(r["file"], r["guard"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())