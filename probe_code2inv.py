"""
Probes the Code2Inv dataset get the required statistics.

Reports per-program facts and an aggregate summary:
  - number and kind of loops (while / for / do)
  - loop-guard shape: nondet-oracle | comparison | other
  - unknown() usage: in condition position vs. as an r-value
  - declared-but-never-assigned scalar variables
  - assume()/assert() call conventions present

"""

import sys
import csv
from pathlib import Path
from collections import Counter

import clang.cindex as ci

PARSE_ARGS = ["-std=c99", "-ferror-limit=0", "-w"]

COMPARE_OPS = {"<", "<=", ">", ">=", "!=", "=="}


# ----------------------------- AST helpers -----------------------------

def tok(node):
    return [t.spelling for t in node.get_tokens()]


def tok_text(node):
    return " ".join(tok(node))


def find_loops(node, out):
    if node.kind in (ci.CursorKind.WHILE_STMT,
                     ci.CursorKind.FOR_STMT,
                     ci.CursorKind.DO_STMT):
        out.append(node)
    for c in node.get_children():
        find_loops(c, out)


def loop_kind(n):
    return {ci.CursorKind.WHILE_STMT: "while",
            ci.CursorKind.FOR_STMT: "for",
            ci.CursorKind.DO_STMT: "do"}[n.kind]


def children(n):
    return list(n.get_children())


def guard_and_body(loop):
    """
    Return (guard_cursor, body_cursor) for a loop.
    while/do: children are [guard, body] (guard is the first non-compound).
    for:      children are up to [init, cond, inc, body]; cond is the guard.
    Robust-ish: pick the last COMPOUND_STMT as body, and the expression
    immediately controlling the loop as guard.
    """
    ch = children(loop)
    body = None
    for c in reversed(ch):
        if c.kind == ci.CursorKind.COMPOUND_STMT:
            body = c
            break
    guard = None
    if loop.kind == ci.CursorKind.FOR_STMT:
        # for: try to find the condition = the middle expression child.
        exprs = [c for c in ch if c.kind != ci.CursorKind.COMPOUND_STMT]
        # heuristic: the condition is typically a BINARY_OPERATOR among them
        for c in exprs:
            if c.kind == ci.CursorKind.BINARY_OPERATOR and any(
                    op in tok(c) for op in COMPARE_OPS):
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


def collect_calls(node, name):
    hits = []
    if node.kind == ci.CursorKind.CALL_EXPR and node.spelling == name:
        hits.append(node)
    for c in node.get_children():
        hits.extend(collect_calls(c, name))
    return hits


def guard_has_memory_access(guard):
    """True if the guard contains an array subscript or pointer dereference."""

    def walk(n):
        if n.kind in (ci.CursorKind.ARRAY_SUBSCRIPT_EXPR,):
            return True
        if n.kind == ci.CursorKind.UNARY_OPERATOR and "*" in tok(n)[:1]:
            return True
        return any(walk(c) for c in n.get_children())

    return walk(guard) if guard is not None else False


def guard_is_nondet_oracle(guard):
    """Guard is essentially unknown()/nondet_bool() with no comparison."""
    if guard is None:
        return False
    t = "".join(tok(guard))
    has_call = ("unknown(" in t) or ("__VERIFIER_nondet_bool(" in t) or \
               ("__VERIFIER_nondet_int(" in t and not any(op in t for op in COMPARE_OPS))
    has_cmp = any(op in tok(guard) for op in COMPARE_OPS)
    return has_call and not has_cmp


def guard_ident_refs(guard):
    """All identifier names referenced in the guard (DECL_REF_EXPR)."""
    names = []

    def walk(n):
        if n.kind == ci.CursorKind.DECL_REF_EXPR:
            names.append(n.spelling)
        for c in n.get_children():
            walk(c)

    if guard is not None:
        walk(guard)
    return names


def guard_literals(guard):
    """Integer literals appearing in the guard."""
    lits = []

    def walk(n):
        if n.kind == ci.CursorKind.INTEGER_LITERAL:
            tt = tok(n)
            if tt:
                lits.append(tt[0])
        for c in n.get_children():
            walk(c)

    if guard is not None:
        walk(guard)
    return lits


# ----------------------- variable-origin analysis -----------------------

def analyze_function(fn):
    """
    Build maps describing each scalar variable:
      declared[name]      = type spelling
      init_is_literal[name] = True if declared with a plain integer-literal init
      assigned_names       = set of names ever written (init or assignment)
      nondet_names         = set of names assigned from a nondet source
    """
    declared = {}
    init_is_literal = {}
    assigned = set()
    nondet = set()

    def rhs_is_nondet(node):
        t = "".join(tok(node))
        return ("unknown(" in t) or ("__VERIFIER_nondet" in t) or ("nondet_" in t)

    def rhs_is_int_literal(node):
        # node is the initializer/RHS expression cursor
        kids = list(node.get_children())
        # direct literal
        if node.kind == ci.CursorKind.INTEGER_LITERAL:
            return True
        # e.g. unary minus on a literal
        tt = [x for x in tok(node) if x not in ("(", ")", "-", "+")]
        return len(tt) == 1 and tt[0].lstrip("-").isdigit()

    def walk(n):
        if n.kind == ci.CursorKind.VAR_DECL:
            declared[n.spelling] = n.type.spelling
            inits = [c for c in n.get_children() if c.kind != ci.CursorKind.TYPE_REF]
            if inits:
                assigned.add(n.spelling)
                init = inits[-1]
                if rhs_is_nondet(init):
                    nondet.add(n.spelling)
                    init_is_literal[n.spelling] = False
                elif rhs_is_int_literal(init):
                    init_is_literal[n.spelling] = True
                else:
                    init_is_literal[n.spelling] = False
        if n.kind == ci.CursorKind.BINARY_OPERATOR:
            tt = tok(n)
            if "=" in tt and "==" not in tt and "!=" not in tt \
                    and "<=" not in tt and ">=" not in tt:
                # assignment; target = first identifier
                kids = list(n.get_children())
                if kids:
                    lhs = kids[0]
                    lhs_names = guard_ident_refs(lhs)
                    if lhs_names:
                        tgt = lhs_names[0]
                        assigned.add(tgt)
                        if len(kids) > 1 and rhs_is_nondet(kids[1]):
                            nondet.add(tgt)
                        # a var re-assigned to non-literal loses literal status
                        if not (len(kids) > 1 and rhs_is_int_literal(kids[1])):
                            init_is_literal[tgt] = False
        for c in n.get_children():
            walk(c)

    walk(fn)
    never_assigned = [v for v in declared if v not in assigned]
    return {
        "declared": declared,
        "init_is_literal": init_is_literal,
        "assigned": assigned,
        "nondet": nondet,
        "never_assigned": never_assigned,
    }


def names_written_in(body):
    """Set of identifier names assigned/updated inside the loop body."""
    written = set()

    def walk(n):
        if n.kind == ci.CursorKind.BINARY_OPERATOR:
            tt = tok(n)
            if "=" in tt and "==" not in tt and "!=" not in tt \
                    and "<=" not in tt and ">=" not in tt:
                kids = list(n.get_children())
                if kids:
                    for nm in guard_ident_refs(kids[0]):
                        written.add(nm)
                        break
        if n.kind == ci.CursorKind.UNARY_OPERATOR:
            tt = tok(n)
            if "++" in tt or "--" in tt:
                for nm in guard_ident_refs(n):
                    written.add(nm)
                    break
        if n.kind == ci.CursorKind.COMPOUND_ASSIGNMENT_OPERATOR:
            kids = list(n.get_children())
            if kids:
                for nm in guard_ident_refs(kids[0]):
                    written.add(nm)
                    break
        for c in n.get_children():
            walk(c)

    if body is not None:
        walk(body)
    return written


# ----------------------------- classifier -----------------------------

def classify_loop(loop, fn_info):
    guard, body = guard_and_body(loop)

    if guard_is_nondet_oracle(guard):
        return "unbounded", "nondet-oracle guard"

    if guard_has_memory_access(guard):
        return "data-dependent", "array/pointer access in guard"

    refs = guard_ident_refs(guard)
    lits = guard_literals(guard)
    written = names_written_in(body)

    # induction variable = a guard ref that is written in the loop
    induction = [r for r in refs if r in written]
    # bound variables = guard refs that are NOT the induction variable
    bound_vars = [r for r in refs if r not in induction]

    # Guard compares induction var to a pure literal (no other var on bound side)
    if lits and not bound_vars:
        return "constant", f"induction vs literal ({','.join(lits)})"

    if not bound_vars:
        # e.g. i != 0 with only induction var and no literal captured, or parse gap
        if lits:
            return "constant", f"literal-only guard ({','.join(lits)})"
        return "unclassified", "no resolvable bound variable"

    # Examine each bound variable's origin; take the "hardest" classification.
    verdicts = []
    for bv in bound_vars:
        if bv in written:
            verdicts.append(("data-dependent", f"{bv} modified in loop"))
        elif bv in fn_info["nondet"] or bv in fn_info["never_assigned"]:
            verdicts.append(("parametric", f"{bv} is nondet/uninitialized input"))
        elif fn_info["init_is_literal"].get(bv, False):
            verdicts.append(("constant", f"{bv} holds literal constant"))
        else:
            verdicts.append(("unclassified", f"{bv} origin unresolved"))

    order = {"data-dependent": 3, "parametric": 2, "constant": 1, "unclassified": 0}
    verdicts.sort(key=lambda v: order[v[0]], reverse=True)
    # If anything is unresolved AND nothing stronger than constant, flag it.
    if verdicts[0][0] == "constant" and any(v[0] == "unclassified" for v in verdicts):
        return "unclassified", "; ".join(r for _, r in verdicts)
    return verdicts[0]


# ------------------------------- driver -------------------------------

def probe_file(path, index):
    tu = index.parse(str(path), args=PARSE_ARGS)
    root = tu.cursor
    loops = []
    find_loops(root, loops)

    fn = None
    for c in root.get_children():
        if c.kind == ci.CursorKind.FUNCTION_DECL and c.is_definition():
            fn = c
            break
    fn_info = analyze_function(fn) if fn else {
        "declared": {}, "init_is_literal": {}, "assigned": set(),
        "nondet": set(), "never_assigned": []}

    if len(loops) != 1:
        return {"file": path.name, "n_loops": len(loops),
                "category": "SKIP(non-single-loop)", "reason": "",
                "guard": "", "unknown_calls": len(collect_calls(root, "unknown"))}

    loop = loops[0]
    guard, _ = guard_and_body(loop)
    category, reason = classify_loop(loop, fn_info)
    return {
        "file": path.name,
        "n_loops": 1,
        "category": category,
        "reason": reason,
        "guard": tok_text(guard) if guard is not None else "",
        "unknown_calls": len(collect_calls(root, "unknown")),
    }

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
            rows.append(probe_file(f, index))
        except Exception as e:
            rows.append({"file": f.name, "n_loops": -1, "category": "PARSE_ERROR",
                         "reason": str(e), "guard": "", "unknown_calls": -1})

    out = root_dir.parent / "code2inv_probe.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nClassified {len(rows)} programs. CSV -> {out}\n")
    cat = Counter(r["category"] for r in rows)
    print("Category distribution:")
    for k, v in cat.most_common():
        print(f"  {k:<26} {v}")

    unclassified = [r["file"] for r in rows if r["category"] == "unclassified"]
    if unclassified:
        print(f"\nUnclassified ({len(unclassified)}) — inspect these guards:")
        for r in rows:
            if r["category"] == "unclassified":
                print(f"  {r['file']:<8} guard: {r['guard']}")
    print("\nInspect the CSV 'reason' and 'guard' columns to spot-check the rules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())