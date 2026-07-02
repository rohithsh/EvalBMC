#!/usr/bin/env python3
"""
Code2Inv-specific normalization -> CBMC-ready programs.

Reads categories from code2inv_classified.csv and, for each source .c file:
  * unknown()  -> __VERIFIER_nondet_bool()  in condition context (loop guard / if)
                  __VERIFIER_nondet_int()   elsewhere (value context)
  * assume(x)  -> __VERIFIER_assume(x)
  * assert     -> left as assert(...)  (adds #include <assert.h>)
  * every declared-but-uninitialised scalar gets an injected
        <var> = __VERIFIER_nondet_int();
    inserted after the declarations, in declaration order

Outputs to per-category subfolders under <out_dir>:
    constant/  parametric/  data-dependent/  unbounded/
with filenames namespaced code2inv_<original>.c

Edits are located via libclang and applied as text splices in descending offset
order, preserving original formatting/comments.

"""

import sys
import csv
from pathlib import Path

import clang.cindex as ci

# ci.Config.set_library_file(r"C:\Program Files\LLVM\bin\libclang.dll")  # if needed

PARSE_ARGS = ["-std=c99", "-ferror-limit=0", "-w"]
COMPARE_OPS = {"<", "<=", ">", ">=", "!=", "=="}
SKIP_FILES = {""}

CATEGORY_DIR = {
    "constant": "constant",
    "parametric": "parametric",
    "data-dependent": "data-dependent",
    "unbounded": "unbounded",
}


# ----------------------------- AST helpers -----------------------------

def tok(n):
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

def find_fn(root):
    return next((c for c in root.get_children()
                 if c.kind == ci.CursorKind.FUNCTION_DECL and c.is_definition()), None)

def find_loops(node, out):
    if node.kind in (ci.CursorKind.WHILE_STMT, ci.CursorKind.FOR_STMT, ci.CursorKind.DO_STMT):
        out.append(node)
    for c in node.get_children():
        find_loops(c, out)


# ---------------- condition-context detection for unknown() ----------------

def condition_offsets(fn):
    """
    Return a set of (start, end) source-offset spans that are 'condition context':
      * the controlling expression of while/do/for
      * the condition of every if
    An unknown() call whose extent falls inside any such span is boolean context.
    """
    spans = []

    def add_span(node):
        if node is not None:
            e = node.extent
            spans.append((e.start.offset, e.end.offset))

    def guard_of(loop):
        ch = list(loop.get_children())
        if loop.kind == ci.CursorKind.FOR_STMT:
            exprs = [c for c in ch if c.kind != ci.CursorKind.COMPOUND_STMT]
            for c in exprs:
                if c.kind == ci.CursorKind.BINARY_OPERATOR and any(op in tok(c) for op in COMPARE_OPS):
                    return c
            # fall back: any expr child that contains a call/comparison
            return exprs[len(exprs) // 2] if len(exprs) >= 2 else (exprs[0] if exprs else None)
        for c in ch:
            if c.kind != ci.CursorKind.COMPOUND_STMT:
                return c
        return None

    def w(n):
        if n.kind in (ci.CursorKind.WHILE_STMT, ci.CursorKind.DO_STMT, ci.CursorKind.FOR_STMT):
            add_span(guard_of(n))
        if n.kind == ci.CursorKind.IF_STMT:
            kids = list(n.get_children())
            if kids:
                add_span(kids[0])  # condition
        for c in n.get_children():
            w(c)

    if fn is not None:
        w(fn)
    return spans

def in_any_span(off, spans):
    return any(a <= off < b for (a, b) in spans)


# ---------------- collect edits ----------------

def collect_unknown_edits(root, cond_spans, edits):
    """Replace each unknown() call's function-name token with the right intrinsic."""
    def w(n):
        if n.kind == ci.CursorKind.CALL_EXPR and n.spelling == "unknown":
            e = n.extent
            call_start = e.start.offset
            # boolean context if the call sits within a condition span
            repl = "__VERIFIER_nondet_bool" if in_any_span(call_start, cond_spans) \
                   else "__VERIFIER_nondet_int"
            # replace just the identifier 'unknown' (first token of the call)
            toks = list(n.get_tokens())
            if toks and toks[0].spelling == "unknown":
                t0 = toks[0].extent
                edits.append((t0.start.offset, t0.end.offset, repl))
        for c in n.get_children():
            w(c)
    w(root)

def collect_assume_edits(root, edits):
    """assume( -> __VERIFIER_assume(  (rename the callee identifier only)."""
    def w(n):
        if n.kind == ci.CursorKind.CALL_EXPR and n.spelling == "assume":
            toks = list(n.get_tokens())
            if toks and toks[0].spelling == "assume":
                t0 = toks[0].extent
                edits.append((t0.start.offset, t0.end.offset, "__VERIFIER_assume"))
        for c in n.get_children():
            w(c)
    w(root)


# ---------------- uninitialised scalar injection ----------------

def scalar_var_decls(fn):
    """Ordered list of (name, decl_node) for scalar int-like locals."""
    out = []
    def w(n):
        if n.kind == ci.CursorKind.VAR_DECL:
            # keep it simple: treat all VAR_DECLs as scalars for Code2Inv
            out.append((n.spelling, n))
        for c in n.get_children():
            w(c)
    if fn is not None:
        w(fn)
    return out

def has_declarator_init(decl):
    return any(c.kind != ci.CursorKind.TYPE_REF for c in decl.get_children())

def first_assignment_offset(fn, name):
    """Offset of the earliest statement-level assignment `name = ...`, or None."""
    best = None
    def w(n):
        nonlocal best
        if n.kind == ci.CursorKind.BINARY_OPERATOR:
            tt = tok(n)
            if "=" in tt and not any(o in tt for o in ("==", "!=", "<=", ">=")):
                kids = list(n.get_children())
                if kids:
                    lhs = idents(kids[0])
                    if lhs and lhs[0] == name:
                        off = n.extent.start.offset
                        if best is None or off < best:
                            best = off
        for c in n.get_children():
            w(c)
    if fn is not None:
        w(fn)
    return best

def referenced_names(fn):
    """All identifier names that appear as a DECL_REF_EXPR anywhere in the function."""
    names = set()
    def w(n):
        if n.kind == ci.CursorKind.DECL_REF_EXPR:
            names.add(n.spelling)
        for c in n.get_children():
            w(c)
    if fn is not None:
        w(fn)
    return names

def uninitialised_scalars(fn):
    """
    Names of scalars that are:
      * declared with no initialiser,
      * never assigned anywhere, AND
      * actually referenced somewhere in the function (not dead declarations),
    in declaration order.
    """
    decls = scalar_var_decls(fn)
    used = referenced_names(fn)
    out = []
    for name, decl in decls:
        if has_declarator_init(decl):
            continue
        if first_assignment_offset(fn, name) is not None:
            continue
        if name not in used:          # dead declaration -> skip injection
            continue
        out.append(name)
    return out

def injection_point(fn, decls):
    """
    Offset just after the last VAR_DECL statement, where we insert the
    nondet initialisation statements. Uses the end of the last decl's line.
    """
    if not decls:
        # after the function's opening brace
        body = next((c for c in fn.get_children()
                     if c.kind == ci.CursorKind.COMPOUND_STMT), None)
        return body.extent.start.offset + 1 if body else None
    last_decl = decls[-1][1]
    return last_decl.extent.end.offset  # just after the declaration node


# ---------------- driver for one file ----------------

def normalize_one(src_path, out_path):
    src = src_path.read_text(encoding="utf-8", errors="replace")
    index = ci.Index.create()
    tu = index.parse(str(src_path), args=PARSE_ARGS,
                     unsaved_files=[(str(src_path), src)])
    root = tu.cursor
    fn = find_fn(root)

    edits = []  # (start_off, end_off, replacement)

    # 1. unknown() rewrites (boolean vs int by condition context)
    cond_spans = condition_offsets(fn)
    collect_unknown_edits(root, cond_spans, edits)

    # 2. assume(...) -> __VERIFIER_assume(...)
    collect_assume_edits(root, edits)

    # 3. inject nondet init for uninitialised scalars
    decls = scalar_var_decls(fn)
    uninit = uninitialised_scalars(fn)
    inj_off = injection_point(fn, decls)
    injected_text = ""
    if uninit and inj_off is not None:
        stmts = "".join("\n  {} = __VERIFIER_nondet_int();".format(v) for v in uninit)
        edits.append((inj_off, inj_off, stmts))

    # apply edits right-to-left so earlier offsets remain valid
    edits.sort(key=lambda e: e[0], reverse=True)
    out = src
    for a, b, repl in edits:
        # semicolons after injected decl end: the decl node extent typically
        # ends before the ';', so we insert after the ';' if present.
        if a == b and repl.startswith("\n") and a < len(out) and out[a:a+1] == ";":
            a2 = a + 1
            out = out[:a2] + repl + out[a2:]
        else:
            out = out[:a] + repl + out[b:]

    # ensure <assert.h> for assert()
    if "assert(" in out and "#include <assert.h>" not in out:
        out = "#include <assert.h>\n" + out

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out, encoding="utf-8")
    return {"uninit_injected": uninit,
            "n_unknown": sum(1 for e in edits if e[2].startswith("__VERIFIER_nondet")),
            "n_assume": sum(1 for e in edits if e[2] == "__VERIFIER_assume")}


def main():
    src_dir = Path("datasets/Code2Inv")
    csv_path = Path("datasets/code2inv_classified.csv")
    out_dir = Path("datasets/Code2Inv_classified")

    # read categories
    cat = {}
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            cat[r["file"]] = r["category"]

    summary = []
    for src in sorted(src_dir.glob("*.c")):
        if src.name in SKIP_FILES:
            print("[skip] {}".format(src.name))
            continue
        category = cat.get(src.name)
        if category is None:
            print("[warn] {} not in CSV; skipping".format(src.name))
            continue
        sub = CATEGORY_DIR.get(category)
        if sub is None:
            print("[warn] {} has non-bucket category '{}'; skipping".format(src.name, category))
            continue
        out_name = "code2inv_{}".format(src.name)
        out_path = out_dir / sub / out_name
        try:
            info = normalize_one(src, out_path)
            summary.append((src.name, category, info))
            print("[ok]  {:<10} -> {}/{}   inj={} unk={} asm={}".format(
                src.name, sub, out_name, info["uninit_injected"],
                info["n_unknown"], info["n_assume"]))
        except Exception as e:
            print("[error] {}: {}".format(src.name, e))

    print("\nNormalized {} files into {}".format(len(summary), out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())