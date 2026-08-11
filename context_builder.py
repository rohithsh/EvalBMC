#!/usr/bin/env python3
"""
context_builder.py

Decides what program text to send to the model. Two modes:

  full    the preprocessed source unchanged.

  slice   a BACKWARD SLICE on the loop control variables. Starting from each
          loop's guard and induction variables, keep the statements that
          determine their values, transitively, plus the control flow those
          statements sit under and the declarations involved. Everything else
          is dropped. <incomplete>
"""

import argparse
import json
import sys
from pathlib import Path

import clang.cindex as ci

COMPARE_OPS = {"<", "<=", ">", ">=", "!=", "=="}
ASSUME_FUNCS = {"__VERIFIER_assume", "assume", "__CPROVER_assume"}
EXIT_FUNCS = {"abort", "exit", "__assert_fail"}


# ------------------------------------------------------------- AST helpers

def toks(n):
    return [t.spelling for t in n.get_tokens()]


def idents(n):
    """Variable names read anywhere in a subtree."""
    out = []

    def w(x):
        if x.kind == ci.CursorKind.DECL_REF_EXPR:
            out.append(x.spelling)
        for c in x.get_children():
            w(c)

    if n is not None:
        w(n)
    return out


def is_assign(n):
    if n.kind == ci.CursorKind.COMPOUND_ASSIGNMENT_OPERATOR:
        return True
    if n.kind == ci.CursorKind.BINARY_OPERATOR:
        t = toks(n)
        return "=" in t and not any(o in t for o in ("==", "!=", "<=", ">="))
    if n.kind == ci.CursorKind.UNARY_OPERATOR:
        return "++" in toks(n) or "--" in toks(n)
    return False


def assign_target(n):
    """Name written by an assignment, or None."""
    if n.kind == ci.CursorKind.UNARY_OPERATOR:
        names = idents(n)
        return names[0] if names else None
    kids = list(n.get_children())
    if not kids:
        return None
    names = idents(kids[0])
    return names[0] if names else None


def assign_sources(n):
    """Names read by an assignment's right-hand side."""
    kids = list(n.get_children())
    if n.kind == ci.CursorKind.UNARY_OPERATOR:
        return idents(n)
    if len(kids) < 2:
        return []
    src = idents(kids[1])
    if n.kind == ci.CursorKind.COMPOUND_ASSIGNMENT_OPERATOR:
        src += idents(kids[0])          # x += y reads x too
    return src


def called_names(n):
    out = []

    def w(x):
        if x.kind == ci.CursorKind.CALL_EXPR and x.spelling:
            out.append(x.spelling)
        for c in x.get_children():
            w(c)

    if n is not None:
        w(n)
    return out


def line_span(n):
    e = n.extent
    return e.start.line, e.end.line


# ------------------------------------------------------------ the slicer

class Slicer:
    """Backward slice from the loop control variables to a fixpoint."""

    def __init__(self, tu, cfile_name, seed_vars):
        self.tu = tu
        self.cfile = cfile_name
        self.relevant_vars = set(seed_vars)
        self.relevant_lines = set()
        self.relevant_funcs = set()
        self.dropped_funcs = set()
        self.rounds = 0

    # --- collection --------------------------------------------------------

    def _in_file(self, n):
        loc = n.location
        return loc.file is not None and Path(str(loc.file)).name == self.cfile

    def _functions(self):
        for c in self.tu.cursor.get_children():
            if c.kind == ci.CursorKind.FUNCTION_DECL and c.is_definition():
                yield c

    def _keep(self, n):
        a, b = line_span(n)
        for ln in range(a, b + 1):
            self.relevant_lines.add(ln)

    def _keep_enclosing_control(self, node, fn):
        """
        Control dependence: a kept statement needs the conditions that decide
        whether it runs. Walk down from the function, tracking the branch/loop
        conditions we are inside, and keep those that enclose a kept line.
        """
        added = set()

        def w(n, guards):
            a, b = line_span(n)
            if n.kind in (ci.CursorKind.IF_STMT, ci.CursorKind.WHILE_STMT,
                          ci.CursorKind.FOR_STMT, ci.CursorKind.DO_STMT,
                          ci.CursorKind.SWITCH_STMT):
                kids = list(n.get_children())
                cond = kids[0] if kids else None
                guards = guards + [(n, cond)]
            if any(ln in self.relevant_lines for ln in range(a, b + 1)):
                for gnode, gcond in guards:
                    ga, gb = line_span(gnode)
                    # keep the header line(s), not the whole body
                    if gcond is not None:
                        ca, cb = line_span(gcond)
                        for ln in range(min(ga, ca), max(ga, cb) + 1):
                            self.relevant_lines.add(ln)
                        added.update(idents(gcond))
                    else:
                        self.relevant_lines.add(ga)
            for c in n.get_children():
                w(c, guards)

        w(fn, [])
        return added

    # --- one pass ----------------------------------------------------------

    def _pass(self):
        """One sweep. Returns True if anything new became relevant."""
        grew = False

        for fn in self._functions():
            if not self._in_file(fn):
                continue
            fname = fn.spelling
            touched = False

            def w(n):
                nonlocal grew, touched

                # a declaration of a relevant variable, with its initialiser
                if n.kind == ci.CursorKind.VAR_DECL and n.spelling in self.relevant_vars:
                    self._keep(n)
                    touched = True
                    inits = [c for c in n.get_children()
                             if c.kind != ci.CursorKind.TYPE_REF]
                    for c in inits:
                        for v in idents(c):
                            if v not in self.relevant_vars:
                                self.relevant_vars.add(v)
                                grew = True
                        for callee in called_names(c):
                            if callee not in self.relevant_funcs:
                                self.relevant_funcs.add(callee)
                                grew = True

                # an assignment writing a relevant variable
                elif is_assign(n):
                    tgt = assign_target(n)
                    if tgt and tgt in self.relevant_vars:
                        self._keep(n)
                        touched = True
                        for v in assign_sources(n):
                            if v not in self.relevant_vars:
                                self.relevant_vars.add(v)
                                grew = True
                        for callee in called_names(n):
                            if callee not in self.relevant_funcs:
                                self.relevant_funcs.add(callee)
                                grew = True

                # a constraint on a relevant variable: assume(...)
                elif n.kind == ci.CursorKind.CALL_EXPR and n.spelling in ASSUME_FUNCS:
                    if set(idents(n)) & self.relevant_vars:
                        self._keep(n)
                        touched = True
                        for v in idents(n):
                            if v not in self.relevant_vars:
                                self.relevant_vars.add(v)
                                grew = True

                # a constraint on a relevant variable: if (!(n < C)) abort();
                elif n.kind == ci.CursorKind.IF_STMT:
                    kids = list(n.get_children())
                    if kids and (set(idents(kids[0])) & self.relevant_vars):
                        body = kids[1] if len(kids) > 1 else None
                        exits = False
                        if body is not None:
                            try:
                                exits = any(
                                    c.kind == ci.CursorKind.RETURN_STMT or
                                    (c.kind == ci.CursorKind.CALL_EXPR and
                                     c.spelling in EXIT_FUNCS)
                                    for c in body.walk_preorder())
                            except Exception:
                                exits = False
                        if exits:
                            self._keep(n)
                            touched = True
                            for v in idents(kids[0]):
                                if v not in self.relevant_vars:
                                    self.relevant_vars.add(v)
                                    grew = True

                # a loop is always kept -- it is what we are bounding
                elif n.kind in (ci.CursorKind.WHILE_STMT, ci.CursorKind.FOR_STMT,
                                ci.CursorKind.DO_STMT):
                    kids = list(n.get_children())
                    body = kids[-1] if kids else None
                    header_end = line_span(body)[0] - 1 if body is not None \
                        else line_span(n)[1]
                    a0 = line_span(n)[0]
                    for ln in range(a0, max(a0, header_end) + 1):
                        self.relevant_lines.add(ln)
                    touched = True
                    for c in kids[:-1] if body is not None else kids:
                        for v in idents(c):
                            if v not in self.relevant_vars:
                                self.relevant_vars.add(v)
                                grew = True

                for c in n.get_children():
                    w(c)

            w(fn)

            # a function whose return value feeds a relevant variable
            if fname in self.relevant_funcs and not touched:
                for c in fn.walk_preorder():
                    if c.kind == ci.CursorKind.RETURN_STMT:
                        self._keep(c)
                        touched = True
                        for v in idents(c):
                            if v not in self.relevant_vars:
                                self.relevant_vars.add(v)
                                grew = True

            if touched:
                if fname not in self.relevant_funcs:
                    self.relevant_funcs.add(fname)
                    grew = True
                extra = self._keep_enclosing_control(fn, fn)
                for v in extra:
                    if v not in self.relevant_vars:
                        self.relevant_vars.add(v)
                        grew = True
                # keep the signature and closing brace
                a, b = line_span(fn)
                self.relevant_lines.add(a)
                self.relevant_lines.add(b)
            else:
                self.dropped_funcs.add(fname)

        return grew

    def run(self, max_rounds=8):
        while self.rounds < max_rounds and self._pass():
            self.rounds += 1
        self.dropped_funcs -= self.relevant_funcs
        return self


# ---------------------------------------------------------------- rendering

def keep_type_decls(tu, cfile_name, lines, keep):
    """
    Keep type, struct, union, enum and typedef declarations, plus file-scope
    prototypes. Dropping them leaves the slice unreadable.
    """
    kinds = (ci.CursorKind.STRUCT_DECL, ci.CursorKind.UNION_DECL,
             ci.CursorKind.ENUM_DECL, ci.CursorKind.TYPEDEF_DECL)
    for c in tu.cursor.get_children():
        loc = c.location
        if loc.file is None or Path(str(loc.file)).name != cfile_name:
            continue
        if c.kind in kinds or (c.kind == ci.CursorKind.FUNCTION_DECL
                               and not c.is_definition()):
            a, b = c.extent.start.line, c.extent.end.line
            for ln in range(a, b + 1):
                keep.add(ln)
        elif c.kind == ci.CursorKind.VAR_DECL:
            # file-scope variables: cheap to keep, often the bound
            a, b = c.extent.start.line, c.extent.end.line
            for ln in range(a, b + 1):
                keep.add(ln)


def render(lines, keep):
    out, gap = [], False
    for n, ln in enumerate(lines, start=1):
        if n in keep:
            if gap:
                out.append("  /* ... */")
                gap = False
            out.append(ln)
        elif ln.strip() and not gap:
            gap = True
    return "\n".join(out)


# ------------------------------------------------------------------- API

def seed_variables(rec):
    """Loop control and induction variables, from the classifier's output."""
    seed = set()
    for l in rec.get("loops", []):
        seed.update(l.get("bound_vars") or [])
        seed.update(l.get("induction_vars") or [])
    return seed


def build_context(rec, dataset, mode="slice", clang_args=None):
    """
    Return (text, info). info records what happened, so a bad slice is visible
    rather than silent.
    """
    cpath = Path(dataset) / rec["dir"] / rec["c_file"]
    info = {"mode": mode, "path": str(cpath)}
    try:
        source = cpath.read_text(errors="replace")
    except Exception as e:
        return None, {**info, "error": "unreadable: {}".format(e)}

    info["source_lines"] = source.count("\n") + 1
    info["source_chars"] = len(source)

    if mode == "full":
        info["kept_lines"] = info["source_lines"]
        return source, info

    seed = seed_variables(rec)
    info["seed_vars"] = sorted(seed)
    if not seed:
        # nothing to slice on (e.g. every loop bounded by a literal)
        info["note"] = "no seed variables; sending full source"
        info["kept_lines"] = info["source_lines"]
        return source, info

    args = clang_args or ["-std=gnu99", "-ferror-limit=0", "-w",
                          "-m32" if rec.get("data_model") == "ILP32" else "-m64"]
    try:
        index = ci.Index.create()
        tu = index.parse(str(cpath), args=args)
    except Exception as e:
        return source, {**info, "error": "libclang: {}".format(e),
                        "note": "parse failed; sending full source"}

    sl = Slicer(tu, cpath.name, seed).run()
    lines = source.splitlines()
    keep = set(sl.relevant_lines)
    keep_type_decls(tu, cpath.name, lines, keep)
    keep = {n for n in keep if 1 <= n <= len(lines)}

    text = render(lines, keep)
    info.update({
        "kept_lines": len(keep),
        "dropped_lines": info["source_lines"] - len(keep),
        "kept_chars": len(text),
        "reduction": round(1.0 - len(text) / max(1, len(source)), 3),
        "relevant_vars": sorted(sl.relevant_vars),
        "relevant_funcs": sorted(sl.relevant_funcs),
        "dropped_funcs": sorted(sl.dropped_funcs),
        "rounds": sl.rounds,
    })
    return text, info


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="loops.json")
    ap.add_argument("--dataset", required=True, help="svcomp_clean")
    ap.add_argument("--mode", choices=["full", "slice"], default="slice")
    ap.add_argument("--show", default=None,
                    help="print the context for one task (its 'task' value)")
    ap.add_argument("--stats", action="store_true",
                    help="report slice reduction across the dataset")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    recs = json.load(open(a.json))

    if a.show:
        hit = [r for r in recs if r["task"] == a.show or a.show in r["task"]]
        if not hit:
            print("no such task", file=sys.stderr)
            return 1
        rec = hit[0]
        text, info = build_context(rec, a.dataset, a.mode)
        print("=" * 70)
        for k, v in info.items():
            if isinstance(v, list) and len(v) > 12:
                v = v[:12] + ["..."]
            print("{:<16} {}".format(k, v))
        print("=" * 70)
        print(text)
        return 0

    if a.stats:
        rows = recs[:a.limit] if a.limit else recs
        red, failed, big = [], 0, 0
        for r in rows:
            text, info = build_context(r, a.dataset, a.mode)
            if text is None or "error" in info:
                failed += 1
                continue
            if "reduction" in info:
                red.append(info["reduction"])
            if info.get("kept_chars", info["source_chars"]) > 60000:
                big += 1
        print("programs        : {}".format(len(rows)))
        print("failed to build : {}".format(failed))
        if red:
            red.sort()
            print("slice reduction : median={:.1%}  min={:.1%}  max={:.1%}".format(
                red[len(red) // 2], red[0], red[-1]))
        print("still >60k chars: {}".format(big))
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())