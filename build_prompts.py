#!/usr/bin/env python3
"""
build_prompts.py

Prompt text and response parsing.
Standalone:
  python3 build_prompts.py --json datasets/cleaned/loops.json \
      --dataset datasets/cleaned/svcomp_clean \
      --print loops/bubble_sort-1.yml --property reach --mode slice
"""

import argparse
import json
import re

PROP_DESC = {
    "reach": ("reachability -- can the call to reach_error() be reached on "
              "some execution?"),
    "memsafety": ("memory safety -- is there an invalid pointer dereference, "
                  "an invalid free, or memory that is allocated and then lost?"),
    "overflow": ("signed integer overflow -- can an operation on a signed "
                 "integer overflow?"),
}

SYSTEM = """You are an expert in bounded model checking for C programs.

A bounded model checker unrolls every loop a fixed number of times and checks \
the property on the unrolled program. Your job is to predict how many \
unrollings are needed to reach a conclusive result.
 
There are two ways a run becomes conclusive, and they need very different \
numbers.
 
FINDING A VIOLATION. If the program has a bug, the checker only needs enough \
unrollings to reach it. A violation often appears on an early iteration, long \
before the loop would finish. Do not answer with the loop's full iteration count \
when the bug is reachable sooner: work out on which iteration the violating state is first reachable.
 
PROVING THE PROPERTY. If the program has no bug, nothing is conclusive until \
every loop has been exhausted, because a shallower run only shows the property \
holds up to that depth. Here the answer is the loop's full iteration count.
 
You are not told which case applies. Decide from the program, and give the \
smallest number of unrollings at which the checker reaches a conclusive result \
either way.
 
Reply with ONE JSON object and nothing else. Three replies are allowed.

1. One bound for every loop:
   {"kind":"uniform","k":<integer>,"confidence":"high|medium|low","reasoning":"<brief>"}

2. A bound per loop, using the loop identifiers given in the question:
   {"kind":"schedule","schedule":{"<loop id>":<integer>,...},"confidence":"high|medium|low","reasoning":"<brief>"}

3. No finite bound exists, because some loop's iteration count is an \
unconstrained program input, so no fixed number of unrollings covers every \
possible input:
   {"kind":"unbounded","confidence":"high|medium|low","reasoning":"<brief>"}

Keep "reasoning" under 60 words and say what determines the bound."""

ASK_BLIND = ("Predict the smallest number of loop unrollings at which a bounded "
             "model checker reaches a conclusive result on this program -- "
             "either it exposes a violation of the property, or it exhausts "
             "every loop and proves the property holds.")

ASK_INFORMED_UNSAFE = (
    "This program VIOLATES the property. Predict the smallest number of loop "
    "unrollings at which the violation first becomes reachable. This is usually "
    "far smaller than the loop's full iteration count.")

ASK_INFORMED_SAFE = (
    "This program SATISFIES the property. Predict the smallest number of loop "
    "unrollings at which every loop is exhausted, so the property is proved "
    "rather than merely checked to that depth.")

SLICE_NOTE = ("The source below is a backward slice: only the statements that "
              "determine the loop control variables are shown, and `/* ... */` "
              "marks removed code.\n\n")

USER = """{ask}

Program: {name}
Property: {prop}
{loops}
```c
{source}
```"""

def loop_id_block(rec):
    """
    Loop identifiers only, so a schedule reply can name them. Guards, bound
    variables and categories are deliberately withheld: supplying those would
    be handing the model the output of the static-analysis strategy it is being
    compared against.
    """
    ids = [l.get("id") for l in rec.get("loops", []) if l.get("id")]
    if not ids:
        return ""
    if len(ids) == 1:
        return "Loop identifier: {}\n".format(ids[0])
    return "Loop identifiers ({}): {}\n".format(len(ids), ", ".join(ids))


def build(rec, prop, verdict, source, informed=False):
    """
    Return (system, user).

    informed=False  the verdict is withheld (fair comparison, default)
    informed=True   the verdict is supplied (capability upper bound)
    """
    if not informed:
        ask = ASK_BLIND
    else:
        ask = ASK_INFORMED_UNSAFE if verdict == "unsafe" else ASK_INFORMED_SAFE

    user = USER.format(
        ask=ask,
        name=rec.get("c_file", "?"),
        prop=PROP_DESC.get(prop, prop),
        loops=loop_id_block(rec),
        source=source)
    return SYSTEM, user


# ------------------------------------------------------------ response parse

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
VALID_CONF = {"high", "medium", "low"}


def parse(text):
    """
    Return (obj, error). obj is None when the reply cannot be used.
    Validation is strict: a malformed answer is an abstention, not a guess.
    """
    if not text or not text.strip():
        return None, "empty response"
    m = JSON_RE.search(text)
    if not m:
        return None, "no JSON object in reply"
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return None, "malformed JSON: {}".format(str(e)[:60])
    if not isinstance(obj, dict):
        return None, "JSON is not an object"

    kind = obj.get("kind")
    if kind == "uniform":
        k = obj.get("k")
        if isinstance(k, bool) or not isinstance(k, int) or k < 0:
            return None, "uniform without a non-negative integer k"
    elif kind == "schedule":
        sched = obj.get("schedule")
        if not isinstance(sched, dict) or not sched:
            return None, "schedule without a schedule object"
        for lid, v in sched.items():
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                return None, "schedule bound for {} is not a non-negative integer".format(lid)
    elif kind == "unbounded":
        pass
    else:
        return None, "unknown kind: {}".format(kind)

    conf = obj.get("confidence")
    if conf is not None and str(conf).lower() not in VALID_CONF:
        obj["confidence"] = None
    return obj, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--print", dest="task", required=True,
                    help="task to render, e.g. loops/bubble_sort-1.yml")
    ap.add_argument("--property", default="reach",
                    choices=["reach", "memsafety", "overflow"])
    ap.add_argument("--mode", choices=["full", "slice"], default="slice")
    a = ap.parse_args()

    import context_builder

    recs = json.load(open(a.json))
    hit = [r for r in recs if r["task"] == a.task or a.task in r["task"]]
    if not hit:
        print("no such task")
        return 1
    rec = hit[0]

    key = {"reach": "unreach-call", "memsafety": "valid-memsafety",
           "overflow": "no-overflow"}[a.property]
    verdict = (rec.get("verdicts") or {}).get(key) or "unsafe"

    source, info = context_builder.build_context(rec, a.dataset, a.mode)
    if source is None:
        print("context failed:", info)
        return 1

    system, user = build(rec, a.property, verdict, source, a.mode)
    print("--- SYSTEM " + "-" * 58)
    print(system)
    print("\n--- USER " + "-" * 60)
    print(user)
    print("\n--- SIZE " + "-" * 60)
    print("system {} chars, user {} chars, ~{} tokens total".format(
        len(system), len(user), (len(system) + len(user)) // 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())