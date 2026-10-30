#!/usr/bin/env python3
"""
predict_bounds.py

Ask a language model to predict loop-unrolling bounds for the preprocessed
SV-COMP programs. Produces predictions only -- scoring against the ground truth
is a separate step, so predictions can be re-scored without re-paying for
inference.

PROVIDER ADAPTERS
  Every backend implements one method, complete(system, user) -> text.
  Adding a provider means adding one class and one registry entry.

    openai      any OpenAI-compatible /v1/chat/completions endpoint.
                Covers vLLM, Ollama (which serves /v1), OpenAI itself,
                Together, Groq, and most hosted providers -- change --base-url.
    anthropic   Anthropic Messages API.
    echo        a dry-run backend that returns a fixed response; use it to
                inspect prompts and test the pipeline without inference.

  Examples:
    vLLM      --provider openai --base-url http://localhost:8000/v1 \
              --model openai/gpt-oss-20b
    Ollama    --provider openai --base-url http://localhost:11434/v1 \
              --model gpt-oss:20b
    OpenAI    --provider openai --model gpt-4o --api-key-env OPENAI_API_KEY
    Anthropic --provider anthropic --model claude-sonnet-4-6 \
              --api-key-env ANTHROPIC_API_KEY

INPUT MODES (--input)
    full        the whole preprocessed source file
    slice       the function containing each loop, plus file-scope
                declarations -- a program slice, not similarity retrieval.
                Bound variables come from loops.json, so the slice is targeted
                at what actually determines the trip count.
    inventory   slice plus the static loop inventory (CBMC loop ids, guards,
                categories, nesting). Stronger, but not a fair baseline against
                static analysis, since the inventory IS static analysis output.

  Preprocessed .i files are large (ntdrivers averages ~7.4k lines), so 'full'
  will exceed the context window on some programs. Those are recorded as
  'too_large' rather than silently truncated.

OUTPUT FORMAT
  The model must answer with JSON. Three answer kinds are allowed so that
  declining is expressible -- roughly 95% of input-bounded programs have no
  finite bound, and a predictor restricted to integers would be scored wrong on
  all of them when abstaining is correct:

    {"kind":"uniform","k":12,"confidence":"high","reasoning":"..."}
    {"kind":"schedule","schedule":{"main.0":3,"main.1":200},...}
    {"kind":"unbounded","reasoning":"trip count is an unconstrained input"}

Usage:
  python3 predict_bounds.py \
      --json    datasets/cleaned/loops.json \
      --dataset datasets/cleaned/svcomp_clean \
      --out     results/predictions_gptoss_slice.jsonl \
      --provider openai --base-url http://localhost:8000/v1 \
      --model openai/gpt-oss-20b \
      --input slice [--only-tasks results/kstar_all_v2.csv]
      [--samples 1] [--temperature 0] [--jobs 4] [--limit 20]
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from bmc_cache import PredictionCache, sha_text, PRED_DB

PROP_KEYS = {
    "reach": "unreach-call",
    "memsafety": "valid-memsafety",
    "overflow": "no-overflow",
}

PROP_DESC = {
    "reach": "reachability: can the call to reach_error() be reached?",
    "memsafety": "memory safety: is there an invalid dereference, an invalid "
                 "free, or memory that is allocated and then lost?",
    "overflow": "signed integer overflow: can any operation on a signed "
                "integer overflow?",
}


# ============================================================ provider adapters
#
# The providers live in llm_client.py, which owns the key ring: it reads every
# API_KEY* variable from .env, rotates to the next key on HTTP 429, and when
# every key is rate limited it waits for the next clock hour. Keeping a second
# copy of the provider classes here is what produced the last round of bugs --
# the prompt fixes went into one copy and the run used the other -- so this
# script now has no provider code of its own.

try:
    from llm_client import (add_provider_args, build_provider,  # noqa: F401
                            ProviderError)
except ImportError as e:
    print("predict_bounds.py needs llm_client.py beside it: {}".format(e),
          file=sys.stderr)
    raise


# ================================================================ prompt build

SYSTEM = """You are analysing C programs for bounded model checking.

A bounded model checker unrolls each loop a fixed number of times, then checks \
the property on the unrolled program. If a loop is unrolled fewer times than it \
can actually iterate, the check is unsound and the tool reports that the bound \
was insufficient. Your task is to predict how many unrollings are needed.

The bound k is ONE MORE than the number of iterations. CBMC's --unwind k makes \
k copies of the loop body and then asserts that the loop has finished, so a \
loop that runs 32 times needs k = 33: the 33rd unrolling is what establishes \
that the loop has exited. Give k, not the iteration count.

Answer with a single JSON object and nothing else. Three answers are allowed:

1. A single bound applied to every loop:
   {"kind":"uniform","k":<integer>,"confidence":"high|medium|low","reasoning":"<brief>"}

2. A bound per loop, using ONLY the loop identifiers listed in the prompt:
   {"kind":"schedule","schedule":{"<loop id>":<integer>,...},"confidence":"...","reasoning":"<brief>"}
   The identifiers are CBMC's own and are NOT in source order: in a nested
   pair the inner loop is given the LOWER index. Match each identifier to a
   loop by the source line printed next to it, never by the order the loops
   appear in the program. Do not invent an identifier.

3. No finite bound exists, because some loop's iteration count is an \
unconstrained program input and no fixed number of unrollings covers every input:
   {"kind":"unbounded","confidence":"...","reasoning":"<brief>"}

Answer 3 when it is correct. A loop such as `while (i < n)` where n is an \
unconstrained input has no finite bound, and claiming one would be wrong.

Keep "reasoning" under 60 words and state what determines the bound."""


TASK = """Program: {name}
Property to check: {prop}
Expected verdict: {verdict}

{ask}

{inventory}Source:
```c
{source}
```"""

ASK_UNSAFE = ("This program violates the property. Predict the smallest "
              "unwinding depth k at which a bounded model checker would expose "
              "the violation. Remember that k is the iteration count plus one.")
ASK_SAFE = ("This program satisfies the property. Predict the smallest unwinding "
            "depth k at which every loop is exhausted, so the property can be "
            "proved rather than merely checked to that depth. Here k is each "
            "loop's full iteration count plus one.")

# A short label for the prompt wording, stored with every cached answer so a
# cache can be read back by prompt version. The cache KEY is the hash of the
# full prompt actually sent, which changes with any edit here anyway.
import hashlib as _hashlib
PROMPT_VERSION = _hashlib.sha256(
    (SYSTEM + TASK + ASK_UNSAFE + ASK_SAFE).encode()).hexdigest()[:12]


def read_source(path):
    try:
        return path.read_text(errors="replace")
    except Exception:
        return None


def resolve_source(dataset, rec):
    """
    Locate a record's C file, or None.

    The expected layout is <dataset>/<dir>/<c_file>, which is what
    build_clean_dataset.py writes. Two fallbacks, because the dataset on disk
    and the JSON can disagree about the extension: build_clean_dataset.py
    copies whatever the .yml's input_files names, which for many SV-COMP
    tasks is a preprocessed .i, while loops.json records a .c. A flat layout
    is tried last.
    """
    d = Path(dataset)
    name = rec.get("c_file") or ""
    stem = Path(name).stem
    cands = [d / rec.get("dir", "") / name]
    for ext in (".i", ".c"):
        cands.append(d / rec.get("dir", "") / (stem + ext))
    cands.append(d / name)
    for c in cands:
        if c.is_file():
            return c
    return None


def slice_source(text, rec):
    """
    A program slice: file-scope declarations plus every function containing a
    loop. Targeted at what determines the trip count, rather than retrieved by
    similarity -- the enclosing function names come from loops.json.

    Deliberately crude and brace-counting based: preprocessed .i files are flat
    C with no templates or classes, so a function body is delimited by matching
    braces at column zero. Anything not confidently parsed is kept.
    """
    lines = text.splitlines()
    want = {l.get("function") for l in rec.get("loops", []) if l.get("function")}
    if not want:
        return text

    keep = [False] * len(lines)
    fn_start = re.compile(r"^[A-Za-z_][\w \t\*\(\)\[\],]*\b(" +
                          "|".join(re.escape(f) for f in want) +
                          r")\s*\(")

    i = 0
    while i < len(lines):
        if fn_start.search(lines[i]):
            depth, j, opened = 0, i, False
            while j < len(lines):
                depth += lines[j].count("{") - lines[j].count("}")
                if "{" in lines[j]:
                    opened = True
                keep[j] = True
                if opened and depth <= 0:
                    break
                j += 1
            i = j + 1
            continue
        i += 1

    # file-scope declarations: non-blank lines outside any function body
    depth = 0
    for n, ln in enumerate(lines):
        if depth == 0 and not keep[n] and ln.strip():
            if not ln.lstrip().startswith("//"):
                keep[n] = True
        depth += ln.count("{") - ln.count("}")
        depth = max(depth, 0)

    out, gap = [], False
    for n, ln in enumerate(lines):
        if keep[n]:
            if gap:
                out.append("/* ... */")
                gap = False
            out.append(ln)
        elif not gap:
            gap = True
    return "\n".join(out)


def loop_id_block(rec, with_nesting=False):
    """
    Loop identifiers with their source line, because CBMC does not number
    loops in source order: in a nested pair the INNER loop is given the LOWER
    index. Listing bare identifiers invites the model to assume source order
    and assign the bounds the wrong way round.

    Emitted in every input mode. Without it a 'schedule' answer has no
    identifiers to key on and the model makes them up.

    Identifier and line both come from `cbmc --show-loops`, which any user of
    --unwindset has to run anyway, so this is not static-analysis input.
    Nesting depth comes from our own parser, so it is added only for the
    inventory arm; the model can read the nesting off the source itself.
    """
    loops = [l for l in rec.get("loops", []) if l.get("id")]
    if not loops:
        return ""
    if len(loops) == 1:
        l = loops[0]
        return "Loop identifier: {} (line {})\n\n".format(l["id"], l.get("line"))
    out = ["Loop identifiers ({}), with the source line each one sits on.".format(len(loops)),
           "These are CBMC's own identifiers and are NOT in source order:",
           "a nested inner loop is given a LOWER index than the loop enclosing it.",
           "Match an identifier to a loop by its line, not by position."]
    for l in sorted(loops, key=lambda x: x.get("line") or 0):
        extra = ""
        if with_nesting:
            extra = "  depth {}{}".format(
                l.get("nesting_depth", 0),
                "  (inside {})".format(l["parent"]) if l.get("parent") else "")
        out.append("  {:<16} line {:<6}{}".format(
            l["id"], l.get("line", "?"), extra))
    return "\n".join(out) + "\n\n"


def inventory_block(rec):
    rows = []
    for l in rec.get("loops", []):
        rows.append(
            "  {id:<14} line {line:<6} {kind:<6} depth={d} category={cat}\n"
            "      guard: {g}\n      bound: {b}{con}".format(
                id=l.get("id", "?"), line=l.get("line", "?"),
                kind=l.get("kind", "?"), d=l.get("nesting_depth", 0),
                cat=l.get("category", "?"), g=(l.get("guard") or "")[:80],
                b=(l.get("bound_expr") or "(literal)")[:60],
                con=("  [constrained: {}]".format((l.get("constraint") or "")[:40])
                     if l.get("bound_constrained") else "")))
    if not rows:
        return ""
    return ("Static loop inventory (loop ids are CBMC's, as used by --unwindset,\n"
            "and are NOT in source order -- go by the line number):\n"
            + "\n".join(rows) + "\n\n")


def build_prompt(rec, prop, verdict, source, mode, max_chars):
    text = source
    if mode in ("slice", "inventory"):
        text = slice_source(source, rec)
    if len(text) > max_chars:
        return None, len(text)
    # the identifier list goes in whatever the mode, otherwise a 'schedule'
    # answer has nothing to key on; the fuller inventory is mode-gated because
    # it is static-analysis output and the full/slice arms must not see it
    inv = loop_id_block(rec, with_nesting=(mode == "inventory"))
    if mode == "inventory":
        inv += inventory_block(rec)
    return TASK.format(
        name=rec["c_file"], prop=PROP_DESC.get(prop, prop), verdict=verdict,
        ask=ASK_UNSAFE if verdict == "unsafe" else ASK_SAFE,
        inventory=inv, source=text), len(text)


# ================================================================== response

_TRAILING_COMMA = re.compile(r",\s*([}\]])")
_LINE_COMMENT = re.compile(r"//[^\n]*")


def _balanced(text, start):
    """End index (exclusive) of the {...} opened at text[start], or None.
    Tracks double-quoted strings so braces inside them do not count."""
    depth, i, in_str, esc = 0, start, False, False
    while i < len(text):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return None


def _load_lenient(s):
    """json.loads, then the same text with the slips models commonly make
    repaired: trailing commas, // comments, Python-style quoting. None if
    nothing works. These repairs change the syntax, never the values."""
    try:
        return json.loads(s)
    except ValueError:
        pass
    fixed = _TRAILING_COMMA.sub(r"\1", _LINE_COMMENT.sub("", s))
    try:
        return json.loads(fixed)
    except ValueError:
        pass
    try:
        import ast
        py = re.sub(r"\btrue\b", "True", fixed)
        py = re.sub(r"\bfalse\b", "False", py)
        py = re.sub(r"\bnull\b", "None", py)
        obj = ast.literal_eval(py)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _as_bound(v):
    """A non-negative int from 33, 33.0 or "33"; None otherwise."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v if v >= 0 else None
    if isinstance(v, float) and v.is_integer() and v >= 0:
        return int(v)
    if isinstance(v, str) and v.strip().isdigit():
        return int(v.strip())
    return None


def parse_response(text):
    """
    Extract the answer object. Returns (parsed_or_None, error_string).

    The answer is the LAST {...} in the reply that carries a "kind" key.
    The first-brace-to-last-brace match used before broke whenever the
    reasoning quoted C code such as `for (...) { ... }` ahead of the JSON:
    the match then spanned the code and failed with "Expecting property
    name enclosed in double quotes", although the answer itself was fine.
    """
    if not text:
        return None, "empty response"
    candidates = []
    for i, c in enumerate(text):
        if c != "{":
            continue
        end = _balanced(text, i)
        if end is None:
            continue
        obj = _load_lenient(text[i:end])
        if isinstance(obj, dict) and "kind" in obj:
            candidates.append(obj)
    if not candidates:
        return None, ("no JSON object in response" if "{" not in text
                      else "malformed JSON: no parseable object with a 'kind'")
    obj = candidates[-1]

    kind = obj.get("kind")
    if kind == "uniform":
        k = _as_bound(obj.get("k"))
        if k is None:
            return None, "uniform without a non-negative integer k"
        obj["k"] = k
    elif kind == "schedule":
        sched = obj.get("schedule")
        if not isinstance(sched, dict) or not sched:
            return None, "schedule without a schedule object"
        clean = {}
        for lid, v in sched.items():
            b = _as_bound(v)
            if b is None:
                return None, "schedule contains a non-integer bound"
            clean[str(lid)] = b
        obj["schedule"] = clean
    elif kind == "unbounded":
        pass
    else:
        return None, "unknown kind: {}".format(kind)
    return obj, ""


# ==================================================================== worker

def predict_one(job):
    (rec, prop, verdict, dataset, mode, max_chars, provider,
     samples, retries, cache) = job

    out = {
        "task": rec["task"], "dir": rec["dir"], "c_file": rec["c_file"],
        "property": prop, "verdict": verdict,
        "category": (lambda c: next(iter(c)) if len(c) == 1 else "mixed")(
            {l["category"] for l in rec.get("loops", [])} or {"none"}),
        "n_loops": len(rec.get("loops", [])),
        "input_mode": mode,
        # stamp the model into every row. Without it a predictions file cannot
        # be attributed afterwards, and resume cannot tell one model's rows
        # from another's, which matters because the plan is to redo the whole
        # evaluation on a different model once the pipeline is settled.
        "provider": provider.name,
        "model": provider.model,
        "temperature": getattr(provider, "temperature", None),
        "predictions": [], "errors": [],
    }

    cpath = resolve_source(dataset, rec)
    source = read_source(cpath) if cpath else None
    if source is None:
        # name the path, otherwise a wrong --dataset looks like a corrupt
        # dataset and every item fails with nothing to act on
        out["errors"].append("source unreadable: {}".format(
            cpath if cpath else Path(dataset) / rec["dir"] / rec["c_file"]))
        return out
    out["source_file"] = str(cpath)

    prompt, nchars = build_prompt(rec, prop, verdict, source, mode, max_chars)
    out["source_chars"] = nchars
    if prompt is None:
        out["errors"].append("too_large")
        return out
    out["prompt_chars"] = len(prompt)

    psha = sha_text(SYSTEM, prompt)
    out["prompt_sha"] = psha
    out["prompt_version"] = PROMPT_VERSION
    temp = getattr(provider, "temperature", 0.0) or 0.0

    for s in range(samples):
        # cache first: the same model, temperature, prompt and sample index
        # gives back the stored answer, with its original latency, so the
        # prediction time in the speedup is the real API time
        hit = cache.get(provider.model, temp, psha, s) if cache else None
        if hit:
            out["predictions"].append({
                "sample": s, "latency_s": hit.get("latency_s"),
                "raw": hit.get("raw"), "parsed": hit["parsed"],
                "parse_error": "", "cache_hit": True,
            })
            continue
        last_err = ""
        for attempt in range(retries + 1):
            try:
                t0 = time.time()
                raw = provider.complete(SYSTEM, prompt)
                dt = round(time.time() - t0, 2)
                parsed, err = parse_response(raw)
                out["predictions"].append({
                    "sample": s, "latency_s": dt, "raw": raw,
                    "parsed": parsed, "parse_error": err,
                })
                if cache and parsed:
                    cache.put(provider.model, temp, psha, s, parsed, raw=raw,
                              latency_s=dt, provider=provider.name,
                              approach=mode, task=rec["task"], prop=prop,
                              prompt_version=PROMPT_VERSION)
                break
            except Exception as e:
                last_err = "{}: {}".format(type(e).__name__, str(e)[:120])
                if attempt < retries:
                    time.sleep(2 ** attempt)
        else:
            out["errors"].append(last_err)
    return out


# ====================================================================== main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="loops.json")
    ap.add_argument("--dataset", required=True, help="svcomp_clean")
    ap.add_argument("--out", required=True, help="JSONL output")

    # the model flags come from llm_client, so every script that talks to a
    # model takes exactly the same options and reads the same key ring
    add_provider_args(ap)

    ap.add_argument("--input", choices=["full", "slice", "inventory"],
                    default="slice")
    ap.add_argument("--max-chars", type=int, default=60000,
                    help="skip a program whose prompt source exceeds this")
    ap.add_argument("--samples", type=int, default=1,
                    help=">1 with temperature>0 measures prediction stability")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=4)

    ap.add_argument("--properties", nargs="+", default=list(PROP_KEYS),
                    choices=list(PROP_KEYS))
    ap.add_argument("--verdicts", nargs="+", default=["unsafe", "safe"],
                    choices=["unsafe", "safe"])
    ap.add_argument("--only-tasks", default=None,
                    help="CSV with task+property columns; restrict to those "
                         "rows (e.g. the k* results, to predict only where "
                         "ground truth exists)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--cache-db", default=str(PRED_DB),
                    help="prediction cache (default {})".format(PRED_DB))
    ap.add_argument("--no-cache", action="store_true",
                    help="neither read nor write the prediction cache")
    ap.add_argument("--no-probe", action="store_true",
                    help="skip the single auth/model check made before the run")
    ap.add_argument("--allow-mixed-models", action="store_true",
                    help="append to an output file that already holds rows "
                         "from a different model (off by default, so a model "
                         "switch cannot silently inherit old answers)")
    a = ap.parse_args()

    if a.provider != "echo" and not a.model:
        print("--model is required", file=sys.stderr)
        return 2

    recs = json.load(open(a.json))

    allow = None
    if a.only_tasks:
        import csv as _csv
        allow = set()
        with open(a.only_tasks) as fh:
            for r in _csv.DictReader(fh):
                allow.add((r["task"], r["property"]))
        print("restricting to {} task/property pairs".format(len(allow)))

    jobs_spec = []
    for r in recs:
        v = r.get("verdicts", {}) or {}
        for prop in a.properties:
            got = v.get(PROP_KEYS[prop])
            if got not in a.verdicts:
                continue
            if allow is not None and (r["task"], prop) not in allow:
                continue
            jobs_spec.append((r, prop, got))

    # ---- pre-flight: can the sources actually be found? ----
    # A wrong --dataset used to fail every single item with "source
    # unreadable", one API-free iteration at a time, after the run had
    # already started. Check a sample up front instead.
    ds = Path(a.dataset)
    if not ds.is_dir():
        print("\n--dataset is not a directory: {}".format(ds.resolve()),
              file=sys.stderr)
        return 2
    if jobs_spec:
        sample = [j[0] for j in jobs_spec[:40]]
        found = [r for r in sample if resolve_source(a.dataset, r)]
        if not found:
            r = sample[0]
            tried = ds / r.get("dir", "") / r.get("c_file", "")
            print("\nNone of the first {} records could be located under "
                  "{}".format(len(sample), ds.resolve()), file=sys.stderr)
            print("  expected layout  <dataset>/<dir>/<c_file>", file=sys.stderr)
            print("  first one tried  {}".format(tried), file=sys.stderr)
            sub = ds / r.get("dir", "")
            if not sub.is_dir():
                here = sorted(p.name for p in ds.iterdir() if p.is_dir())[:8]
                print("  '{}' does not exist under the dataset. Directories "
                      "present: {}".format(r.get("dir", ""),
                                           ", ".join(here) or "(none)"),
                      file=sys.stderr)
            else:
                here = sorted(p.name for p in sub.iterdir())[:8]
                print("  '{}' exists but does not hold '{}'. It holds: "
                      "{}".format(r.get("dir", ""), r.get("c_file", ""),
                                  ", ".join(here) or "(empty)"),
                      file=sys.stderr)
            print("\nEither --dataset points at the wrong tree, or --json is "
                  "not the loops file that matches it.", file=sys.stderr)
            return 2
        if len(found) < len(sample):
            print("warning: {} of the first {} records have no source file "
                  "under {}".format(len(sample) - len(found), len(sample), ds))

    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    # One writer per output file. Two runs started together on the same file
    # both see it empty, both pass the model check, and then interleave their
    # rows -- which is how a 27b run and a 397b run once ended up in one file.
    # The lock is held until this process exits.
    import fcntl
    lock_path = outp.with_name(outp.name + ".lock")
    lock_fh = open(lock_path, "w")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("\nanother predict_bounds.py run is already writing {}".format(outp),
              file=sys.stderr)
        print("wait for it to finish, or give this run its own --out", file=sys.stderr)
        return 2
    lock_fh.write(str(os.getpid()))
    lock_fh.flush()
    if outp.exists() and not a.restart:
        # Resume is keyed on (task, property, model). Keying it on
        # (task, property) alone would let a run on one model inherit
        # another model's answers, which is exactly the mistake waiting to
        # happen when the whole evaluation is redone on a different model.
        done, other_models, legacy = set(), {}, 0
        with open(outp) as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                m = d.get("model")
                if m is None:
                    legacy += 1
                elif m == a.model:
                    done.add((d["task"], d["property"]))
                else:
                    other_models[m] = other_models.get(m, 0) + 1

        if (other_models or legacy) and not a.allow_mixed_models:
            print("\n{} already holds rows from another run:".format(outp),
                  file=sys.stderr)
            for m, n in sorted(other_models.items(), key=lambda kv: -kv[1]):
                print("  {:<28} {} rows".format(m, n), file=sys.stderr)
            if legacy:
                print("  {:<28} {} rows (written before the model was "
                      "recorded)".format("(unknown model)", legacy),
                      file=sys.stderr)
            print("\nCurrent model is {}. Appending here would mix models in "
                  "one file.".format(a.model), file=sys.stderr)
            print("Use a per-model output path, or --restart to overwrite, or "
                  "--allow-mixed-models if that is really what you want.",
                  file=sys.stderr)
            return 2

        before = len(jobs_spec)
        jobs_spec = [j for j in jobs_spec if (j[0]["task"], j[1]) not in done]
        print("resuming: {} done for {}, {} to go".format(
            before - len(jobs_spec), a.model, len(jobs_spec)))
    elif a.restart and outp.exists():
        outp.unlink()

    if a.limit:
        jobs_spec = jobs_spec[:a.limit]
    if not jobs_spec:
        print("nothing to do")
        return 0

    provider = build_provider(a)
    print("provider: {}".format(provider.describe()))

    # one cheap call before spawning workers. A bad key or an unknown model
    # name otherwise fails once per item, which is how a 401 turned into 60
    # identical lines.
    if a.provider != "echo" and not a.no_probe:
        # The probe only has to show that the key and model name are accepted,
        # so it runs with its own small limits. Inheriting the run's
        # --max-tokens and --request-timeout once let a reasoning model spend
        # 15 minutes on "Ready?" and stop the whole run.
        saved = (provider.max_tokens, provider.timeout)
        provider.max_tokens, provider.timeout = 256, 180
        msg = ""
        try:
            for attempt in range(3):
                try:
                    provider.complete("Reply with the single word OK.", "Ready?",
                                      retries=0)
                    msg = ""
                    break
                except Exception as e:
                    msg = str(e)
                    if "empty content" in msg:
                        # HTTP 200 with the budget spent thinking: the key and
                        # the model were accepted, which is all this checks
                        msg = ""
                        break
                    if "Timeout" not in msg and "timed out" not in msg:
                        break                      # a real refusal: stop
                    print("auth probe: no answer within 180 s (attempt {}/3)"
                          .format(attempt + 1))
        finally:
            provider.max_tokens, provider.timeout = saved
        if not msg:
            print("auth probe: ok")
        else:
            print("\nauth probe failed: {}".format(msg[:300]), file=sys.stderr)
            if "401" in msg or "Unauthorized" in msg:
                print("\nThe endpoint rejected the key. llm_client reads every "
                      "API_KEY* variable from .env, so check that", file=sys.stderr)
                print("  * .env sits in the directory you are running from",
                      file=sys.stderr)
                print("  * python-dotenv is installed in this venv, otherwise "
                      "the .env is never read", file=sys.stderr)
                print("  * the key is valid for {}".format(a.base_url),
                      file=sys.stderr)
                print("The key count printed above says how many were found.",
                      file=sys.stderr)
            elif "Timeout" in msg or "timed out" in msg:
                print("\nThe key was not rejected; the endpoint did not answer "
                      "a one-line request in 3 x 180 s,\nso it is overloaded or "
                      "down. Try again later.", file=sys.stderr)
            elif "404" in msg or "model" in msg.lower():
                print("\nThe endpoint may not serve '{}' under that name."
                      .format(a.model), file=sys.stderr)
            print("\nNothing was written. Use --no-probe to skip this check.",
                  file=sys.stderr)
            return 2

    print("input mode: {}   items: {}   samples: {}".format(
        a.input, len(jobs_spec), a.samples))

    cache = None if a.no_cache else PredictionCache(a.cache_db)
    if cache:
        print("prediction cache: {}  ({} answers stored)".format(
            a.cache_db, cache.count()))
    jobs = [(rec, prop, verdict, a.dataset, a.input, a.max_chars,
             provider, a.samples, a.retries, cache)
            for (rec, prop, verdict) in jobs_spec]

    n_done = ok = bad = skipped = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(predict_one, j): (j[0]["task"], j[1]) for j in jobs}
        for f in as_completed(futs):
            task, prop = futs[f]
            try:
                res = f.result()
            except Exception as e:
                res = {"task": task, "property": prop, "predictions": [],
                       "errors": ["worker: {}".format(str(e)[:100])]}
            with open(outp, "a") as fh:
                fh.write(json.dumps(res) + "\n")

            n_done += 1
            if "too_large" in res.get("errors", []):
                skipped += 1
                tag = "too_large"
            elif res.get("predictions") and res["predictions"][0].get("parsed"):
                ok += 1
                p = res["predictions"][0]["parsed"]
                tag = p.get("kind")
                if p.get("kind") == "uniform":
                    tag += " k={}".format(p.get("k"))
            else:
                bad += 1
                tag = (res["predictions"][0].get("parse_error")
                       if res.get("predictions") else
                       (res.get("errors") or ["?"])[0])[:30]
            print("[{}/{}] {:<40} {:<10} {:<22} ({:.0f}m)".format(
                n_done, len(jobs), res.get("task", task)[:40],
                res.get("property", prop), tag, (time.time() - t0) / 60.0),
                flush=True)

    print("\nparsed ok: {}   unparsable/error: {}   skipped (too large): {}"
          .format(ok, bad, skipped))
    if cache:
        print("prediction cache: {} hits, {} API calls, {} new answers stored"
              .format(cache.hits, cache.misses, cache.writes))
    print("predictions -> {}".format(outp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())