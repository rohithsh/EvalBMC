#!/usr/bin/env python3
"""
bmc_cache.py

Two caches, so that nothing expensive is ever computed twice.

PREDICTION CACHE  (results/cache/predictions.sqlite)
    One row per model answer.
    Key:  model, temperature, sample index, and the SHA-256 of the exact
          prompt sent (system prompt + user prompt).
    The user prompt contains the program source, the property, the expected
    verdict and the loop identifiers, and its wording is the prompt version,
    so a change to any of them is a different key and a fresh call. Nothing
    has to be invalidated by hand.
    Only answers that parsed are stored. A timeout, a server error or a reply
    with no usable JSON is never cached, so the next run tries it again.

VERIFICATION CACHE  (results/cache/verify.sqlite)
    One row per CBMC run at a given bound.
    Key:  task, property, the SHA-256 of the program file, the bound, the
          CBMC version, and the SHA-256 of the CBMC flags.
    The bound is the key, not the approach or the model that proposed it:
    if the syntactic baseline and Qwen both propose k = 33 for the same
    program, CBMC runs once.
    A conclusive result (works, insufficient, no_violation, other_property,
    inconclusive) is reused under any time budget. A timeout is reused only
    when it was recorded under a budget at least as large as the one asked
    for, and never with --retry-timeouts. Errors are never cached.

Commands
--------
  python3 bmc_cache.py stats

  # fill the prediction cache from a predictions file that already exists,
  # rebuilding each prompt with the current predict_bounds.py
  python3 bmc_cache.py seed-predictions \
      --jsonl results/pred_qwen397b_full.clean.jsonl \
      --json datasets/cleaned/loops.json --dataset datasets/cleaned/svcomp_clean \
      --input full --max-chars 900000

  # one row per task/property: k*, then the bound and the result for each
  # scored approach/model
  python3 bmc_cache.py export-wide \
      --kstar results/kstar_final.csv \
      --score syntactic=results/score_syn \
      --score qwen397b=results/score_qwen397b_full.csv \
      --out results/bounds_wide.csv
"""

import argparse
import csv
import hashlib
import json
import sqlite3
import subprocess
import threading
import time
from pathlib import Path

DEFAULT_DIR = Path("results/cache")
PRED_DB = DEFAULT_DIR / "predictions.sqlite"
VERIFY_DB = DEFAULT_DIR / "verify.sqlite"

CONCLUSIVE = {"works", "insufficient", "no_violation", "other_property",
              "inconclusive"}


# ------------------------------------------------------------------ hashing

def sha_text(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def bound_key(kind, k=None, sched=None):
    """A canonical text form of a bound, so equal bounds compare equal."""
    if kind == "uniform" and k is not None:
        return "u:{}".format(int(k))
    if kind == "schedule" and sched:
        return "s:" + ",".join("{}={}".format(lid, int(v))
                               for lid, v in sorted(sched.items()))
    return None


_CBMC_VERSION = {}


def cbmc_version(cbmc="cbmc"):
    if cbmc not in _CBMC_VERSION:
        try:
            p = subprocess.run([cbmc, "--version"], capture_output=True,
                               text=True, timeout=30)
            _CBMC_VERSION[cbmc] = (p.stdout or p.stderr).strip().splitlines()[0]
        except Exception:
            _CBMC_VERSION[cbmc] = "unknown"
    return _CBMC_VERSION[cbmc]


# ------------------------------------------------------------------ storage

class _DB:
    SCHEMA = ""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(str(self.path), timeout=60,
                                    check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()
        self.hits = self.misses = self.writes = 0

    def _one(self, sql, args):
        with self.lock:
            cur = self.conn.execute(sql, args)
            return cur.fetchone(), [d[0] for d in cur.description or []]

    def _write(self, sql, args):
        """
        Best effort. A cache that cannot be written (disk full, database
        locked) must never stop a run or make a caller retry work it has
        already done -- a full disk once turned saved model answers into
        repeated API calls. The failure is reported once and counted; the
        caller's own output file still receives the result.
        """
        with self.lock:
            try:
                self.conn.execute(sql, args)
                self.conn.commit()
                self.writes += 1
            except sqlite3.Error as e:
                try:
                    self.conn.rollback()
                except sqlite3.Error:
                    pass
                self.write_errors = getattr(self, "write_errors", 0) + 1
                if self.write_errors == 1:
                    import sys
                    print("[bmc_cache] WARNING: cannot write {} ({}). The run "
                          "continues; results go to the output file but are "
                          "not cached.".format(self.path, e), file=sys.stderr,
                          flush=True)
                return False
        return True

    def count(self):
        with self.lock:
            return self.conn.execute("SELECT COUNT(*) FROM {}".format(
                self.TABLE)).fetchone()[0]

    def close(self):
        with self.lock:
            self.conn.close()


class PredictionCache(_DB):
    TABLE = "predictions"
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS predictions (
        model        TEXT NOT NULL,
        temperature  REAL NOT NULL,
        prompt_sha   TEXT NOT NULL,
        sample       INTEGER NOT NULL,
        provider     TEXT,
        approach     TEXT,
        task         TEXT,
        property     TEXT,
        prompt_version TEXT,
        raw          TEXT,
        parsed       TEXT NOT NULL,
        latency_s    REAL,
        created      TEXT,
        PRIMARY KEY (model, temperature, prompt_sha, sample)
    );
    CREATE INDEX IF NOT EXISTS pred_task ON predictions (task, property);
    """

    def get(self, model, temperature, prompt_sha, sample):
        row, cols = self._one(
            "SELECT * FROM predictions WHERE model=? AND temperature=? "
            "AND prompt_sha=? AND sample=?",
            (model, float(temperature or 0.0), prompt_sha, int(sample)))
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        d = dict(zip(cols, row))
        d["parsed"] = json.loads(d["parsed"])
        return d

    def put(self, model, temperature, prompt_sha, sample, parsed, raw="",
            latency_s=None, provider="", approach="", task="", prop="",
            prompt_version=""):
        if not parsed:
            return False          # only valid answers are kept
        return self._write(
            "INSERT OR REPLACE INTO predictions VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (model, float(temperature or 0.0), prompt_sha, int(sample),
             provider, approach, task, prop, prompt_version, raw,
             json.dumps(parsed), latency_s, time.strftime("%Y-%m-%d %H:%M:%S")))


class VerifyCache(_DB):
    TABLE = "verifications"
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS verifications (
        task         TEXT NOT NULL,
        property     TEXT NOT NULL,
        file_sha     TEXT NOT NULL,
        bound_key    TEXT NOT NULL,
        cbmc_version TEXT NOT NULL,
        flags_sha    TEXT NOT NULL,
        status       TEXT NOT NULL,
        detail       TEXT,
        seconds      REAL,
        timeout      INTEGER,
        steps        INTEGER,
        variables    INTEGER,
        clauses      INTEGER,
        created      TEXT,
        PRIMARY KEY (task, property, file_sha, bound_key, cbmc_version, flags_sha)
    );
    """

    def get(self, task, prop, file_sha, bkey, version, flags_sha, timeout,
            retry_timeouts=False):
        row, cols = self._one(
            "SELECT * FROM verifications WHERE task=? AND property=? AND "
            "file_sha=? AND bound_key=? AND cbmc_version=? AND flags_sha=?",
            (task, prop, file_sha, bkey, version, flags_sha))
        if row is None:
            self.misses += 1
            return None
        d = dict(zip(cols, row))
        if d["status"] in CONCLUSIVE:
            self.hits += 1
            return d
        if (d["status"] == "timeout" and not retry_timeouts
                and (d.get("timeout") or 0) >= int(timeout)):
            self.hits += 1
            return d
        self.misses += 1
        return None

    def put(self, task, prop, file_sha, bkey, version, flags_sha, status,
            detail="", seconds=None, timeout=None, steps=None, variables=None,
            clauses=None):
        if status not in CONCLUSIVE and status != "timeout":
            return False          # errors, impractical, not_run: never cached
        return self._write(
            "INSERT OR REPLACE INTO verifications VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (task, prop, file_sha, bkey, version, flags_sha, status, detail,
             seconds, timeout, steps, variables, clauses,
             time.strftime("%Y-%m-%d %H:%M:%S")))


# ----------------------------------------------------------------- commands

def cmd_stats(a):
    for label, cls, path in (("predictions", PredictionCache, a.pred_db),
                             ("verifications", VerifyCache, a.verify_db)):
        if not Path(path).exists():
            print("{:<14} {}  (not created yet)".format(label, path))
            continue
        db = cls(path)
        print("{:<14} {}  {} rows".format(label, path, db.count()))
        with db.lock:
            if label == "predictions":
                q = ("SELECT model, approach, COUNT(*) FROM predictions "
                     "GROUP BY model, approach ORDER BY 3 DESC")
            else:
                q = ("SELECT status, COUNT(*) FROM verifications "
                     "GROUP BY status ORDER BY 2 DESC")
            for r in db.conn.execute(q):
                print("    " + "   ".join(str(x) for x in r))
        db.close()


def cmd_seed_predictions(a):
    import predict_bounds as pb
    recs = {r["task"]: r for r in json.load(open(a.json))}
    cache = PredictionCache(a.pred_db)
    seen = added = skipped = 0
    for line in open(a.jsonl):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        seen += 1
        rec = recs.get(row.get("task"))
        model = row.get("model")
        if not rec or not isinstance(model, str):
            skipped += 1
            continue
        src_path = pb.resolve_source(a.dataset, rec)
        source = pb.read_source(src_path) if src_path else None
        if source is None:
            skipped += 1
            continue
        mode = row.get("input_mode") or a.input
        prompt, _ = pb.build_prompt(rec, row["property"], row.get("verdict"),
                                    source, mode, a.max_chars)
        if prompt is None:
            skipped += 1
            continue
        psha = sha_text(pb.SYSTEM, prompt)
        for s in row.get("predictions") or []:
            if cache.put(model, row.get("temperature") or 0.0, psha,
                         s.get("sample", 0), s.get("parsed"),
                         raw=s.get("raw") or "", latency_s=s.get("latency_s"),
                         provider=row.get("provider") or "", approach=mode,
                         task=row["task"], prop=row["property"],
                         prompt_version=pb.PROMPT_VERSION):
                added += 1
    print("seed-predictions: {} rows read, {} answers cached, {} rows skipped "
          "(no record, unreadable source, or prompt too large)".format(
              seen, added, skipped))
    print("Note: prompts are rebuilt with the current predict_bounds.py. If the")
    print("prompt has changed since these predictions were made, the cached")
    print("entries will simply never be hit, so seeding is always safe.")
    print("cache now holds {} answers".format(cache.count()))


def _pred_text(r):
    kind = r.get("pred_kind")
    if kind == "uniform":
        return str(r.get("pred_k", ""))
    if kind == "schedule":
        return r.get("pred_schedule", "")
    if kind == "unbounded":
        return "unbounded"
    return ""


def cmd_export_wide(a):
    truth = {(r["task"], r["property"]): r for r in csv.DictReader(open(a.kstar))}
    labels, scores = [], {}
    for spec in a.score:
        label, _, path = spec.partition("=")
        labels.append(label)
        scores[label] = {(r["task"], r["property"]): r
                         for r in csv.DictReader(open(path))}
    cols = ["task", "property", "verdict", "category", "k_star"]
    for lab in labels:
        cols += ["k_pred_" + lab, "score_" + lab]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for key in sorted(truth):
            t = truth[key]
            out = {"task": key[0], "property": key[1],
                   "verdict": t.get("verdict", ""),
                   "category": t.get("category", ""),
                   "k_star": (t.get("uniform_kstar") if t.get("uniform_outcome") == "found"
                              else t.get("uniform_outcome", ""))}
            for lab in labels:
                s = scores[lab].get(key, {})
                out["k_pred_" + lab] = _pred_text(s)
                out["score_" + lab] = s.get("score_b", "")
            w.writerow(out)
    print("wrote {}  ({} rows, {} approaches: {})".format(
        a.out, len(truth), len(labels), ", ".join(labels)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-db", default=str(PRED_DB))
    ap.add_argument("--verify-db", default=str(VERIFY_DB))
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("stats")

    s = sub.add_parser("seed-predictions")
    s.add_argument("--jsonl", required=True)
    s.add_argument("--json", required=True)
    s.add_argument("--dataset", required=True)
    s.add_argument("--input", default="full")
    s.add_argument("--max-chars", type=int, default=900000)

    e = sub.add_parser("export-wide")
    e.add_argument("--kstar", required=True)
    e.add_argument("--score", action="append", required=True,
                   help="label=path to a score CSV; repeat per approach/model")
    e.add_argument("--out", required=True)

    a = ap.parse_args()
    return {"stats": cmd_stats, "seed-predictions": cmd_seed_predictions,
            "export-wide": cmd_export_wide}[a.cmd](a) or 0


if __name__ == "__main__":
    raise SystemExit(main())