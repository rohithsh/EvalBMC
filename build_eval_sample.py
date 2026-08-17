#!/usr/bin/env python3
"""
build_eval_sample.py

Draw the fixed evaluation sample. Every strategy -- the LLM, syntactic
counting, static analysis, the learned model, and the iterative-deepening
baseline -- is measured on this same set, so the sample is drawn once, saved,
and reused. Re-running with the same --seed reproduces it exactly.

WHY STRATIFY
  Drawing at random from the programs with a ground-truth bound would be
  dominated by whatever is most numerous rather than most informative:

    - safe programs outnumber unsafe ones roughly three to one, and the two are
      different prediction tasks (a safe bound is the loop's trip count; an
      unsafe one is the depth at which the violation first appears)
    - data-dependent loops outnumber input-bounded ones roughly eighteen to one
    - one directory, nla-digbench-scaling, contributes a large share of the
      population, and its bounds often appear as literals in the source, so it
      is the easiest family in the dataset

  The sample is therefore stratified on verdict x loop category, spread across
  the k* range inside each stratum, and capped per directory.

STRATA
  verdict     unsafe / safe
  category    constant / input-bounded / data-dependent / mixed
  k*          within each stratum, programs are ordered by k* and drawn at even
              intervals, so the full range from shallow to deep is represented
              rather than clustered at whichever end is more populous

  Allocation is proportional to stratum size, subject to a floor
  (--min-per-cell) so small strata such as input-bounded are not lost, and a
  ceiling at what the stratum actually contains.

TIMED-OUT PROGRAMS
  Programs whose k* search never finished have no ground truth, so accuracy and
  tightness are undefined for them -- but a predicted bound can still be run
  through CBMC to see whether it works. These are the programs where bounded
  model checking failed outright, which is where a predictor is most useful, so
  --n-timeout draws a separate labelled group of them. They are reported apart
  from the main sample, never pooled with it.

  Programs with k* = 0 are excluded: the violation or the proof needs no loop
  iteration at all, so no bound is being predicted.

Usage:
  python3 build_eval_sample.py \
      --kstar   results/kstar_all_v2.csv \
      --out     results/eval_sample.csv \
      [--n 200] [--n-timeout 40] [--max-per-dir 15]
      [--min-per-cell 8] [--seed 0]
"""

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

CATEGORIES = ["constant", "input-bounded", "data-dependent", "mixed"]
VERDICTS = ["unsafe", "safe"]


def num(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def spread_pick(rows, want, key):
    """
    Take `want` rows spread evenly across the range of `key`, rather than at
    random, so the sample covers shallow and deep bounds alike instead of
    clustering wherever the population is densest.
    """
    if want <= 0 or not rows:
        return []
    ordered = sorted(rows, key=key)
    if want >= len(ordered):
        return ordered
    step = len(ordered) / float(want)
    return [ordered[min(int(i * step), len(ordered) - 1)] for i in range(want)]


def allocate(sizes, total, floor):
    """
    Split `total` across strata in proportion to size, with a floor for small
    strata and a ceiling at what each stratum holds. Any shortfall or excess is
    settled against the strata that still have room.
    """
    live = {k: v for k, v in sizes.items() if v > 0}
    if not live:
        return {}
    pool = sum(live.values())
    alloc = {}
    for k, v in live.items():
        want = int(round(total * v / float(pool)))
        alloc[k] = min(v, max(min(floor, v), want))

    # settle the difference
    for _ in range(1000):
        diff = total - sum(alloc.values())
        if diff == 0:
            break
        if diff > 0:
            room = [k for k in alloc if alloc[k] < live[k]]
            if not room:
                break
            for k in sorted(room, key=lambda x: -live[x]):
                if diff == 0:
                    break
                alloc[k] += 1
                diff -= 1
        else:
            room = [k for k in alloc if alloc[k] > min(floor, live[k])]
            if not room:
                room = [k for k in alloc if alloc[k] > 0]
            if not room:
                break
            for k in sorted(room, key=lambda x: alloc[x]):
                if diff == 0:
                    break
                alloc[k] -= 1
                diff += 1
    return alloc


def apply_dir_cap(picked, cap, pool_by_cell, already, rng):
    """
    Enforce a per-directory ceiling. A row over the cap is swapped for another
    from the same stratum whose directory still has room; if none exists the row
    is dropped, since exceeding the cap would let one family dominate.
    """
    if not cap:
        return picked, []
    kept, dropped = [], []
    counts = Counter(already)
    chosen = {(r["task"], r["property"]) for r in picked}

    for r in picked:
        d = r["dir"]
        if counts[d] < cap:
            kept.append(r)
            counts[d] += 1
            continue
        # find a replacement from the same stratum
        cell = (r["verdict"], r["category"])
        swapped = False
        candidates = [c for c in pool_by_cell.get(cell, [])
                      if (c["task"], c["property"]) not in chosen
                      and counts[c["dir"]] < cap]
        if candidates:
            rng.shuffle(candidates)
            c = candidates[0]
            kept.append(c)
            chosen.add((c["task"], c["property"]))
            counts[c["dir"]] += 1
            swapped = True
        if not swapped:
            dropped.append(r)
    return kept, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kstar", required=True, help="derive_kstar_all CSV")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200,
                    help="programs with a finite k* > 0")
    ap.add_argument("--n-timeout", type=int, default=40,
                    help="programs whose k* search timed out (no ground truth)")
    ap.add_argument("--max-per-dir", type=int, default=15,
                    help="ceiling per source directory; 0 disables")
    ap.add_argument("--min-per-cell", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    rows = list(csv.DictReader(open(a.kstar)))

    usable, zero, timeouts = [], [], []
    for r in rows:
        k = num(r.get("uniform_kstar"))
        if r.get("uniform_outcome") == "found" and k is not None:
            (usable if k > 0 else zero).append(r)
        elif r.get("uniform_outcome") in ("timeout", "cap_reached"):
            timeouts.append(r)

    print("population")
    print("   with finite k* > 0 : {}".format(len(usable)))
    print("   k* = 0 (excluded)  : {}".format(len(zero)))
    print("   no k* (timed out)  : {}".format(len(timeouts)))

    # ---- cross-tabulate the usable population -----------------------------
    pool = defaultdict(list)
    for r in usable:
        cat = r.get("category") or "mixed"
        if cat not in CATEGORIES:
            cat = "mixed"
        pool[(r["verdict"], cat)].append(r)

    print("\navailable, verdict x category")
    print("   {:<10}".format("") + "".join("{:>16}".format(c) for c in CATEGORIES))
    for v in VERDICTS:
        print("   {:<10}".format(v) +
              "".join("{:>16}".format(len(pool[(v, c)])) for c in CATEGORIES))

    sizes = {cell: len(v) for cell, v in pool.items()}
    alloc = allocate(sizes, a.n, a.min_per_cell)

    print("\nallocated")
    print("   {:<10}".format("") + "".join("{:>16}".format(c) for c in CATEGORIES))
    for v in VERDICTS:
        print("   {:<10}".format(v) +
              "".join("{:>16}".format(alloc.get((v, c), 0)) for c in CATEGORIES))

    picked = []
    for cell, want in alloc.items():
        picked += spread_pick(pool[cell], want,
                              key=lambda r: num(r["uniform_kstar"]) or 0)

    picked, dropped = apply_dir_cap(picked, a.max_per_dir, pool, [], rng)
    if dropped:
        print("\ndropped {} rows that exceeded the per-directory cap "
              "with no replacement available".format(len(dropped)))

    # ---- timed-out group ---------------------------------------------------
    to_pool = defaultdict(list)
    for r in timeouts:
        cat = r.get("category") or "mixed"
        if cat not in CATEGORIES:
            cat = "mixed"
        to_pool[(r["verdict"], cat)].append(r)
    to_alloc = allocate({c: len(v) for c, v in to_pool.items()},
                        a.n_timeout, max(2, a.min_per_cell // 3))
    to_picked = []
    for cell, want in to_alloc.items():
        cand = to_pool[cell][:]
        rng.shuffle(cand)
        to_picked += cand[:want]
    dir_counts = Counter(r["dir"] for r in picked)
    to_picked, to_dropped = apply_dir_cap(to_picked, a.max_per_dir,
                                          to_pool, list(dir_counts.elements()), rng)

    # ---- write -------------------------------------------------------------
    out_rows = []
    for r in picked:
        out_rows.append({
            "task": r["task"], "property": r["property"],
            "group": "kstar", "verdict": r["verdict"],
            "category": r.get("category", ""), "dir": r["dir"],
            "n_loops": r.get("n_loops", ""),
            "kstar": r.get("uniform_kstar", ""),
            "true_cost": r.get("uniform_cost", ""),
        })
    for r in to_picked:
        out_rows.append({
            "task": r["task"], "property": r["property"],
            "group": "timeout", "verdict": r["verdict"],
            "category": r.get("category", ""), "dir": r["dir"],
            "n_loops": r.get("n_loops", ""),
            "kstar": "", "true_cost": "",
        })

    cols = ["task", "property", "group", "verdict", "category", "dir",
            "n_loops", "kstar", "true_cost"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)

    # ---- report ------------------------------------------------------------
    print("\n" + "=" * 62)
    print("SAMPLE: {} with ground truth, {} timed out, {} total".format(
        len(picked), len(to_picked), len(out_rows)))
    print("=" * 62)

    print("\nverdict x category (ground-truth group)")
    c = Counter((r["verdict"], r["category"]) for r in picked)
    print("   {:<10}".format("") + "".join("{:>16}".format(x) for x in CATEGORIES))
    for v in VERDICTS:
        print("   {:<10}".format(v) +
              "".join("{:>16}".format(c[(v, x)]) for x in CATEGORIES))

    ks = sorted(num(r["uniform_kstar"]) or 0 for r in picked)
    if ks:
        q = lambda p: ks[min(int(p * len(ks)), len(ks) - 1)]
        print("\nk* spread: min={} p25={} median={} p75={} p90={} max={}".format(
            ks[0], q(.25), q(.5), q(.75), q(.9), ks[-1]))

    print("\ntop directories (all groups)")
    for d, n in Counter(r["dir"] for r in picked + to_picked).most_common(10):
        print("   {:<30} {}".format(d, n))

    print("\nloop counts")
    lc = Counter("single" if num(r.get("n_loops")) == 1
                 else ("double" if num(r.get("n_loops")) == 2 else "multi")
                 for r in picked)
    for k_ in ("single", "double", "multi"):
        print("   {:<8} {}".format(k_, lc[k_]))

    print("\nsample -> {}".format(a.out))
    print("use it with:  --only-tasks {}".format(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())