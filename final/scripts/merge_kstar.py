#!/usr/bin/env python3
"""
merge_kstar.py -- build the k* table under the exposure rule.

Safe pairs keep their rows from the existing k* table (the rule does not
change them); unsafe pairs take their rows from the new exposure run.

  python3 merge_kstar.py --old results/kstar_final.csv \
      --unsafe results/kstar_unsafe_exposure.csv --out results/kstar_v2.csv
"""
import argparse
import csv
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--old", required=True)
ap.add_argument("--unsafe", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

old = list(csv.DictReader(open(a.old)))
new = {(r["task"], r["property"]): r for r in csv.DictReader(open(a.unsafe))}
cols = list(old[0].keys())
for c in next(iter(new.values())).keys():
    if c not in cols and c is not None:
        cols.append(c)

out, missing = [], 0
for r in old:
    k = (r["task"], r["property"])
    if r["verdict"] == "unsafe":
        if k in new:
            out.append(new[k])
        else:
            missing += 1
            out.append(r)
    else:
        out.append(r)
with open(a.out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(out)

def summary(rows, label):
    u = [r for r in rows if r["verdict"] == "unsafe"]
    ks = [int(r["uniform_kstar"]) for r in u
          if r["uniform_outcome"] == "found" and str(r["uniform_kstar"]).isdigit()]
    print("{:<9} unsafe {}  outcomes {}  k*=0 {}  k*>0 {}".format(
        label, len(u), dict(Counter(r["uniform_outcome"] for r in u)),
        sum(x == 0 for x in ks), sum(x > 0 for x in ks)))

summary(old, "complete")
summary(out, "exposure")
if missing:
    print("WARNING: {} unsafe pairs had no row in --unsafe; kept their old row".format(missing))
print("->", a.out)