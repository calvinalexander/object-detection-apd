"""Fit the cuts to the manual curation, honestly: fit on one half, report on the other.

The 1,214 curated events are split by CLIP HOUR, so the same stretch of footage
cannot appear in both halves. Coordinate descent over (known_above,
present_below) per item, maximising MCC on the fit half, optionally under a cap
on false alarms; the numbers printed are from the half the fit never saw.

    python scripts/148_fit_cuts_curation.py <scores.json> [--name v4]
"""
import argparse, csv, json, re, collections
from pathlib import Path
import numpy as np
from sklearn.metrics import matthews_corrcoef

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("scores"); ap.add_argument("--name", default="model")
a = ap.parse_args()
ITEMS = ["head_cap", "mask", "gloves"]

sc = {r["event"]: r for r in json.load(open(a.scores))}
by8 = {e[:8]: e for e in sc}
lab = {}
for r in csv.DictReader(open(ROOT/"artifacts/_ss/passed_filenames.csv")):
    if r["Status"].strip() in ("skip", "flag"):
        lab[re.sub(r"\.jpg$", "", r["Filename"].strip()).split("_")[-1]] = r["Status"].strip() == "flag"
rows = [(sc[by8[k]], v) for k, v in lab.items() if k in by8]
# split by hour of the day: even hours fit, odd hours report
fit = [r for r in rows if int(r[0]["time"][11:13]) % 2 == 0]
rep = [r for r in rows if int(r[0]["time"][11:13]) % 2 == 1]
print(f"{len(rows)} curated events -> fit {len(fit)} ({sum(y for _,y in fit)} flag), "
      f"held out {len(rep)} ({sum(y for _,y in rep)} flag)")


def arrs(g):
    P = {i: np.array([r["p"][k] for r, _ in g]) for k, i in enumerate(ITEMS)}
    K = {i: np.array([r["p"][3+k] for r, _ in g]) for k, i in enumerate(ITEMS)}
    return P, K, np.array([y for _, y in g], int)


Pf, Kf, Yf = arrs(fit); Pr, Kr, Yr = arrs(rep)


def alerts(P, K, c):
    q = np.zeros(len(next(iter(P.values()))), bool)
    for i in ITEMS:
        ka, pb = c[i]; q |= (K[i] >= ka) & (P[i] < pb)
    return q


def score(P, K, Y, c):
    q = alerts(P, K, c)
    return (matthews_corrcoef(Y, q) if len(set(q.tolist())) > 1 else 0.0,
            int((q & (Y == 1)).sum()), int((q & (Y == 0)).sum()))


KA = np.round(np.arange(0.05, 1.0, 0.025), 3)
PB = np.round(np.arange(0.05, 1.0, 0.025), 3)
START_MCC = {i: (0.5, 0.5) for i in ITEMS}
START_CAP = {i: (1.01, 0.0) for i in ITEMS}   # alerts on nobody: always satisfies a cap


def fit_cuts(cap=None, passes=3):
    best = dict(START_MCC if cap is None else START_CAP)
    for _ in range(passes):
        for it in ITEMS:
            top = (-9, best[it])
            for ka in KA:
                for pb in PB:
                    t = {**best, it: (float(ka), float(pb))}
                    m, tp, fp = score(Pf, Kf, Yf, t)
                    ok = cap is None or fp <= cap
                    val = tp if cap is not None else m
                    if ok and val > top[0]: top = (val, (float(ka), float(pb)))
            best[it] = top[1]
    return best


print(f"\n{a.name}: cuts fitted on the even hours, scored on the odd hours")
print(f"{'setting':22} {'catches':>12} {'false alarms':>13} {'MCC':>7}")
nsf = int((1-Yf).sum())
for name, cap in (("max MCC", None), ("fp<=10% of skips", int(0.10*nsf)), ("fp<=20% of skips", int(0.20*nsf))):
    c = fit_cuts(cap)
    m, tp, fp = score(Pr, Kr, Yr, c)
    nf, ns = int(Yr.sum()), int((1-Yr).sum())
    print(f"{name:22} {tp:5d}/{nf:<6d} {fp:6d}/{ns:<6d} {m:7.3f}   " +
          json.dumps({i: [round(v, 3) for v in c[i]] for i in ITEMS}))
