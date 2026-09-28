"""Fit the v3 mask/gloves/head_cap cuts to the manual curation of the passed folder.

The curator went through artifacts/filter16/passed image by image and marked
"flag" (this person should have been alerted) or "skip" (correctly passed).
It is the first ground truth we hold for THIS camera, and it is item-agnostic,
so scoring is at event level: the system alerts when any item is a violation.

Each item's rule is the union of two rectangles in (p_present, p_known):
    violation  <=>  (p_known >= known_above  and p_present < present_below)
                 or (p_known >= known_floor  and p_present < present_strong)
Coordinate descent over both rectangles, two passes, maximising MCC -- and the
same search under a cap on false alarms.

CAVEAT every curated event is one the model passed, so this set says nothing
about the alerts it already raises. It can only push toward flagging more, and
the false-alarm count below is a floor measured on the 852 "skip" rows.

    python scripts/139_tune_on_curation.py
"""
import csv, json, re
from pathlib import Path
import numpy as np
from sklearn.metrics import matthews_corrcoef

ROOT = Path(__file__).resolve().parents[1]
ITEMS = ["head_cap", "mask", "gloves"]

lab = {}
for r in csv.DictReader(open(ROOT/"artifacts/_ss/passed_filenames.csv")):
    if r["Status"].strip() in ("skip", "flag"):
        lab[re.sub(r"\.jpg$", "", r["Filename"].strip()).split("_")[-1]] = r["Status"].strip() == "flag"
idx = {r["event_id"][:8]: r for r in csv.DictReader(open(ROOT/"artifacts/filter16/index.csv"))}
rows = [(idx[k], v) for k, v in lab.items() if k in idx]
Y = np.array([v for _, v in rows], dtype=int)
P = {it: np.array([float(r[f"p_{it}"]) for r, _ in rows]) for it in ITEMS}
K = {it: np.array([float(r[f"known_{it}"]) for r, _ in rows]) for it in ITEMS}
print(f"{len(rows)} curated events: {Y.sum()} flag, {(1-Y).sum()} skip\n")

J = json.loads((ROOT/"deploy/apd_person/apd_ours_person_ppe_v3.json").read_text())["cuts"]
cur = {it: (J[it]["known_above"], J[it]["present_below"],
            J[it].get("known_floor", 1.1), J[it].get("present_strong", 0.0)) for it in ITEMS}


def alerts(c):
    q = np.zeros(len(Y), bool)
    for it in ITEMS:
        ka, pb, kf, ps = c[it]
        q |= ((K[it] >= ka) & (P[it] < pb)) | ((K[it] >= kf) & (P[it] < ps))
    return q.astype(int)


def score(c):
    q = alerts(c)
    return (matthews_corrcoef(Y, q), int((q & (Y == 1)).sum()), int((q & (Y == 0)).sum()))


def show(c, name):
    m, tp, fp = score(c)
    print(f"{name:32} MCC {m:.3f}  catches {tp:3d}/{Y.sum()} flags, raises {fp:3d}/{(1-Y).sum()} on skips")
    return m, tp, fp


KA = np.round(np.arange(0.30, 1.00, 0.02), 3)
PB = np.round(np.arange(0.05, 0.85, 0.025), 3)


def search(cap=None, passes=2):
    best = dict(cur)
    for _ in range(passes):
        for it in ITEMS:
            for rect in (0, 1):                     # 0 = (known_above, present_below), 1 = override
                top = (-9, best[it])
                for ka in KA:
                    for pb in PB:
                        v = list(best[it])
                        v[0 if rect == 0 else 2] = float(ka); v[1 if rect == 0 else 3] = float(pb)
                        m, tp, fp = score({**best, it: tuple(v)})
                        ok = (fp <= cap) if cap is not None else True
                        val = tp if cap is not None else m
                        if ok and val > top[0]: top = (val, tuple(v))
                if top[1]: best[it] = top[1]
    return best


show(cur, "current bundle")
b = search(); show(b, "fitted (max MCC)")
print("   " + json.dumps({i: list(b[i]) for i in ITEMS}))
for cap in (49, 85, 128):
    b2 = search(cap=cap)
    show(b2, f"fitted (false alarms <= {cap})")
    print("   " + json.dumps({i: list(b2[i]) for i in ITEMS}))
