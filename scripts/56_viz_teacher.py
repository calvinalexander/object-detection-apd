"""Render the teacher's verdict -- what the labelling stack actually decided.

The existing directories show each voter separately, which is what "its labels"
means for a single model but is not what the teacher is: the teacher is the
agreement of voter3 and voter4, abstaining wherever they differ. That verdict is
what the training labels were built from, so it is the one worth reviewing.

Drawn from the recorded consensus rather than by re-running the models, so what
appears here is exactly what was written, not a fresh inference that might drift.
apd_mbg is shown only as a dissent marker: it was excluded from the teacher
because it cannot be retrained and was trained on a split we validate against.
"""
import argparse, collections, json, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apd.overlay import clean
from apd.schema import assert_contract

assert_contract()
ATTRS = ("head_cap", "mask", "gloves")
OK, BAD, UNK = (80,200,120), (60,60,235), (150,155,160)

ap = argparse.ArgumentParser()
ap.add_argument("--consensus", default="artifacts/consensus_ci3b.jsonl")
ap.add_argument("--out", default="artifacts/viz_ci3/teacher")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

recs = [json.loads(l) for l in Path(a.consensus).read_text().splitlines()]
if a.limit: recs = recs[:a.limit]
tally = collections.Counter()

def draw(im, people):
    H, W = im.shape[:2]
    CW, CH = 96, 48
    placed = []
    def free(x, y):
        if x < 2 or y < 28 or x+CW > W-2 or y+CH > H-2: return False
        return all(not (x < px+CW and x+CW > px and y < py+CH and y+CH > py) for px,py in placed)
    for box, v, dissent in sorted(people, key=lambda t: t[0][0]):
        x1,y1,x2,y2 = [int(t) for t in box]
        col = BAD if "absent" in v.values() else (OK if "present" in v.values() else UNK)
        cv2.rectangle(im,(x1,y1),(x2,y2),col,2)
        for cx,cy in ((x1,y1-CH-4),(x2+4,y1),(x1-CW-4,y1),(x1,y2+4),(x2+4,y2-CH),(x1,y1+4)):
            if free(cx,cy): break
        else:
            cx,cy = min(max(2,x1), W-CW-2), 28
            while not free(cx,cy) and cy+CH < H-2: cy += CH+3
        placed.append((cx,cy))
        cv2.line(im,(x1+2,y1+2),(cx+CW//2,cy+CH//2),col,1)
        cv2.rectangle(im,(cx,cy),(cx+CW,cy+CH),(18,24,28),-1)
        cv2.rectangle(im,(cx,cy),(cx+CW,cy+CH),col,1)
        for j,k in enumerate(ATTRS):
            s = v[k]; mark = "OK" if s=="present" else ("X" if s=="absent" else "?")
            c2 = OK if s=="present" else (BAD if s=="absent" else UNK)
            txt = f"{mark} {k}" + ("  *" if k in dissent else "")
            cv2.putText(im,txt,(cx+5,cy+14+13*j),cv2.FONT_HERSHEY_SIMPLEX,0.35,c2,1)
    return im

for n, r in enumerate(recs, 1):
    src = Path("data/sites")/r["site"]/f"{r['id']}.jpg"
    if not src.exists(): continue
    im = clean(cv2.imread(str(src)))
    if im is None: continue
    people = []
    for p in r["persons"]:
        v, dissent = {}, set()
        for at in ATTRS:
            votes = p["attrs"][at]["votes"]
            v3, v4, mb = votes["voter3"], votes["voter4"], votes["apd_mbg"]
            dec = v3 if (v3 == v4 and v3 != "unclear") else "unclear"
            v[at] = dec
            tally[f"{at}:{dec}"] += 1
            if dec != "unclear" and mb != "unclear" and mb != dec:
                dissent.add(at); tally[f"{at}:apd_mbg_dissent"] += 1
        people.append((p["box"], v, dissent))
    im = draw(im, people)
    H_, W_ = im.shape[:2]
    cv2.rectangle(im,(0,0),(W_,26),(18,24,28),-1)
    cv2.putText(im,f"teacher  voter3 AND voter4 agree   {r['site']}",(8,18),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(235,240,242),1)
    sub = f"flag: {r.get('flagged','')}   persons: {len(people)}   * = apd_mbg disagreed"
    (tw,_),_ = cv2.getTextSize(sub,cv2.FONT_HERSHEY_SIMPLEX,0.42,1)
    cv2.putText(im,sub,(max(8,W_-tw-8),18),cv2.FONT_HERSHEY_SIMPLEX,0.42,(150,200,220),1)
    d = Path(a.out)/r["site"]; d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d/f"{r['id']}.jpg"), im)
    if n % 300 == 0: print(f"  {n}/{len(recs)}", flush=True)

print(f"\n{a.out}")
for at in ATTRS:
    tot = sum(tally[f"{at}:{v}"] for v in ("present","absent","unclear")) or 1
    print(f"  {at:9} present {tally[f'{at}:present']:5d}  absent {tally[f'{at}:absent']:5d}  "
          f"unclear {tally[f'{at}:unclear']:5d}   apd_mbg dissented on {tally[f'{at}:apd_mbg_dissent']}")
