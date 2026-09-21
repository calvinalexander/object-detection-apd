"""Can the gloves head tell 'bare hand' from 'no hand in the crop'?

The only gate on gloves is whether a wrist keypoint exists. Nothing checks that a
hand ended up inside the ROI, and the classifier's own probability never triggers
abstention -- so a crop of a table edge is scored, comes back low, and is reported
as ABSENT, which downstream is a violation.

This prints the probability distribution and dumps the crops behind the calls.
"""
import json, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from PIL import Image, ImageDraw
from sklearn.linear_model import LogisticRegression
from transformers import AutoImageProcessor, AutoModel
from apd.schema import Value

DEV, MID = "cuda", "facebook/dinov2-base"; CROPS = Path("artifacts/crops")
proc = AutoImageProcessor.from_pretrained(MID); enc = AutoModel.from_pretrained(MID).to(DEV).eval()
@torch.no_grad()
def embed(ps,b=64):
    o=[]
    for i in range(0,len(ps),b):
        ims=[Image.open(CROPS/p).convert("RGB") for p in ps[i:i+b]]
        o.append(enc(**proc(images=ims,return_tensors="pt").to(DEV)).last_hidden_state[:,0].float().cpu().numpy())
    return np.concatenate(o)

sp=json.loads(Path("artifacts/splits.json").read_text())["splits"]
fs={f:k for k,v in sp.items() for f in v}
idx=[json.loads(l) for l in Path("artifacts/crops_index.jsonl").read_text().splitlines()]
hnd={(r["person_id"],int(r["crop"].rsplit("_h",1)[1].split(".")[0])):r["crop"] for r in idx if r["part"]=="hand"}

tr_p,tr_y,ho_p,ho_y=[],[],[],[]
for r in (json.loads(l) for l in Path("artifacts/seed_labels.jsonl").read_text().splitlines()):
    split=fs.get(r["frame"])
    for h in r.get("hands",[]):
        c=hnd.get((r["person_id"],h["hand"]))
        if not c or h["value"]==Value.UNCLEAR.value: continue
        (tr_p if split=="train" else ho_p).append(c)
        (tr_y if split=="train" else ho_y).append(int(h["value"]=="present"))
clf=LogisticRegression(max_iter=3000,C=0.05,class_weight="balanced").fit(embed(tr_p),np.array(tr_y))
P=clf.predict_proba(embed(ho_p))[:,1]; Y=np.array(ho_y)
print(f"held-out hand crops: {len(P)}  ({int(Y.sum())} gloved / {int((~Y.astype(bool)).sum())} bare)\n")

print("probability distribution -- a confident model separates, a guessing one piles at 0.5")
bins=[(0,.2),(.2,.35),(.35,.5),(.5,.65),(.65,.8),(.8,1.01)]
print(f"  {'p(gloved)':<14}{'n':>5}{'actually gloved':>18}{'verdict':>10}")
for lo,hi in bins:
    k=(P>=lo)&(P<hi)
    if not k.sum(): continue
    print(f"  {f'{lo:.2f}-{hi:.2f}':<14}{int(k.sum()):>5}{100*Y[k].mean():>17.0f}%"
          f"{('ABSENT' if hi<=.5 else 'PRESENT'):>10}")
near=((P>=.35)&(P<=.65)).mean()
print(f"\n  {100*near:.0f}% of decisions fall in the 0.35-0.65 band -- reported as")
print( "  confidently as anything else, because no confidence gate exists.")

# the crops behind the most-confident ABSENT calls
order=np.argsort(P)[:12]
C=170; sheet=Image.new("RGB",(C*6,(C+16)*2+22),(235,238,239)); d=ImageDraw.Draw(sheet)
d.text((6,6),"lowest p(gloved) -- these are the strongest ABSENT calls",fill=(30,38,42))
for n,i in enumerate(order):
    im=Image.open(CROPS/ho_p[i]).convert("RGB").resize((C-4,C-4),Image.LANCZOS)
    x,y=(n%6)*C,22+(n//6)*(C+16)
    col=(46,160,108) if Y[i] else (206,64,52)
    d.rectangle([x,y,x+C-1,y+C+14],fill=col); sheet.paste(im,(x+2,y+2))
    d.text((x+4,y+C),f"p={P[i]:.2f}  truth={'GLOVED' if Y[i] else 'bare'}",fill=(255,255,255))
sheet.save("artifacts/glove_absent_calls.png")
print("\nwrote artifacts/glove_absent_calls.png")
