"""Head crops the audit judged unjudgeable -> judgeability=0 rows for the person model.

A det7 box that is simply absent means "nobody labelled it", which says nothing
about visibility. The head-crop audit does say it: 'unclear' means a reviewer
looked and could not tell. Those become known=0 supervision.

    python scripts/133_det7_unclear_from_audit.py   ->  data/clean/det7_unclear.csv
"""
import csv, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
heads = list(csv.DictReader(open(ROOT/"artifacts/headcap_audit/heads.csv")))
ver = {r["id"]: r["verdict"] for r in csv.DictReader(open(ROOT/"artifacts/headcap_audit/verdicts.csv"))}
stems = {p.stem for p in (ROOT/"data/clean/det7/images").rglob("*.jpg")}
rows = []
for r in heads:
    if ver[r["id"]] != "unclear": continue
    m = re.match(r"(.+)_(\d+)$", re.sub(r"^sdk_", "", r["base"]))
    if not m: continue
    stem = f"{m.group(1)}_p{m.group(2)}"
    if stem in stems: rows.append({"image": stem, "item": "head_cap"})
out = ROOT/"data/clean/det7_unclear.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["image", "item"]); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} det7 people marked unjudgeable for head_cap -> {out}")
