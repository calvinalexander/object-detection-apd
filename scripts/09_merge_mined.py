"""Merge a reviewed mining sheet into the label store.

Mined labels carry `source: mined` and the sheet they came from. That provenance
is not decoration: labels found by ranking are a biased sample, so any statistic
computed over the store has to be able to exclude them. They are training fuel,
never evidence about prevalence.
"""
import json, sys, collections, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apd.schema import Attribute, Value

ap = argparse.ArgumentParser()
ap.add_argument("--sheet", required=True)
ap.add_argument("--verdicts", required=True)
ap.add_argument("--attribute", default=Attribute.MASK.value)
ap.add_argument("--mode", default="ranked")
args = ap.parse_args()

STORE = Path("artifacts/mined_labels.jsonl")
slots = json.loads(Path(args.sheet).read_text())
verdicts = json.loads(Path(args.verdicts).read_text())
existing = {}
if STORE.exists():
    for line in STORE.read_text().splitlines():
        r = json.loads(line)
        existing[(r["person_id"], r["attribute"])] = r

added = collections.Counter()
for s in slots:
    v = verdicts.get(str(s["slot"]))
    if v is None:
        continue
    Value(v)  # reject anything outside the contract
    key = (s["person_id"], args.attribute)
    if key in existing:
        added["already_labelled"] += 1
        continue
    existing[key] = {"person_id": s["person_id"], "frame": s["frame"],
                     "split": s["split"], "attribute": args.attribute, "value": v,
                     "source": "mined", "sampling": args.mode,
                     "sheet": Path(args.sheet).name}
    added[v] += 1

STORE.write_text("".join(json.dumps(r) + "\n" for r in existing.values()))
print(f"added: {dict(added)}")

tot = collections.Counter()
for r in existing.values():
    if r["attribute"] == args.attribute:
        tot[(r["split"], r["value"])] += 1
seed = [json.loads(l) for l in Path("artifacts/seed_labels.jsonl").read_text().splitlines()]
splits = json.loads(Path("artifacts/splits.json").read_text())["splits"]
frame_split = {f: k for k, v in splits.items() for f in v}
for r in seed:
    v = r["labels"][args.attribute]
    if v != Value.UNCLEAR.value:
        tot[(frame_split.get(r["frame"]), v)] += 1

print(f"\n{args.attribute} labels after merge (seed + mined)")
print(f"{'split':<20}{'present':>9}{'absent':>9}")
for k in ("train", "val", "test_unseen_scene", "test_same_camera"):
    print(f"{k:<20}{tot[(k,'present')]:>9}{tot[(k,'absent')]:>9}")
print(f"{'TOTAL':<20}{sum(v for (s,x),v in tot.items() if x=='present'):>9}"
      f"{sum(v for (s,x),v in tot.items() if x=='absent'):>9}")
