"""Apply a re-review, overwriting prior values and recording why.

A rule change invalidates every label made under the old wording. Overwriting is
correct here; losing the fact that it happened is not, so each revision keeps the
value it replaced.
"""
import json, sys, argparse, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apd.schema import Value

ap = argparse.ArgumentParser()
ap.add_argument("--sheet", required=True); ap.add_argument("--verdicts", required=True)
ap.add_argument("--attribute", default="mask"); ap.add_argument("--reason", required=True)
args = ap.parse_args()

STORE = Path("artifacts/mined_labels.jsonl")
store = {}
for line in STORE.read_text().splitlines():
    r = json.loads(line); store[(r["person_id"], r["attribute"])] = r
slots = json.loads(Path(args.sheet).read_text())
verdicts = json.loads(Path(args.verdicts).read_text())

changed = collections.Counter()
for s in slots:
    new = verdicts.get(str(s["slot"]))
    if new is None: continue
    Value(new)
    key = (s["person_id"], args.attribute)
    rec = store.get(key)
    if rec is None: continue
    old = rec["value"]
    if old != new:
        changed[f"{old}->{new}"] += 1
        rec["previous_value"] = old
        rec["revision_reason"] = args.reason
        rec["value"] = new
    else:
        changed["unchanged"] += 1

STORE.write_text("".join(json.dumps(r) + "\n" for r in store.values()))
n_ch = sum(v for k, v in changed.items() if k != "unchanged")
print(f"reviewed {sum(changed.values())}, changed {n_ch} "
      f"({100*n_ch/sum(changed.values()):.0f}%)")
for k, v in sorted(changed.items()): print(f"  {k}: {v}")
tot = collections.Counter(r["value"] for r in store.values() if r["attribute"] == args.attribute)
print(f"\n{args.attribute} mined store now: {dict(tot)}")
