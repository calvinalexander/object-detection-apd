"""Phase 1 state: what exists, what is labelled, what the binding constraint is."""
import json, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apd.schema import Attribute, Value

splits = json.loads(Path("artifacts/splits.json").read_text())
frame_split = {f: k for k, v in splits["splits"].items() for f in v}
persons = [json.loads(l) for l in Path("artifacts/persons.jsonl").read_text().splitlines()]
seed = {r["person_id"]: r for r in
        (json.loads(l) for l in Path("artifacts/seed_labels.jsonl").read_text().splitlines())}

per_split = collections.Counter(frame_split.get(p["frame"], "?") for p in persons)
auto = collections.Counter()
for p in persons:
    s = frame_split.get(p["frame"], "?")
    if not p["head_collision"] and p["head_localizable"]:
        auto[s] += 1

print("PERSON INSTANCES BY SPLIT")
print(f"{'split':<22}{'persons':>9}{'auto-labelable':>16}")
for k in ("train", "val", "test_unseen_camera", "test_same_camera"):
    print(f"{k:<22}{per_split[k]:>9}{auto[k]:>16}")
print(f"{'TOTAL':<22}{sum(per_split.values()):>9}{sum(auto.values()):>16}")

print("\nSEED LABELS BY SPLIT (from the 592 human boxes)")
rows = collections.defaultdict(lambda: collections.Counter())
for pid, r in seed.items():
    s = frame_split.get(r["frame"], "?")
    for a, v in r["labels"].items():
        if v != Value.UNCLEAR.value:
            rows[a][(s, v)] += 1
hdr = f"{'attribute':<11}" + "".join(f"{k[:14]:>16}" for k in ("train","val","test_unseen_camera","test_same_camera"))
print(hdr)
for a in Attribute:
    line = f"{a.value:<11}"
    for k in ("train", "val", "test_unseen_camera", "test_same_camera"):
        p_ = rows[a.value][(k, "present")]; n_ = rows[a.value][(k, "absent")]
        line += f"{f'{p_}+/{n_}-':>16}"
    print(line)

mask_pos = sum(v for (s, val), v in rows["mask"].items() if val == "present")
print(f"\nBINDING CONSTRAINT: mask has {mask_pos} positive examples in total.")
print("Everything else has enough to start; mask sets the labelling budget.")
