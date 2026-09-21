"""Undo the hat sanitation, then re-audit the reverted crops against the rule.

The pass moved 92 unique crops from `present` to `absent` on a geometry
heuristic. Where the three-view teacher had an opinion it was overridden and
wrong 25 times out of 37, and the sampled images are overwhelmingly green
surgical caps -- food-grade PPE, not street hats.

Reverting is step one. Step two re-audits every reverted crop with the same
method that produced the rest of the labels: the rule text, read per crop, by
the model that scored 95% agreement with the seed labels. A brim is legible to
a reader given the rule; it was not legible to the heuristic.
"""
import argparse, json, shutil, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apd.schema import HEAD_CAP_RULE, assert_contract

assert_contract()
ap = argparse.ArgumentParser()
ap.add_argument("--log", default="artifacts/sanitized_head_cap_hats.txt")
ap.add_argument("--roots", default="data/crops_ci_inpaint,data/crops_plus_sdk,data/crops_cocoheld")
ap.add_argument("--dry-run", type=int, default=0)
a = ap.parse_args()

files = [l.strip() for l in Path(a.log).read_text().splitlines()
         if l.strip() and not l.startswith("#")]
names = {Path(f).name for f in files}
print(f"{len(files)} files in the sanitation log ({len({n.split('__')[0] for n in names})} unique crops)\n")

moved, missing = Counter(), 0
for root in a.roots.split(","):
    for split in ("train", "val"):
        src_dir = Path(root) / "head_cap" / split / "absent"
        dst_dir = Path(root) / "head_cap" / split / "present"
        if not src_dir.is_dir():
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        for n in names:
            src = src_dir / n
            if not src.exists():
                continue
            if a.dry_run:
                moved[f"{root}/{split}"] += 1
                continue
            shutil.move(str(src), str(dst_dir / n))
            moved[f"{root}/{split}"] += 1

print("reverted absent -> present:")
for k, v in sorted(moved.items()):
    print(f"  {k:34} {v}")
print(f"  {'TOTAL':34} {sum(moved.values())}")
if a.dry_run:
    print("\n(dry run — nothing moved)")
