"""Re-audit the reverted crops against HEAD_CAP_RULE, per crop, with the rule text.

The heuristic could not tell a brimless surgical cap from a peaked street hat.
A reader given the rule can. Crops are shown a few at a time on a contact sheet,
the rule is injected verbatim, and only crops the reader calls a street hat are
moved back to `absent` -- `unclear` leaves the crop where the teacher put it,
because an abstention is not evidence of a violation.
"""
import argparse, json, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apd.labeller import label_sheet, render_sheet
from apd.schema import assert_contract

assert_contract()
ap = argparse.ArgumentParser()
ap.add_argument("--log", default="artifacts/sanitized_head_cap_hats.txt")
ap.add_argument("--root", default="data/crops_ci_inpaint")
ap.add_argument("--per-sheet", type=int, default=8)
ap.add_argument("--model", default="opus")
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--apply", type=int, default=0)
ap.add_argument("--out", default="artifacts/headcap_reaudit.jsonl")
a = ap.parse_args()

names = {Path(l.strip()).name for l in Path(a.log).read_text().splitlines()
         if l.strip() and not l.startswith("#")}
originals = sorted(n for n in names if n.endswith("__o.jpg"))
found = []
for n in originals:
    for split in ("train", "val"):
        p = Path(a.root) / "head_cap" / split / "present" / n
        if p.exists():
            found.append((p, split)); break
print(f"{len(found)} reverted originals to re-audit (of {len(originals)} in the log)\n")

work = Path("artifacts/reaudit_sheets"); work.mkdir(parents=True, exist_ok=True)
groups = [found[i:i + a.per_sheet] for i in range(0, len(found), a.per_sheet)]

def run(ig):
    i, grp = ig
    sheet = work / f"r{i:03d}.jpg"
    render_sheet([p for p, _ in grp], sheet, cell=300, cols=4)
    return grp, label_sheet(sheet, len(grp), "head_cap", model=a.model)

rows, cost, errs = [], 0.0, 0
with ThreadPoolExecutor(a.workers) as ex:
    for grp, res in ex.map(run, enumerate(groups)):
        cost += res.cost_usd or 0.0
        if res.error:
            errs += 1; print(f"  sheet error: {res.error}"); continue
        for k, (p, split) in enumerate(grp):
            v = res.verdicts.get(str(k + 1))
            if v:
                rows.append({"file": str(p), "split": split, "verdict": v})

Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
tally = Counter(r["verdict"] for r in rows)
print(f"re-audit of {len(rows)} crops: {dict(tally)}   ${cost:.2f}  errors={errs}")
print(f"\nthe sanitation had called ALL {len(found)} of these street hats.")
print(f"the rule-based re-audit calls {tally['absent']} of them absent "
      f"({tally['absent']/max(1,len(rows)):.0%}) -- the rest were mislabelled.")

if a.apply:
    n = 0
    for r in rows:
        if r["verdict"] != "absent":
            continue
        src = Path(r["file"]); stem = src.name.replace("__o.jpg", "")
        for root in ("data/crops_ci_inpaint", "data/crops_cocoheld", "data/crops_plus_sdk"):
            for split in ("train", "val"):
                d = Path(root) / "head_cap" / split / "present"
                if not d.is_dir(): continue
                for f in d.glob(stem + "__*.jpg"):
                    dst = d.parent / "absent"; dst.mkdir(exist_ok=True)
                    shutil.move(str(f), str(dst / f.name)); n += 1
    print(f"\napplied: moved {n} files (incl. augmentation copies) back to absent")
else:
    print("\n(--apply 0: nothing moved; rerun with --apply 1 to act on the verdicts)")
