"""Label the mined site crops with Opus, in parallel, resumable.

Sonnet scored 27% agreement with the seed labels against Opus's 95%, so the
model is not a knob to economise on here.
"""
import argparse, json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apd.labeller import label_sheet, render_sheet
from apd.schema import Attribute, assert_contract

assert_contract()
ap = argparse.ArgumentParser()
ap.add_argument("--attribute", default=Attribute.HEAD_CAP.value)
ap.add_argument("--per-sheet", type=int, default=12)
ap.add_argument("--model", default="opus")
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--root", default="artifacts/site_crops")
args = ap.parse_args()

root = Path(args.root) / args.attribute
sel = [json.loads(l) for l in (root / "selection.jsonl").read_text().splitlines()]
store = root / "labels.jsonl"
done = {json.loads(l)["crop"] for l in store.read_text().splitlines()} if store.exists() else set()
todo = [r for r in sel if r["crop"] not in done]
print(f"{len(sel)} selected, {len(done)} already labelled, {len(todo)} to do")

sheets = [todo[i:i + args.per_sheet] for i in range(0, len(todo), args.per_sheet)]
(root / "sheets").mkdir(exist_ok=True)

def run(idx_group):
    idx, group = idx_group
    sheet = root / "sheets" / f"s{idx:04d}.jpg"
    render_sheet([root / "pool" / r["crop"] for r in group], sheet)
    res = label_sheet(sheet, len(group), args.attribute, model=args.model)
    if res.error:
        return idx, [], res.error, res.cost_usd
    rows = [r | {"value": res.verdicts[str(n + 1)], "labeller": args.model}
            for n, r in enumerate(group) if str(n + 1) in res.verdicts]
    return idx, rows, None, res.cost_usd

t0, cost, errs = time.time(), 0.0, 0
with store.open("a") as fh, ThreadPoolExecutor(args.workers) as ex:
    for n, (idx, rows, err, c) in enumerate(ex.map(run, enumerate(sheets)), 1):
        cost += c or 0.0
        if err:
            errs += 1
            print(f"  sheet {idx}: {err}", flush=True)
        for r in rows:
            fh.write(json.dumps(r) + "\n")
        fh.flush()
        if n % 5 == 0 or n == len(sheets):
            print(f"  {n}/{len(sheets)} sheets  ${cost:.2f}  {time.time()-t0:.0f}s  errors={errs}", flush=True)

rows = [json.loads(l) for l in store.read_text().splitlines()]
tally = {}
for r in rows:
    tally[r["value"]] = tally.get(r["value"], 0) + 1
print(f"\n{args.attribute}: {len(rows)} labelled  {tally}  ${cost:.2f}")
