"""Three-way comparison on the audit pool: seed vs a bulk model vs adjudication.

The bulk pass agreed with the seed only 27% of the time while a careful pass
agreed 86%, so the question is whether that gap is the task or the model. Same
crops, same prompt, same rule -- only the model changes.
"""
import argparse, collections, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apd.labeller import label_sheet, render_sheet

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="opus")
ap.add_argument("--per-sheet", type=int, default=12)
args = ap.parse_args()

pool = json.loads(Path("artifacts/audit_pool.json").read_text())
crops = {r["person_id"]: r["crop"] for r in (json.loads(l) for l in
         Path("artifacts/crops_index.jsonl").read_text().splitlines()) if r["part"] == "head"}
mine = json.loads(Path("artifacts/audit_adjudication.json").read_text())

got, cost = {}, 0.0
for s in range(0, len(pool), args.per_sheet):
    batch = pool[s:s + args.per_sheet]
    sheet = Path(f"artifacts/sheets/bakeoff_{args.model}_{s//args.per_sheet}.png")
    render_sheet([Path("artifacts/crops") / crops[b["person_id"]] for b in batch], sheet)
    res = label_sheet(sheet, len(batch), "mask", model=args.model)
    cost += res.cost_usd or 0.0
    for n, b in enumerate(batch, start=1):
        if str(n) in res.verdicts:
            got[b["slot"]] = res.verdicts[str(n)]
    print(f"  sheet {s//args.per_sheet}: {len(res.verdicts)}/{len(batch)}, ${res.cost_usd or 0:.3f}"
          f"{'  ERR ' + res.error if res.error else ''}", flush=True)

def score(pred, truth_of, label):
    dec = [s for s in pred if pred[s] != "unclear" and truth_of(s) != "unclear"]
    ok = sum(1 for s in dec if pred[s] == truth_of(s))
    ab = sum(1 for s in pred if pred[s] == "unclear")
    print(f"  vs {label:14s} decided {len(dec):2d}  agree {ok:2d}  "
          f"= {100*ok/len(dec) if dec else 0:3.0f}%   abstained {ab}/{len(pred)}")

seed_of = {p["slot"]: p["seed"] for p in pool}
print(f"\n{args.model.upper()} on the audit pool (n={len(got)}), cost ${cost:.3f}")
score(got, lambda s: seed_of[s], "seed")
score(got, lambda s: mine[str(s)], "adjudication")
print(f"\nreference -- careful adjudication vs seed: 18/21 = 86%, abstained 15/36")
