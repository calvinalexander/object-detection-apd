"""Random-sampled bulk labelling of the held-out splits.

Held-out splits must be sampled at random. Ranking them by predicted positivity
would raise the yield and destroy the prevalence, and every precision figure
computed against them afterwards would be fiction -- so `--mode ranked` is
refused for anything but train, in code rather than in a comment.

Resumable: sheets already recorded in the output store are skipped, so an
interrupted run continues rather than re-paying for work already done.
"""
import argparse, collections, json, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apd.labeller import label_sheet, render_sheet
from apd.schema import Attribute, Value, assert_contract
from apd.pose import frontality

assert_contract()
CROPS = Path("artifacts/crops")
SHEETS = Path("artifacts/sheets")
STORE = Path("artifacts/bulk_labels.jsonl")

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True)
ap.add_argument("--attribute", default=Attribute.MASK.value)
ap.add_argument("--sheets", type=int, default=2)
ap.add_argument("--per-sheet", type=int, default=12)
ap.add_argument("--per-scene", type=int, default=2)
ap.add_argument("--model", default="sonnet")
ap.add_argument("--mode", choices=["random", "ranked"], default="random")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--frontality", type=float, default=0.5,
                help="geometric face-on score; keypoint confidence does not measure this")
ap.add_argument("--person-conf", type=float, default=0.35)
ap.add_argument("--audit-per-sheet", type=int, default=2,
                help="crops with known seed labels mixed into each sheet, so the "
                     "run scores itself instead of being trusted")
args = ap.parse_args()

if args.split != "train" and args.mode != "random":
    raise SystemExit("held-out splits must be sampled at random: ranking distorts prevalence")

splits = json.loads(Path("artifacts/splits.json").read_text())["splits"]
frame_split = {f: k for k, v in splits.items() for f in v}
index = [json.loads(l) for l in Path("artifacts/crops_index.jsonl").read_text().splitlines()]

part = "head" if args.attribute in (Attribute.HEAD_CAP.value, Attribute.MASK.value) else "hand"
elig = [r for r in index if r["part"] == part and frame_split.get(r["frame"]) == args.split]
if part == "head":
    elig = [r for r in elig if not r.get("head_collision")]
    if args.attribute == Attribute.MASK.value:
        persons = {r["person_id"]: r for r in (json.loads(l) for l in
                   Path("artifacts/persons.jsonl").read_text().splitlines())}
        elig = [r for r in elig
                if frontality(persons[r["person_id"]]["kp"]) >= args.frontality
                and persons[r["person_id"]]["person_conf"] >= args.person_conf]

# Crops whose seed label is known and decided -- the audit pool.
seed_label = {}
for r in (json.loads(l) for l in Path("artifacts/seed_labels.jsonl").read_text().splitlines()):
    v = r["labels"].get(args.attribute)
    if v and v != Value.UNCLEAR.value:
        seed_label[r["person_id"]] = v
audit_pool = [r for r in elig if r["person_id"] in seed_label]

done = {}
if STORE.exists():
    for line in STORE.read_text().splitlines():
        r = json.loads(line); done[(r["crop"], r["attribute"])] = r
todo = [r for r in elig if (r["crop"], args.attribute) not in done]

rng = random.Random(args.seed)
rng.shuffle(todo)
seen, picked = collections.Counter(), []
for r in todo:
    if seen[r["scene_id"]] >= args.per_scene:
        continue
    seen[r["scene_id"]] += 1
    picked.append(r)
    if len(picked) >= args.sheets * args.per_sheet:
        break

rng.shuffle(audit_pool)
print(f"{args.split}/{args.attribute}: {len(elig)} eligible, {len(todo)} unlabelled, "
      f"{len(picked)} selected across {len(seen)} scenes; "
      f"audit pool {len(audit_pool)}")
SHEETS.mkdir(parents=True, exist_ok=True)

total_cost, added, errors = 0.0, collections.Counter(), []
audit_seen = collections.Counter()
audit_i = 0
for s in range(0, len(picked), args.per_sheet):
    batch = picked[s:s + args.per_sheet]
    # Mix in known-answer crops at random positions. The model cannot tell them
    # apart from the rest, so agreement on them measures the run itself.
    audit_slots = {}
    for _ in range(min(args.audit_per_sheet, len(audit_pool) - audit_i)):
        a = audit_pool[audit_i]; audit_i += 1
        pos = rng.randrange(len(batch) + 1)
        batch.insert(pos, a)
        audit_slots = {}  # recomputed below once positions are final
    for n, r in enumerate(batch, start=1):
        if r["person_id"] in seed_label and (r["crop"], args.attribute) not in done:
            audit_slots[n] = seed_label[r["person_id"]]
    name = f"{args.split}_{args.attribute}_{args.seed}_{s // args.per_sheet:03d}"
    sheet = SHEETS / f"{name}.png"
    render_sheet([CROPS / r["crop"] for r in batch], sheet)
    t0 = time.time()
    res = label_sheet(sheet, len(batch), args.attribute, model=args.model)
    total_cost += res.cost_usd or 0.0
    if res.error:
        errors.append(f"{name}: {res.error}")
    for n, r in enumerate(batch, start=1):
        v = res.verdicts.get(str(n))
        if v is None:
            continue
        if n in audit_slots:
            audit_seen["total"] += 1
            audit_seen["agree" if v == audit_slots[n] else "disagree"] += 1
            audit_seen[f"{audit_slots[n]}->{v}"] += 1
            continue                      # audit crops never enter the store
        added[v] += 1
        done[(r["crop"], args.attribute)] = {
            "crop": r["crop"], "person_id": r["person_id"], "frame": r["frame"],
            "scene_id": r["scene_id"], "split": args.split, "attribute": args.attribute,
            "value": v, "source": "bulk_claude", "sampling": args.mode,
            "model": args.model, "sheet": name}
    print(f"  {name}: {len(res.verdicts)}/{len(batch)} verdicts, "
          f"{time.time()-t0:.0f}s, ${res.cost_usd or 0:.4f}"
          f"{'  ERR ' + res.error if res.error else ''}", flush=True)

STORE.write_text("".join(json.dumps(r) + "\n" for r in done.values()))
print(f"\nadded: {dict(added)}   cost: ${total_cost:.4f}   store: {len(done)} labels")
if audit_seen["total"]:
    # Abstention is not error. Conflating them made a run that never once
    # contradicted the seed look like 55% agreement, so the two are reported apart:
    # accuracy is measured only where both sides committed to an answer.
    t = audit_seen["total"]
    abst = sum(v for k, v in audit_seen.items() if k.endswith("->unclear"))
    both = t - abst
    a = audit_seen["agree"]
    print(f"\nSELF-AUDIT vs seed labels ({t} known-answer crops)")
    print(f"  agreement where BOTH decided : {a}/{both} = "
          f"{100*a/both if both else 0:.0f}%")
    print(f"  run abstained                : {abst}/{t} = {100*abst/t:.0f}%")
    for k, v in sorted(audit_seen.items()):
        if "->" in k: print(f"  seed {k}: {v}")
    print("  This is the run's own accuracy. Labels below inherit it as their ceiling.")
if errors:
    print("errors:"); [print("  " + e) for e in errors]
if added:
    per = total_cost / max(sum(added.values()), 1)
    print(f"\n${per:.5f}/label -> full {args.split} pass ({len(elig)} crops) "
          f"would cost about ${per*len(elig):.2f}")
