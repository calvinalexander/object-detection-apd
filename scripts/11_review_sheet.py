"""Render specific person_ids for re-review after a rule change."""
import json, sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("--attribute", default="mask")
ap.add_argument("--value", default="present")
ap.add_argument("--out", default="artifacts/review.png")
args = ap.parse_args()

CROPS = Path("artifacts/crops")
head = {r["person_id"]: r["crop"] for r in (json.loads(l) for l in
        Path("artifacts/crops_index.jsonl").read_text().splitlines()) if r["part"] == "head"}
mined = [json.loads(l) for l in Path("artifacts/mined_labels.jsonl").read_text().splitlines()]
picked = [m for m in mined if m["attribute"] == args.attribute and m["value"] == args.value]

CELL, COLS = 320, 5
ROWS = (len(picked) + COLS - 1) // COLS
sheet = Image.new("RGB", (COLS * CELL, ROWS * (CELL + 24)), (233, 237, 238))
dr = ImageDraw.Draw(sheet)
for n, m in enumerate(picked):
    im = Image.open(CROPS / head[m["person_id"]]).convert("RGB").resize((CELL, CELL), Image.LANCZOS)
    x, y = (n % COLS) * CELL, (n // COLS) * (CELL + 24)
    sheet.paste(im, (x, y + 24))
    dr.rectangle([x, y, x + CELL - 1, y + 23], fill=(19, 26, 29))
    dr.text((x + 7, y + 6), f"R{n+1:02d}", fill=(255, 255, 255))
sheet.save(args.out)
Path(args.out).with_suffix(".json").write_text(json.dumps(
    [{"slot": n + 1, "person_id": m["person_id"], "frame": m["frame"],
      "split": m["split"], "sheet": m["sheet"]} for n, m in enumerate(picked)], indent=1))
print(f"{len(picked)} crops to re-review -> {args.out}")
