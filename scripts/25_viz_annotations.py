"""Draw the human COCO annotations on the frames they belong to.

This is the raw training annotation -- the 592 boxes an annotator drew over 53
frames -- not the derived crops. Colour per class, count per frame in the header.
"""
import argparse, collections, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageDraw
from apd.schema import LEGACY_COCO_CATEGORIES

COL = {"person": (80,170,250), "head_cap": (250,170,60),
       "gloves": (90,220,130), "mask": (230,110,220)}
ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=0, help="which page of frames")
ap.add_argument("--per-page", type=int, default=6)
ap.add_argument("--cols", type=int, default=2)
ap.add_argument("--width", type=int, default=760)
ap.add_argument("--out", default=None)
args = ap.parse_args()

ROOT = Path("data/sppg")
coco = json.loads((ROOT/".exports/coco-1778210446.4251382.json").read_text())
frame_of = {i["id"]: i["file_name"] for i in coco["images"]}
by_frame = collections.defaultdict(list)
for a in coco["annotations"]:
    by_frame[frame_of[a["image_id"]]].append((LEGACY_COCO_CATEGORIES[a["category_id"]], a["bbox"]))
frames = sorted(by_frame)
print(f"{len(frames)} annotated frames, {sum(len(v) for v in by_frame.values())} boxes total")

start = args.batch * args.per_page
page = frames[start:start + args.per_page]
if not page: raise SystemExit(f"batch {args.batch} is past the end")

W = args.width; COLS = args.cols
panels = []
for fn in page:
    im = Image.open(ROOT/fn).convert("RGB")
    sc = W/im.width
    im = im.resize((W, int(im.height*sc)), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    counts = collections.Counter()
    for name, (x,y,w,h) in by_frame[fn]:
        counts[name] += 1
        b = [x*sc, y*sc, (x+w)*sc, (y+h)*sc]
        d.rectangle(b, outline=COL[name], width=2)
        tag = name
        tw = 6*len(tag)+6
        d.rectangle([b[0], max(0,b[1]-12), b[0]+tw, max(12,b[1])], fill=COL[name])
        d.text((b[0]+3, max(0,b[1]-11)), tag, fill=(20,20,20))
    band = Image.new("RGB", (W, 20), (20,26,30))
    ImageDraw.Draw(band).text((5,5),
        f"{fn[:44]}  {im.width}x{im.height}  " +
        "  ".join(f"{k}:{v}" for k,v in sorted(counts.items())), fill=(232,238,240))
    p = Image.new("RGB", (W, im.height+20), (235,238,239))
    p.paste(band,(0,0)); p.paste(im,(0,20))
    panels.append(p)

rows = (len(panels)+COLS-1)//COLS
hmax = max(p.height for p in panels)
sheet = Image.new("RGB", (W*COLS, hmax*rows), (235,238,239))
for i,p in enumerate(panels):
    sheet.paste(p, ((i%COLS)*W, (i//COLS)*hmax))
out = args.out or f"artifacts/annotations_batch{args.batch}.png"
sheet.save(out)
print(f"batch {args.batch}: frames {start}-{start+len(page)-1} -> {out}")
