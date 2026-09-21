"""Test the glyph-level overlay removal vs the old block-fill approach.

Produces side-by-side comparisons: original | new clean (inpainted glyphs).
"""
import glob, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apd.overlay import detect, clean

imgs = sorted(glob.glob("data/sites/*/*.jpg"))
# Pick a spread of sites
sites = {}
for p in imgs:
    s = Path(p).parent.name
    if s not in sites:
        sites[s] = p
    if len(sites) >= 6:
        break

rows = []
for site, path in sites.items():
    img = cv2.imread(path)
    if img is None:
        continue
    h, w = img.shape[:2]
    ov, boxes = detect(img)
    if not ov.panels:
        print(f"  {site}: no panels found, skipping")
        continue
    cleaned = clean(img, (ov, boxes))

    # Mark panels on original for reference
    marked = img.copy()
    for px1, py1, px2, py2 in ov.panels:
        cv2.rectangle(marked, (px1, py1), (px2, py2), (0, 255, 255), 2)

    # Scale down for the comparison sheet
    scale = 640 / max(w, 1)
    sz = (640, int(h * scale))
    row = np.hstack([
        cv2.resize(marked, sz),
        cv2.resize(cleaned, sz),
    ])
    rows.append(row)
    print(f"  {site}: {len(ov.panels)} panels, image {w}x{h}")

if rows:
    # Pad rows to same width
    max_w = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        if r.shape[1] < max_w:
            r = np.hstack([r, np.zeros((r.shape[0], max_w - r.shape[1], 3), np.uint8)])
        padded.append(r)
    sheet = np.vstack(padded)
    out = "artifacts/_overlay_v2_test.jpg"
    cv2.imwrite(out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"\nWrote {out}  ({sheet.shape[1]}x{sheet.shape[0]})")
else:
    print("No panels found in any sample")
