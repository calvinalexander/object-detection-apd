"""Methodology figure for the kick-off deck (slide "Metodologi"), Use Case 1.

Draws the deployed inference path with real images and the real model output:

    frame CCTV -> YOLO11s-pose (Metis/AIPU) -> crop orang (+18%, 320x320)
               -> EfficientNet-B0 (CPU, satu inferensi) -> 3 probabilitas -> verdict

Portrait, sized for half a 16:9 slide. Text is Indonesian to match the deck.

    python scripts/122_methodology_figure.py [--person 2453] [--out artifacts/report/metodologi.png]
"""
import argparse, csv, json, sys
from pathlib import Path
import cv2, numpy as np, torch, torchvision
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"deploy/apd_person"))
ap = argparse.ArgumentParser()
ap.add_argument("--person", default="373")
ap.add_argument("--out", default="artifacts/report/metodologi.png")
ap.add_argument("--w", type=int, default=1600)
a = ap.parse_args()

F = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
NAVY, GREY, GREEN, RED, AMBER, LINE = (27, 42, 87), (90, 98, 115), (22, 140, 78), (198, 40, 40), (245, 190, 60), (205, 210, 220)
font = lambda s, b=False: ImageFont.truetype(FB if b else F, s)

# ---------------------------------------------------------------- real data
rows = {r["n"]: r for r in csv.DictReader(open(ROOT/"data/label_sppg/labels.csv"))}
rec = rows[a.person]
frame = cv2.imread(str(ROOT/"data/sppg"/rec["frame"]))
H, W = frame.shape[:2]
box = [int(v) for v in json.loads(rec["box"])]

from ultralytics import YOLO
pose = YOLO(str(ROOT/"yolo11s-pose.pt"))
r = pose.predict(frame, imgsz=640, conf=0.40, verbose=False)[0]
boxes = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))

meta = json.loads((ROOT/"runs/person_ppe/export/apd_ours_person_ppe.json").read_text())
net = torchvision.models.efficientnet_b0(weights=None)
net.classifier[-1] = torch.nn.Linear(net.classifier[-1].in_features, 3)
net.load_state_dict(torch.load(ROOT/"runs/person_ppe/efficientnet_b0_320_sppg/best.pt", map_location="cpu"))
net.eval()

pw, ph = box[2]-box[0], box[3]-box[1]
c = frame[int(max(0, box[1]-.18*ph)):int(min(H, box[3]+.18*ph)), int(max(0, box[0]-.18*pw)):int(min(W, box[2]+.18*pw))]
s = 320/max(c.shape[:2])
rs = cv2.resize(c, (round(c.shape[1]*s), round(c.shape[0]*s)))
lb = np.full((320, 320, 3), 114, np.uint8)
y0, x0 = (320-rs.shape[0])//2, (320-rs.shape[1])//2
lb[y0:y0+rs.shape[0], x0:x0+rs.shape[1]] = rs
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
x = ((cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32)/255 - MEAN)/STD).transpose(2, 0, 1)
with torch.no_grad(): p = torch.sigmoid(net(torch.from_numpy(x)[None]))[0].numpy()
TH = [meta["thresholds"][k] for k in ("head_cap", "mask", "gloves")]
print({k: round(float(v), 3) for k, v in zip(("head_cap", "mask", "gloves"), p)}, "thresholds", TH)

# ---------------------------------------------------------------- canvas
def bgr2pil(im, w):
    im = cv2.resize(im, (w, round(im.shape[0]*w/im.shape[1])), interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))


def iou(p, q):
    ix = max(0, min(p[2], q[2])-max(p[0], q[0])); iy = max(0, min(p[3], q[3])-max(p[1], q[1])); i = ix*iy
    return i/max(1e-6, (p[2]-p[0])*(p[3]-p[1])+(q[2]-q[0])*(q[3]-q[1])-i)


W0, HH, PAD, GAP = a.w, 1400, 46, 54
L, R = PAD, a.w//2 + 26                      # two columns, fits half a 16:9 slide
COLW = W0//2 - PAD - 26
canvas = Image.new("RGB", (W0, HH), "white")
d = ImageDraw.Draw(canvas)
font = lambda s, b=False: ImageFont.truetype(FB if b else F, s)


def step(x, y, n, title, sub):
    d.ellipse([x, y, x+46, y+46], fill=NAVY)
    d.text((x+23, y+23), str(n), font=font(25, True), fill="white", anchor="mm")
    d.text((x+62, y+1), title, font=font(31, True), fill=NAVY)
    d.text((x+62, y+38), sub, font=font(22), fill=GREY)
    return y + 78


def vart(x, y, h=40):
    d.line([x+23, y, x+23, y+h-16], fill=LINE, width=6)
    d.polygon([(x+10, y+h-18), (x+36, y+h-18), (x+23, y+h)], fill=LINE)
    return y + h + 10


# ---- left column: detect -> crop
y = step(L, PAD, 1, "Deteksi orang", "YOLO11s-pose  \u2022  akselerator Metis (AIPU)")
vis = frame.copy()
for b in boxes:
    bi = b.astype(int)
    hit = iou(bi, box) > 0.5
    cv2.rectangle(vis, (bi[0], bi[1]), (bi[2], bi[3]), (0, 220, 255) if hit else (250, 205, 110), 7 if hit else 5)
img = bgr2pil(vis, COLW)
canvas.paste(img, (L, y)); d.rectangle([L, y, L+img.width, y+img.height], outline=LINE, width=3)
d.text((L+14, y+img.height-34), f"{len(boxes)} orang terdeteksi", font=font(22, True), fill="white",
       stroke_width=4, stroke_fill=(20, 20, 20))
y = vart(L, y + img.height + 12)
y = step(L, y, 2, "Potong area orang", "kotak orang + 18%  \u2192  320 \u00d7 320 piksel")
crop_img = bgr2pil(lb, 250)
canvas.paste(crop_img, (L, y)); d.rectangle([L, y, L+250, y+250], outline=LINE, width=3)
tx = L + 250 + 28
for i, t in enumerate(["Satu crop = satu orang.", "Tanpa ROI kepala/tangan:", "model melihat orang", "seutuhnya."]):
    d.text((tx, y+30+i*46), t, font=font(24, True if i == 0 else False), fill=NAVY if i == 0 else GREY)
LY = y + 250                       # bottom of the left column (crop box is 250 px tall)

# arrow between the columns
ax, ay = W0//2 - 16, PAD + 300
d.line([ax-30, ay, ax+18, ay], fill=LINE, width=6)
d.polygon([(ax+14, ay-13), (ax+14, ay+13), (ax+36, ay)], fill=LINE)

# ---- right column: model -> verdicts
y = step(R, PAD, 3, "Klasifikasi APD", "EfficientNet-B0 (CPU)  \u2022  1 inferensi = 3 jawaban")
d.rounded_rectangle([R, y, R+COLW, y+120], 16, fill=(244, 246, 250), outline=LINE, width=3)
d.text((R+COLW//2, y+44), "EfficientNet-B0", font=font(34, True), fill=NAVY, anchor="mm")
d.text((R+COLW//2, y+88), "~35 ms/orang  \u2022  15 MB  \u2022  ONNX Runtime", font=font(21), fill=GREY, anchor="mm")
y = vart(R, y + 132)
y = step(R, y, 4, "Hasil per orang", "probabilitas APD terpakai vs ambang")
for i, (nm, prob, th) in enumerate(zip(["Tutup kepala", "Masker", "Sarung tangan"], p, TH)):
    ok = float(prob) >= th
    ry = y + i*104
    d.rounded_rectangle([R, ry, R+COLW, ry+88], 14, fill="white", outline=LINE, width=3)
    d.text((R+20, ry+52), nm, font=font(24, True), fill=NAVY, anchor="lm")
    gx, gw = R+260, 165
    d.rounded_rectangle([gx, ry+40, gx+gw, ry+62], 11, fill=(232, 235, 240))
    d.rounded_rectangle([gx, ry+40, gx+max(6, int(gw*float(prob))), ry+62], 11, fill=GREEN if ok else RED)
    d.line([gx+int(gw*th), ry+32, gx+int(gw*th), ry+70], fill=NAVY, width=4)
    d.text((gx+int(gw*th), ry+20), f"ambang {th:.2f}", font=font(17), fill=NAVY, anchor="mm")
    d.text((gx+gw+14, ry+52), f"{float(prob):.2f}", font=font(24, True), fill=NAVY, anchor="lm")
    d.text((R+COLW-14, ry+52), "SESUAI" if ok else "PELANGGARAN", font=font(22, True),
           fill=GREEN if ok else RED, anchor="rm")
y = y + 3*104 + 6
d.text((R, y), "Pelanggaran dikonfirmasi lewat voting 5 frame", font=font(21), fill=GREY)
d.text((R, y+30), "sebelum notifikasi dikirim.", font=font(21), fill=GREY)

canvas = canvas.crop((0, 0, W0, min(HH, max(LY, y + 30 + 30) + PAD)))   # trim to the taller column
out = ROOT/a.out; out.parent.mkdir(parents=True, exist_ok=True)
canvas.save(out, quality=95)
print("wrote", out, canvas.size)
