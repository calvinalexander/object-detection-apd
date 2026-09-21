"""Remove the production dashboard's burned-in annotations from site frames.

Text glyphs across all font colors (red, green, yellow, grey) and their dark semi-transparent
background boxes are detected and inpainted back to the background scene color.
"""
from dataclasses import dataclass, field
from pathlib import Path
import cv2, numpy as np

MODEL = Path(__file__).resolve().parents[1] / ".work" / "textdet.onnx"
_MODEL = None


def _model():
    """PP-OCRv3 detection, as configured by the OpenCV Zoo demo."""
    global _MODEL
    if _MODEL is None:
        if not MODEL.exists():
            raise FileNotFoundError(f"text detector missing: {MODEL}")
        m = cv2.dnn.TextDetectionModel_DB(str(MODEL))
        m.setBinaryThreshold(0.3).setPolygonThreshold(0.5)
        m.setUnclipRatio(2.0).setMaxCandidates(200)
        m.setInputParams(1 / 255.0, (736, 736),
                         (122.67891434, 116.66876762, 104.00698793), True)
        _MODEL = m
    return _MODEL


def _saturated(img):
    """Pure red / green / yellow palette."""
    b, g, r = [c.astype(np.int16) for c in cv2.split(img)]
    red = (r > 140) & (g < 90) & (b < 90) & (r - np.maximum(g, b) > 60)
    grn = (g > 140) & (r < 110) & (b < 110) & (g - np.maximum(r, b) > 60)
    yel = (r > 150) & (g > 150) & (b < 110) & (np.minimum(r, g) - b > 60)
    return ((red | grn | yel).astype(np.uint8)) * 255


@dataclass
class Overlay:
    box_mask: np.ndarray
    panels: list[tuple[int, int, int, int]] = field(default_factory=list)

    def covers(self, box, frac: float = 0.05) -> bool:
        x1, y1, x2, y2 = [int(v) for v in box]
        area = max(1, (x2 - x1) * (y2 - y1))
        for px1, py1, px2, py2 in self.panels:
            iw = max(0, min(x2, px2) - max(x1, px1))
            ih = max(0, min(y2, py2) - max(y1, py1))
            if iw * ih / area > frac:
                return True
        return False


def _glyph_components(sat):
    n, lab, st, _ = cv2.connectedComponentsWithStats(sat, 8)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if 3 <= bh <= 26 and 1 <= bw <= 34 and area >= 4:
            out.append((x, y, x + bw, y + bh))
    return out


def detect(img: np.ndarray, pad: int = 3, upscale: int = 2) -> 'tuple[Overlay, list]':
    h, w = img.shape[:2]
    sat = _saturated(img)

    src = (cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
           if upscale > 1 else img)
    anchors = []
    try:
        quads, _ = _model().detect(src)
    except cv2.error:
        quads = []
    for q in quads or []:
        q = np.array(q).reshape(-1, 2) / upscale
        x1, y1 = q[:, 0].min(), q[:, 1].min()
        x2, y2 = q[:, 0].max(), q[:, 1].max()
        x1, y1 = max(0, int(x1) - pad), max(0, int(y1) - pad)
        x2, y2 = min(w, int(x2) + pad), min(h, int(y2) + pad)
        if x2 <= x1 or y2 <= y1:
            continue
        if (sat[y1:y2, x1:x2] > 0).mean() > 0.005:
            anchors.append((x1, y1, x2, y2))

    glyphs = _glyph_components(sat)
    boxes = list(anchors)
    if anchors:
        near = 26
        for _ in range(6):
            grew = False
            for g in glyphs:
                if any(g[0] < b[2] + near and g[2] > b[0] - near
                       and g[1] < b[3] + near and g[3] > b[1] - near for b in boxes):
                    if g not in boxes:
                        boxes.append(g)
                        grew = True
            if not grew:
                break
        boxes = [(max(0, x1 - pad), max(0, y1 - pad),
                  min(w, x2 + pad), min(h, y2 + pad)) for x1, y1, x2, y2 in boxes]

    panels = []
    for b in sorted(boxes, key=lambda t: (t[1], t[0])):
        for i, p in enumerate(panels):
            if (b[0] < p[2] + 24 and b[2] > p[0] - 24
                    and b[1] < p[3] + 14 and b[3] > p[1] - 14):
                panels[i] = (min(p[0], b[0]), min(p[1], b[1]),
                              max(p[2], b[2]), max(p[3], b[3]))
                break
        else:
            panels.append(b)

    red = sat.copy()
    for x1, y1, x2, y2 in panels:
        red[y1:y2, x1:x2] = 0
    horiz = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((1, 21), np.uint8))
    vert = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((21, 1), np.uint8))
    thick = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    lines = cv2.bitwise_and(cv2.bitwise_or(horiz, vert), cv2.bitwise_not(thick))
    box_mask = cv2.dilate(lines, np.ones((3, 3), np.uint8), iterations=2)
    return Overlay(box_mask=box_mask, panels=panels), boxes


def clean(img: np.ndarray, ov=None) -> np.ndarray:
    """Inpaint caption overlay panels and detection boxes using surrounding background context."""
    if ov is None:
        ov, _ = detect(img)
    elif isinstance(ov, tuple):
        ov = ov[0]
    out = img.copy()

    # Combine panel regions and bounding box lines into a single mask to inpaint
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    bm = ov.box_mask
    if bm.any() and (bm > 0).mean() < 0.02:
        mask = cv2.bitwise_or(mask, bm)

    for px1, py1, px2, py2 in ov.panels:
        # Fill panel region in mask
        mask[py1:py2, px1:px2] = 255

    if mask.any():
        # Inpaint entire panel and red line areas seamlessly using Telea algorithm from border pixels
        out = cv2.inpaint(out, mask, 5, cv2.INPAINT_TELEA)

    return out
