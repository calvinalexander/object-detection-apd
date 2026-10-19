"""Background-subtraction tray detector for the static-camera s14 render (stand-in for tc_2272)."""
import cv2, numpy as np
HERE = __file__.rsplit("/", 1)[0]
BG = cv2.GaussianBlur(cv2.imread(HERE + "/bg.png"), (5, 5), 0)
A_SLOPE, A_ICPT = map(float, open(HERE + "/area_fit.txt").read().split())   # single-tray area vs centre y

def _split(mm, n):
    """Split a blob of n touching trays into n parts along its long axis (equal pixel counts)."""
    ys, xs = np.nonzero(mm)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    mu = pts.mean(0); _, _, vt = np.linalg.svd(pts - mu, full_matrices=False)
    proj = (pts - mu) @ vt[0]
    cuts = np.quantile(proj, np.linspace(0, 1, n + 1)[1:-1])
    lab = np.searchsorted(cuts, proj)
    parts = []
    for k in range(n):
        p = np.zeros_like(mm); p[ys[lab == k], xs[lab == k]] = 1
        parts.append(p)
    return parts

def tray_masks(img, thr=40, min_area=5000, erode=17):
    d = cv2.absdiff(cv2.GaussianBlur(img, (5, 5), 0), BG).max(axis=2)
    m = (d > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))   # fill food holes inside a tray
    # seeds: erode until touching trays separate, then watershed back to the full mask
    core = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode, erode)))
    k, seeds = cv2.connectedComponents(core)
    markers = seeds.astype(np.int32) + 1
    markers[(m == 1) & (core == 0)] = 0                       # unknown band -> decided by watershed
    markers[m == 0] = 1                                       # background
    cv2.watershed(img, markers)
    out = []
    for lab in range(2, k + 1):
        mm = (markers == lab).astype(np.uint8)
        a = int(mm.sum())
        if a < min_area: continue
        x, y, w, h = cv2.boundingRect(mm)
        n = int(round(a / (A_SLOPE * (y + h / 2) + A_ICPT)))
        for p in (_split(mm, n) if n >= 2 else [mm]):
            x, y, w, h = cv2.boundingRect(p)
            out.append(((x, y, x + w, y + h), p, int(p.sum())))
    return out
