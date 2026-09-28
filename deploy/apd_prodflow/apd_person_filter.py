"""Person-crop PPE classifier as a second opinion on production's own events.

Production's flow is untouched: apd_v3 on the AIPU, ByteTrack, vote window,
cooldown. When that vote fires an event, the tracked person's box is cropped
from the CLEAN stream frame (not the snapshot, which has the table drawn on it),
padded 18%, letterboxed to 320 and judged by one ONNX network with six outputs:
present_{head_cap,mask,gloves}, known_{head_cap,mask,gloves}.

This is the same decision measured offline in artifacts/filter16 (v3) and
artifacts/filter16_v5 (v5): an event is kept when the classifier calls any item
a violation, and it then reports the classifier's items, not production's.

    APD_PERSON_ONNX   model file (default: apd_ours_person_ppe_v5.onnx beside this file or in
                      apd_person/ beside it, else v3); its cuts come from the .json next to it
    APD_PERSON_MODE   filter (default) = drop events the classifier clears
                      shadow           = keep production's events as they are, only log
    APD_PERSON_THREADS  ONNX Runtime intra-op threads (default 2)
"""
import json, os, time
from pathlib import Path
import cv2, numpy as np

HERE = Path(__file__).resolve().parent
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
ITEMS = (("headcap", "head_cap"), ("mask", "mask"), ("gloves", "gloves"))   # monitor group, model item
PAD = 0.18
PX = 320


def _letterbox_square(im, s, pad=114):
    h, w = im.shape[:2]; r = s/max(h, w)
    im = cv2.resize(im, (max(1, round(w*r)), max(1, round(h*r))),
                    interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    out = np.full((s, s, 3), pad, np.uint8); y, x = (s-im.shape[0])//2, (s-im.shape[1])//2
    out[y:y+im.shape[0], x:x+im.shape[1]] = im
    return out


def decide(p, cuts):
    """Six scores -> {group: ok|violation|unknown}. Same rule as scripts/134."""
    v = {}
    for k, (grp, item) in enumerate(ITEMS):
        c = cuts[item]; pp, pk = float(p[k]), float(p[3+k])
        strong = ("present_strong" in c and pp < c["present_strong"]
                  and pk >= c.get("known_floor", c["known_above"]))
        v[grp] = ("violation" if strong else
                  "unknown" if pk < c["known_above"] else
                  "violation" if pp < c["present_below"] else "ok")
    return v


class PersonFilter:
    def __init__(self, log_path):
        import onnxruntime as ort
        # installed into production the models sit in apd_person/ beside this file
        pm = os.environ.get("APD_PERSON_ONNX") or next(
            (str(d/f) for f in ("apd_ours_person_ppe_v5.onnx", "apd_ours_person_ppe_v3.onnx")
             for d in (HERE, HERE/"apd_person") if (d/f).exists()), "")
        if not pm or not Path(pm).exists():
            raise FileNotFoundError(f"person PPE model not found: {pm!r}")
        opts = ort.SessionOptions(); opts.log_severity_level = 3
        opts.intra_op_num_threads = int(os.environ.get("APD_PERSON_THREADS", 2))
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(pm, opts, providers=["CPUExecutionProvider"])
        self.cuts = json.loads(Path(pm).with_suffix(".json").read_text())["cuts"]
        self.name = Path(pm).stem
        self.mode = os.environ.get("APD_PERSON_MODE", "filter")
        assert self.mode in ("filter", "shadow"), f"APD_PERSON_MODE={self.mode!r}"
        self.log_path = log_path
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        # the padded crop the network judged, native resolution, no overlay;
        # letterboxing it to 320 reproduces the model input exactly
        self.crop_dir = Path(log_path).parent/"person_crops"
        self.last_crop = None

    def describe(self):
        c = {k: {kk: v for kk, v in c.items() if kk in ("known_above", "present_below", "present_strong", "known_floor")}
             for k, c in self.cuts.items()}
        return f"model={self.name} mode={self.mode} cuts={c}"

    def classify(self, frame, bbox):
        """(verdicts, six scores) for one person box, or (None, None) if the crop is unusable."""
        self.last_crop = None
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = [int(t) for t in bbox]; pw, ph = x2-x1, y2-y1
        c = frame[int(max(0, y1-PAD*ph)):int(min(H, y2+PAD*ph)), int(max(0, x1-PAD*pw)):int(min(W, x2+PAD*pw))]
        if c.size == 0 or min(c.shape[:2]) < 8:
            return None, None
        self.last_crop = c.copy()
        x = (cv2.cvtColor(_letterbox_square(c, PX), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0 - MEAN)/STD
        p = self.sess.run(None, {"person_crops": x.transpose(2, 0, 1)[None]})[0][0]
        return decide(p, self.cuts), p

    def log(self, rec):
        rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "model": self.name, "mode": self.mode, **rec}
        if self.last_crop is not None:
            try:
                d = self.crop_dir/str(rec.get("camera_id", "cam"))
                d.mkdir(parents=True, exist_ok=True)
                # PNG, not JPEG: on ~60 px wide crops JPEG q92 moved the scores by up to 0.23
                f = d/f"{rec['event_id']}.png"
                if cv2.imwrite(str(f), self.last_crop, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                    rec["crop_path"] = str(f)
            except Exception:
                pass                               # a lost crop must never cost the event
            self.last_crop = None
        with open(self.log_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
