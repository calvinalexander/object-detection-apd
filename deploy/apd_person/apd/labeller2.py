"""Context-panel labelling: show the labeller the scene, not just a crop.

A 128px hand crop showing a bucket is unanswerable -- whose hand, doing what,
is it even a hand. The earlier protocol abstained on 84% of gloves crops for
that reason. Each slot here is a pair: the whole frame with the person and the
region marked, beside the region enlarged. The scene resolves which person is
meant and what they are doing; the enlargement resolves what they are wearing.

Two independent votes per slot, with slot order shuffled between them. A slot
the two votes disagree on becomes `unclear` -- disagreement under reordering is
evidence the evidence was not there, and is cheaper than a wrong label.
"""
from __future__ import annotations
from dataclasses import dataclass
import json, os, re, subprocess
from pathlib import Path
import cv2
import numpy as np

from apd.labeller import resolve_token
from apd.schema import ATTRIBUTE_RULE, Value

CYAN, YELLOW, DARK = (230, 200, 0), (0, 215, 255), (19, 26, 29)

SYSTEM_PROMPT = """\
You label personal protective equipment in CCTV frames from commercial kitchens.

The image is a contact sheet. Each numbered row shows the SAME person twice:
  LEFT  -- the whole camera frame. The person to judge is outlined in cyan and
           the body region to judge is outlined in yellow. Use this to see who
           is meant, which way they face, and what they are doing.
  RIGHT -- that yellow region enlarged. Use this to see the equipment itself.

Judge only the cyan-outlined person in each row, using both views together.

{rule}

Some frames contain flat grey rectangles. Those are regions we deliberately
erased because an automated system had printed its own guess there. That guess
is unreliable and is exactly what you are replacing. Never infer a verdict from
an erased region, from any leftover coloured text, or from any box drawn in the
footage that is not the cyan or yellow outline we added.

Answer `unclear` when the evidence is not there -- the region is blurred beyond
reading, the body part is turned away or occluded, or the outline does not
contain the part it should. `unclear` is a correct answer, not a failure;
guessing injects noise that is worse than a missing label.

Return ONLY a JSON object mapping row number (as a string) to one of
"present", "absent", "unclear". No prose, no code fence.
Example for 3 rows: {{"1":"present","2":"unclear","3":"absent"}}
"""

_FENCE = re.compile(r"^```[a-z]*\n|\n```$", re.M)


@dataclass
class PanelResult:
    verdicts: dict[str, str]
    cost_usd: float | None
    error: str | None = None


def _fit(img, w, h):
    s = min(w / img.shape[1], h / img.shape[0])
    out = np.full((h, w, 3), 245, np.uint8)
    r = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))))
    y, x = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


def render_panels(items: list[dict], out: Path, ctx_w: int = 620, zoom_w: int = 380,
                  row_h: int = 300) -> None:
    """items: {frame (BGR ndarray), person_box, roi_box}"""
    band = 26
    sheet = np.full((len(items) * (row_h + band), ctx_w + zoom_w + 12, 3), 236, np.uint8)
    for n, it in enumerate(items):
        img = it["frame"].copy()
        px1, py1, px2, py2 = [int(v) for v in it["person_box"]]
        rx1, ry1, rx2, ry2 = [int(v) for v in it["roi_box"]]
        lw = max(2, int(0.004 * max(img.shape[:2])))
        cv2.rectangle(img, (px1, py1), (px2, py2), CYAN, lw)
        cv2.rectangle(img, (rx1, ry1), (rx2, ry2), YELLOW, lw)
        ctx = _fit(img, ctx_w, row_h)

        pad = int(0.45 * max(rx2 - rx1, ry2 - ry1))
        crop = it["frame"][max(0, ry1 - pad):ry2 + pad, max(0, rx1 - pad):rx2 + pad]
        if crop.size == 0:
            crop = it["frame"]
        zoom = _fit(cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC),
                    zoom_w, row_h)

        y0 = n * (row_h + band)
        sheet[y0:y0 + band] = DARK
        cv2.putText(sheet, f"{n + 1}", (8, y0 + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
        cv2.putText(sheet, "whole frame (cyan = person, yellow = region)",
                    (34, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 210, 215), 1)
        cv2.putText(sheet, "region enlarged", (ctx_w + 20, y0 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 210, 215), 1)
        sheet[y0 + band:y0 + band + row_h, :ctx_w] = ctx
        sheet[y0 + band:y0 + band + row_h, ctx_w + 12:] = zoom
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 93])


def label_panels(sheet: Path, n_slots: int, attribute: str, model: str = "opus",
                 timeout: int = 900) -> PanelResult:
    prompt = (f"Read the image at {sheet.resolve()} and label all {n_slots} numbered "
              f"rows for `{attribute}`. Return the JSON object and nothing else.")
    cmd = ["claude", "-p", "--output-format", "json", "--model", model,
           "--allowedTools", "Read",
           "--exclude-dynamic-system-prompt-sections",
           "--system-prompt-snapshot", "on",
           "--system-prompt", SYSTEM_PROMPT.format(rule=ATTRIBUTE_RULE[attribute])]
    env = dict(os.environ)
    if (tok := resolve_token()):
        env["CLAUDE_CODE_OAUTH_TOKEN"] = tok
    try:
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                              timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return PanelResult({}, None, f"timeout after {timeout}s")
    if proc.returncode != 0:
        return PanelResult({}, None, f"exit {proc.returncode}: {proc.stderr.strip()[:200]}")
    try:
        env_json = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return PanelResult({}, None, "envelope was not JSON")
    if env_json.get("is_error") or env_json.get("subtype") != "success":
        return PanelResult({}, env_json.get("total_cost_usd"),
                           f"{env_json.get('subtype')} {env_json.get('api_error_status') or ''}".strip())
    body = _FENCE.sub("", (env_json.get("result") or "").strip())
    try:
        raw = json.loads(body)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", body, re.S)
        if not m:
            return PanelResult({}, env_json.get("total_cost_usd"), "no JSON in result")
        try:
            raw = json.loads(m.group(0))
        except json.JSONDecodeError:
            return PanelResult({}, env_json.get("total_cost_usd"), "unparseable JSON")
    allowed = {v.value for v in Value}
    verdicts = {}
    for slot in range(1, n_slots + 1):
        v = raw.get(str(slot))
        if isinstance(v, str) and v.strip().lower() in allowed:
            verdicts[str(slot)] = v.strip().lower()
    return PanelResult(verdicts, env_json.get("total_cost_usd"))
