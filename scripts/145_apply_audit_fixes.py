"""Turn the one-by-one audit verdicts into label overrides for training.

Input  artifacts/v3_audit/verdicts.csv   (source, n, id, item, label, verdict, note)
Output data/clean/audit_v3_overrides.csv (source, id, item, old, new, reason)

Mapping, from the note the auditor wrote:
  item is plainly there / worn          -> flip to the other presence value
  body part not inside the crop, blurred, censored, cut off, too small
                                        -> unclear  (no presence, judgeability = 0)
  label belongs to another person in the crop
                                        -> unclear
  navy beret labelled absent            -> present   (agreed rule)
  fabric mask labelled present          -> absent    (schema: cloth masks are absent)
  beanie labelled present               -> absent    (schema)

    python scripts/145_apply_audit_fixes.py
"""
import csv, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader(open(ROOT/"artifacts/v3_audit/verdicts.csv")))
FLIP = {"present": "absent", "absent": "present"}

PRESENT_WORDS = ("clearly worn", "clearly visible", "on the face", "on both hands",
                 "visible but labelled absent", "worn over the hijab", "appear to be worn",
                 "may be worn", "a mask is on the face", "a mask looks present",
                 "a mask appears", "a light mask", "a white mask", "blue mask", "yellow gloves",
                 "black gloves", "blue gloves", "white gloves", "dark gloves", "a light hairnet",
                 "white hairnet", "a green cap")
UNCLEAR_WORDS = ("unsupportable", "not assessable", "out of frame", "not visible", "cut off",
                 "not identifiable", "too small", "blurred", "blur", "censored", "pixelated",
                 "belongs to", "another person", "someone else", "fits only one", "edge of the crop",
                 "trolley", "tray", "not resolvable", "ambiguous", "partly hidden", "face hidden",
                 "face down", "not attributable", "not clearly visible", "bag, not a cap")

out, n = [], {"flip": 0, "unclear": 0, "rule": 0}
for r in rows:
    if r["verdict"] == "ok" or r["label"] in ("", "-"):
        continue
    note = r["note"].lower()
    new = reason = None
    if "navy beret" in note:
        new, reason = "present", "rule: navy beret counts as head_cap"
    elif "fabric mask" in note:
        new, reason = "absent", "rule: cloth/fabric masks are absent"
    elif "beanie" in note and r["label"] == "present":
        new, reason = "absent", "rule: beanies are absent"
    elif any(w in note for w in UNCLEAR_WORDS):
        new, reason = "unclear", "item not judgeable from this crop"
    elif any(w in note for w in PRESENT_WORDS):
        new, reason = FLIP.get(r["label"]), "item is visibly " + ("present" if r["label"] == "absent" else "absent")
    if new is None or new == r["label"]:
        new, reason = "unclear", "audit flagged it, reason not machine-readable"
    n["rule" if reason.startswith("rule") else "unclear" if new == "unclear" else "flip"] += 1
    out.append({"source": r["source"], "id": r["id"], "item": r["item"],
                "old": r["label"], "new": new, "reason": reason})

p = ROOT/"data/clean/audit_v3_overrides.csv"
with open(p, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["source", "id", "item", "old", "new", "reason"])
    w.writeheader(); w.writerows(out)
import collections
print(f"{len(out)} overrides -> {p}")
print("  by action:", n)
print("  by item  :", dict(collections.Counter(r["item"] for r in out)))
print("  by new   :", dict(collections.Counter(r["new"] for r in out)))
