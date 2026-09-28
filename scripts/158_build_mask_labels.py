"""Turn the mined-mask review into training labels.

artifacts/mined_mask_audit/verdicts.csv holds a judgement for all 508 rows where
v5 disputed production's `mask: ok`:
    covering -> mask PRESENT   (the profile views the old 642 positives lacked)
    chin     -> mask ABSENT    (chin-worn: the confusion that costs us most)
    hidden   -> mask UNCLEAR   (judgeability negative; presence not supervised)

Writes data/mined_positives/mask_labels.csv, which scripts/159 reads.

    python scripts/158_build_mask_labels.py [--agree include|site23|exclude]
`--agree` decides what to do with the 843 rows where v5 and production already
agree: they were not reviewed one by one, and the review showed production's
`ok` is 94% reliable at site23 but only 37% at dev16.
"""
import argparse, csv, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--agree", default="site23", choices=["include", "site23", "exclude"])
a = ap.parse_args()
D = ROOT/"data/mined_positives"
MAP = {"covering": "present", "chin": "absent", "hidden": "unclear"}

ver = {r["file"]: r["verdict"] for r in csv.DictReader(open(ROOT/"artifacts/mined_mask_audit/verdicts.csv"))}
rows, n = [], collections.Counter()
for r in csv.DictReader(open(D/"candidates.csv")):
    if r["item"] != "mask": continue
    if r["file"] in ver:
        lab = MAP[ver[r["file"]]]; src = "reviewed"
    else:                                            # v5 agreed with production, not reviewed
        if a.agree == "exclude": continue
        if a.agree == "site23" and r["site"] != "site23": continue
        lab, src = "present", "agreed-" + r["site"]
    rows.append({"file": r["file"], "site": r["site"], "mask": lab, "source": src,
                 "v5_p": r["v5_p"], "event_id": r["event_id"]})
    n[(src.split("-")[0], lab)] += 1
with open(D/"mask_labels.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} mask labels -> {D}/mask_labels.csv")
for k in sorted(n): print(f"   {k[0]:8} {k[1]:8} {n[k]}")
