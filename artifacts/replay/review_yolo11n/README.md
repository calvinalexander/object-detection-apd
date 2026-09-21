# Blind alert review: ours vs production device, same recordings (2026-09-17)

`scripts/108_blind_alert_review.py`. 91 clips the ours replay had finished.
Device alerts: the 103 in those clips that script 107 placed in the recording by
picture. Ours: all 123 of its replay alerts in the same clips. Each alert is cut
from the raw frame (no overlay from either arm), shuffled; the arm is only in
key.csv. Each claimed violation was judged against apd/schema.py rules (a hijab
without a food-grade cap is a violation; the blue bouffant cap is a cap; one
visibly bare hand is a gloves violation) as real / wrong / unclear.
Reviewer: Claude, from sheet_*.jpg, single pass -- check labels.json against the sheets.

| | ours | production device |
|---|---|---|
| alerts raised | 123 | 103 |
| alerts with at least one real violation | **36** | 26 |
| headcap precision (clear cases) | **25/39 = 64%** | 24/55 = 44% (Fisher p=0.04) |
| headcap alerts on a blue cap / green hairnet | **6 / 4** | 23 / 8 |
| mask precision | 9/15 = 60% | 6/6 = 100% |
| gloves precision | 2/7 | 1/5 |
| alerts whose box holds no person (fan, wall, empty floor/table) | 8 | 0 |

Unclear dominates mask and gloves (hands 10-40 px, transparent gloves), so those
two cannot be compared on this footage. Ours loses mask precision to its pose
network boxing a wall fan and empty floor; production never did.
