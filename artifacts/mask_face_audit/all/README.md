# Audit: data/mask_face/g0.8 (the deployed mask trunk's data), 2026-09-17

`python scripts/112_audit_mask_face.py --all [--score]`. Every labelled face,
one tile per face: the 128 px training crop, the annotator's box, the frame.
1073 faces (train 639, val 111, test 323). Sources: SDK apd_v3 val (~80%),
dump apd_v2/v3 (voter3). No site snapshots, no COCO.
Reviewed by five parallel Claude reviewers with one written rule set; the lead
spot-checked #156, #160, #201 (agreed). verdicts.csv holds a note per face.

| label | faces | ok | wrong class | wrong-face crop | unclear |
|---|---|---|---|---|---|
| mask | 482 | 341 | 6 | 12 | 123 |
| no_mask | 591 | 385 | 0 | 44 | 162 |
| all | 1073 | 726 (68%) | 6 (0.6%) | 56 (5%) | 285 (27%) |

By split: wrong-face train 32 / val 10 / test 14; wrong class train 1 / test 5;
unclear train 174 / val 20 / test 91.

Findings
- Labels are clean: 6 wrong, all "mask" on a mask below the nose / under the
  chin / a bare face (#156, #276, #278, #296, #722, #847). 5 of them are in TEST.
- 5% of crops are the wrong person or no face (scripts/101 matches a face box to
  the first pose person whose head region contains the box centre; a foreground
  head's region swallows the face behind it). Worst kind: #201, labelled no_mask,
  crop shows another woman wearing a mask.
- 27% cannot be judged (tiny, bowed, hand over mouth) but are trained as certain.
- The held-out numbers (MCC 0.826) were scored on a test split carrying 14
  wrong-face crops, 5 wrong labels and 91 unjudgeable faces.
