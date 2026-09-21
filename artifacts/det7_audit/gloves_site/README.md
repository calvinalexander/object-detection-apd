# Audit: site-sourced gloves / no_gloves boxes in det7 (2026-09-17)

`scripts/111_audit_site_gloves.py`. Every gloves (1) and no_gloves (4) box in
det7 train+val whose frame is a production snapshot (data/sites, labels from
artifacts/consensus.jsonl): 359 boxes, one tile each in sheet_NN.jpg.
Reviewer: Claude, one pass, verdicts in verdicts.csv (ok / wrong_class /
not_hand / unclear). Re-score with `--score`.

| class | boxes | ok | wrong class | not a hand | unclear | wrong |
|---|---|---|---|---|---|---|
| gloves | 138 | 60 | 1 | 12 | 65 | 9% |
| no_gloves | 221 | 111 | **12** | 16 | 82 | 13% |
| all | 359 | 171 | 13 | 28 | 147 | 11% |

no_gloves on a visible glove (12): #36 #74 #80 #109 #148 #218 #241 #242 #287 #326 #338 #345
  - blue nitrile: 74 80 109 241 326 345; black: 36 338; yellow: 218 242;
    white: 148; transparent: 287
gloves on a bare hand (1): #352

Box on something other than a hand (28): feet/shoes 116 153 297, face 57, ear 172,
knee 324, hairnet/head 186 205 206, paper cup 302 303, shirt patch 111, sleeves
11 12 13 32 332, food/objects 73 84 94 321, whole person 101 301, floor/blur
9 106 107 149 305.

Unclear (41%): median box 24 px, blurred hands in food; nobody can say gloved or
not at that size, yet they train the detector as if certain.
Duplicates: many boxes appear 2-4x (same frame, overlapping person crops).
