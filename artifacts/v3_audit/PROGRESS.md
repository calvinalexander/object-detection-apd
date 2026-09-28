# v3 training-data audit

One person at a time over everything v3 trains on (5,576 people). Verdicts in
`artifacts/v3_audit/verdicts.csv`, sheets in `artifacts/v3_audit/<source>/`.

## Done: det7_export (dashboard-export crops, never audited before)

1,093 people, 1501 item-labels: **1387 ok, 31 wrong, 83 questionable** (8% flagged)

| item | labels | wrong | questionable |
|---|---|---|---|
| head_cap | 995 | 12 | 39 |
| mask | 209 | 11 | 22 |
| gloves | 297 | 8 | 22 |

### The wrong labels

| # | item | label | what is actually in the crop |
|---|---|---|---|
| 4 | head_cap | absent | head out of frame; absent is unsupportable, should be unjudgeable |
| 19 | head_cap | absent | wears a light surgical-style cap but labelled absent; crop also contains the dashboard red box |
| 35 | mask | absent | 54x63 crop, head barely in frame, clock overlay across it; absent unsupportable |
| 52 | mask | absent | white mask visible on the face but labelled absent |
| 100 | head_cap | present | central person is bare-headed; the green cap belongs to the person below |
| 124 | head_cap | present | crop is a bending back with dashboard text burned in; cap belongs to another person |
| 128 | gloves | absent | blue gloves clearly on both hands but labelled absent |
| 132 | mask | absent | mask is on the face but labelled absent |
| 132 | gloves | absent | black gloves on both hands but labelled absent |
| 149 | mask | absent | blue mask on the face but labelled absent |
| 153 | head_cap | absent | navy beret-style kitchen cap = present under the agreed rule |
| 271 | head_cap | absent | white hairnet clearly visible but labelled absent |
| 291 | mask | absent | white mask on the face but labelled absent |
| 304 | mask | absent | white mask on the face but labelled absent |
| 396 | mask | absent | blue mask clearly on the face but labelled absent |
| 423 | mask | absent | blue mask on the face but labelled absent |
| 435 | head_cap | absent | a green cap is worn over the hijab; cap over hijab = present |
| 521 | mask | absent | blue mask on the face but labelled absent |
| 627 | gloves | absent | yellow gloves on both hands but labelled absent |
| 650 | gloves | absent | yellow gloves on both hands but labelled absent |
| 667 | mask | absent | blue mask on the face but labelled absent |
| 678 | head_cap | absent | white hairnet worn over the hijab but labelled absent |
| 706 | head_cap | absent | white hairnet clearly worn but labelled absent |
| 710 | head_cap | absent | white hairnet worn over the hijab but labelled absent |
| 789 | head_cap | absent | a light hairnet appears to be worn but labelled absent |
| 842 | gloves | absent | white gloves on both hands but labelled absent |
| 929 | gloves | absent | blue gloves on both hands but labelled absent |
| 938 | gloves | absent | black gloves on both hands but labelled absent |
| 1007 | gloves | absent | blue gloves on both hands but labelled absent |
| 1022 | head_cap | absent | white hairnet visible but labelled absent |
| 1033 | mask | absent | a mask is on the face but labelled absent |

### Patterns

1. **Item visibly present, labelled absent** - the largest group, and worst for
   mask and gloves: masks on the face and blue/yellow/black gloves on both hands
   labelled `absent`. This trains the presence head against the truth.
2. **Label asserted on a body part outside the crop** - heads cut off, 54x63
   crops with the clock overlay across the face, still carrying a verdict.
   This is what teaches the judgeability head that anything is judgeable.
3. **Two people in one crop, the label fitting the other one** - about a third
   of the questionable cases.
4. **Rule mismatches** - navy beret and white hairnet labelled absent; a blue
   beanie labelled present at #657 and absent at #373.

## Next

det7_site (1,078) -> det7_cam1 (479) -> sppg (2,619) -> coco (299).
