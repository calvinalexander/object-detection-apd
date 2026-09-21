# PPE retrain on audited data, 2026-09-17

## Result

Best model for all three items: one network per person,
**efficientnet_b0 at 320 px, trained on the cleaned det7 set plus the 2,629
reviewed sppg people**. It is not deployed yet.

TEST = COCO human labels. These 53 frames were never used for training or
selection. The pipeline runs end to end on the device's geometry
(yolo11s-pose at 0.40, then the model). A violation counts as the positive
result, and a missed or undecided person counts as "no alert".

| item | model | precision | recall | MCC | p vs deployed |
|---|---|---|---|---|---|
| headcap (202 people) | **person effnet_b0 +sppg** | 93.9% | 49.5% | **0.543** | 0.001 |
| | deployed resnet34 | 75.5% | 39.8% | 0.335 | – |
| | production apd_mbg | 65.8% | 26.9% | 0.191 | |
| mask (44 people) | **person effnet_b0 +sppg** | 100.0% | 50.0% | **0.346** | 0.37 |
| | deployed effnet_b0 face | 94.4% | 44.7% | 0.196 | – |
| | production apd_mbg | 93.3% | 36.8% | 0.146 | |
| gloves (122 people) | **person effnet_b0 +sppg** | 89.7% | 38.0% | **0.228** | 0.002 |
| | deployed yolo26n detector | 100.0% | 12.0% | 0.180 | – |
| | production apd_mbg | 81.8% | 9.8% | 0.047 | |

Full tables with every variant: `runs/retrain_clean/coco_test_final_clean.txt`.

- **Headcap:** the gain is significant.
- **Gloves:** the person model catches three times as many violations as the
  deployed detector at 90% precision.
- **Mask:** the gain is not significant. Only 25 of the 44 mask people are
  located, so this test cannot separate the models.

## What changed, step by step (headcap MCC on COCO)

| step | headcap | mask | gloves |
|---|---|---|---|
| deployed | 0.335 | 0.196 | 0.180 |
| same recipes on the audited data (data/clean) | 0.363–0.408 | 0.095–0.257 | 0.098–0.146 (detectors) |
| person model, data/clean only | 0.420 | 0.230 | 0.135 |
| + 2,629 reviewed sppg people (headcap/mask classifiers) | 0.465–0.533 | 0.079–0.331 | – |
| **+ sppg, person model** | **0.543** | **0.346** | **0.228** |

Most of the gain comes from the new labelled people, not from the audit or the
architecture. Every architecture jumps once they are added.

**Caveat.** The sppg people and the COCO frames come from the same frame pool
(data/sppg). Every COCO frame and every frame from the same 24 camera scenes
was excluded, but the sites are similar. Part of the gain is domain match,
which is the point of labelling our own kind of footage. The device camera
(data/recording) is a different site again.

## Data work behind it

- **Audits.** Every audit uses one rule set (apd/schema.py) and five to seven
  parallel reviewers.
  - mask_face: 1,073 faces. 6 wrong labels, 56 wrong-person crops, 285
    unjudgeable.
  - head_cap: 1,867 heads. 30 wrong labels (including the 16 black-skullcap
    heads relabelled after your decision), 6 non-heads, 68 unjudgeable, and 5
    heads that sat in both class folders.
  - det7 gloves: site 11% wrong, voter3 2% wrong.
  - det7 mask (site): 9% wrong.
- **Rule decisions (yours).** Navy beret, chef toque, and black or striped
  chef skullcap all count as headcap.
- **Cleaned copies.** All cleaned data is in `data/clean/` (images are hard
  links; originals untouched). `data/clean/CHANGES.csv` has one row per change.
- **New labels.** `data/label_sppg/labels.csv` holds 2,629 people, pre-labelled
  by the deployed models and then reviewed item by item.
  - Headcap: present 1,305 / absent 935 / unclear 389.
  - Mask: present 227 / absent 525 / unclear 1,877.
  - Gloves: present 253 / absent 271 / unclear 2,105.
  - Deployed models vs the reviewers where both decided: headcap wrong 18%
    (MCC 0.64), mask 14% (0.69), gloves 11% (0.74).

## Deployable file (prepared, not in the bundle)

`runs/person_ppe/export/apd_ours_person_ppe.onnx`, 16 MB, with its settings in
`apd_ours_person_ppe.json`.

- **Input.** Person crop (pose box + 18% padding), letterboxed to 320 with
  pad value 114, RGB, ImageNet normalisation, dynamic batch.
- **Output.** P(present) for [head_cap, mask, gloves]. A violation is flagged
  when p is below the threshold: headcap 0.65, mask 0.20, gloves 0.525.
- **Parity.** Matches PyTorch to 2e-7.
- **CPU cost.** Measured on this 4-core server while the replay was running,
  so the numbers are pessimistic:
  - 35–41 ms for one person;
  - about 106 ms for three people, with 1–2 threads.

  This replaces the packed classifiers, the mask model and the gloves detector
  (about 58 ms per person today).

## What deploying it would need (not done)

1. Perception change: one call per frame on the person crops, with the three
   thresholds from the JSON. The pose network and config.yaml stay the same,
   so no recompile is needed.
2. Replay and blind review again on data/recording before shipping.
3. Label a set from the device camera (data/recording) with the same
   pre-label-and-review method, to confirm the gain holds there.

Scripts: 111–113 (audits), 115 (data/clean), 116 (classifier retrain), 117
(COCO test), 118 (person model), 119/120 (sppg pre-labels and sheets), 121
(sppg crops for the classifiers).

## Replay check of the current bundle (yolo11s-pose), for context

Full replay, 358 recordings, blind review of 237 of our alerts and 128 device
alerts (`artifacts/replay/SUMMARY.md`):

- **All items.** Ours has 133 real violations at 62.4% precision; the device
  has 65 at 59.1%.
- **Headcap.** Ours 61.5%, device 50.0%.
- **Mask.** Ours 65.7%, device 90.5%.
- **No-person boxes.** 15% of our alerts sit on the wall fan or on empty
  floor. They come from the pose model, and no PPE model can fix them.

Before deploying anything new, add a person-validity filter. Examples:
require visible head/shoulder keypoints, or add a static exclusion mask for
the fan.

## Three-way benchmark (2026-09-18)

Systems, each run as the device would:

- **PERSON (new):** one network per person, efficientnet_b0 320 px, trained on
  cleaned det7 + sppg labels + this camera's labels
  (`runs/person_ppe/efficientnet_b0_320_rec2`).
- **PACKED (new):** the same shape as today's bundle but retrained on the
  audited data plus sppg: head_cap resnet34 and mask efficientnet_b0 (both
  chosen on VAL), gloves yolo26n crop detector on the cleaned det7.
- **PACKED (deployed):** the bundle running today.
- **PRODUCTION:** apd_mbg Dec-2025, its own detector on the whole frame.

### TEST A: the device camera, 195 people from 8 clips no model trained on

`scripts/124_benchmark_three.py` (MCC, violation positive; precision/recall in brackets)

| system | head_cap | mask | gloves |
|---|---|---|---|
| **PERSON (new)** | **0.921** (100%/90%) | **0.712** (85%/90%) | **0.413** (66%/86%) |
| PACKED (new) | 0.744 (89%/78%) | 0.515 (86%/63%) | −0.012 (50%/21%) |
| PRODUCTION | 0.601 (84%/62%) | 0.632 (93%/68%) | 0.402 (79%/52%) |
| PACKED (deployed) | 0.519 (85%/49%) | 0.498 (91%/53%) | 0.122 (62%/30%) |

- **head_cap:** the person model beats production decisively (p=0.000).
- **gloves:** only the person model matches production here. Both packed
  variants fail, because the gloves crop detector never saw this site's white
  and blue gloves.
- **mask:** 35 judged people only, so nothing here is significant.
- Production scores far better on its own camera than on COCO, which is worth
  remembering when quoting the COCO table.

### TEST B: COCO human labels, 53 frames

| system | head_cap | mask | gloves |
|---|---|---|---|
| **PERSON (new)** | **0.534** | 0.213 | **0.228** |
| PACKED (new) | 0.465 | 0.213 | 0.146 |
| PACKED (deployed) | 0.335 | 0.196 | 0.180 |
| PRODUCTION | 0.191 | 0.146 | 0.047 |

The person model wins or ties on both test sets, and it is one 15 MB network at
about 35-40 ms per person instead of three at about 58 ms.
Files: `runs/person_ppe/export/apd_ours_person_ppe_rec2.onnx` (+ .json with the
thresholds 0.30 / 0.175 / 0.20) and `runs/person_ppe/benchmark_device_camera.json`.
