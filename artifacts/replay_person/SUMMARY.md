# Replay with the one-network person model (2026-09-18)

## What ran

`deploy/apd_person` is a staging copy of the device bundle whose perception step
calls ONE network per person instead of three
(`runs/person_ppe/export/apd_ours_person_ppe.onnx`, efficientnet_b0 320 px,
outputs head_cap / mask / gloves). `deploy/apd_ours` and the device are
untouched. Replay: all 358 recordings, same monitor, tracker, vote and cooldown
as the device.

    APD_BUNDLE=deploy/apd_person APD_REPLAY_OUT=artifacts/replay_person \
      python scripts/105_replay_recordings.py ours

**The model in this run is the `+sppg` one**, trained before the device
camera's own labels existed. The newer `+rec2` model
(`apd_ours_person_ppe_rec2.onnx`) is not in this replay.

## Violations detected, 348 dated recordings

| system | events | headcap | mask | gloves |
|---|---|---|---|---|
| deployed bundle (packed classifiers + gloves detector) | 239 | 131 | 101 | 72 |
| one-network person model (+sppg) | 306 | 204 | 188 | 285 |
| production device | 207 | 137 | 79 | 92 |

The person model alerts far more often. Two reasons:

- **No abstention.** The packed pipeline says "unknown" when it cannot see a
  head or face; this model answers for every person the pose model finds.
- **Gloves on this camera.** It reads the cleaning crew's white and blue gloves
  as bare hands, which is exactly the misfiring reported after the first check.
  On 195 held-out people from this camera it scores 0.145 (MCC), against 0.421
  for the `+rec2` model trained with this camera's labels.

## Accuracy of this model family

| test | +sppg (this replay) | +rec2 (newer) | deployed stack |
|---|---|---|---|
| device camera, 195 people, 8 unseen clips: head_cap / mask / gloves | 0.788 / 0.795 / 0.145 | **0.932 / 0.712 / 0.421** | 0.778 / 0.795 / 0.143 |
| COCO, 53 frames: head_cap / mask / gloves | 0.543 / 0.346 / 0.228 | 0.534 / 0.213 / 0.228 | 0.335 / 0.196 / 0.180 |

Scores are MCC, violation positive, missed or undecided people counted as no
alert.

## Side-by-side videos

Same two clips as the deployed-bundle renders, right pane labelled
"OURS (one-network person model)":

- `artifacts/replay/renders/2026-09-16_00-07-46_person/side_by_side.mp4`
- `artifacts/replay/renders/2026-09-16_00-02-11_person/side_by_side.mp4`

The deployed-bundle versions are in the same folders without the `_person`
suffix, and the older yolo11n-pose ones end in `_yolo11n`.

## What this run does not answer

- The alerts have not been blind-reviewed, so these are counts, not precision.
- The replay should be repeated with `+rec2` before any deployment decision.
- Neither model fixes the pose model's false people: 12 wall fans were found
  among the 793 labelled people from this camera, and 15% of the deployed
  bundle's alerts in the earlier review sat on no person at all.
