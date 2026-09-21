# Replay summary: ours vs production device

## Violations detected, full replay with yolo11s-pose (348 dated recordings)

| source | events | headcap | mask | gloves |
|---|---|---|---|---|
| ours (yolo11s-pose bundle) | 239 | 131 | 101 | 72 |
| device, all its events in those clips | 207 | 137 | 79 | 92 |

One event can carry more than one violation, so the three item columns add up
to more than the event count.

Earlier partial run with yolo11n-pose, same 172 clips on both sides:

| source | events | headcap | mask | gloves |
|---|---|---|---|---|
| ours (ours_yolo11n) | 192 | 113 | 67 | 61 |
| device, all its events in those clips | 179 | 120 | 60 | 87 |

## Blind review of the alerts (yolo11s-pose run)

- **What was reviewed.** 237 of our alerts and the 128 device alerts that could
  be placed in the recording by picture.
- **How.** Each alert was cut from the raw frame, shuffled, and judged without
  knowing which system raised it. Four reviewers used the apd/schema.py rules,
  including the navy beret and chef skullcap decisions.
- **Precision.** Precision below counts only claims that could be judged:
  real / (real + wrong).

| item | ours: real / wrong / unclear | ours precision | device: real / wrong / unclear | device precision |
|---|---|---|---|---|
| headcap | 67 / 42 / 22 | **61.5%** | 35 / 35 / 15 | 50.0% |
| mask | 46 / 24 / 29 | 65.7% | 19 / 2 / 16 | **90.5%** |
| gloves | 20 / 14 / 38 | 58.8% | 11 / 8 / 41 | 57.9% |
| all | 133 / 80 / 89 | **62.4%** | 65 / 45 / 72 | 59.1% |

**Ours finds twice as many real violations** (133 vs 65) at similar or better
precision, except for mask.

**Our biggest error is boxes with no person in them.** 36 of our 237 alerts
(15%) sit on the wall fan (13) or on empty floor or table (23). The device has
none. They come from the pose model, not the PPE models, and they account for
most of our wrong mask alerts.

**Other wrong headcap claims (both systems):**

- navy beret-style caps over a hijab;
- purple and blue bouffant caps;
- green hairnets;
- black chef skullcaps.

Files: `review/sheet_*.jpg`, `review/key.csv`, `review/labels.json`, and the
per-reviewer notes in `review/_parts/`. The yolo11n review is kept in
`review_yolo11n/`.

## Accuracy on the COCO human-labelled test set

53 frames labelled by hand and never used for training. Every person goes
through the full pipeline: find the person, then judge each item. A violation
counts as the positive result, and a person the pipeline misses or leaves
undecided counts as "no alert".

- **ours (packed model):** the bundle as deployed. yolo11s-pose at conf 0.40,
  then the packed head_cap classifier, the face-crop mask classifier and the
  yolo26n gloves detector.
- **production:** apd_mbg Dec-2025. The device's apd_v3 weights are not
  available, so this model stands in for it.

| item | model | precision | recall | F1 | MCC |
|---|---|---|---|---|---|
| headcap (202 people) | ours (packed model) | 75.5% | 39.8% | 0.521 | **0.335** |
| | production | 65.8% | 26.9% | 0.382 | 0.191 |
| mask (44 people) | ours (packed model) | 94.4% | 44.7% | 0.607 | **0.196** |
| | production | 93.3% | 36.8% | 0.528 | 0.146 |
| gloves (122 people) | ours (packed model) | 100.0% | 12.0% | 0.214 | **0.180** |
| | production | 81.8% | 9.8% | 0.175 | 0.047 |

MCC runs from −1 to 1, and 0 is no better than chance. The retrained
per-person model scores higher still (headcap 0.543, mask 0.346, gloves 0.228);
see `artifacts/RETRAIN_REPORT.md`.

Side-by-side videos (yolo11s-pose), with the yolo11n versions kept alongside:

- `renders/2026-09-16_00-07-46/side_by_side.mp4`
- `renders/2026-09-16_00-02-11/side_by_side.mp4`
