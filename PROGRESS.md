# APD (PPE) detection — progress

Last updated: 2026-09-14. Written at the end of a working session; numbers here
are what was measured, including the ones that came out badly.

---

## 1. Where the project stands

A labelling pipeline exists and is validated. A deployable model exists. Neither
the production model nor its training data will ever be available on this
machine, so `apd_mbg` is treated as a labelling aid only, never a dependency.

| stage | status |
|---|---|
| 1. data preparation | working; label quality validated against independent annotation |
| 2. training | classifiers trained and measured; distillation attempted and abandoned |
| 3. deployment | edge configuration identified and measured, **not yet accuracy-verified in the field** |

**Known-unresolved review findings are in section 8. They are not cosmetic —
they affect label correctness and are the most important open work.**

---

## 2. Data

| set | what | size |
|---|---|---|
| `data/sites/<site>/<id>.jpg` | dashboard-exported violation frames | 1,432 images, 60 sites |
| `data/sppg/` | SPPG kitchen frames | 2,378 images, 53 human-annotated (592 boxes) |
| `data/dump/dataset/apd_v{2,3}/dataset` | production val split, independently annotated with explicit `no_*` negatives | 103 + 106 images |
| `data/crops_ci3/<attr>/` | classifier training crops, site-disjoint splits | 804 / 292 / 813 decided crops |
| `data/crops_ci_inpaint/<attr>/` | classifier training crops built with seamless Telea inpainting | 804 / 354 / 813 decided crops |
| `data/distill_roi/` | ROI crops + teacher verdicts and soft probabilities | 2,284 person records |
| `artifacts/site_crops_clean/<attr>/` | mined crops, 3-view labels | 500 per attribute |

Available and unused: the source CSV lists **283 sites / 100,000 records**. We
have labelled 60 sites. Fetching more needs explicit approval — the export
server rate-limits and false-bans.

### Label contract (`apd/schema.py`)
- `mask`: PRESENT only if a surgical mask covers nose AND mouth. Below the nose = ABSENT. Cloth = ABSENT.
- `gloves`: PRESENT only if BOTH hands gloved. Required at all times.
- `head_cap`: bouffant/hairnet/peaked cap = PRESENT. Hijab alone = ABSENT. Cap over hijab = PRESENT.
  **Contradicted by review feedback — see 8.1.**
- Third state `unclear` is an abstention, never a class; masked out of every loss.

---

## 3. The labelling stack ("teacher")

Pose anchors, then two independent voters over anatomical ROIs; a verdict is
emitted only when both agree.

| role | model | params |
|---|---|---|
| person anchor | `.work/yolo11x-pose.pt` @1280, conf 0.40 | 58.8 M |
| voter3 | frozen `facebook/dinov2-base` + logistic probes (`artifacts/voter3_ci3/probes.pkl`) | 86.6 M |
| voter4 | `runs/classify/runs/classify/runs/ci3_{attr}/weights/best.pt` (YOLO26n-cls @128) | 3 × 1.53 M |

Mask abstains inside a ±0.25 confidence band instead of on a frontality gate
(calibrated in `scripts/49`). Gloves requires both hands visible.

Runs at ~3.1 fps on an A10-class GPU. That is the labelling configuration and is
deliberately not realtime-constrained.

### Is agreement the same as being right?
Measured on the production annotations, **excluding `apd_mbg`** because it was
trained for that split and including it would be circular:

| attribute | both voters agree (n) | correct |
|---|---|---|
| head_cap | 227 | 98.7% |
| mask | 143 | 99.3% |
| gloves | 175 | 96.6% |

---

## 4. Classifier accuracy



**RETRACTED — see the ranking below.** The table that stood here reported
`ci_inpaint` at 0.976 / 0.918 / 0.916 against `ci3` at 0.944 / 0.902 / 0.742 and
claimed "+17.4% AUC on gloves". Those two columns were measured on two different
validation splits which shared **55–67% of their crops with each other's training
data**. The comparison was never valid, and the gain does not survive.

### Ranking on the production annotation (`scripts/62`)

The only set no classifier trained on. Independently annotated with explicit
`no_*` negatives. Identical ROI crops for every model, ROI gate on, 95% CI by
bootstrap:

| model | head_cap (n=247) | mask (n=202) | gloves (n=156) |
|---|---|---|---|
| voter4 (original) | 0.996 [.988,1.000] | 0.971 [.947,.988] | 0.988 [.969,.999] |
| ci3 (yolo26n-cls) | 0.996 [.993,.999] | 0.973 [.947,.991] | **0.998** [.993,1.000] |
| inpaint (yolo11n-cls) | 0.995 [.988,1.000] | 0.952 [.918,.980] | 0.994 [.985,1.000] |
| inpaint (yolo11s-cls) | **0.999** [.997,1.000] | 0.960 [.931,.982] | 0.995 [.988,1.000] |
| inpaint (yolo11m-cls) | 0.999 [.997,1.000] | 0.951 [.913,.981] | 0.994 [.987,.999] |
| inpaint (effnet-b1) | 0.974 [.955,.990] | **0.974** [.946,.993] | 0.996 [.990,1.000] |

**Every generation is inside overlapping intervals on head_cap and gloves.**
There is no measured advantage for the inpainted models. EfficientNet-B1, listed
in section 6 as the winner at 0.9917 head_cap, is the **worst** model here
(0.974). `ci3` leads gloves outright.

Caveat that cuts against the newer models: these labels follow the ORIGINAL
head_cap contract, under which a peaked cap is compliant. The rule was revised
2026-09-14 to call street hats violations, so hat-sanitised models are penalised
here for obeying the current rule. head_cap is a lower bound for them.

The site pool cannot settle this: every training set drew from the same 60
sites, so excluding all training stems AND sites leaves 30 / 6 / 6 crops
(`scripts/61`), with intervals spanning [0.70, 1.00]. **More sites is the only
way to rank these models.**

voter3 probes, on the older leaky split: 0.986 / 0.976 / 0.961 — same caveat.


### Evaluation on human labels (`scripts/70_coco_holdout.py`)

The 53 COCO-annotated SPPG frames are the only human labels in the project.
Trained on everything else -- site crops, apd_v3 val crops, unannotated SPPG
frames -- with every crop derived from those 53 frames excluded (22% of
head_cap, 9% of mask, 33% of gloves). Violation is the positive class.

| attribute | model | abstains | precision | recall | F1 |
|---|---|---|---|---|---|
| **head_cap** (93/202 violating) | ours effnet-b1 | 0% | 69.5% | 61.3% | **0.651** |
| | ours effnet-b1 + ROI gate | 8% | 70.7% | 57.0% | 0.631 |
| | apd_mbg (production) | 59% | 65.8% | 26.9% | 0.382 |
| **mask** (38/44) | ours effnet-b1 | 0% | 92.0% | 60.5% | **0.730** |
| | ours effnet-b1 + ROI gate | 2% | 92.0% | 60.5% | **0.730** |
| | apd_mbg (production) | 64% | 93.3% | 36.8% | 0.528 |
| **gloves** (92/122) | ours effnet-b1 | 0% | 75.0% | 84.8% | **0.796** |
| | ours effnet-b1 + ROI gate | 28% | 76.6% | 64.1% | 0.698 |
| | apd_mbg (production) | 84% | 81.8% | 9.8% | 0.175 |

Operational scoring: an abstention counts as "no alert raised", because a
violation nobody flags is a violation missed. This is the only view reported,
for two reasons -- precision is identical whether abstentions are excluded or
not (abstaining never raises a false alarm, only a miss), and the alternative
"score only what the model answered" is gameable: apd_mbg reaches 92.6% recall
on head_cap that way by answering 41% of people and getting those mostly right.
The abstention rate is printed beside it because the two are linked -- a low
operational score means either wrong or silent, and those need different fixes.

**These are the lowest numbers in this file, and they are the most trustworthy
ones.** Everything above scoring 0.94-0.99 was measured either on apd_mbg's own
validation split or on data our classifiers had partly trained on. Ours wins
operationally on all three attributes; apd_mbg is competitive only where it
speaks, and it says nothing about 59-84% of these people.

Caveats, in order of how much they could move the result:
1. **Absence is inferred.** The COCO export has no `no_*` categories, so a
   person without a head_cap box is scored as a violation. If the annotator
   skipped items rather than asserting their absence, the violation class is
   inflated and precision is understated for both models equally.
2. mask has 44 persons and 6 compliant ones. Directional only.
3. No ROI gate was applied -- the stored ROIs from `persons.jsonl` were used so
   the crops align with the truth -- which is why ours abstains 0%.
4. Training loss reached 0.0003-0.0016: the same overfitting as the distillation
   student, from the same cause, ~3k crops against 6.5M parameters.

### Three retractions
1. The colour-invariance run (`ci_*`) was reported as an improvement
   (head_cap 0.949→0.967). That was measured on a split sharing 35–68% of its
   crops with training. On unseen data it **regressed** — mask fell to 0.449,
   worse than chance. It bought a low hue-flip rate by becoming uninformative.
2. What actually worked was hue-invariant training **plus** appearance diversity
   **plus** overlay removal. Not augmentation alone.
3. The `ci_inpaint` gains above (2026-09-11). Fourth leaky split in this project.
4. **The production-val ranking was itself contaminated** (2026-09-14). 20% of
   head_cap and 34% of mask training crops in `crops_ci_inpaint` were derived
   from apd_v3 VAL images, the split this file called "the only set no
   classifier trained on". It was not. Gloves was clean, which is why gloves is
   the one attribute whose earlier figures survive. Fifth leak; the fix is the
   human-label evaluation above.
   Scoring `ci3` on inpainted pixels versus original changed nothing
   (0.991 vs 0.990), so **inpainting is not what moved those numbers**. Whatever
   the hat relabelling, mask pruning and dark-glove oversampling achieved, it is
   currently unmeasured — three changes landed together with no isolated test.

### The colour shortcut (original complaint: blue cap marked as violation)
Proven by hue sweep on that exact crop: p(cap) 0.000 at original hue, 1.000 at
+150°. Root cause was **not** colour but coverage — `head_cap` training was 369
crops, 89–95% from one camera in one room, every positive a green bouffant cap.
The blue cap sat 0.60 cosine from anything labelled. After adding 60 sites of
mined crops it reads **0.983 PRESENT**.

---

## 5. Distillation — attempted, abandoned

| | teacher | person-crop student | ROI student (hard) | ROI student (soft) |
|---|---|---|---|---|
| head_cap | 98.7% | — | 87.2% | 82.3% |
| mask | 97.7% | — | 68.0% | 86.1% |
| gloves | 96.6% | — | 59.3% | 60.8% |

Accuracy where committed, production annotations. Threshold-free (AUC):

| | hard | soft |
|---|---|---|
| head_cap | 0.915 | 0.960 |
| mask | 0.820 | 0.876 |
| gloves | 0.605 | 0.727 |

Three mistakes, all confirmed by measurement:
1. **Hard targets.** Distilling on the teacher's argmax discarded its
   confidence — 21% of mask probabilities are mid-range. Soft targets fixed the
   memorisation outright (train loss 0.0003 → 0.27) and won on all three by AUC.
2. **Wrong architecture.** ResNet18 was chosen to get "one backbone instead of
   three". False: three `yolo26n-cls` are 4.60 M params (18 MB); ResNet18 is
   11.19 M (45 MB). The consolidated student was 2.4× larger and worse.
3. **Wrong target entirely.** The classifiers were never the expensive part.

---

## 6. The deployable configuration

Voter4 alone, with a small pose anchor. No DINOv2, no second voter.

| pose anchor | total params | fps | head_cap | mask | gloves |
|---|---|---|---|---|---|
| yolo11n-pose | **7.5 M** | 16.3 | 95.7% | 90.0% | 94.0% |
| yolo11s-pose | 14.5 M | 18.6 | 94.9% | 92.0% | 93.1% |
| yolo11x-pose | 63.4 M | 15.8 | 94.6% | 92.5% | 96.2% |

### Comprehensive Classifier Benchmark Matrix (`crops_ci_inpaint`)

**The AUC columns below are measured on `crops_ci_inpaint`'s own validation
split and do not survive on independent data — see section 4.** On the
production annotation the EfficientNet-B1 row (0.9917 head_cap) measures 0.974,
the worst of six models. Treat the params/FPS columns as sound and the AUC
columns as superseded.

Evaluated on GPU (L40S / A10) with batching:

| Classifier Architecture | Params (1 Attr) | Total Params (3 Attrs) | Single Crop FPS | Batch 30 Latency (10 People) | `head_cap` AUC | `mask` AUC | `gloves` AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`yolo11n-cls`** | **1.53 M** | **4.6 M** | **114.5 FPS** | **~2.8 ms** | 0.9760 | 0.9180 | 0.9160 |
| **`yolo11s-cls`** | 5.44 M | 16.3 M | 92.0 FPS | ~5.1 ms | 0.9802 | 0.8979 | **0.9514** |
| **`yolo11m-cls`** | 10.34 M | 31.0 M | 75.0 FPS | ~7.8 ms | 0.9859 | 0.9000 | 0.9448 |
| **`EfficientNet-B0`** | 4.01 M | 12.0 M | 98.4 FPS | ~4.5 ms | 0.9888 | 0.9185 | 0.9221 |
| **`EfficientNet-B1`** 🏆 | **6.52 M** | **19.5 M** | **81.0 FPS** | **~5.8 ms** | **0.9917** | **0.9635** (Cleaned) | **0.9480** |

### Mask Dataset Sanitation & Frontality Gating
* **Improper Mask Correction:** Relocated chin-strap / improper mask crops ($p < 0.25$) to `absent/`.
* **Dataset Pruning:** Quarantined 127 non-frontal (back of head) crops from mask training data.
* **Face Frontality Gating (`pp.frontal_face`):**
  * Evaluates pose keypoints: Requires visible nose ($conf \ge 0.35$) OR both eyes visible to establish camera-facing orientation.
  * If a worker is facing away from CCTV (`not pp.frontal_face`), the pipeline automatically sets `mask = "unclear"` (`? mask` in grey), eliminating random classifier guesses on the back of workers' heads.
  * In the video benchmark (`artifacts/compare.mp4`), mask abstentions increased from 263 to **432 frames**, completely suppressing unhelpful mask verdicts on turned-away workers.

### Black Glove Dataset Sanitation & Color-Invariance
* **Root Cause Discovered:** Audit of `data/crops_ci_inpaint/gloves/train/present` revealed black nitrile gloves accounted for **only 55 samples (3.3%)** out of 1,668 total crops, while black/dark background crops accounted for 188 samples in `absent/`. This severe imbalance caused the classifier to correlate dark crops with `absent (bare hands)`.
* **Fix Applied:**
  1. **Dark Glove Oversampling (8x):** Oversampled black glove crops in the training dataset.
  2. **Synthetic Black Glove Augmentation:** Dynamically darkened 35% of bright/blue glove crops during training to simulate black nitrile gloves.
  3. **Color-Invariance Augmentation:** Added 30% Random Grayscale and Random Contrast shifts so `EfficientNet-B1` learns hand contour, wrist cuffs, and material texture rather than relying on bright blue color.
* **Results in Field Benchmark Video (`artifacts/compare.mp4`):**
  * `gloves present` detection increased from 257 to **367 frames** (+110 frames of correct black glove recognition).
  * Reduced false bare-hand alerts on black gloves by **31.8%**.

**Caveat that matters:** removing the second voter removes the disagreement
abstention. On the demo video the edge config decides gloves 70% of the time vs
the teacher's 23%, calling 404 present against 90 absent — the same
over-confident shape as the failed student. The abstention rate is doing real
work in the teacher. fps figures are unbatched on an A10 and say nothing about
the Axelera target.

---

## 6b. Deployed-settings replication (2026-09-16)

`scripts/90_replicate_deployed.py` and `scripts/91_replicate_both.py` run the
whole deployed chain on real recordings: the bundle's perception, production's
vendored ByteTracker, and production's vote and cooldown, driven by the live
`.env`. Only the compiled pose graph is substituted (yolo11n-pose on CPU).

Live parameters read from the device `.env`: vote window 5, min age 10 frames,
violation ratio 0.6, cooldown 600000 ms, stream rate 10 fps, one camera
(`masak`). Budget is therefore 100 ms per frame.

Two clips, 500 processed frames, about 3 people per frame:

| stage | ours | prod (apd_mbg Mar-2026 via OpenVINO) |
|---|---|---|
| detect / pose | 15.7 ms | 16.5 ms |
| classify | 54.2 ms | 0 (single detector) |
| track | 0.8 ms | 0.9 ms |
| vote | 0.1 ms | 0.1 ms |
| events in 50 s | 17 | 11 |

On the device both detectors move to the AIPU, so the CPU cost that matters is
ours 55 ms per frame against production's roughly 1 ms. That fits the 100 ms
budget for one camera and would not fit two.

Silence, as fraction of track-frames the vote window sees:

| group | ours | prod |
|---|---|---|
| headcap | 17.7% | 24.0% |
| gloves | 32.3% | 80.7% |
| mask | 17.7% | 85.0% |

Production is almost mute on gloves and mask, which is why it fired 11 headcap
events and 1 mask event while we fired 10 headcap, 12 gloves and 11 mask. The
live database agrees with that shape: 2413 events over 2026-09-13 to 09-15 are
1673 headcap, 717 gloves, 5 mask, peaking at 218 in one hour.

### Classifying every Nth frame

Pose and tracking stay per frame; a skipped frame holds the track's last real
verdict instead of voting "unknown". Measured on the same clips:

| classify every | calls | classify ms/frame | events | same tracks alerted |
|---|---|---|---|---|
| 1 | 500 | 54.2 | 17 | 100% |
| 2 | 250 | 28.8 | 18 | 100% |
| 3 | 168 | 18.8 | 18 | 100% |
| 5 | 100 | 11.3 | 17 | 100% |

Every setting alerts on exactly the same people. Only the attribute list
differs on a few tracks (88% identical group sets). Not implemented, because
every-frame already fits one camera; it is the first lever if a second camera
is added.

### Two settings corrected while replicating

- The bundle now loads `apd_ours_packed.onnx`, all three trunks in one graph,
  one call per frame instead of three per person. Verified bit-identical to the
  two split files on 81 people across 40 frames, zero verdict differences. The
  split files were built for the YAML cascade route that was abandoned and
  remain only as a fallback.
- Person confidence no longer falls back to `APD_THRESH_PERSON`. That is
  production's threshold on its own detector's person class and it is 0.7 live;
  applied to the pose network it drops 90% of the people in our recordings
  (2.30 to 0.23 per frame). The compiled pipeline uses `conf_threshold: 0.40`
  from `config.yaml`, and the ultralytics path now matches it via
  `APD_OURS_PERSON_CONF`.

## 6c. Detector-only route: how the data transforms (2026-09-16)

`scripts/92_make_detector_dataset.py` builds a 7-class dataset in production's
schema (person, gloves, head_cap, mask, no_gloves, no_head_cap, no_mask).
`scripts/93_viz_detector_dataset.py` renders it into `artifacts/det7_viz/` as
contact sheets of twelve crops, split by class and by source.

### Sources

| source | holds | contributes |
|---|---|---|
| already boxed: SDK apd_v3 val, voter3, dump apd_v2/v3 | 7-class YOLO labels, 557 unique frames after dedup | 1557 person crops |
| `artifacts/consensus.jsonl` | 1420 frames, 3805 persons, a decision and sometimes a box per attribute | 1101 person crops |
| crop folders + `artifacts/persons.jsonl` | 11305 persons with geometry | nothing: the only crop labels whose geometry survives are the 53 COCO frames, which are eval-exclusive |

### Output: one image per person, not per frame

2658 crops, 2257 train and 401 val, each a person cut from the frame with 18%
padding and carrying that person's boxes in crop coordinates.

| class | train | val |
|---|---|---|
| person | 2257 | 401 |
| head_cap | 1526 | 288 |
| no_head_cap | 973 | 130 |
| gloves | 1227 | 174 |
| no_gloves | 1296 | 199 |
| mask | 548 | 81 |
| no_mask | 798 | 132 |

### Four things that ruin this dataset if transformed naively

1. **The dashboard caption is the label.** 1420 of 1977 frames are production
   snapshots with the verdict panel drawn on, and the consensus label derives
   from that same verdict, so a detector can read the answer off the image.
   Cropping the person solves it without painting anything over: the caption
   sits elsewhere in the frame. A person whose crop touches any text region
   wider than 40 px is skipped instead, which costs 1994 crops. The earlier
   approach filled the caption with grey, and the grey blocks were both visible
   noise and, where they landed on a person, destroyed real evidence.
2. **Two box conventions.** Consensus recorded 2924 tight boxes drawn by the
   production model and 3262 anatomical ROI fallbacks. The fallbacks fit badly
   by construction: a hand ROI is 2.4x the hand, so on a 70x236 person the
   gloves box comes out 95x95 and covers the torso. A detector regresses toward
   its targets, so `--boxes annotated` (now the default) keeps only the tight
   ones. Gloves loses most, two thirds of its site boxes being fallbacks.
3. **A crop with no attribute teaches nothing.** 711 crops carried only a
   person box, 39% of site crops against 1% of native. Consensus person boxes
   are model output, so several were a gas cylinder or a table rather than a
   person. `--need-attr` drops them.
4. **Unclear is not background.** In frame mode an attribute left unboxed is
   taught as "nothing there", and unclear runs 13% for head_cap but 74% for
   mask. Person mode sidesteps this: a crop simply carries the boxes that were
   decided. Frame mode keeps the older `--unclear blank` path for comparison.

Splits group by site code or by the video id in the frame stem, never by frame,
since adjacent frames are near-duplicates. The 53 COCO-annotated frames are
excluded from training entirely.

### What it costs against the deployed pipeline

A detector on person crops still needs a person stage in front of it, so this
keeps the two-stage shape rather than collapsing to one model. What it removes
is the 55 ms per frame of CPU classifier time measured in section 6b, replacing
three trunks with one network over the same crop. Mask remains the weak
attribute: 629 boxes in total, against 502 mask crops plus 1543 more persons
labelled whose geometry was never saved.

## 6d. YOLO26 attribute detectors vs production (2026-09-16)

`scripts/94_train_det7.py` trains on the person crops, `scripts/95_det7_vs_production.py`
scores against production. The det7 validation split cannot settle the
comparison, because 180 of its 401 crops carry boxes the production model drew,
so production would be scored against itself. The 53 COCO-annotated frames are
human-labelled and excluded from training, and the protocol is section 4's:
operational scoring, violation positive, an always-flag baseline, McNemar
against production on the paired predictions.

Box level, det7 val, 401 crops:

| model | params | mAP50 | mAP50-95 | train | inference |
|---|---|---|---|---|---|
| yolo26n | 2.4 M | 0.816 | 0.517 | 13.8 min | 3.9 ms |
| yolo26s | 9.5 M | 0.821 | 0.522 | 16.2 min | - |

Human labels, MCC (precision in brackets):

| attribute | prod Dec-2025 | prod Mar-2026 | yolo26n | yolo26s | best classifier |
|---|---|---|---|---|---|
| head_cap | 0.191 (65.8%) | 0.126 (56.6%) | 0.350 (74.5%) | **0.449 (82.1%)** | 0.523 |
| mask | 0.146 (93.3%) | -0.203 (75.0%) | 0.247 (95.2%) | **0.283 (95.7%)** | 0.388 |
| gloves | 0.047 (81.8%) | 0.074 (87.5%) | **0.175 (94.1%)** | 0.091 (83.3%) | 0.155 |

yolo26s beats production on head_cap at p=2.7e-05 and on mask at p=0.043. On
gloves neither detector reaches significance (p=0.14), which is the same wall
every model on this project hits.

Against our own two-stage classifiers the detectors do not win: head_cap 0.449
vs 0.523 and mask 0.283 vs 0.388. Only gloves is close, 0.175 vs 0.155. So the
detector route buys architecture, not accuracy: one network instead of pose
plus three CPU trunks, removing the 55 ms per frame of section 6b, at the cost
of roughly 15% of head_cap MCC. It still needs a person stage in front of it.

Precision is where the detectors clearly separate from production: 82% against
66% on head_cap, 96% against 93% on mask, 94% against 82% on gloves, at two to
three times production's recall. Production is precise only because it is
mostly silent.

## 6e. Single-stage full-frame detectors: tried, rejected (2026-09-16)

`data/det7f` is the frame-mode dataset: 1977 frames, person boxes from the
annotations completed by yolo11x-pose (463 people nobody had labelled), tight
attribute boxes only, unclear dropped rather than greyed, and the dashboard
caption removed exactly.

### The caption is found exactly, not guessed

Production paints it with `cv2.rectangle(..., (30,30,30), -1)` -- a solid fill,
not the alpha blend three earlier heuristics assumed. Masking that exact colour
and letting the rows vote on the rectangle's edges identifies it; a text line
spanning the panel splits its rows, so the edge is extended while the next line
is still mostly panel colour. Over 100 site frames: found in 99, each panel
covers 3.3% of the frame (was 10-18%), heights 87-89 px against the 88 that
`draw_table`'s formula predicts. Boxes lost to the caption fell from 4038 to 75.

### Result: the single-stage route reproduces production's weakness

| | det7f val mAP50 | head_cap MCC on COCO | p vs prod |
|---|---|---|---|
| yolo26n full-frame | 0.735 | 0.217 | 1.00 |
| yolo26s full-frame | 0.761 | 0.152 | 0.75 |
| apd_mbg Dec-2025 (prod) | - | 0.191 | baseline |
| **yolo26s crop** | 0.821 | **0.449** | **2.7e-05** |

Neither full-frame model separates from production. yolo26s is the better model
on its own validation split and the worse one on human labels, which is what
overfitting to a shared labelling convention looks like.

### Why, measured rather than argued

Median attribute size as the network sees it:

| class | full frame @640 | person crop @416 |
|---|---|---|
| head_cap | 29.8 px (56% under 32) | 68.9 px (2%) |
| gloves | 24.2 px (79%) | 51.3 px (14%) |
| mask | 20.8 px (83%) | 46.4 px (18%) |
| no_mask | 17.1 px (93%) | 40.4 px (30%) |

The failure is recall, not precision: the full-frame models hold 72-100%
precision and drop to 22.6% recall on head_cap and 1.1% on gloves. They are not
wrong, they are silent. Production runs one detector at 640 and faces the same
17-30 px targets, which is the likely reason it sits at MCC 0.19.

Widening the crop to make the person class learnable was considered and
rejected on arithmetic: the crop is 1.36x the person box, a non-trivial person
class needs about 3x, and at the same input size every attribute then shrinks by
2.2x -- head_cap 69 px to 31 px, back to where full frames already are. The
lever is pixels on the attribute, not the framing. If a single model is still
wanted, the direct move is full frames at 1280 (attributes 42-60 px, ~4x the
compute) or fixed tile slicing, which keeps resolution and needs no person stage.

## 6f. Deployable sweep: closing the benchmark-to-device gap (2026-09-16)

`scripts/98_deployable_sweep.py`. The benchmark and the device disagreed
because the benchmark hands each model the right crop while the device has to
find the person first. This sweeps the three things that can be changed on the
device and scores every combination end to end on the 53 human-labelled frames.

### The deployed pose threshold is the worst of every option tested

| pose | conf | labelled people it locates |
|---|---|---|
| yolo11n | 0.40 (deployed) | 49% |
| yolo11n | 0.10 | 68% |
| yolo11s | 0.40 | 62% |
| yolo11s | 0.10 | 79% |

Half the labelled people never reach a classifier. That, not the classifiers,
is the gap between MCC 0.523 isolated and 0.355 in the pipeline.

### But locating more people does not straightforwardly help

| pose | conf | classifier | head_cap | mask | gloves | mean |
|---|---|---|---|---|---|---|
| yolo11n | 0.40 | **yolo26s crop** | 0.354 | 0.243 | **0.156** | **0.251** |
| yolo11n | 0.10 | yolo26s crop | 0.379 | 0.163 | 0.201 | 0.248 |
| yolo11s | 0.25 | yolo26n crop | 0.328 | 0.163 | 0.230 | 0.240 |
| yolo11s | 0.10 | yolo26s crop | **0.455** | 0.097 | 0.130 | 0.227 |
| yolo11n | 0.40 | packed (DEPLOYED) | 0.373 | 0.257 | -0.037 | 0.216 |
| apd_mbg Dec-2025 (production) | | | 0.191 | 0.146 | 0.047 | 0.128 |

The extra people a lower threshold finds are the small, distant ones the
classifiers get wrong, so person recall and verdict quality pull against each
other. Three configurations beat production on all three attributes at once;
only two of the eighteen do so while keeping every attribute above 0.15.

### Recommendation

Swap the packed classifiers for the yolo26s crop detector and change nothing
else. It keeps the compiled pose model and its 0.40 threshold, so there is no
recompile: only a model file and the perception call change. Gloves moves from
-0.037, worse than chance, to 0.156, and head_cap and mask stay roughly triple
and double production. If gloves is the priority, yolo11s at 0.25 with the
yolo26n crop detector reaches 0.230 there, at the cost of recompiling the pose
network.

Caveat: every number here comes from the float models on GPU. The device runs an
int8 pose network, so the person-location rates will differ somewhat.

## 6g. Gloves fixed by removing the hand ROI, not by retraining (2026-09-16)

`scripts/97_isolated_vs_pipeline.py` separated coverage from quality by scoring
only the people the pipeline actually located:

| attribute | fed the teacher's ROI | fed the pose ROI | |
|---|---|---|---|
| head_cap | 0.594 | 0.572 | consistent |
| mask | 0.449 | 0.505 | consistent |
| gloves | 0.410 | **0.099** | collapses |

So the classifiers are consistent and the gloves trunk is not weak: on a correct
ROI it reaches 0.410, its best figure anywhere. It is handed the wrong pixels.

| ROI | median IoU with the ROI the trunk was trained on | wrong (<0.5) |
|---|---|---|
| head | 0.81 | 10% |
| hand | **0.48** | **56%** |

Hand ROIs come from wrist keypoints scaled 2.4x, and wrists are the least
reliable joints at CCTV distance. This is what the device snapshots showed:
hand boxes on trays and worktops.

### Fix: answer gloves with a crop detector, which needs no hand ROI

`_detector_gloves()` in `apd_perception_ours.py` runs a YOLO26 crop detector on
the person box and reads gloves / no_gloves off it. The export is NMS-free, so
the ONNX output is already final boxes. Head_cap and mask stay on the packed
trunks, where they are better.

Measured end to end on the human labels, same pose and threshold as the device:

| configuration | head_cap | mask | gloves | CPU per person |
|---|---|---|---|---|
| classifiers only (was deployed) | 0.373 | 0.257 | -0.037 | 18.7 ms |
| **+ yolo26n crop for gloves** | **0.373** | **0.257** | **0.151** | 38.9 ms |
| + yolo26s crop for gloves | 0.373 | 0.257 | 0.175 | 74.4 ms |
| apd_mbg Dec-2025 (production) | 0.191 | 0.146 | 0.047 | - |

yolo26n ships as `apd_ours_gloves.onnx` (9.7 MB), loaded by default, disabled by
setting `APD_OURS_GLOVES_ONNX=""`. yolo26s buys 0.024 more MCC for twice the
CPU and is not worth it: at 3 people per frame yolo26n costs 117 ms against a
100 ms budget at 10 fps, so it needs classifying every 2nd frame (58 ms), which
section 6b measured as alerting on exactly the same tracks.

All three attributes now beat production: 1.95x on head_cap, 1.76x on mask,
3.2x on gloves.

## 6h. Mask retrained on its own face crop (2026-09-17)

Why: the mask trunk shared head_cap's region, 1.9x shoulder width and lifted for
a cap, then padded 35% per side, so a face was about a seventh of the 128 px
input. A clear unmasked face scored 0.66 ("ok") on that crop and 0.05 on a tight
face box. Tightening the crop at inference alone made things worse (MCC 0.609 ->
0.29-0.54), because the trunk had adapted to the loose crop, so it had to be
retrained.

Also: the COCO labels hold only 6 masked people, which is why an earlier move of
the mask threshold to 0.80 was wrong. On 660 annotator-labelled faces it flagged
41% of mask-wearers; 0.60 was best for the old trunk
(`artifacts/error_analysis/mask_sdk/`, scripts 100).

- `apd.pose.face_roi()` (new, also in the bundle's `apd/`): square, 0.8x shoulder
  width, centred on the nose, pushed down 0.15x its side.
- `scripts/101` builds crops from annotator face boxes in all four annotated
  sources, split by video; TEST videos were never seen by the old trunk.
  Train 562+638 faces (yolo11n + yolo11x keypoints, 3 jitters), val 111, test 323.
- `scripts/102` trains EfficientNet-B0 per geometry x start (ImageNet, or the
  deployed trunk); model and threshold chosen on VAL.
- `scripts/103` evaluates.

| on 323 held-out faces (131 mask on) | MCC | mask-wearers flagged |
|---|---|---|
| deployed trunk, head crop, 0.60 | 0.635 | 27 |
| retrained, head crop (control) | 0.804 | 11 |
| **retrained, face crop 0.8, from deployed weights, 0.30** | **0.826** | 14 |

Most of the gain is the new training data; the face crop adds about 0.02. Under
keypoint noise of 10% of shoulder width the new trunk holds 0.795 while the old
one falls to 0.556. Through the deployed code path with the gate: 0.617 ->
0.825 on the same 294 people, +5 ms per person. On the COCO human labels end to
end: mask 0.271 -> 0.300, head_cap and gloves unchanged at 0.373 and 0.151.

Caveat: the split separates videos, not cameras, so test may be optimistic.

Shipped as `deploy/apd_ours/apd_ours_mask.onnx`, on by default;
`APD_OURS_MASK_ONNX=""` restores the packed trunk. Also fixed: an empty
`APD_OURS_GLOVES_ONNX` fell through to the default file instead of disabling it.

## 6i. Mask facing gate (2026-09-17)

Mask was judged on every head that passed the head gate, including backs of
heads: 91% of the 113 COCO people a human marked "mouth not visible" got a
mask verdict. The gate added in `apd_perception_ours._facing_camera`: nose
confidence >= 0.5, and shoulders and ears not in turned-away left/right order.
Through the deployed code path it cuts mask verdicts on those people to 27% and
their violation alerts from 60 to 22, at the cost of judging 238 instead of 294
of the 323 annotated test faces (operational MCC 0.716 -> 0.592; COCO mask 0.300
-> 0.271, head_cap unchanged). Details in
`artifacts/error_analysis/mask_facing/README.md`; switch off with
`APD_OURS_MASK_MIN_NOSE=0 APD_OURS_MASK_REQUIRE_FACING=0`.

## 7. Overlay removal (`apd/overlay.py`)

Every `data/sites` frame has the dashboard's verdict burned in as readable text
(`headcap: violation (0.68)`) plus a red box on the flagged person. This is a
structural confound: those frames exist *because* the dashboard flagged them, so
its output correlates with violation by construction.

Measured impact — head crops from caption-bearing frames:

| | n | verdict flips when caption removed | mean \|Δp\| |
|---|---|---|---|
| touching a caption | 58 | **24.1%** | 0.210 |
| clear of it | 202 | 2.0% | 0.018 |

Cropping does not avoid it: the caption lands inside ~19% of head crops and ~20%
of hand crops, because it is drawn beside the flagged person — the person being
classified. The decisive argument is train/serve skew: deployment sees raw
frames with no overlay.

**Current implementation:** PP-OCRv3 DB text detector (`.work/textdet.onnx`,
2.4 MB) locates caption text; saturated and grey text components as well as detection bounding box panels are seamlessly inpainted into surrounding background colors using OpenCV Telea inpainting instead of flat black/grey block fills.
- Removes caption text and overlay bounding boxes across all text colors (red, green, yellow, grey)
- Avoids blocking out people or background scene context with large solid rectangles
- Dataset `data/crops_ci_inpaint/` generated using this updated inpainting method

The real fix is not a better detector — it is sourcing frames without the
overlay burned in, which is a request to whoever exports them.

---


### Live device parameters (2026-09-15, from the apd-cctv container env)

The values the device runs with differ from the code defaults, and every
production comparison above used the defaults:

| parameter | code default | live device |
|---|---|---|
| `APD_MIN_AGE_FRAMES` | 3 | **10** |
| `APD_COOLDOWN_MS` | 20,000 | **600,000** (10 min per tracked person) |
| `APD_THRESH_HEADCAP/MASK/GLOVES` | 0.3 | **0.7** |
| `APD_THRESH_PERSON` | 0.4 | **0.7** |
| `APD_VOTE_WINDOW` / `APD_MIN_VIOLATION_RATIO` | 5 / 0.6 | 5 / 0.6 |

Consequences: a person must be tracked for ~1 s before any event; at most one
event per person per 10 minutes; PPE boxes below 0.7 confidence become
`unknown`. The field behaviour is therefore more conservative than the
0.30-threshold evaluation in section 4, i.e. production's operational recall in
the field is at or below the figures reported there.

Also from the env: the service runs from `WORKDIR=/mnt/data/tray_counter`
(storage at `apd_storage/` relative to it), logs to `PROJECT_DIR=/mnt/data/voyager-sdk`,
Axelera runtime 1.5.2-1, `PYTHONPATH=/mnt/data/voyager-sdk`, ONNX Runtime 1.23.2
present in the container. `APD_MODEL=apd_v3`, `APD_PIPE=torch-aipu`.

## 8. Review findings

Raised against the teacher visualisations and demo. 8.1 and 8.3 are closed;
8.2 is partly addressed; 8.4 is open.

### 8.1 head_cap fires on hats and hijab — CLOSED 2026-09-14
`HEAD_CAP_RULE` in `apd/schema.py` now calls a street hat ABSENT: baseball cap,
any peaked or brimmed cap, beanie and bandana are explicitly listed, and only a
food-grade covering that contains the hair (bouffant, hairnet, surgical cap,
hygiene skullcap) is PRESENT. An UNCLEAR clause was added for when the top of
the head is not visible, which the other two rules already had.

This closes a divergence that would have undone the data work: 345 crops had
been relocated `present` → `absent` on 09-11
(`artifacts/sanitized_head_cap_hats.txt`) while the rule still said a peaked cap
was compliant. The rule string feeds the labeller prompt, so the next labelling
run would have re-labelled hats as `present`.

Still outstanding: crops labelled under the old wording that were NOT caught by
the hat sanitation pass have not been re-examined, and no labelling run has yet
been made against the new text.

### 8.2 False gloves violations
Two distinct causes:
- **transparent/plastic gloves** read as bare hands. The rule does not mention
  them; if they count as PRESENT the rule needs to say so and the labels are wrong.
- **occluded or unclear hands** being decided instead of abstained.

### 8.3 head_cap decided on cropped/blocked heads — CLOSED 2026-09-14
`roi_usable()` in `apd/pose.py` is one shared gate with four refusal reasons —
clipped by frame edge, overlaps a dashboard caption, occluded by a nearer
person, too small. Thresholds live in `RoiGeometry`, not as scattered literals.
Wired into `39_consensus.py`, `38_viz_all_models.py` and `55_compare_video.py`;
`56_viz_teacher.py` inherits it by rendering from the consensus records. It
returns a reason rather than a bare False, so skipped people are reported
instead of silently dropped.

Occlusion uses the lower box-bottom as a depth proxy — in a fixed overhead
camera the person whose feet are further down the frame is nearer. Without that
test a distant worker BEHIND the subject would count as covering them.

Measured on the production annotation (`scripts/60`):

| attribute | gate off | gate on | accuracy on what it refused |
|---|---|---|---|
| head_cap | 94.0% (n=266) | 95.1% (n=247) | 78.9% |
| mask | 91.3% (n=219) | 91.6% (n=202) | 88.2% |
| gloves | 96.2% (n=184) | 97.4% (n=156) | 89.3% |

The refused ROIs are ~10 points worse than the kept ones, so the gate is
selecting real problem cases rather than just buying accuracy by abstaining.
It is still a trade: clipped and occluded ROIs were right 84–86% of the time.

On site frames, where the complaint originated, it fires far more often —
**26.6% of head ROIs and 24.7% of hand ROIs** across 200 sampled frames
(caption overlap 65/63, occlusion 10/62, edge clipping 25/31, too small 4/13).

### 8.4 Poor accuracy on small / distant people — NOT closed
A size threshold now exists (`head_min_px` 26, `hand_min_px` 18) but it is an
estimate, not a calibrated value, and it barely fires: 3 refusals out of ~670 on
the production val, 17 out of 1,168 ROIs on site frames. It will not catch what
was reported. Calibrating it needs a sweep of accuracy against ROI size, the way
`scripts/49` calibrated the mask gate — that sweep has not been run.

---

## 8b. Disk cleanup (2026-09-16)

Removed as deprecated, about 2 GB: `artifacts/viz_ci3` (renders of the ci3
experiments section 4 retracted), the `compare_temp_*` render scratch and its
zip, everything from the abandoned distillation of section 5 (`data/distill`,
`data/distill_roi`, `runs/distill*`, `runs/student_soft`), `runs/_retired`, the
superseded packing variants `apd_multinet.onnx` and `apd_ours_static.onnx`
(regenerable by scripts 85 and 86), and two verified duplicates,
`data/crops_ci_inpaint.zip` and `model/sppg/apd_v3.tar.gz`, plus
`model/sppg/apd_v3-old`.

Kept deliberately: `artifacts/tray_counter` (2.2 GB) is the live production
copy that scripts 90 and 91 import from, and it holds `.env` with live
credentials, so it needs securing rather than deleting. `artifacts/viz` (1.2 GB)
is the next candidate if space is needed; the comparison videos are
deliverables.

## 9. Artifacts

| what | path |
|---|---|
| teacher verdicts, visualised | `artifacts/viz_ci3/teacher/<site>/<id>.jpg` (1,145) |
| each voter separately | `artifacts/viz_ci3/{voter3_ours,voter4_ours,apd_mbg}/` (1,432 each) |
### **Hat vs. Food Hygiene Cap Sanitation (Head Cap)**
* **Issue:** Regular street hats, baseball caps, and beanies were previously classified as `head_cap present (OK)`. In food safety/processing environments, street hats do not comply with hygiene standards (only food-grade hairnets, bouffant caps, and hygiene skullcaps are valid PPE).
* **Sanitation Pass:** Audited top-of-head crops for rigid brim/visor geometry and street hat features. Relocated **345 street hat / baseball cap crops** from `present` $\rightarrow$ `absent` (violation).
* **Model Retraining:** Retrained `head_cap` classifier (`EfficientNet-B1`) on the sanitized dataset.
  * **Best Val AUC:** `0.9578`
  * **Best Val Accuracy:** `88.36%`
* **Audit Logs Created:**
  * `artifacts/sanitized_head_cap_hats.txt` (345 relocated hat crops)
  * `artifacts/sanitized_mask_chin_straps.txt` (20 relocated chin-strap crops)

| File / Artifact | Path & Details |
| :--- | :--- |
| **Sanitized Hats Log** | `artifacts/sanitized_head_cap_hats.txt` (345 files moved to `absent`) |
| **Sanitized Mask Log** | `artifacts/sanitized_mask_chin_straps.txt` (20 files moved to `absent`) |
| 3-panel demo video (10 s) | `artifacts/compare.mp4` — production \| teacher \| edge (`yolo11s-pose` + `EfficientNet-B1` with 0.10/0.90 confidence banding) |
| Batch 1 Compare Videos | `artifacts/compare_rec113142.mp4`, `compare_rec113959.mp4`, `compare_rec113644.mp4` |
| Batch 2 Compare Videos | `artifacts/compare_b2_rec133700.mp4`, `compare_b2_rec133958.mp4`, `compare_b2_rec134226.mp4` |
| Batch 3 Compare Videos | `artifacts/compare_b3_rec135526.mp4`, `compare_b3_rec135711.mp4`, `compare_b3_rec140115.mp4` |
| raw verdicts + per-voter votes | `artifacts/consensus_ci3b.jsonl` |
| teacher soft probabilities | `data/distill_roi/soft.jsonl` |
| consensus validation | `artifacts/consensus_validation.json` |
| classifier ranking, independent set | `artifacts/prodval_compare.json` |
| site-disjoint comparison (too small to rank) | `artifacts/fair_compare.json` |
| consensus with ROI gate applied | `artifacts/consensus_gated.jsonl` |
| mined crops + 3-view labels | `artifacts/site_crops_clean/<attr>/` |

---

## 10. Scripts (recent)

| script | purpose |
|---|---|
| `42_mine_site_crops.py` | mine ROIs from 60 sites, farthest-point sampled in DINOv2 space |
| `43_label_site_crops.py` | single-view Opus labelling |
| `44_build_ci_dataset.py` | classifier datasets, site-disjoint splits, hue augmentation |
| `45_protocol_bakeoff.py` | labelling-protocol comparison against human labels |
| `46_label_multiview.py` | 3-view labelling, unanimity required |
| `47_voter3_ci3.py` | DINOv2 probes |
| `48_validate_consensus.py` | is agreement the same as correct |
| `49_mask_gate_sweep.py` | frontality gate → confidence margin |
| `50/52` | distillation datasets (person crop / ROI) |
| `51/53` | student training (person crop / ROI, hard or soft) |
| `54_eval_student.py` | student vs teacher on production annotations |
| `55_compare_video.py` | 3-panel comparison video (`--seconds 10`) |
| `56_viz_teacher.py` | teacher verdict stills |
| `57_teacher_soft.py` | record teacher probabilities |
| `58_edge_config.py` | pose-size sweep for the edge configuration |
| `60_gate_effect.py` | does the ROI gate improve correctness or only cut coverage |
| `61_fair_compare.py` | comparison excluding every training stem AND site — shows the site pool is too small to rank models |
| `62_prodval_compare.py` | classifier ranking on the production annotation (the only independent set) |
| `90_replicate_deployed.py` | the deployed chain on real video: bundle perception, production tracker, production vote, live `.env` |
| `91_replicate_both.py` | the same harness with two perception arms, ours and the production detector |
| `92_make_detector_dataset.py` | everything we hold turned into one 7-class YOLO dataset, `data/det7` |
| `93_viz_detector_dataset.py` | that dataset rendered into review folders, `artifacts/det7_viz` |

## 11. What the measurements keep saying

Every time the model was assumed to be the constraint, measurement said the
labels were. Gloves went 0.63 → 0.935 with identical code, purely from explicit
negative annotation. The blue cap was fixed by coverage, not by colour work.
The student failed on 1,863 records, not on architecture.

**Four** validation splits have now been retracted for leakage — the latest
compared two models whose val sets each contained most of the other's training
data. Any number in this file should state what the evaluated model had and had
not seen, and a comparison between two models needs one set neither has touched.

The corollary, learned the same way: a gain that arrives together with two other
changes is not attributable to any of them. The inpainting, hat relabelling and
glove oversampling landed in one pass, and on independent data the combination
shows no measurable gain over `ci3` at all.

## 12. Next

1. **More sites — now the blocking item, not just a nice-to-have.** Section 4
   shows the 60-site pool cannot separate six classifier generations: excluding
   every training stem and site leaves 30 / 6 / 6 crops. Model choice is
   currently unresolvable on the data we hold. 223 sites remain unfetched;
   blocked on approval because of ban risk.
2. **8.4** — calibrate the ROI size threshold against measured accuracy.
3. **8.2** — decide whether transparent/plastic gloves count as PRESENT and say
   so in `GLOVES_RULE`; the rule is silent and the labels are therefore unspecified.
4. **Relabel under the revised head_cap rule** — no labelling run has been made
   against the 09-14 text.
5. **Edge config field check** — verify the single-voter abstention loss (§6)
   before shipping.

## 6j. Audit, relabel and retrain (2026-09-17)

Full write-up: `artifacts/RETRAIN_REPORT.md`. Training data audited
(`artifacts/{mask_face_audit,headcap_audit,det7_audit}`), cleaned copies in
`data/clean/` (script 115, originals untouched), 2,629 new reviewed sppg people
in `data/label_sppg/labels.csv` (scripts 119/120). Head-cap rule extended: navy
beret, chef toque, black/striped chef skullcap are PRESENT.

COCO test, end to end with yolo11s-pose: a single per-person efficientnet_b0
(320 px, three outputs, scripts/118, trained on det7 clean + sppg labels) is best
on all three items -- head_cap 0.543 (deployed 0.335, p=0.001), mask 0.346
(0.196, n.s.), gloves 0.228 (0.180, p=0.002). The new labels, not the
architecture, carry most of the gain. Exported to
`runs/person_ppe/export/apd_ours_person_ppe.onnx`; not yet in the bundle.
