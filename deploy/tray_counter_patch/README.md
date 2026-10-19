# Tray counter fixes

Three fixes to the tray counter's `tlabcode/`. These are the "Patched" results on the
Tray Counter Dry Run page (https://claude.ai/artifact/BkMJnYtAMjNfGsEc6h6x1k).
The original production snapshot in `artifacts/tray_counter/` is not modified.

| File | Change |
|---|---|
| `tlabcode/line_counter.py` | IN/OUT comes from which side of the ROI line the tray moved from and to (`recent_sides`), not from the average motion of everything on screen. Fixes: a returned tray counted OUT when nothing else moves, and a pulled-back under-filled tray logging a second violation. |
| `tlabcode/tracker.py` | Tracks match across classes, and the tray's class is chosen by majority vote. Fixes: a fill-level flicker starting a new track, which caused double counts and false violations. |
| `tlabcode/tracker.py` | Overlap is measured against each track's predicted box (last box moved by its velocity for the frames since it was seen), and tracks seen in the previous frame are matched before stale ones. Fixes: on a fast belt, tracks left behind by earlier trays took over the next tray's ID, so its crossing was never counted (8 of 95 missed at 4x belt speed, now 3). |

`tray_counter_fixes.diff` is the unified diff against the originals.

## Deploy

Copy the two files over `tlabcode/line_counter.py` and `tlabcode/tracker.py` on the device
(`/mnt/data/tray_counter/tlabcode/`) and restart the `counter-tray` container. No config change is needed.

**Check each camera's line orientation before relying on IN/OUT.** OUT is now the side the line's normal
points to, meaning the right-hand side when walking from the first ROI point to the second. The old code
ignored point order. If a camera reports IN and OUT swapped, swap the two line points in the dashboard.

## Simulation (`sim/`)

- `crops.py` cuts tray sprites from the `tc_2272_v11s_640` val set into `sim/sprites/`.
- `sim.py --runs 30` gives production-vs-patched stats per detector profile.
- `sim.py --runs 0 --render realistic --variant patched` renders the annotated video.
- `verify_disk.py` runs the files in this folder and compares them with `results_30runs.json`.

In `sim.py` the "patched" variant applies the first two changes as in-memory text edits, so it
tests the logic without touching files. `verify_disk.py` runs the files in this folder, all three fixes included.
With the third fix it gives the same results within run-to-run noise (realistic: 45.9 of 46 crossings right vs 45.8;
harsh: 45.2 vs 45.1, violations caught 6.9 vs 7.0).

## Test on the s14 conveyor render (`s14/`)

Test video: `artifacts/tray_counter/data/s14_all_cases_elev45_az35_dist1700_hfov55.mp4`. The real model isn't
available, and YOLOE/YOLO-World were not usable on this render, so `det.py` stands in for it: background subtraction
against `bg.png` (fixed camera), watershed, then a size-based split of touching trays. `fill.py` maps the share of
food-coloured pixels to a fill class. Stacks are not recognised.

Run from `s14/`: `python detect_all.py`, then `python run_counter.py [--render]`, then `python reference.py`.

Counting line: `[(485, 524), (794, 210)]`, rail to rail across both lanes. It follows the belt's true cross direction
in this perspective view: the belt edges meet at the vanishing point (-1840, -868), which sits on the horizon of a
45-degree camera, so the cross-belt vanishing point is (1858, -868) and the line runs through it. That makes it
parallel to the belt slats and the trays' short edges. See `make_line()` in `run_counter.py`.

| | Reference | Production | Patched |
|---|---|---|---|
| OUT / IN | 92 / 3 | 89 / 5 | 91 / 3 |
| Crossings with the right direction (of 95) | 95 | 87 | 94 |
| Missed / extra crossings | | 2 / 1 | 1 / 0 |
| Violations caught (of 26) / false violations | | 21 / 2 | 23 / 1 |

Production's six wrong directions (40.9 s twice, 154.9, 169.9, 185.7 and 230.6 s) were checked frame by frame, and
the patched result was correct each time. The one patched miss, at about 260 s, comes from the stand-in detector
splitting a stacked tray into overlapping boxes. The reference picks up an extra crossing from those boxes that is
not a real tray.

An earlier run used a shorter, steeper line, `[(563, 608), (702, 288)]`: 94 reference crossings, production 89
right, patched 94 right.

### At 4x belt speed

`s14_x4.mp4` is the same render with every 8th source frame kept, played at 15 fps. The belt runs 4x faster
(about 25 px per frame instead of about 6) and the clip is 71 s. The stand-in detector was run on that clip
(`python detect_clip.py s14_x4.mp4 dets_x4.json`), and the counters on its detections
(`python run_counter.py --dets dets_x4.json --video s14_x4.mp4 --tag _x4 --src-step 1 [--render]`).

Scoring is done by the tuner app's engine (`tuner-app/src/engine.ts`, which gives exactly the same events as the
Python; `npm run test:engine` checks that). The reference is the 1x offline tracks with their times divided by 4,
minus tracks shorter than 60 frames (the only one is a junk box the stand-in draws beside a stacked tray), with
a stacked tray tracked twice counted once: 93 crossings. Counts are matched to reference crossings by where the
tray crossed the line (so two trays crossing together in two lanes stay apart) and by time. A count may come
from one frame before to 1.5 s after the crossing; the counter's smoothing makes it count about 0.4 s late, and
trays in one lane are only 0.6-0.9 s apart, so nearest-time matching would pair counts with the wrong tray.
(`score_speed.py` is the earlier time-only scorer and over-reports misses.)

| 93 reference crossings | Production | Patched (all three fixes) |
|---|---|---|
| Right direction | 85 | 91 |
| Wrong direction | 2 | 0 |
| Missed | 6 | 2 |
| Extra counts | 0 | 1 |
| Violations caught (of 25) / false | 22 / 2 | 23 / 1 |

Changing the `.env` tuning settings barely helps (the sweep is `tuner-app/scripts/sweep.ts`). Production's misses
come from the tracker: a tray that leaves a track behind keeps that track alive for 30 frames at its last box. At 4x,
the next tray lands on those stale boxes, overlaps them more than its own track's box from one frame earlier, and
takes over their IDs. Its ID then hops between old tracks, none of which sees a side change, so the crossing is
never counted. The third fix (predicted boxes, live tracks matched first) removes these. Patched's 2 misses are a
tray going back upstream while another moves downstream (36.5 s) and a tray crossing 0.4 s before the clip ends;
its extra count is the junk box at 65.4 s.
