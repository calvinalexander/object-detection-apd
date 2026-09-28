# Person-crop classifier (v5 / v3) inside the production `apd-cctv` service

Installed INTO the running `apd-cctv`: no second container, no second camera
session, no second AIPU network, no compile. Production's pipeline runs as it
does today (`apd_v3` on the AIPU, person threshold, ByteTrack,
`assign_apd_smart`, vote window, `MIN_AGE_FRAMES`, cooldown). The one change:
when that vote fires an event, the track's person box is cropped from the
**clean stream frame** (+18% pad, letterbox 320) and judged by the person-PPE
network on the CPU. This is the decision measured offline in
`artifacts/filter16_v5` (held out: v5 140/219 catches at 78/444 false alarms,
v3 131/219 at 115/444); the offline run cropped the saved snapshot, which has the
table drawn on it, while here the crop has no overlay.

| classifier says | `filter` (default) | `shadow` |
|---|---|---|
| any item `violation` | event saved and uploaded as usual; `violation` = the classifier's items, snapshot + `apd_detail` show its verdicts | production's event, unchanged |
| nothing is a violation | **no event row, nothing uploaded**; clean frame + verdicts to `apd_storage/snapshots_passed/<cam>/` | production's event, unchanged |
| crop unusable / error | production's event, unchanged (fail open) | same |

Production's cooldown and dedupe state advance exactly as before, so each person
is judged once per cooldown -- the same unit as the offline numbers. Every event
the detector fires is one line in `apd_storage/person_filter.jsonl` (production's
vote, classifier verdicts, six scores, kept/passed), in both modes.

## What is recorded (under `apd_storage/`)

| path | when | content |
|---|---|---|
| `person_filter.jsonl` | every fired event, both modes | event_id, camera, track, `person_bbox`, production's vote, classifier verdicts, six scores `p`, `decision` (kept/passed/unscored), `crop_path` |
| `person_crops/<cam>/<event_id>.png` | every scored event, both modes | the padded crop the network judged, native resolution, no overlay. PNG on purpose: JPEG q92 moved scores by up to 0.23 on these ~60 px crops; the PNG re-scores to the logged `p` (within the log's 4-decimal rounding) |
| `snapshots/<cam>/<event_id>.jpg` | kept events (filter) / all events (shadow) | production's snapshot; in filter mode drawn with the classifier's verdicts. Uploaded, then deleted by production's retention |
| `snapshots_passed/<cam>/<event_id>.jpg` | passed events, filter mode | full frame drawn with the classifier's verdicts; never uploaded |

`person_crops/` and `snapshots_passed/` are not in the DB, so production's
retention never deletes them: `DAYS=7 steps.sh prune`. Crops are about 35 KB each.

## Bundle

| file | role |
|---|---|
| `patch_monitor.py` | adds the filter to production's `apd_monitor_axelera_multi.py` (31 lines); every anchor asserted |
| `apd_person_filter.py` | crop, ONNX Runtime call, cut rule, log |
| `apd_ours_person_ppe_v5.onnx/.json` | default; cuts fitted on the manual curation |
| `apd_ours_person_ppe_v3.onnx/.json` | v3, with the 2026-09-22 mask confident-absence override |
| `steps.sh` | check / install / mode / model / status / logs / prune / rollback |

    rsync -avz deploy/apd_prodflow/ root@<sppg>:/mnt/data/apd_prodflow/

## Install

    bash /mnt/data/apd_prodflow/steps.sh check                  # read-only: shows the exact diff it will apply
    MODE=filter MODEL=v5 bash /mnt/data/apd_prodflow/steps.sh install
    bash /mnt/data/apd_prodflow/steps.sh status

`install` does, in `<compose project>` (`/mnt/data/tray_counter`):

1. keeps production's monitor as `apd_monitor_axelera_multi.py.pre_person` (once; a re-install patches that, never a patched file)
2. re-derives the monitor from it with production's own python, inside production's image
3. copies `apd_person_filter.py` beside it and the models into `apd_person/`
4. sets `APD_PERSON_MODE` and `APD_PERSON_ONNX` in `.env` (original kept as `.env.pre_person`)
5. recreates `apd-cctv` (`docker compose up -d --force-recreate --no-deps apd-cctv`) -- a restart does not re-read `.env`

The worker log then shows
`Perception: PRODUCTION (apd_v3) + person filter model=apd_ours_person_ppe_v5 mode=filter cuts={...}`.

## Operate

| | |
|---|---|
| `steps.sh mode shadow` / `mode filter` | switch behaviour (recreates apd-cctv) |
| `steps.sh model v3` / `model v5` | switch model (recreates apd-cctv) |
| `steps.sh rollback` | production's original monitor back, recreate |
| `DAYS=7 steps.sh prune` | delete old `snapshots_passed/` frames and `person_crops/` |

If production's monitor is updated later, `install` again: it sees an
unpatched file, keeps it as the new original and patches that. If an anchor no
longer matches, the patch fails before anything is copied.

## Cost and limits

- One CPU inference per fired event (a few per 10 minutes), not per frame.
- The filter only removes or relabels production's events; a person production
  never votes on is never judged.
- v5 export checked on the 16 Sep events against the offline scores: max score
  difference 0.008 (CPU vs GPU), 1 of 3,104 events flips.
