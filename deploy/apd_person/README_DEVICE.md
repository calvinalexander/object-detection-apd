# Our model through the production flow, on SPPG-3881

Same image, same supervisor (`apd_monitor_axelera_service.py` -> `StreamSupervisor`),
same ByteTrack, vote window, cooldown, SQLite schema and snapshot code. Only the
perception stage is ours. Production's files are mounted read-only and never edited.

How the swap works: `worker_entry` does `from apd_monitor_axelera_multi import
apd_monitor_main`. The supervisor script is launched from THIS directory, so the
import resolves to our patched `apd_monitor_axelera_multi.py` (derived from the
live file by `patch_monitor.py`, 70 lines changed, all anchors asserted).
Verified off-device with the supervisor's exact `SimpleNamespace` and the live
`apd_utils` / `apd_database` / `apd_tracker` / `apd_config`.

## Bundle (copy the whole directory to `/mnt/data/apd_ours` on the host)

| file | role |
|---|---|
| `apd_monitor_axelera_multi.py` | live monitor, patched (perception swapped) |
| `apd_monitor_axelera_service.py` | supervisor entry, byte-identical copy of production's |
| `apd_perception_ours.py` | pose -> anatomical ROI -> packed classifiers; `assign_ours()` |
| `apd/` | ROI geometry + gate |
| `apd_ours_head.onnx`, `apd_ours_hand.onnx` | classifiers, run on CPU via ONNX Runtime 1.23.2 (in the image) |
| `yolo11s-pose.onnx` + `config.yaml` | pose network to compile for the AIPU (since 2026-09-17; `yolo11n-pose.onnx` kept for rollback) |
| `docker-compose.apd-ours-test.yml` | overlay service `apd-cctv-test` |
| `patch_monitor.py` | re-derive the monitor if production's file changes |

    rsync -avz deploy/apd_ours/ root@<sppg>:/mnt/data/apd_ours/

## 1. Compile the pose network (once)

`customers/` and `build/` inside the container are bind-mounts of
`/mnt/data/tray_counter/{customers,build}` on the host, so:

    mkdir -p /mnt/data/tray_counter/customers/apd_ours
    cp -r /mnt/data/apd_ours/config.yaml /mnt/data/apd_ours/yolo11s-pose.onnx /mnt/data/apd_ours/dataset /mnt/data/tray_counter/customers/apd_ours/
    # compile in a THROWAWAY container, not inside the live apd-cctv: the compiler
    # uses every core and up to 8 GB, which would starve the running worker.
    docker run --rm --name apd-ours-compile --cpus=4 \
      -v /mnt/data/tray_counter/customers:/mnt/data/voyager-sdk/customers \
      -v /mnt/data/tray_counter/build:/mnt/data/voyager-sdk/build \
      -v 43bb1a0ee8fcad73a94b83cae2fd5971a4adb298443b5c14847135ea5e9073eb:/home/cpu-tower/.local \
      sipgn-git.bgn.go.id:axelera-v1.5 \
      bash -lc "cd /mnt/data/voyager-sdk && $PY ./deploy.py apd_ours --model apd_ours_pose --metis pcie 2>&1 | tail -40"
    # $PY = the interpreter production runs, e.g. PY=$(docker inspect apd-cctv --format '{{index .Config.Cmd 0}}')
    #  (the venv path differs per box; "source venv/bin/activate" is NOT portable). Prefer: bash sppg_steps.sh compile
    # --model X alone = "compile only X, no pipeline deploy" (mutually exclusive
    #  with --models-only). --metis pcie names the target board so deploy.py does
    #  NOT need to open the Metis card: without it, compile-only fails with
    #  "failed to open device 0 metis-0:1:0" in a container that has no /dev --
    #  and we deliberately give the compile container no /dev, so it can never
    #  touch the card the live service is using.

`dataset/` (200 kitchen frames + `calib.json` with COCO-17 keypoints) ships in
the bundle; it is only used to calibrate the int8 quantisation. If the compile
fails, send back the last 40 lines it prints.

Output lands in `/mnt/data/tray_counter/build/apd_ours/`. Budget: 0.25 (one
core's resources) -- the same as `apd_v3`. The classifiers are not compiled.

## 2. Start beside production

From the compose project directory (`/mnt/data/tray_counter`), with the overlay
file copied next to the live compose file:

    cp /mnt/data/apd_ours/docker-compose.apd-ours-test.yml /mnt/data/tray_counter/
    cd /mnt/data/tray_counter
    docker compose -f docker-compose-axelera-x86-64.yml -f docker-compose.apd-ours-test.yml up -d apd-cctv-test
    docker logs -f apd-cctv-test

It reads the same `.env` (camera config, device id, thresholds, S3/API keys) so
the flow is identical -- except:

| | production | apd-cctv-test |
|---|---|---|
| `APD_MODEL` | apd_v3 | **apd_ours** |
| `APD_STORAGE_PATH` | apd_storage (under tray_counter) | **/mnt/data/apd_ours/apd_storage** |
| snapshots | tray_counter/apd_storage/snapshots | **/mnt/data/apd_ours/apd_storage/snapshots** |
| upload | apd-api reads the production DB | **never sees our DB** |
| `AXELERA_CONFIGURE_DEVICE` | unset | **0** (do not reconfigure the card) |
| restart | always | **no** |
| tray_counter mount | rw | **ro** |

The supervisor will fetch the same camera from the config API and open a second
RTSP session on it. That link already shows HEVC `Could not find ref with POC`
loss; if production degrades, stop the test container first.

Live env values our run inherits: `APD_MIN_AGE_FRAMES=10`, `APD_COOLDOWN_MS=600000`,
`APD_VOTE_WINDOW=5`, `APD_MIN_VIOLATION_RATIO=0.6`. Expect one event per person
per 10 minutes, by design.

## 3. Check

    ls /mnt/data/apd_ours/apd_storage/snapshots/*/ | wc -l
    sqlite3 /mnt/data/apd_ours/apd_storage/apd_events.db "select ppe_type,status,count(*) from apd_detail group by 1,2;"
    ls /mnt/data/tray_counter/apd_storage/snapshots/masak/ | wc -l     # production must keep rising as before
    docker exec apd-cctv-test axdevice                                  # read-only

## Known limits

- Gloves is the weak attribute (MCC 0.155 on human labels; dark gloves worst).
- `config.yaml` classifier entries are unused in this route; they are there for
  the alternative of compiling the classifiers onto the AIPU (0.75 total).
- Shutdown prints `cvDestroyAllWindows ... not implemented` on headless OpenCV
  builds; it is production's own `finally:` and occurs after everything is saved.


## Classifier file (updated)

`apd_ours_packed.onnx` holds all three trunks in one graph and takes a whole
frame's head crops and hand crops as two batches, so the CPU does one call per
frame instead of three per person. Outputs are bit-identical to the two older
files (`apd_ours_head.onnx`, `apd_ours_hand.onnx`), verified on 81 people across
40 SPPG frames: zero verdict differences. The two files stay as a fallback and
are used automatically if the packed file is missing.

Thread count is `APD_OURS_THREADS`, default 2. Four threads oversubscribe a
4-core host once the decoder and pose stage are also running.

## Calibration pool for the compile

By default the SDK quantises against its own 400 generic COCO photos. To use the
200 SPPG kitchen frames in `dataset/images/` instead:

    CAL=ours bash sppg_steps.sh compile

`bash sppg_steps.sh verify` prints which pool the finished build actually used.

## Mask model (2026-09-17)

`apd_ours_mask.onnx` is a dedicated mask trunk (EfficientNet-B0, fine-tuned from
the previous mask trunk) that judges a **face crop** instead of the shared head
crop. The head crop is sized for caps (1.9x shoulder width, lifted), which left
the mouth a few pixels of the model's input. The face crop comes from
`apd.pose.face_roi`: 0.8x shoulder width, centred on the nose, pushed toward the
mouth. It needs no new pose output and no recompile.

Requires the updated `apd/pose.py` and `apd/schema.py` in this bundle.

| switch | default | effect |
|---|---|---|
| `APD_OURS_MASK_ONNX` | `apd_ours_mask.onnx` | `""` falls back to the packed head trunk |
| `APD_OURS_MASK_THRESH` | 0.30 with the face trunk, 0.60 without | "ok" when p(mask) >= this |
| `APD_OURS_GLOVES_ONNX` | `apd_ours_gloves.onnx` | `""` falls back to the hand-ROI trunk |

The startup log line reports which path is live, e.g.
`mask=face-crop trunk gloves=crop-detector thresh={'headcap': 0.5, 'mask': 0.3, 'gloves': 0.5}`.

Deploying it: copy the files, then `bash sppg_steps.sh stop` and `start`. No
`place`, no compile.

## Mask facing gate (2026-09-17)

Mask is not judged on people whose mouth cannot be seen. A person passes when
the nose keypoint confidence is at least `APD_OURS_MASK_MIN_NOSE` (0.5) and,
with `APD_OURS_MASK_REQUIRE_FACING=1`, neither the shoulders nor the ears are in
turned-away order (facing the camera, the left side appears on the image's
right). Otherwise mask is `unknown`, which never alerts; head_cap and gloves are
still judged. On human labels this cut mask violation alerts on people whose
mouth nobody could see from 60 to 22. `0` and `0` restore the old behaviour.
The startup log shows it as `mask_gate=(nose>=0.5, facing=True)`.

## Pose model: yolo11s-pose (2026-09-17)

The compiled pose network is now `yolo11s-pose.onnx` (9.9 M params, 23.2 GFLOPs,
about 3x yolo11n). Same input/output as before ([1,3,640,640] -> [1,56,8400]),
same model name `apd_ours_pose`, same calibration images, so only a recompile is
needed:

    bash sppg_steps.sh place      # copies config.yaml + yolo11s-pose.onnx + dataset
    bash sppg_steps.sh compile
    bash sppg_steps.sh stop && bash sppg_steps.sh start

Why: on 316 recording frames at conf 0.40, s found 98 people against 82 for n,
and the extra ones are real (bent over, partly hidden); n found only 2 that s
missed. On the COCO-labelled frames s locates 62% of people against 49%.
Cost: about 20% more people through the CPU classifiers. Production's own
detector is yolo11s-sized on the same Metis, so AIPU load is comparable.

Rollback: set `weight_path: yolo11n-pose.onnx` in config.yaml, then place +
compile again.
