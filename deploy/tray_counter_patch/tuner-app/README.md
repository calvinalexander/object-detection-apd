# Tray Counter Tuner

A self-hosted web app (Vite + React + TypeScript) for setting up the tray counter. You can drag the counting line, flip its OUT side, change which classes may go out, and move the `.env` tuning sliders. Each change reruns the whole s14 conveyor clip, sped up to 4× belt speed (71 s), in the browser and scores the counts against an offline reference.

The app is fully static and needs no backend. `dist/` is about 2.5 MB, including the video.

## Run

Needs Node 18+, and for processing uploads Python 3 with OpenCV and NumPy, plus ffmpeg.

```bash
npm install
npm run build
npm start            # http://<host>:5180, serves the app and saves runs in ./runs
```

For development, run the server and Vite side by side; Vite proxies `/api` and `/runs` to the server:

```bash
npm run dev:server   # API on :5181
npm run dev          # app with hot reload on :5173
```

Or use Docker. The `-v` keeps saved runs when the container is replaced:

```bash
docker build -t tray-tuner . && docker run -p 5180:5180 -v tray-runs:/app/runs tray-tuner
```

`dist/` on its own still works on any static host, but then only the built-in run is available: no uploads,
no saving. The server has no login, so only expose it on a network where everyone may upload and delete runs.

Settings: `PORT` (5180), `HOST` (0.0.0.0), `RUNS_DIR` (`./runs`), `PYTHON` (`python3`), `MAX_UPLOAD_MB` (2048),
`TUNER_KEEP_WORK=1` to keep the 1280×720 working copy of each upload.

## Runs

A run is a video plus its detections and, optionally, saved settings. The built-in run is the s14 render at 4×
belt speed, which has a reference to score against. Uploaded runs have no reference, so they show counts only.
Turn on **Side by side** to see where production and patched disagree, and check those moments by eye.

An upload is processed in the background by `server/process_run.py`, one at a time, at roughly twice the video's
length:

1. It converts the video to what `recorder.py` writes: 1280×720, 15 fps (letterboxed if the aspect ratio differs).
   It also writes a 640×360 copy for the browser.
2. It builds a background from the 20th percentile of 120 frames, not the median. Trays are brighter than the belt,
   and on a busy belt they cover each spot more than half the time, so a median would bake trays into the
   background. The camera must be fixed.
3. It measures one tray's size: the most common blob size, since most blobs on a belt are single trays.
   It scales the clean-up kernels, the seed erosion that separates touching trays, and the smallest blob kept
   (a quarter of a tray) to that size. At s14's scale (about 24,000 px² per tray) this gives almost the
   hand-tuned s14 settings. On a far camera (about 6,600 px² per tray) the old fixed settings dropped single
   trays and merged neighbours, which made boxes blink.
4. It fits one tray's area against image height to split touching trays, and falls back to the measured size
   when the fit is implausible.
5. It runs the stand-in detector on every frame. Fill level comes from the share of food-coloured pixels.
   Stacks are not recognised.

**Download** (per run):
- **Original video:** the uploaded file, as uploaded.
- **Annotated video:** an MP4 rendered on the server by `server/render_run.py` with the same overlay as the app,
  for the version shown or both side by side. It uses the line and settings on screen, saved or not, at
  1280×720 (1920×540 side by side). It takes a few seconds per minute of video, and each run keeps its 3 newest renders.
- **Counts (CSV):** every counted tray: version, time, frame, direction, fill class, violation, track id.

**Expected results:** `python scripts/attach_expected.py <expected_result_tray_v4.xlsx>` attaches the expected
crossings to each run whose file is named `kasusNN_…` (or whose run name is `case NN`). Side by side then shows a small
**EXPECTED** overlay between existing and patched: the running OUT / IN / violation count at the current video
time. Side-by-side downloads include it too. It is the renderer's list, not ground truth.

**Reprocess** runs detection again on an upload, for example after the detector changes. Saved settings are kept.

**Save settings** stores the line, version, allowed classes and tuning with the run (`runs/<id>/settings.json`).
They load next time anyone opens that run. New uploads start with a vertical line through the middle. If
patched counts most trays IN, the app says so, because that usually means the line's OUT side is flipped.

| API | |
|---|---|
| `GET /api/runs` | list runs, newest first |
| `POST /api/runs?name=…` | body = the raw video, header `x-filename`; returns the new run |
| `GET /api/runs/:id` | one run, with processing progress |
| `PUT /api/runs/:id/settings` | save settings (JSON) |
| `POST /api/runs/:id/render` | render an annotated MP4 from the overlay the browser counted; returns its URL |
| `GET /runs/:id/original` | the uploaded file (download) |
| `POST /api/runs/:id/reprocess` | run detection again on an upload (keeps saved settings) |
| `DELETE /api/runs/:id` | delete an uploaded run |
| `GET /runs/:id/{video.mp4,poster.jpg,data.json}` | run files (Range supported) |

## What's inside

| File | Purpose |
|---|---|
| `src/engine.ts` | Port of `tlabcode/tracker.py` and `tlabcode/line_counter.py`, both production and patched. It also holds the reference crossings and the scoring. |
| `src/Panel.tsx` | One view of the video: frame, boxes, counts and the draggable line (two of them side by side) |
| `src/Transport.tsx` | Play/pause, speed and the timeline, one row per version shown |
| `src/api.ts` | Client for the server's run API |
| `server/server.mjs` | Serves the app and run files, stores uploads, queues processing |
| `server/render_run.py` | Draws the overlay onto the video for the annotated download |
| `server/process_run.py` | Turns an upload into a run: convert, background, tray-size fit, detections |
| `src/App.tsx` | Runs, upload, settings panels, results, side-by-side comparison, event lists |
| `src/settings.ts` | Device defaults (`.env`), class names, default line |
| `public/s14_x4_data.json` | Stand-in detections on the 4× clip at 15 fps, plus the 1× offline reference tracks at a quarter of the time |
| `public/s14_x4_640.mp4` | The 4× clip at 640×360, 15 fps |
| `npm run build:standalone-engine` | Bundles `src/engine.ts` as a plain script (global `TrayEngine`) for the single-file claude.ai version of the tuner |
| `scripts/sweep.ts` | Prints the score for a range of tuning settings on one clip |
| `scripts/check-engine.ts` | Checks that the port gives the same events as the Python |
| `scripts/prep_data.py` | Rebuilds `s14_x4_data.json` from `../s14/dets15.json` and `../s14/dets_x4.json` |

Check the port against the Python counter:

```bash
npm run test:engine
# production events=87 python=87 identical=true right=85/93 ...
# patched    events=92 python=92 identical=true right=91/93 ...
```

## Limits

- The boxes come from the background-subtraction stand-in, not tc_2272. Fill classes are rough, and stacks never appear.
- `CONF_THRESHOLD` isn't shown, because every stand-in box has the same score.
- OUT in the reference means the tray moved downstream along the s14 belt (`BELT_DIR` in `settings.ts`). A new clip needs its own `BELT_DIR`, its own default line and its own data file.
