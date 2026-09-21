# SIPAI Dashboard — APD Tray Counter

> **Internal Developer Documentation**
> Last updated: 2026-02-22

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Tech Stack & Frameworks](#2-tech-stack--frameworks)
3. [Architecture & Data Flow](#3-architecture--data-flow)
4. [Design Decisions: Recording vs. Stream](#4-design-decisions-recording-vs-stream)
5. [Module Reference](#5-module-reference)
6. [AI Features](#6-ai-features)
7. [AI Models — Training, Quantization & Inference](#7-ai-models--training-quantization--inference)
8. [Deployment Architecture](#8-deployment-architecture)
9. [API Integration](#9-api-integration)
10. [Configuration Reference](#10-configuration-reference)

---

## 1. Project Overview

**SIPAI Dashboard APD Tray** is an edge-deployed computer-vision system that serves two primary functions:

| Function | Description |
|---|---|
| **Tray Counter** | Counts food trays crossing a configurable line-of-interest in video streams from kitchen/canteen cameras. Tracks objects, logs per-minute counts, records violations, and migrates data to a cloud API. |
| **APD Monitor** (Alat Pelindung Diri / Personal Protective Equipment) | Detects whether personnel in camera feeds are wearing required PPE — **head cap**, **face mask**, and **gloves** — and raises violation events when any item is missing. |

Both subsystems share a common edge infrastructure: cameras ➜ recorder ➜ inference ➜ tracking ➜ event storage ➜ cloud sync.

---

## 2. Tech Stack & Frameworks

### Core Language & Runtime

| Component | Version |
|---|---|
| Python | 3.10 |
| OS (target) | Ubuntu 22.04 (ARM64 / x86-64) |

### AI & Computer Vision

| Library / Framework | Purpose |
|---|---|
| **Ultralytics (YOLO)** | Model training (YOLOv8s / YOLO11s), ONNX export, Ultralytics inference backend |
| **Axelera Voyager SDK v1.4** | Quantization, compilation, and on-device inference for Axelera Metis AIPU |
| **OpenVINO** | Inference backend for Intel Arc GPU / Intel CPU |
| **OpenCV (`opencv-python`)** | Video capture, frame manipulation, drawing overlays, video writing |
| **PyTorch** | Tensor operations within tracker and Ultralytics inference |
| **NumPy** | Array operations for detection results, IoU computation, Kalman filter |

### Tracking

| Library | Purpose |
|---|---|
| **ByteTrack** (bundled in `apd_tracker/`) | Multi-object tracking with Kalman filter for APD person tracking |
| **AdvancedObjectTracker** (bundled in `tlabcode/tracker.py`) | Custom IoU + Kalman tracker for tray counting |

### Storage & Networking

| Technology | Purpose |
|---|---|
| **SQLite** (WAL mode) | Local event/session/violation storage with retry-safe writes |
| **MinIO / S3** (`boto3`) | Object storage for violation snapshots and setup-region frames |
| **Requests** | REST API communication for config pull and data migration |

### Infrastructure

| Technology | Purpose |
|---|---|
| **Docker / Docker Compose** | Containerized deployment across Axelera ARM64, Axelera x86-64, Intel Arc |
| **systemd** | Native service management (alternative to Docker) |
| **GStreamer** | Hardware-accelerated video decode pipeline (used by Axelera SDK) |
| **FFmpeg** (`ffmpeg-python`) | Fallback video encoding when OpenCV codecs are unavailable |
| **python-dotenv** | Environment variable management |

### Additional Dependencies

| Package | Purpose |
|---|---|
| `argon2-cffi` | Password hashing |
| `pycryptodome` | Cryptographic utilities |
| `python-dateutil` | Date/time parsing |

---

## 3. Architecture & Data Flow

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         EDGE DEVICE                                 │
│                                                                     │
│  ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌─────────────┐  │
│  │  CCTV /  │───>│ Recorder  │───>│ Recorded │───>│   Counter   │  │
│  │  RTSP    │    │ (recorder │    │  Videos  │    │ (counter.py)│  │
│  │  Cameras │    │  .py)     │    │ (SQLite) │    │             │  │
│  └──────────┘    └───────────┘    └──────────┘    └──────┬──────┘  │
│       │                                                   │         │
│       │ (live stream)                            inference results  │
│       │                                                   │         │
│       └──────────────────────┐                   ┌────────▼──────┐  │
│                              │                   │  Tray Counter │  │
│                              │                   │  Database     │  │
│                              │                   │  (SQLite)     │  │
│                              ▼                   └───────┬───────┘  │
│                     ┌────────────────┐                   │          │
│                     │  APD Monitor   │           ┌───────▼───────┐  │
│                     │ (apd_monitor_  │           │  Migrations   │  │
│                     │  axelera_      │           │  Orchestrator │  │
│                     │  multi.py)     │           └───────┬───────┘  │
│                     └───────┬────────┘                   │          │
│                             │                            │          │
│                    ┌────────▼─────────┐                  │          │
│                    │   APD Database   │                  │          │
│                    │   (SQLite)       │                  │          │
│                    └────────┬─────────┘                  │          │
│                             │                            │          │
│                    ┌────────▼─────────┐                  │          │
│                    │   APD Service    │                  │          │
│                    │  (apd_service.py)│                  │          │
│                    └────────┬─────────┘                  │          │
│                             │                            │          │
└─────────────────────────────┼────────────────────────────┼──────────┘
                              │                            │
                    ┌─────────▼────────────────────────────▼──────────┐
                    │              CLOUD / BACKEND                     │
                    │  ┌─────────────┐    ┌────────────────────┐      │
                    │  │  REST API   │    │   S3 / MinIO       │      │
                    │  │  (config +  │    │   (snapshots +     │      │
                    │  │   events)   │    │    violations)     │      │
                    │  └─────────────┘    └────────────────────┘      │
                    └─────────────────────────────────────────────────┘
```

### Tray Counter Pipeline

```
Video File ──▶ Axelera Inference Stream ──▶ Object Detections
                                                  │
                                                  ▼
                                    AdvancedObjectTracker.update()
                                                  │
                                                  ▼
                                    SmartLineCrossingCounter.update()
                                          │               │
                                     IN events        OUT events
                                          │               │
                                          ▼               ▼
                                   DatabaseManager (log_minute, log_violation)
                                                  │
                                                  ▼
                                     MigrationOrchestrator
                                       │              │
                                  S3 Upload       API POST
                                 (violations)    (session data)
```

### APD Monitor Pipeline

```
RTSP Streams ──▶ Inference Stream ──▶ Raw Detections
  (multi-cam)     (Axelera/YOLO)       │
                                        ├── Person detections ──▶ ByteTracker
                                        │                            │
                                        ├── PPE detections           │
                                        │   (gloves, mask,           │
                                        │    head_cap, etc.)         │
                                        │                            │
                                        ▼                            ▼
                                  assign_apd_smart()          Active Tracks
                                  (IoU + center-inside         (with IDs)
                                   assignment)                     │
                                        │                          │
                                        ▼                          │
                                  Per-track voting ◀───────────────┘
                                  (sliding window)
                                        │
                                        ▼
                               Violation detected?
                               (ratio ≥ threshold)
                                   │         │
                                  YES        NO
                                   │         └── skip
                                   ▼
                             Save to SQLite
                             Save snapshot (.jpg)
                                   │
                                   ▼
                             APD Service
                             (S3 upload + API POST)
```

---

## 4. Design Decisions: Recording vs. Stream

The system employs a split architecture: the **Tray Counter** processes local recordings, while the **APD Monitor** processes live streams. This choice is based on the distinct goals of each subsystem:

### 4.1 Tray Counter (Recording-Based)
**Requirement**: 100% Accuracy & Auditability.

*   **Frame Integrity**: Counting logic is sensitive to frame drops. If a camera stutters for 1 second, a tray might "jump" over the line undetected. Processing local `.mp4` files ensures 0% frame loss and perfect sequence.
*   **Global Flow Analysis**: The tray counter calculates a "global flow direction" (average movement of all objects) to determine conveyor direction. Stable tracking over many frames is required, which is far more reliable on local disk storage than noisy network streams.
*   **Audit Trail**: The system saves processed `_out.mp4` videos with counting overlays. This provides visual proof for every count reported in the dashboard.
*   **Batch Processing**: Tray counts are aggregated into "sessions" (usually 2 minutes). Local recording fits this batch-based reporting naturally.

### 4.2 APD Monitor (Stream-Based)
**Requirement**: Real-Time Safety & Low-Latency Alerts.

*   **Immediate Alerting**: PPE violations (missing mask, gloves, etc.) are safety/hygiene concerns that require near-instant notifications. Waiting for a 2-minute recording to finish is too slow for active intervention.
*   **Noise Filtering**: Instead of relying on perfectly sequential frames, the APD monitor uses a `VOTE_WINDOW` (multi-frame voting) and `MIN_VIOLATION_RATIO`. This design accommodates minor live-stream jitter while maintaining detection accuracy.
*   **Resource Efficiency**: Monitoring 4–8 cameras live is more resource-efficient than constantly reading and writing large temporary video files to the disk for every camera.
*   **Event-Based Logic**: APD detections are treated as discrete "events" rather than inventory tallies, making them a perfect fit for live-stream processing.

---

## 5. Module Reference

### 5.1 Root-Level Scripts

| File | Description |
|---|---|
| `counter.py` | **Tray Counter Service** — main loop that polls the recordings database for ready videos, runs Axelera inference, tracks objects via `AdvancedObjectTracker`, counts line crossings via `SmartLineCrossingCounter`, logs per-minute reports and violations, then triggers data migration. Includes `ShutdownHandler`, `ProcessingWatchdog`, `FFMPEGVideoWriter`, and `ResourceCleaner`. |
| `recorder.py` | **Video Recorder** — fetches camera config from the API, starts the `RecorderOrchestrator` for continuous recording with segment rotation, and runs `SetupRegionWorker` for first-frame capture and upload to MinIO. |
| `apd_monitor_axelera_multi.py` | **APD Monitor (Multi-Stream)** — the core real-time PPE detection loop. Supports both `axelera` and `ultralytics` inference backends. Manages per-stream state (trackers, frame counters, writers), performs person–PPE assignment via `assign_apd_smart()`, implements sliding-window voting, and saves violation events to SQLite. |
| `apd_monitor_axelera_service.py` | **APD Stream Supervisor Launcher** — thin wrapper that starts `apd_stream_service.py` on a configurable interval. |
| `apd_stream_service.py` | **Stream Supervisor** — periodically polls the API for camera configuration, detects changes (add/remove cameras), and starts/stops worker processes running `apd_monitor_axelera_multi.py`. Implements graceful worker lifecycle management. |
| `apd_service.py` | **APD Event Sync Service** — background service that reads unsent events from `apd_events.db`, uploads snapshot images to S3, POSTs event payloads to the cloud API, marks events as sent, and runs periodic retention cleanup of old data. |
| `apd_config.py` | **APD Configuration Manager** — unified configuration class with precedence: ENV > YAML > DEFAULT. Contains `Paths`, `S3`, `API`, `Service`, and `APD` config sections. |
| `apd_database.py` | **APD Event Database** — SQLite database wrapper with WAL mode, retry-safe insert (`save_event`), tables: `events`, `apd_detail`. |
| `apd_utils.py` | **APD Utility Functions** — frame conversion (`to_numpy`), class-to-group mapping (`build_class_maps`), smart person–PPE assignment (`assign_apd_smart` with center-inside + IoU + head-distance heuristics), bounding box utilities, and overlay drawing (`draw_text`, `draw_table`, `draw_icon`, `draw_debug_boxes`). |
| `config_apd.py` | **APD Class Definitions** — canonical class-name map (7 classes), APD groups (headcap, mask, gloves), enabled groups, per-group IoU thresholds. Legacy file, mostly superseded by `apd_config.py`. |
| `config_apd_service.yaml` | **YAML Configuration** — optional YAML config for APD service (database, S3, API, service settings). |
| `video_importer.py` | **External Video Importer** — CLI tool to import external video files into the recordings database for processing. Supports single and batch import, duplicate checking, file organization, and statistics. |

### 4.2 `apd_inference/` — Inference Abstraction Layer

| File | Description |
|---|---|
| `base_stream.py` | Abstract `BaseInferenceStream` iterator interface with `__iter__`, `__next__`, `stop`. |
| `factory.py` | Factory function `create_inference_stream(backend, **kwargs)` — routes to `AxeleraInferenceStream` or `UltralyticsInferenceStream`. |
| `ultralytics_stream.py` | `UltralyticsInferenceStream` — wraps Ultralytics YOLO model for multi-source inference via OpenCV `VideoCapture`. Iterates round-robin across camera sources, runs YOLO prediction per frame, packages results as `FrameResult` with `DetectionResult`. |
| `detection_result.py` | `DetectionResult` data class — holds `boxes` (N×4), `scores` (N,), `class_ids`. |
| `frame_result.py` | `FrameResult` data class — holds `image`, `meta` (dict of detection results), `stream_id`. |

### 4.3 `apd_tracker/` — ByteTrack Multi-Object Tracker

| File | Description |
|---|---|
| `basetrack.py` | `BaseTrack` base class with track state enum (`New`, `Tracked`, `Lost`, `Removed`) and auto-incrementing IDs. |
| `byte_tracker.py` | `BYTETracker` — full ByteTrack implementation. `STrack` class with Kalman-filtered bounding box state. Two-stage association: high-confidence first, then low-confidence detections matched to unmatched tracks. |
| `kalman_filter.py` | `KalmanFilter` — full Kalman filter implementation in `(cx, cy, ar, h, vx, vy, var, vh)` state space for bounding box motion prediction. |
| `matching.py` | Association utilities — linear assignment via `lap` or `scipy`, IoU distance matrix, fuse score/IoU, embedding distance. |

### 4.4 `tlabcode/` — Tray Counter Core Logic

| File | Description |
|---|---|
| `config.py` | `ConfigManager` — pulls configuration from a remote API, caches locally to JSON, supports hot-reload via background thread. `load_config_from_env()` loads all tray-counter config from `.env` variables. |
| `tracker.py` | `AdvancedObjectTracker` — custom multi-object tracker using IoU + Euclidean distance + velocity prediction. Includes `Kalman1D` for smoothing signed distance in line-crossing counter. |
| `line_counter.py` | `SmartLineCrossingCounter` — counts objects crossing a configurable line. Uses Kalman-smoothed signed distance, anti-bounce cooldown, minimum trajectory length, direction detection (IN/OUT), violation clip recording. |
| `database.py` | `RecordingsDB` — tracks recording file lifecycle (ready → processing → processed). `DatabaseManager` — session-based logging with tables: `video_sessions`, `minute_reports`, `violations`. Supports migration status tracking. |
| `migrations.py` | `MigrationOrchestrator` — uploads violation images to S3, builds session payload, POSTs to cloud API. `S3Manager`, `APIClient` classes for storage and API communication. |
| `visualizer.py` | Drawing functions for counting line, detection bounding boxes, statistics overlay, and violation highlights. |
| `utils.py` | Frame conversion (`to_numpy`) and health file writing (`write_counter_health`). |

### 4.5 `tlabrecord/` — Camera Recording Module

| File | Description |
|---|---|
| `config.py` | `Config` class — loads recording configuration from API (cameras, schedules, ROI). Supports hot-reload. |
| `orchestrator.py` | `RecorderOrchestrator` — manages per-camera recorder threads. Watches for camera additions/removals every 15 seconds. Runs daily storage cleanup. |
| `camera_recorder.py` | `CameraRecorder` — handles single camera recording: video source detection, segment rotation, scheduled recording hours, frame timing, file loop detection. Uses FFmpeg or OpenCV for video writing. |
| `storage_manager.py` | `StorageManager` — SQLite-backed recording lifecycle management. Registers/finalizes recordings, provides pending recordings for the counter, and cleans up old files by retention policy. |
| `video_source_detector.py` | `VideoSourceDetector` — auto-detects source type (RTSP, USB webcam, file) and validates connectivity. |
| `setup_region_handler.py` | `SetupRegionHandler` — captures a frame from a video file and uploads to MinIO for ROI configuration via the dashboard. |
| `logger.py` | Shared logging utility with file and stdout handlers. |

---

## 6. AI Features

### 6.1 Food Tray Detection & Counting

The tray counter detects food trays in camera feeds and counts them as they cross a user-defined line. Key features:

- **Object Detection**: YOLO-based detection of tray classes (e.g. full tray, empty tray).
- **Multi-Object Tracking**: `AdvancedObjectTracker` fuses IoU matching, Euclidean distance, and velocity prediction with a scoring formula: `0.5 × IoU + 0.25 × normalized_distance + 0.25 × normalized_predicted_distance`.
- **Line Crossing Logic**: `SmartLineCrossingCounter` uses Kalman-smoothed signed distance from each tracked object to a counting line. Anti-bounce cooldown and minimum trajectory length prevent false triggers.
- **Direction Detection**: Objects are classified as moving "IN" or "OUT" based on signed distance change across the line.
- **Per-Minute Reporting**: In/out counts are logged per minute to the database and migrated to the cloud.
- **Violation Clip Recording**: When violations are detected, short video clips are saved for review.

### 6.2 PPE / APD Detection (Personal Protective Equipment)

The APD monitor performs real-time PPE compliance checking. Key features:

- **7-Class Object Detection**: Detects `person`, `gloves`, `head_cap`, `mask`, `no_gloves`, `no_head_cap`, `no_mask`.
- **APD Group Logic**: Classes are grouped into 3 enforceable categories:

  | Group | Positive (OK) | Negative (Violation) |
  |---|---|---|
  | **headcap** | `head_cap` | `no_head_cap` |
  | **mask** | `mask` | `no_mask` |
  | **gloves** | `gloves` | `no_gloves` |

- **Smart Person–PPE Assignment** (`assign_apd_smart`): A multi-heuristic algorithm that assigns detected PPE items to tracked persons using:
  1. **Center-inside check** — whether the PPE center point is within the person bounding box.
  2. **IoU overlap** — per-group adaptive IoU thresholds.
  3. **Head-distance fallback** — for headcap assignments, checks pixel distance from headcap center to the person's head-top region.

- **Sliding Window Voting**: Per-track APD status is accumulated over a configurable window (`VOTE_WINDOW`, default 5 frames). A violation is only confirmed if the violation ratio exceeds `MIN_VIOLATION_RATIO` (default 0.6) and the person has been visible for at least `MIN_AGE_FRAMES` (default 3).

- **Cooldown & Deduplication**: Events for the same track are suppressed for `COOLDOWN_MS` (default 60s) and duplicate states are skipped.

- **Multi-Stream Support**: Independent tracking state, frame counters, and video writers per camera stream.

### 6.3 Multi-Object Tracking

Two tracking implementations are used:

| Tracker | Module | Used By | Algorithm |
|---|---|---|---|
| **ByteTrack** | `apd_tracker/byte_tracker.py` | APD Monitor | Two-pass association with high/low confidence splits, Kalman filter (8D state: cx, cy, ar, h + velocities) |
| **AdvancedObjectTracker** | `tlabcode/tracker.py` | Tray Counter | IoU + Euclidean + velocity-prediction scoring, Kalman1D smoothing for line-crossing |

### 6.4 Dynamic Configuration & Hot Reload

- Camera configurations, counting lines (ROI), and model settings are fetched from a remote API.
- `ConfigManager` runs a background thread that periodically refreshes config.
- Counting line geometry can be updated every 30 frames during processing.
- Camera additions/removals are detected every 15 seconds by the `RecorderOrchestrator`.

---

## 7. AI Models — Training, Quantization & Inference

### 7.1 Model Architecture

All models are based on the **Ultralytics YOLO** architecture:

| Model | Base Architecture | Input Size | Classes | Task |
|---|---|---|---|---|
| `apd_mbg` | YOLOv8s | 640×640 | 7 (person, gloves, head_cap, mask, no_gloves, no_head_cap, no_mask) | Object Detection |
| `apd_mbg_yolo11s` | YOLO11s | 640×640 | 7 (same) | Object Detection |
| `apd_v2` | YOLOv8s / YOLO11s | 640×640 | 7 (same) | Object Detection |
| `apd_v3` | YOLOv8s / YOLO11s | 640×640 | 7 (same) | Object Detection |
| `tc_2272_v11s_640` | YOLO11s | 640×640 | Tray classes | Object Detection |

### 7.2 Training

Models are trained using the **Ultralytics** training framework:

- **Dataset format**: YOLOv8 format with YAML dataset configurations located in `customers/<model_name>/<model_name>/dataset.yaml`.
- **Training command** (typical):
  ```bash
  yolo train model=yolo11s.pt data=dataset.yaml imgsz=640
  ```
- **Output**: A `best.pt` PyTorch checkpoint.

### 7.3 ONNX Export

Trained `.pt` models are exported to ONNX for deployment:

```bash
yolo export model=best.pt imgsz=640 dynamic=True simplify=True opset=17 format=onnx
```

ONNX models are stored in `customers/<model_name>/`:
- e.g., `customers/apd_mbg/apd_mbg.onnx`, `customers/apd_mbg/apd_mbg_yolo11s.onnx`

### 7.4 Quantization & Compilation

Models go through different optimization pipelines depending on the target hardware:

#### Axelera AIPU (Metis, Omega/Sigma chips)

The Axelera Voyager SDK compiles ONNX models into quantized + compiled `.axnet` packages. The flow:

```
.pt (PyTorch) ──▶ .onnx (ONNX) ──▶ Quantize ──▶ Compile ──▶ .axnet (Axelera binary)
```

**Quantization details** (from `compile_config.json`):

| Setting | `apd_mbg` value | `apd_mbg_yolo11s` value |
|---|---|---|
| Mode | `quantize_and_lower` | `quantize_and_lower` |
| Quantization scheme | `per_tensor_histogram` | `per_tensor_min_max` |
| Target hardware | Omega AIPU | Omega AIPU |
| AIPU cores used | 1 | 1 (configurable up to 4) |
| Frequency | 800 MHz | 800 MHz |
| Weight buffer optimization | `ignore_weight_buffers: true` | `ignore_weight_buffers: false` |

**Compilation artifacts** (stored in `build/<model_name>/`):
- `<model_name>.axnet` — Axelera network descriptor.
- `<model_name>/<model_name>/` — compiled binary directory.
- `<model_name>/<model_name>/quantized/` — quantized ONNX and postprocess graph.
- `<model_name>/<model_name>/compile_config.json` — full compilation configuration.
- `<model_name>/<model_name>/model_info.json` — model metadata.

**Axelera model pipeline config** (in `customers/<model_name>/config.yaml`):
- Defines preprocessing: letterbox resize to 640×640.
- Defines postprocessing: YOLO decode with NMS (IoU threshold: 0.45, conf threshold: 0.1–0.25).
- References the ONNX weight file and dataset.

#### OpenVINO (Intel Arc GPU / Intel CPU)

For Intel hardware, the ONNX model is exported to OpenVINO Intermediate Representation (IR) format:

**Artifacts** (in `build/apd_mbg_openvino_model/`):
- `apd_mbg_yolo11s.xml` — OpenVINO IR model structure (315 KB).
- `apd_mbg_yolo11s.bin` — OpenVINO IR weights (~36 MB).
- `metadata.yaml` — model metadata (YOLO11s, FP32, no INT8 quantization, no dynamic shapes).

**Current status**: FP32 inference (no INT8 quantization applied). Potential for future INT8 optimization using OpenVINO's POT or NNCF tools.

### 7.5 Inference Backends

The system supports multiple inference backends via the factory pattern:

| Backend | When to Use | Pipeline | Device Config |
|---|---|---|---|
| **`axelera`** | Axelera Metis M.2 / PCIe accelerator | `torch-aipu` or `gst` | Set `APD_BACKEND=axelera`, `APD_PIPE=gst` or `torch-aipu` |
| **`ultralytics`** | Intel Arc GPU, CPU, CUDA GPU | Direct Ultralytics predict | Set `APD_BACKEND=ultralytics`, `APD_DEVICE=intel:gpu` / `cpu` / `cuda` |

**Inference flow**:

1. `create_inference_stream()` creates a backend-specific stream.
2. The stream iterates frames from video sources (RTSP / file).
3. Each iteration yields a `FrameResult` containing the raw image and detection metadata.
4. Detections are parsed into `[x1, y1, x2, y2, confidence, class_id]` arrays.

**Axelera-specific pipeline types**:
- `gst` — GStreamer pipeline with hardware video decode. Recommended for production.
- `torch-aipu` — PyTorch-based pipeline routing computation to AIPU. Useful for debugging.
- `torch` — CPU-only PyTorch inference (no accelerator).

---

## 8. Deployment Architecture

### 8.1 Docker Compose Services

The system deploys as 4 application containers + 1 utility container:

| Service | Container | Script | Purpose |
|---|---|---|---|
| `counter-tray` | `counter-tray` | `counter.py` | Processes recorded videos for tray counting |
| `counter-tray-recorder` | `counter-tray-recorder` | `recorder.py` | Records RTSP streams from cameras |
| `apd-api` | `apd-api` | `apd_service.py` | Syncs APD events to cloud (S3 + API) |
| `apd-cctv` | `apd-cctv` | `apd_monitor_axelera_service.py` | Live APD monitoring on RTSP streams |
| `filebrowser` | `filebrowser` | FileBrowser v2.52 | Web UI for browsing recorded files and snapshots |

All application containers use the image `git.tlab.co.id:5050/bgn/voyager-sdk:v1.4-vsdk-installed-gstreamer` and run with `privileged: true` for hardware access.

### 8.2 Docker Compose Variants

| File | Target Hardware |
|---|---|
| `docker-compose-axelera.yml` | Axelera Metis M.2 on ARM64 (e.g., Firefly) |
| `docker-compose-axelera-x86-64.yml` | Axelera Metis on x86-64 host |
| `docker-compose-intel-arc.yml` | Intel Arc GPU (x86-64) |

### 8.3 Dockerfile Variants

| File | Target |
|---|---|
| `Dockerfile-axelera` | ARM64 with Axelera runtime 1.4.2 (Omega/Sigma) |
| `Dockerfile-axelera-v1.5-x86` | x86-64 with Axelera runtime v1.5 |
| `Dockerfile-axelera-x86-64` | x86-64 with Axelera runtime 1.4.2 |
| `Dockerfile-intel-arc` | Intel Arc with OpenVINO |

Key points about the Docker environment:
- Base: Ubuntu 22.04 with Python 3.10 compiled from source.
- Installs Axelera SDK components: `axelera_runtime`, `axelera_compiler`, `axelera_tvm`, etc.
- Installs PyTorch CPU build from `download.pytorch.org/whl/cpu`.
- Clones Voyager SDK to `/mnt/data/voyager-sdk`.
- Runs as `firefly` user (UID 1000) with video/render/kvm/axelera groups.

### 8.4 Systemd Services (Non-Docker)

For bare-metal or VM deployments without Docker:

```bash
# Generate service files
chmod +x setup_systemd.sh
./setup_systemd.sh

# Install
sudo cp recorder.service counter.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable recorder.service counter.service
sudo systemctl start recorder.service counter.service
```

Service files are templated with the correct project path, user, and virtual environment.

### 8.5 Volume Mounts

| Host Path | Container Path | Purpose |
|---|---|---|
| `./customers` | `/mnt/data/voyager-sdk/customers` | Model definitions, ONNX weights, datasets |
| `./build` | `/mnt/data/voyager-sdk/build` | Compiled model artifacts |
| `./` | `/mnt/data/tray_counter` | Application source code |
| `./recordings` | `/srv` (filebrowser) | Recorded video files |
| `./apd_storage/snapshots` | `/srv/snapshots_apd` (filebrowser) | APD violation snapshot images |

---

## 9. API Integration

### 9.1 Configuration API

**Purpose**: Fetches camera configuration, counting line ROI, and device settings.

| Detail | Value |
|---|---|
| Base URL | `API_BASE_URL` (e.g., `https://api-bgn.tlabdemo.com`) |
| Auth | `X-Auth-Api-Key` header |
| Used by | `recorder.py`, `tlabcode/config.py`, `apd_stream_service.py` |

**Endpoints used**:
- **GET** camera configuration for an edge device — returns camera list with RTSP URLs, camera types, schedules.
- **GET** counting line ROI — returns line coordinates for the tray counter.
- **POST** last-capture notification — notifies the API that setup-region frame has been captured.

**Data flow**:
1. On startup, `ConfigManager` fetches config from the API.
2. Config is cached locally to `config_cache.json` as fallback.
3. A background thread refreshes config every 60 seconds.
4. If the API is unreachable, the cached config is used.

### 9.2 Tray Counter Data API

**Purpose**: Receives tray counting session data and violation reports.

| Detail | Value |
|---|---|
| Endpoint | `API_ENDPOINT` (e.g., `https://api-bgn.tlabdemo.com/tray-counter`) |
| Auth | `X-Auth-Api-Key` header |
| Used by | `tlabcode/migrations.py` |

**Payload** (POST): JSON containing:
- `session_id`, `device_id`, `sppg_code`, `camera_id`
- `video_file`, `start_time`, `end_time`, `duration`, `fps`, `total_frames`
- `total_in`, `total_out`, `total_violations`
- `minute_reports[]` — array of per-minute in/out counts
- `violations[]` — array of violation details with S3 image paths

### 9.3 APD Event API

**Purpose**: Receives PPE violation events.

| Detail | Value |
|---|---|
| Base URL | `API.URL` from `apd_config.py` (e.g., `https://api-bgn.testlab.id/`) |
| Auth | `X-Auth-Api-Key` header |
| Used by | `apd_service.py` |

**Payload** (POST per event): JSON containing:
- `event_id` (UUID), `timestamp`, `device_id`, `camera_id`, `session_id`
- `track_id`, `person_bbox`, `violation` (list of violated groups)
- `apd_detail` — per-group status and bounding box
- `image_path` — S3 key of the uploaded snapshot

**Sync flow**:
1. `apd_service.py` polls the local SQLite database for unsent events.
2. For each event, uploads the snapshot JPEG to S3.
3. POSTs the event payload to the API.
4. Marks the event as `sent` in the database.
5. Retries with exponential backoff on failure.

### 9.4 S3 / MinIO Storage

**Purpose**: Object storage for violation images and setup-region frames.

| Detail | Value |
|---|---|
| Endpoint | `S3_ENDPOINT_URL` (MinIO or AWS S3) |
| Bucket | `S3_BUCKET` (e.g., `bgn-dev`) |
| Credentials | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` |
| Used by | `apd_service.py`, `tlabcode/migrations.py`, `recorder.py` |

**Stored objects**:
- APD violation snapshots: `<device_id>/<camera_id>/<event_id>.jpg`
- Tray violation images: `<sppg_code>/<session_id>/violations/<violation_id>.jpg`
- Setup-region frames: frames captured for ROI configuration

---

## 10. Configuration Reference

All configuration is managed via `.env` file (primary) with optional `config_apd_service.yaml` fallback. Priority: **ENV > YAML > DEFAULT**.

### 10.1 API & Device

| Variable | Default | Where to Change | Description |
|---|---|---|---|
| `API_BASE_URL` | `https://api-bgn.tlabdemo.com` | Per environment | Base URL for config + data APIs |
| `API_KEY` | — | Per environment | API authentication key |
| `EDGE_DEVICE_ID` | — | Per device | Unique edge device identifier (shown in dashboard) |
| `SPPG_ID` | — | Per device | SPPG identifier |
| `SPPG_CODE` | — | Per device | SPPG code for data association |
| `CAMERA_ID` | — | Per camera | Camera identifier |
| `API_ENDPOINT` | `https://api-bgn.tlabdemo.com/tray-counter` | Per environment | Tray counter data API endpoint |
| `CONFIG_API_URL` | `https://api-bgn.tlabdemo.com` | Per environment | Config fetch API |

### 10.2 S3 / MinIO Storage

| Variable | Default | Where to Change | Description |
|---|---|---|---|
| `AWS_ACCESS_KEY_ID` | — | Per environment | S3/MinIO access key |
| `AWS_SECRET_ACCESS_KEY` | — | Per environment | S3/MinIO secret key |
| `AWS_REGION` | `us-east-1` | Per environment | S3 region |
| `S3_BUCKET` | `bgn-dev` | Per environment | Target S3 bucket |
| `S3_ENDPOINT_URL` | — | Per environment | MinIO/S3 endpoint URL |
| `IS_MINIO` | `True` | Per environment | Set `False` for non-MinIO S3 (changes addressing style) |

### 10.3 Recorder Configuration

| Variable | Default | Where to Change | Description |
|---|---|---|---|
| `RECORDINGS_PATH` | `./recordings` | Per device | Root directory for recorded video files |
| `RECORD_START_HOUR` | `0` | Per installation | Hour to start recording (0–23) |
| `RECORD_END_HOUR` | `23` | Per installation | Hour to stop recording (0–23) |
| `RETENTION_DAYS` | `3` | Per device (disk space) | Days to keep recordings before cleanup |
| `ROTATE_MINUTES` | `2` | Per device | Segment rotation interval in minutes |
| `TEST_MODE` | `true` | Per environment | Enables frequent cleanup (every 60s vs. daily) |
| `VIDEO_WIDTH` | `1280` | Per camera | Recording width in pixels |
| `VIDEO_HEIGHT` | `720` | Per camera | Recording height in pixels |
| `VIDEO_FPS` | `15` | Per camera | Target recording FPS |
| `FFMPEG_DEBUG` | `false` | Debug only | Enable FFmpeg debug output |

### 10.4 Tray Counter / Analysis

| Variable | Default | Where to Change | Description |
|---|---|---|---|
| `MODEL_NAME` | `tray_counter_v8s_640` | When changing model | Axelera model name (must match `customers/` and `build/` dirs) |
| `COUNTING_LINE` | `[(937,547),(579,714)]` | Per camera ROI | Line crossing coordinates (auto-updated from API if `USE_REMOTE_CONFIG=True`) |
| `DB_PATH` | `./recordings/per_minute_report.db` | Per device | Path to tray counter SQLite database |
| `VIOLATION_DIR` | `./recordings/violations` | Per device | Directory for violation images |
| `FULL_TRAY_CLASS_ID` | `1` | Per model | Class ID for full tray |
| `CONF_THRESHOLD` | `0.1` | Tune per environment | Minimum detection confidence |
| `MIN_TRAY_AREA` | `2000` | Tune per camera | Minimum tray area in pixels² |
| `USE_REMOTE_CONFIG` | `True` | Per environment | Enable API-based config fetch |
| `MIN_TRAJECTORY_POINTS` | `1` | Tune if false positives | Minimum trajectory length for crossing count |
| `CROSSING_BUFFER_ZONE` | `5` | Tune if missed crossings | Pixel buffer zone near counting line |
| `ANTI_BOUNCE_COOLDOWN` | `30` | Tune if double counts | Frames to wait before re-counting same object |
| `IOU_THRESHOLD` | `0.2` | Tune if tracking breaks | Minimum IoU for tracker matching |
| `MAX_DISTANCE` | `100` | Tune per resolution | Max pixel distance for object matching |
| `MAX_POSITION_JUMP` | `100` | Tune per resolution | Max pixel jump to accept for same track |
| `PIPE` | `gst` | Per hardware | Axelera pipeline: `gst`, `torch-aipu`, `torch` |

### 10.5 APD Configuration

| Variable | Default | Where to Change | Description |
|---|---|---|---|
| `APD_MODEL` | `apd_mbg` | When changing model | Model name (must match `build/` and `customers/`) |
| `APD_PIPE` | `gst` | Per hardware | Pipeline type: `gst` / `torch-aipu` / `torch` |
| `APD_MODE` | `stream` | Per use case | `stream` for live RTSP, `recording` for file-based |
| `APD_BACKEND` | `axelera` | Per hardware | Inference backend: `axelera` / `ultralytics` |
| `APD_DEVICE` | `cpu` | Per hardware | Compute device: `intel:gpu` / `intel:cpu` / `cpu` / `cuda` / `0` |

### 10.6 APD Detection Thresholds

| Variable | Default | Where to Change | Description |
|---|---|---|---|
| `APD_THRESH_HEADCAP` | `0.4` | Tune per environment | Min confidence for headcap detection |
| `APD_THRESH_MASK` | `0.35` | Tune per environment | Min confidence for mask detection |
| `APD_THRESH_GLOVES` | `0.3` | Tune per environment | Min confidence for gloves detection |
| `APD_THRESH_PERSON` | `0.4` | Tune per environment | Min confidence for person detection |
| `APD_VOTE_WINDOW` | `5` | Tune for sensitivity | Number of frames in voting window |
| `APD_MIN_AGE_FRAMES` | `3` | Tune for sensitivity | Min frames visible before triggering event |
| `APD_MIN_VIOLATION_RATIO` | `0.6` | Tune for sensitivity | Required violation ratio within window (0.0–1.0) |
| `APD_COOLDOWN_MS` | `60000` | Tune for event frequency | Anti-spam timeout per track (milliseconds) |

### 10.7 APD Service & Retention

| Variable | Default | Where to Change | Description |
|---|---|---|---|
| `APD_STORAGE_PATH` | `apd_storage` | Per device | Root directory for APD data |
| `APD_SNAPSHOTS_DIR` | `snapshots` | Per device | Subdirectory for violation snapshots |
| `APD_DB_PATH` | `apd_events.db` | Per device | APD event database filename |
| `APD_DATABASE_CHECK_INTERVAL` | `5` | Tune for responsiveness | Seconds between event-sync checks |
| `APD_SERVICE_MAX_WORKERS` | `5` | Tune per CPU | Max concurrent upload workers |
| `APD_SERVICE_RETRY_ATTEMPTS` | `3` | Rarely change | Max retry attempts for S3/API operations |
| `APD_SERVICE_RETRY_BASE_DELAY` | `1.0` | Rarely change | Base delay (seconds) for exponential backoff |
| `APD_RETENTION_DAYS` | `3` | Per device (disk space) | Days to keep APD events before deletion |
| `APD_RETENTION_INTERVAL` | `3600` | Rarely change | Seconds between retention cleanup runs |
| `APD_RETENTION_BATCH_SIZE` | `100` | Rarely change | Number of old events to delete per batch |

### 10.8 Debug & Display

| Variable | Default | Where to Change | Description |
|---|---|---|---|
| `DEBUG_MODE` | `false` | Debug only | Enable verbose debug logging |
| `DEBUG_CONSOLE` | `True` | Debug only | Print debug to console |
| `DEBUG_FILE` | `True` | Debug only | Write debug to file |
| `DRAW_DEBUG_INFO` | `True` | Debug only | Draw debug overlays on frames |
| `SHOW_DISPLAY` | `True` | Local testing only | Show `cv2.imshow()` windows (requires display) |
| `SAVE_VIDEO` | `True` | Per environment | Save annotated output video |
