import type { Line, Pt } from "./engine";

export const FRAME_W = 1280;
export const FRAME_H = 720;
export const FPS = 15;
// The built-in s14 run: downstream direction of its belt and its counting line (uploaded runs carry their own).
export const BELT_DIR: Pt = [0.918, 0.397];
// Rail to rail across both lanes, along the belt's cross direction in this view (see s14/run_counter.py make_line).
export const DEFAULT_LINE: Line = [[485, 524], [794, 210]];
export const DEFAULT_VALID = [1, 5]; // VALID_OUT_CLASSES = {FULL_TRAY_CLASS_ID, 5}

// Box colour runs from green (full) to red (empty); a stack is allowed out like a full tray, so it is green too.
export const CLASSES = [
  { id: 0, name: "empty", color: "#ef3b2c" },
  { id: 1, name: "full", color: "#1fbf4a" },
  { id: 2, name: "half", color: "#f2c418" },
  { id: 3, name: "less than half", color: "#f47c20" },
  { id: 4, name: "more than half", color: "#8fd13a" },
  { id: 5, name: "stack", color: "#1fbf4a" },
];
// fill order for the colour legend, full first
export const FILL_ORDER = [1, 4, 2, 3, 0];

export type KnobKey =
  | "MIN_TRAY_AREA" | "MAX_DISTANCE" | "MAX_POSITION_JUMP" | "IOU_THRESHOLD"
  | "CROSSING_BUFFER_ZONE" | "ANTI_BOUNCE_COOLDOWN" | "MIN_TRAJECTORY_POINTS";

// Defaults are the device .env values.
export const KNOBS: { key: KnobKey; def: number; min: number; max: number; step: number; unit: string; help: string }[] = [
  { key: "MIN_TRAY_AREA", def: 2000, min: 0, max: 20000, step: 250, unit: "px²",
    help: "Boxes smaller than this are never counted. Raise it if hands or scraps get counted; scale it with camera distance." },
  { key: "MAX_DISTANCE", def: 100, min: 10, max: 250, step: 5, unit: "px",
    help: "How far a tray may move between frames and keep its ID. Too low and fast trays get a new ID mid-crossing." },
  { key: "MAX_POSITION_JUMP", def: 100, min: 10, max: 250, step: 5, unit: "px",
    help: "Same check as MAX_DISTANCE; the stricter of the two wins." },
  { key: "IOU_THRESHOLD", def: 0.2, min: 0, max: 0.8, step: 0.05, unit: "",
    help: "Minimum box overlap between frames to keep an ID. Raise it and fast trays lose their tracks." },
  { key: "CROSSING_BUFFER_ZONE", def: 5, min: 0, max: 60, step: 1, unit: "px",
    help: "Dead band either side of the line. Raise it if trays that stop on the line flicker IN/OUT." },
  { key: "ANTI_BOUNCE_COOLDOWN", def: 30, min: 0, max: 150, step: 5, unit: "frames",
    help: "After a count, the same tray ID can't count again for this many frames (30 = 2 s)." },
  { key: "MIN_TRAJECTORY_POINTS", def: 1, min: 1, max: 15, step: 1, unit: "",
    help: "A tray must be seen this many times before it can count." },
];

export type Knobs = Record<KnobKey, number>;
export const DEFAULT_KNOBS = Object.fromEntries(KNOBS.map((k) => [k.key, k.def])) as Knobs;

export const roiString = (l: Line) => `[(${l[0][0]}, ${l[0][1]}), (${l[1][0]}, ${l[1][1]})]`;
export const fmtTime = (t: number) => `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, "0")}`;
