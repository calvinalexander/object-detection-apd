// Checks the TypeScript port against the Python counter's events on s14.
// Usage: npm run test:engine -- [events json] [data json]   (defaults: the 4x clip)
import { readFileSync } from "node:fs";
import { run, reference, score, type Config, type Line } from "../src/engine.ts";

const eventsPath = process.argv[2] ?? "../s14/s14_x4_events.json";
const dataPath = process.argv[3] ?? new URL("../public/s14_x4_data.json", import.meta.url);
const data = JSON.parse(readFileSync(dataPath, "utf8"));
const py = JSON.parse(readFileSync(eventsPath, "utf8"));
const line = py.line as Line;
const base: Omit<Config, "version"> = {
  LINE: line, VALID_OUT_CLASSES: [1, 5], MIN_TRAY_AREA: 2000, IOU_THRESHOLD: 0.2, MAX_DISTANCE: 100,
  MAX_POSITION_JUMP: 100, CROSSING_BUFFER_ZONE: 5, ANTI_BOUNCE_COOLDOWN: 30, MIN_TRAJECTORY_POINTS: 1, TRACK_HISTORY_LEN: 30,
};
const ref = reference(data.ref, line, [1, 5], [0.918, 0.397]);
let failed = false;
for (const v of ["production", "patched"] as const) {
  const { events } = run(data.dets, { ...base, version: v });
  const want = py[v] as { frame: number; dir: string; cls: number }[];
  const same = events.length === want.length &&
    want.every((e, i) => e.frame === events[i].k && e.dir === events[i].dir && e.cls === events[i].cls);
  const s = score(ref, events);
  console.log(`${v.padEnd(10)} events=${events.length} python=${want.length} identical=${same} ` +
    `right=${s.right}/${s.refN} violations caught=${s.vTP}/${s.refViol} false=${s.vFP}`);
  failed ||= !same;
}
process.exit(failed ? 1 : 0);
