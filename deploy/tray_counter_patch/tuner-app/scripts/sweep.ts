// Sweep tuning settings on a clip and print the score for each (patched and production).
// Usage: node --experimental-strip-types scripts/sweep.ts public/s14_x4_data.json
import { readFileSync } from "node:fs";
import { run, reference, score, type Config } from "../src/engine.ts";
import { BELT_DIR, DEFAULT_KNOBS, DEFAULT_LINE } from "../src/settings.ts";

const data = JSON.parse(readFileSync(process.argv[2], "utf8"));
const ref = reference(data.ref, DEFAULT_LINE, [1, 5], BELT_DIR);
const base = { ...DEFAULT_KNOBS, LINE: DEFAULT_LINE, VALID_OUT_CLASSES: [1, 5], TRACK_HISTORY_LEN: 30 };
const show = (label: string, over: Partial<Config>) => {
  for (const version of ["production", "patched"] as const) {
    const { events } = run(data.dets, { ...base, ...over, version } as Config);
    const s = score(ref, events);
    console.log(`${label.padEnd(34)} ${version.padEnd(10)} right=${s.right}/${s.refN} wrong=${s.wrong} missed=${s.missed} extra=${s.extra} viol=${s.vTP}/${s.refViol} false=${s.vFP}`);
  }
};
show("device defaults", {});
for (const v of [60, 150, 200]) show(`MAX_DISTANCE=MAX_POSITION_JUMP=${v}`, { MAX_DISTANCE: v, MAX_POSITION_JUMP: v });
for (const v of [0, 0.05, 0.1]) show(`IOU_THRESHOLD=${v}`, { IOU_THRESHOLD: v });
for (const v of [0, 2, 10]) show(`CROSSING_BUFFER_ZONE=${v}`, { CROSSING_BUFFER_ZONE: v });
for (const v of [8, 15]) show(`ANTI_BOUNCE_COOLDOWN=${v}`, { ANTI_BOUNCE_COOLDOWN: v });
for (const v of [0, 1000]) show(`MIN_TRAY_AREA=${v}`, { MIN_TRAY_AREA: v });
