import { readFileSync, writeFileSync } from "node:fs";
import { run, type Config } from "/home/ubuntu/tryan/apd/deploy/tray_counter_patch/tuner-app/src/engine.ts";
import { DEFAULT_KNOBS, DEFAULT_LINE } from "/home/ubuntu/tryan/apd/deploy/tray_counter_patch/tuner-app/src/settings.ts";
const d = JSON.parse(readFileSync("/home/ubuntu/tryan/apd/deploy/tray_counter_patch/tuner-app/public/s14_x4_data.json", "utf8"));
const out: any = {};
for (const version of ["production", "patched"] as const) {
  const { events, frames } = run(d.dets, { ...DEFAULT_KNOBS, LINE: DEFAULT_LINE, VALID_OUT_CLASSES: [1, 5], TRACK_HISTORY_LEN: 30, version } as Config);
  // per-frame classes the detector gave this track (from the tracker snapshots), to see flicker
  out[version] = events.map((e) => {
    const seen: number[] = [];
    frames.forEach((f, k) => { for (const s of f) if (s[0] === e.oid) {
      const raw = d.dets[k].find((x: number[]) => x[0] === s[1] && x[1] === s[2] && x[2] === s[3] && x[3] === s[4]);
      if (raw) seen.push(raw[4]);
    } });
    return { ...e, seen };
  });
}
writeFileSync(process.argv[2], JSON.stringify(out));
console.log(out.patched.length, "patched events,", out.production.length, "existing events");
