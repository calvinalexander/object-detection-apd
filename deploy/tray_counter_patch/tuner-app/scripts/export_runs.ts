// Export what the tuner shows in side by side, for every run whose name matches the filter (default "case"):
//   <run name> - existing-vs-patched.mp4   the side-by-side video with the EXPECTED overlay (same as the website download)
//   <run name> - tray counts.xlsx          per run: its totals, then the per-tray list of the website's
//                                          "Existing vs patched -> All" table (time, existing, patched, difference)
// Counting uses each run's saved line and settings. Writes exports/tray_side_by_side_<date>.zip.
// Usage: npm run export -- [name filter]
import { readFileSync, writeFileSync, readdirSync, existsSync, mkdirSync, rmSync } from "node:fs";
import { join } from "node:path";
import { execFileSync } from "node:child_process";
import { run, diffRuns, type Config, type CountEvent, type DiffKind } from "../src/engine.ts";
import { CLASSES, DEFAULT_KNOBS, DEFAULT_VALID } from "../src/settings.ts";

const APP = new URL("..", import.meta.url).pathname, RUNS = join(APP, "runs"), OUT = join(APP, "exports");
const PYTHON = process.env.PYTHON ?? "python3";
const filter = (process.argv[2] ?? "case").toLowerCase();
const stamp = new Date().toISOString().slice(0, 10).replace(/-/g, "");
const stage = join(OUT, `tray_side_by_side_${stamp}`);
rmSync(stage, { recursive: true, force: true }); mkdirSync(join(stage, ".jobs"), { recursive: true });

// same wording as the website's comparison table
const KIND: Record<DiffKind, string> = { same: "Same", direction: "Different direction", violation: "Violation differs",
  class: "Fill class differs", onlyA: "Only existing counted", onlyB: "Only patched counted" };
const tag = (e?: CountEvent) => (e ? `${e.dir.toUpperCase()}${e.viol ? " + violation" : ""} · ${CLASSES[e.cls].name}` : "—");

const book: any = { summary: [], cases: [] };
const runs = readdirSync(RUNS).map((id) => ({ id, meta: existsSync(join(RUNS, id, "meta.json")) ? JSON.parse(readFileSync(join(RUNS, id, "meta.json"), "utf8")) : null }))
  .filter((r) => r.meta && !r.meta.builtin && r.meta.name.toLowerCase().includes(filter) && existsSync(join(RUNS, r.id, "data.json")))
  .sort((a, b) => a.meta.name.localeCompare(b.meta.name));

for (const { id, meta } of runs) {
  const dir = join(RUNS, id);
  const data = JSON.parse(readFileSync(join(dir, "data.json"), "utf8"));
  const st = existsSync(join(dir, "settings.json")) ? JSON.parse(readFileSync(join(dir, "settings.json"), "utf8"))
    : { line: meta.defaultLine, knobs: DEFAULT_KNOBS, valid: DEFAULT_VALID };
  const exp = existsSync(join(dir, "expected.json")) ? JSON.parse(readFileSync(join(dir, "expected.json"), "utf8")) : null;
  const res: Record<string, ReturnType<typeof run>> = {};
  for (const version of ["production", "patched"] as const)
    res[version] = run(data.dets, { ...DEFAULT_KNOBS, ...st.knobs, LINE: st.line, VALID_OUT_CLASSES: st.valid, TRACK_HISTORY_LEN: 30, version } as Config, data.fps);
  const E = res.production.events, P = res.patched.events;
  const tot = (ev: CountEvent[]) => { const o = ev.filter((e) => e.dir === "out").length; return [o, ev.length - o, ev.filter((e) => e.viol).length]; };

  // side-by-side video, rendered by the same script the website uses
  const job = { fps: data.fps, line: st.line,
    panels: [["EXISTING", res.production], ["PATCHED", res.patched]].map(([label, r]: any) => ({ label, frames: r.frames,
      events: r.events.map((e: CountEvent) => ({ k: e.k, dir: e.dir, viol: e.viol, oid: e.oid })) })),
    expected: exp ? { out: exp.out, inn: exp.inn, viol: exp.viol, crossings: exp.crossings.map((c: any) => ({ t: c.t, dir: c.dir, viol: c.viol })) } : undefined };
  const jobPath = join(stage, ".jobs", `${id}.json`);
  writeFileSync(jobPath, JSON.stringify(job));
  const upload = readdirSync(dir).find((f) => f.startsWith("upload."))!;
  const base = (meta.originalName ?? meta.name).replace(/\.[^.]+$/, "");
  const video = `${meta.name} - ${base} - existing-vs-patched.mp4`.replace(/[\\/:*?"<>|]/g, "_");
  process.stdout.write(`${meta.name}: rendering… `);
  execFileSync(PYTHON, [join(APP, "server", "render_run.py"), join(dir, upload), jobPath, join(stage, video)], { stdio: ["ignore", "ignore", "inherit"] });

  book.summary.push({ name: meta.name, file: meta.originalName ?? "", line: JSON.stringify(st.line), existing: tot(E), patched: tot(P),
    expected: exp ? [exp.out, exp.inn, exp.viol] : null, differences: diffRuns(E, P).filter((d) => d.kind !== "same" && d.kind !== "class").length, video });
  book.cases.push({ name: meta.name, rows: diffRuns(E, P).map((d) => ({ t: d.t, existing: tag(d.a), patched: tag(d.b), kind: KIND[d.kind], same: d.kind === "same" || d.kind === "class" })) });
  console.log(`existing ${tot(E).join("/")}  patched ${tot(P).join("/")}${exp ? `  expected ${exp.out}/${exp.inn}/${exp.viol}` : ""}`);
}

// the workbook, written by openpyxl
const bookPath = join(stage, ".jobs", "book.json");
writeFileSync(bookPath, JSON.stringify(book));
execFileSync(PYTHON, ["-c", `
import json, os, sys
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
b = json.load(open(sys.argv[1])); out = sys.argv[2]
bold, head = Font(bold=True), PatternFill("solid", fgColor="DCE6F0")
diff = PatternFill("solid", fgColor="FCE4D6")
for s, c in zip(b["summary"], b["cases"]):
    wb = Workbook(); sh = wb.active; sh.title = c["name"].title()[:31]
    e = s["expected"] or ["", "", ""]
    sh.append(["", "OUT", "IN", "Violations"])
    for x in sh[1]: x.font, x.fill = bold, head
    sh.append(["Existing", *s["existing"]]); sh.append(["Patched", *s["patched"]]); sh.append(["Expected", *e])
    sh.append(["Differences existing vs patched", s["differences"]]); sh.append(["Counting line", s["line"]])
    sh.append(["Original file", s["file"]]); sh.append([])
    sh.append(["Time (s)", "Existing", "Patched", "Difference"])
    for x in sh[sh.max_row]: x.font, x.fill = bold, head
    top = sh.max_row
    for r in c["rows"]:
        sh.append([round(r["t"], 2), r["existing"], r["patched"], r["kind"]])
        if not r["same"]:
            for x in sh[sh.max_row]: x.fill = diff
    for col, w in zip("ABCD", (30, 34, 34, 24)): sh.column_dimensions[col].width = w
    sh.freeze_panes = f"A{top + 1}"
    wb.save(os.path.join(out, s["video"].replace(" - existing-vs-patched.mp4", " - tray counts.xlsx")))
`, bookPath, stage]);

rmSync(join(stage, ".jobs"), { recursive: true, force: true });
const zip = `${stage}.zip`;
rmSync(zip, { force: true });
execFileSync("zip", ["-r", "-q", "-0", "-X", zip, "."], { cwd: stage });   // store only: the videos don't compress
rmSync(stage, { recursive: true, force: true });
console.log(`\nwrote ${zip}`);
