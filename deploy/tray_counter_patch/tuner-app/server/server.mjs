// Tray Counter Tuner server: serves the built app (dist/) and saved runs, and turns uploaded videos into runs.
// No dependencies beyond Node 18+. Processing is done by process_run.py (Python 3 with OpenCV and NumPy, plus ffmpeg).
//
//   node server/server.mjs            PORT (5180), HOST (0.0.0.0), RUNS_DIR (./runs), PYTHON (python3), MAX_UPLOAD_MB (2048)
//
// API
//   GET    /api/runs                  list runs, newest first (the built-in s14 run is always there)
//   POST   /api/runs?name=…           body = the raw video file (header x-filename: original name); returns the run
//   GET    /api/runs/:id              one run, with processing progress
//   PUT    /api/runs/:id/settings     body = JSON settings to save with the run (line, version, knobs, allowed classes)
//   POST   /api/runs/:id/reprocess    run detection again on an uploaded run (keeps its saved settings)
//   POST   /api/runs/:id/render       body = JSON overlay the browser counted ({fps, line, panels}); renders an annotated
//                                     MP4 and returns {url, filename} once it is done
//   DELETE /api/runs/:id              delete an uploaded run
//   GET    /runs/:id/:file            run files (video.mp4, poster.jpg, data.json), with Range support for seeking
//   GET    /runs/:id/original         the uploaded video as it was uploaded (download)
//   GET    /runs/:id/renders/:file    a rendered annotated video (?download=name to save it under that name)
//   GET    /exports/:file.zip         a bundle of runs made by `npm run export`
import { createServer } from "node:http";
import { spawn } from "node:child_process";
import { createReadStream, createWriteStream, existsSync, mkdirSync, readFileSync, readdirSync, renameSync, rmSync, statSync, writeFileSync } from "node:fs";
import { extname, join, resolve, dirname } from "node:path";
import { randomBytes } from "node:crypto";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const APP = resolve(HERE, "..");
const DIST = join(APP, "dist");
const RUNS = resolve(process.env.RUNS_DIR ?? join(APP, "runs"));
const PORT = +(process.env.PORT ?? 5180), HOST = process.env.HOST ?? "0.0.0.0";
const PYTHON = process.env.PYTHON ?? "python3";
const MAX_UPLOAD = +(process.env.MAX_UPLOAD_MB ?? 2048) * 1024 * 1024;
const BUILTIN = "s14_x4";

const EXPORTS = join(APP, "exports");
const TYPES = { ".zip": "application/zip", ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css", ".json": "application/json",
  ".mp4": "video/mp4", ".jpg": "image/jpeg", ".png": "image/png", ".svg": "image/svg+xml", ".ico": "image/x-icon" };
const ID = /^[a-z0-9_-]{1,64}$/i, FILE = /^[a-z0-9_.-]{1,64}$/i;

mkdirSync(RUNS, { recursive: true });
const readJson = (p, fallback = null) => { try { return JSON.parse(readFileSync(p, "utf8")); } catch { return fallback; } };
const writeJson = (p, v) => { writeFileSync(p + ".tmp", JSON.stringify(v, null, 1)); renameSync(p + ".tmp", p); };

// The built-in run is the s14 render at 4x belt speed shipped in public/; its files live next to index.html.
if (!existsSync(join(RUNS, BUILTIN, "meta.json"))) {
  mkdirSync(join(RUNS, BUILTIN), { recursive: true });
  writeJson(join(RUNS, BUILTIN, "meta.json"), {
    id: BUILTIN, name: "s14 conveyor render, 4× belt speed", builtin: true, created: "2026-10-12T00:00:00Z",
    files: { data: "s14_x4_data.json", video: "s14_x4_640.mp4", poster: "poster_x4.jpg" },
    beltDir: [0.918, 0.397], defaultLine: [[485, 524], [794, 210]], start: 10,
    note: "Stand-in detections on the sped-up render, with an offline reference to score against.",
  });
}

function runInfo(id) {
  const dir = join(RUNS, id), meta = readJson(join(dir, "meta.json"));
  if (!meta) return null;
  const st = meta.builtin ? { stage: "ready" } : readJson(join(dir, "status.json"), { stage: "queued" });
  const files = meta.builtin ? meta.files
    : { data: `runs/${id}/data.json`, video: `runs/${id}/video.mp4`, poster: `runs/${id}/poster.jpg` };
  return { ...meta, files, status: st.stage, progress: { done: st.done ?? 0, total: st.total ?? 0, fps: st.fps },
    error: st.error, settings: readJson(join(dir, "settings.json")),
    expected: existsSync(join(dir, "expected.json")) ? `runs/${id}/expected.json` : null };
}
const listRuns = () => readdirSync(RUNS).filter((d) => ID.test(d)).map(runInfo).filter(Boolean)
  .sort((a, b) => (a.builtin ? 1 : 0) - (b.builtin ? 1 : 0) || b.created.localeCompare(a.created));

// ---------- processing queue: one video at a time ----------
const queue = [];
let busy = false;
function enqueue(id) { if (!queue.includes(id)) queue.push(id); pump(); }
function pump() {
  if (busy || !queue.length) return;
  busy = true;
  const id = queue.shift(), dir = join(RUNS, id);
  const log = createWriteStream(join(dir, "process.log"));
  const p = spawn(PYTHON, [join(HERE, "process_run.py"), dir], { stdio: ["ignore", "pipe", "pipe"] });
  p.stdout.pipe(log); p.stderr.pipe(log);
  p.on("error", (e) => writeJson(join(dir, "status.json"), { stage: "error", error: `Could not start ${PYTHON}: ${e.message}` }));
  p.on("close", () => { busy = false; pump(); });
}
// runs left unfinished by a restart are processed again
for (const r of listRuns()) if (!r.builtin && !["ready", "error"].includes(r.status)) enqueue(r.id);

const attachment = (name) => ({ "content-disposition": `attachment; filename="${name.replace(/[^\w.() -]/g, "_")}"; filename*=UTF-8''${encodeURIComponent(name)}` });
const uploadFile = (id) => readdirSync(join(RUNS, id)).find((f) => f.startsWith("upload."));
const sourceVideo = (info) => (info.builtin ? join(DIST, info.files.video) : join(RUNS, info.id, uploadFile(info.id) ?? "missing"));

// one render at a time; each run keeps its 3 newest renders
let rendering = Promise.resolve();
function render(info, job) {
  const dir = join(RUNS, info.id, "renders");
  mkdirSync(dir, { recursive: true });
  const rid = new Date().toISOString().replace(/[-:T.Z]/g, "").slice(0, 14) + "-" + randomBytes(2).toString("hex");
  const jobPath = join(dir, `${rid}.json`), out = join(dir, `${rid}.mp4`);
  writeFileSync(jobPath, JSON.stringify(job));
  const task = rendering.then(() => new Promise((ok, bad) => {
    let log = "";
    const p = spawn(PYTHON, [join(HERE, "render_run.py"), sourceVideo(info), jobPath, out], { stdio: ["ignore", "pipe", "pipe"] });
    p.stdout.on("data", (c) => (log += c)); p.stderr.on("data", (c) => (log += c));
    p.on("error", (e) => bad(new Error(`Could not start ${PYTHON}: ${e.message}`)));
    p.on("close", (code) => {
      rmSync(jobPath, { force: true });
      if (code === 0 && existsSync(out)) ok(rid); else bad(new Error(log.trim().split("\n").pop() || "Rendering failed."));
    });
  }));
  rendering = task.catch(() => {});
  return task.then((r) => {
    const old = readdirSync(dir).filter((f) => f.endsWith(".mp4")).sort().reverse().slice(3);
    for (const f of old) rmSync(join(dir, f), { force: true });
    return r;
  });
}

// ---------- http ----------
const send = (res, code, body, type = "application/json") => {
  res.writeHead(code, { "content-type": type, "cache-control": "no-store" });
  res.end(typeof body === "string" ? body : JSON.stringify(body));
};
const fail = (res, code, message) => send(res, code, { error: message });

function sendFile(req, res, path, extra = {}) {
  let st;
  try { st = statSync(path); } catch { return fail(res, 404, "Not found"); }
  if (!st.isFile()) return fail(res, 404, "Not found");
  const type = TYPES[extname(path)] ?? "application/octet-stream";
  const range = /bytes=(\d*)-(\d*)/.exec(req.headers.range ?? "");
  if (range) {
    const start = range[1] ? +range[1] : st.size - +range[2], end = range[1] && range[2] ? Math.min(+range[2], st.size - 1) : st.size - 1;
    if (start >= st.size || start > end) { res.writeHead(416, { "content-range": `bytes */${st.size}` }); return res.end(); }
    res.writeHead(206, { ...extra, "content-type": type, "accept-ranges": "bytes", "content-range": `bytes ${start}-${end}/${st.size}`, "content-length": end - start + 1 });
    return createReadStream(path, { start, end }).pipe(res);
  }
  res.writeHead(200, { ...extra, "content-type": type, "accept-ranges": "bytes", "content-length": st.size,
    "cache-control": path.startsWith(join(DIST, "assets")) ? "public, max-age=31536000, immutable" : "no-cache" });
  createReadStream(path).pipe(res);
}

function readBody(req, limit) {
  return new Promise((ok, bad) => {
    const chunks = []; let n = 0;
    req.on("data", (c) => { n += c.length; if (n > limit) { bad(new Error("too large")); req.destroy(); } else chunks.push(c); });
    req.on("end", () => ok(Buffer.concat(chunks))); req.on("error", bad);
  });
}

function upload(req, res, url) {
  const name = (url.searchParams.get("name") ?? "").trim().slice(0, 120);
  const original = decodeURIComponent(req.headers["x-filename"] ?? "video.mp4").slice(0, 200);
  const ext = (extname(original).toLowerCase().match(/^\.[a-z0-9]{1,5}$/) ?? [".mp4"])[0];
  const size = +(req.headers["content-length"] ?? 0);
  if (size > MAX_UPLOAD) return fail(res, 413, `The video is larger than ${MAX_UPLOAD / 1048576} MB.`);
  const id = new Date().toISOString().replace(/[-:T.Z]/g, "").slice(0, 14) + "-" + randomBytes(3).toString("hex");
  const dir = join(RUNS, id);
  mkdirSync(dir);
  const out = createWriteStream(join(dir, "upload" + ext));
  let received = 0, aborted = false;
  const abort = (code, msg) => { if (aborted) return; aborted = true; out.destroy(); rmSync(dir, { recursive: true, force: true }); if (!res.headersSent) fail(res, code, msg); };
  req.on("data", (c) => { received += c.length; if (received > MAX_UPLOAD) { abort(413, "The video is too large."); req.destroy(); } });
  req.on("aborted", () => abort(400, "Upload interrupted."));
  req.pipe(out);
  out.on("finish", () => {
    if (aborted) return;
    if (!received) return abort(400, "The upload was empty.");
    writeJson(join(dir, "meta.json"), { id, name: name || original.replace(/\.[^.]+$/, ""), originalName: original, size: received,
      created: new Date().toISOString(), builtin: false, defaultLine: [[640, 120], [640, 600]], start: 0 });
    writeJson(join(dir, "status.json"), { stage: "queued" });
    enqueue(id);
    send(res, 201, runInfo(id));
  });
  out.on("error", () => abort(500, "Could not save the upload."));
}

createServer(async (req, res) => {
  const url = new URL(req.url, "http://x");
  const parts = url.pathname.split("/").filter(Boolean).map(decodeURIComponent);
  try {
    if (parts[0] === "api" && parts[1] === "runs") {
      const id = parts[2];
      if (!id) {
        if (req.method === "GET") return send(res, 200, listRuns());
        if (req.method === "POST") return upload(req, res, url);
        return fail(res, 405, "Method not allowed");
      }
      if (!ID.test(id) || !existsSync(join(RUNS, id, "meta.json"))) return fail(res, 404, "No such run");
      if (parts[3] === "settings" && req.method === "PUT") {
        const body = JSON.parse((await readBody(req, 64 * 1024)).toString("utf8"));
        writeJson(join(RUNS, id, "settings.json"), { ...body, saved: new Date().toISOString() });
        return send(res, 200, runInfo(id));
      }
      if (parts[3] === "render" && req.method === "POST") {
        const info = runInfo(id);
        if (info.status !== "ready") return fail(res, 409, "The run isn't ready yet.");
        const job = JSON.parse((await readBody(req, 200 * 1024 * 1024)).toString("utf8"));
        if (!Array.isArray(job.panels) || !job.panels.length || job.panels.length > 2) return fail(res, 400, "Nothing to render.");
        try {
          const rid = await render(info, job);
          const kind = job.panels.length === 2 ? "existing-vs-patched" : (job.panels[0].label ?? job.version ?? "counted").toLowerCase();
          const base = (info.originalName ?? info.name).replace(/\.[^.]+$/, "");
          return send(res, 200, { url: `runs/${id}/renders/${rid}.mp4`, filename: `${base} - ${kind}.mp4` });
        } catch (e) { return fail(res, 500, e.message); }
      }
      if (parts[3] === "reprocess" && req.method === "POST") {           // run detection again, e.g. after a detector update
        if (runInfo(id).builtin) return fail(res, 400, "The built-in run can't be reprocessed.");
        if (queue.includes(id) || !["ready", "error"].includes(runInfo(id).status)) return fail(res, 409, "This run is already being processed.");
        writeJson(join(RUNS, id, "status.json"), { stage: "queued" });
        enqueue(id);
        return send(res, 200, runInfo(id));
      }
      if (!parts[3] && req.method === "GET") return send(res, 200, runInfo(id));
      if (!parts[3] && req.method === "DELETE") {
        if (runInfo(id).builtin) return fail(res, 400, "The built-in run can't be deleted.");
        if (queue.includes(id)) queue.splice(queue.indexOf(id), 1);
        rmSync(join(RUNS, id), { recursive: true, force: true });
        return send(res, 200, { deleted: id });
      }
      return fail(res, 405, "Method not allowed");
    }
    if (parts[0] === "exports" && parts.length === 2 && /^[\w.-]+\.zip$/.test(parts[1]))   // bundles made by npm run export
      return sendFile(req, res, join(EXPORTS, parts[1]), attachment(parts[1]));
    if (parts[0] === "runs" && parts.length === 3 && ID.test(parts[1]) && parts[2] === "original") {
      const info = runInfo(parts[1]), f = info && !info.builtin && uploadFile(parts[1]);
      if (!f) return fail(res, 404, "No original video for this run.");
      return sendFile(req, res, join(RUNS, parts[1], f), attachment(info.originalName ?? f));
    }
    if (parts[0] === "runs" && parts.length === 4 && ID.test(parts[1]) && parts[2] === "renders" && /^[0-9a-f-]+\.mp4$/.test(parts[3])) {
      const name = url.searchParams.get("download");
      return sendFile(req, res, join(RUNS, parts[1], "renders", parts[3]), name ? attachment(name) : {});
    }
    if (parts[0] === "runs" && parts.length === 3 && ID.test(parts[1]) && FILE.test(parts[2])
        && ["video.mp4", "poster.jpg", "data.json", "expected.json"].includes(parts[2])) return sendFile(req, res, join(RUNS, parts[1], parts[2]));
    // the app itself; unknown paths fall back to index.html
    const file = resolve(DIST, "." + url.pathname);
    if (file.startsWith(DIST) && existsSync(file) && statSync(file).isFile()) return sendFile(req, res, file);
    return sendFile(req, res, join(DIST, "index.html"));
  } catch (e) {
    return fail(res, 400, e.message);
  }
}).listen(PORT, HOST, () => console.log(`Tray Counter Tuner on http://${HOST}:${PORT}  (runs in ${RUNS})`));
