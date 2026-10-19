// Port of tlabcode/tracker.py + tlabcode/line_counter.py (production and patched).
// Runs a whole clip on cached detections; returns counted events and per-frame track snapshots.
// scripts/check-engine.ts checks it gives the same events as the Python.

export type Pt = [number, number];
export type Line = [Pt, Pt];
export type Det = [number, number, number, number, number]; // x1, y1, x2, y2, cls
export type Version = "patched" | "production";

export interface Config {
  CLASS_MODE?: "vote" | "existing" | "latest"; // patched only; experiment switch, default vote
  version: Version;
  LINE: Line;
  VALID_OUT_CLASSES: number[];
  MIN_TRAY_AREA: number;
  IOU_THRESHOLD: number;
  MAX_DISTANCE: number;
  MAX_POSITION_JUMP: number;
  CROSSING_BUFFER_ZONE: number;
  ANTI_BOUNCE_COOLDOWN: number;
  MIN_TRAJECTORY_POINTS: number;
  TRACK_HISTORY_LEN: number;
}

export interface CountEvent {
  k: number; // frame index into the detections
  along: number; // where the tray crossed: 0 at line point 1, 1 at point 2 (for matching to the reference)
  t: number; // seconds
  dir: "in" | "out";
  cls: number;
  oid: number;
  bbox: number[];
  viol: boolean;
}

export type Snapshot = [id: number, x1: number, y1: number, x2: number, y2: number, cls: number];

export interface RefTrack { k: number[]; p: number[]; fill: number }
export interface RefCrossing { t: number; dir: "in" | "out"; cls: number; viol: boolean; along: number }

export type Mark = "ok" | "wrong" | "falseviol" | "missedviol" | "extra";
export interface Score {
  right: number; wrong: number; missed: number; extra: number;
  refN: number; refViol: number; vTP: number; vFN: number; vFP: number;
  marks: Map<number, Mark>; missedRefs: RefCrossing[];
  matchOf: number[]; // per reference crossing: index of the matched event, or -1
}

class Kalman1D {
  x: number; P = 1.0; Q = 5e-3; R = 2e-1;
  constructor(x: number) { this.x = x; }
  update(m: number) {
    const Pp = this.P + this.Q, K = Pp / (Pp + this.R);
    this.x = this.x + K * (m - this.x); this.P = (1 - K) * Pp; return this.x;
  }
}

const iou = (a: number[], b: number[]) => {
  const xi1 = Math.max(a[0], b[0]), yi1 = Math.max(a[1], b[1]);
  const xi2 = Math.min(a[2], b[2]), yi2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, xi2 - xi1) * Math.max(0, yi2 - yi1);
  const uni = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return uni > 0 ? inter / uni : 0;
};
const bottom = (b: number[]): Pt => [(b[0] + b[2]) / 2, b[3]];
const dist = (p: Pt, q: Pt) => Math.hypot(p[0] - q[0], p[1] - q[1]);

function velocity(traj: Pt[], window: number): Pt {
  if (traj.length < 2) return [0, 0];
  const pts = traj.slice(-Math.min(window, traj.length));
  if (pts.length < 2) return [0, 0];
  let vx = 0, vy = 0;
  for (let i = 1; i < pts.length; i++) { vx += pts[i][0] - pts[i - 1][0]; vy += pts[i][1] - pts[i - 1][1]; }
  const n = pts.length - 1;
  return [vx / n, vy / n];
}

interface Obj { bbox: number[]; cls: number; votes: Map<number, number> }

export function run(D: Det[][], cfg: Config, fps = 15) {
  const patched = cfg.version === "patched";
  const valid = new Set(cfg.VALID_OUT_CLASSES);
  const [p1, p2] = cfg.LINE;
  const L = Math.hypot(p2[0] - p1[0], p2[1] - p1[1]) || 1;
  const lv = [(p2[0] - p1[0]) / L, (p2[1] - p1[1]) / L], normal = [-lv[1], lv[0]];
  const sd = (p: Pt) => ((p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])) / L;
  const alongOf = (p: Pt) => ((p[0] - p1[0]) * lv[0] + (p[1] - p1[1]) * lv[1]) / L;
  // point where a trajectory last changed side of the line (interpolated), else its point nearest the line
  const crossAlong = (t: Pt[]) => {
    for (let i = t.length - 1; i > 0; i--) {
      const a = sd(t[i - 1]), b = sd(t[i]);
      if (Math.sign(a) !== Math.sign(b) && a !== b) {
        const f = a / (a - b);
        return alongOf([t[i - 1][0] + f * (t[i][0] - t[i - 1][0]), t[i - 1][1] + f * (t[i][1] - t[i - 1][1])]);
      }
    }
    return alongOf(t.reduce((m, q) => (Math.abs(sd(q)) < Math.abs(sd(m)) ? q : m)));
  };

  // tracker state (Map keeps insertion order like the Python dicts)
  const objects = new Map<number, Obj>(), disappeared = new Map<number, number>();
  const traj = new Map<number, Pt[]>(), vel = new Map<number, Pt>();
  let nextId = 0;
  // counter state
  const counted = new Map<number, number>(), kalman = new Map<number, Kalman1D>(), sideHist = new Map<number, number[]>();
  const lastDir = new Map<number, "in" | "out">(); // patched: last counted direction per tray
  const anchor = new Map<number, number>();         // patched: the side of the line each tray was last clearly on
  const events: CountEvent[] = [], frames: Snapshot[][] = new Array(D.length);
  const MAXD = 30, HIST = cfg.TRACK_HISTORY_LEN;

  const register = (bbox: number[], cls: number) => {
    const id = nextId++;
    objects.set(id, { bbox, cls, votes: new Map([[cls, 1]]) });
    disappeared.set(id, 0); traj.set(id, [bottom(bbox)]); vel.set(id, [0, 0]);
  };
  const deregister = (id: number) => {
    objects.delete(id); disappeared.delete(id); traj.delete(id); vel.delete(id);
    kalman.delete(id); sideHist.delete(id);
  };
  const pushTraj = (id: number, p: Pt) => { const t = traj.get(id)!; t.push(p); if (t.length > HIST) t.shift(); };

  function trackerUpdate(dets: Det[]) {
    if (!dets.length) {
      for (const id of [...disappeared.keys()]) {
        disappeared.set(id, disappeared.get(id)! + 1);
        if (disappeared.get(id)! > MAXD) deregister(id);
      }
      return;
    }
    const boxes = dets.map((d) => d.slice(0, 4)), classes = dets.map((d) => d[4]), centers = boxes.map(bottom);
    if (!objects.size) { boxes.forEach((b, j) => register(b, classes[j])); return; }
    const ids = [...objects.keys()];
    const oCent = ids.map((id) => { const t = traj.get(id)!; return t.length ? t[t.length - 1] : bottom(objects.get(id)!.bbox); });
    const nO = ids.length, nD = boxes.length;
    const I: number[][] = [], Dm: number[][] = [], Pm: number[][] = [];
    let maxD = 0, maxPD = 0;
    for (let i = 0; i < nO; i++) {
      const id = ids[i], t = traj.get(id)!, v = vel.get(id)!;
      const pred: Pt | null = t.length ? [t[t.length - 1][0] + v[0], t[t.length - 1][1] + v[1]] : null;
      I.push([]); Dm.push([]); Pm.push([]);
      for (let j = 0; j < nD; j++) {
        // float32 like the numpy matrices in tracker.py
        // patched: compare with the predicted box (last box moved by velocity x frames since last seen)
        const gap = disappeared.get(id)! + 1, ob = objects.get(id)!.bbox;
        const pb = patched ? [ob[0] + v[0] * gap, ob[1] + v[1] * gap, ob[2] + v[0] * gap, ob[3] + v[1] * gap] : ob;
        I[i][j] = Math.fround(iou(pb, boxes[j]));
        Dm[i][j] = Math.fround(dist(oCent[i], centers[j]));
        Pm[i][j] = pred ? Math.fround(dist(pred, centers[j])) : Dm[i][j];
        maxD = Math.max(maxD, Dm[i][j]); maxPD = Math.max(maxPD, Pm[i][j]);
      }
    }
    maxD = Math.max(maxD, 1); maxPD = Math.max(maxPD, 1);
    const matches: [number, number, number, number, number][] = [];
    for (let i = 0; i < nO; i++) for (let j = 0; j < nD; j++) {
      if ((!patched || cfg.CLASS_MODE === "existing") && objects.get(ids[i])!.cls !== classes[j]) continue; // production: same class only
      const sc = 0.5 * I[i][j] + 0.25 * (1 - Dm[i][j] / maxD) + 0.25 * (1 - Pm[i][j] / maxPD);
      matches.push([sc, i, j, I[i][j], Dm[i][j]]);
    }
    // patched: tracks seen last frame match first, so stale tracks can't take the next tray's ID
    const stale = (i: number) => (patched && disappeared.get(ids[i])! > 0 ? 1 : 0);
    matches.sort((a, b) => stale(a[1]) - stale(b[1]) || b[0] - a[0] || b[1] - a[1] || b[2] - a[2]);
    const usedD = new Set<number>(), usedO = new Set<number>();
    for (const [sc, i, j, io, ds] of matches) {
      if (sc > 0.3 && io > cfg.IOU_THRESHOLD && ds < cfg.MAX_DISTANCE && Dm[i][j] < cfg.MAX_POSITION_JUMP &&
          !usedD.has(j) && !usedO.has(i)) {
        const id = ids[i], o = objects.get(id)!;
        o.bbox = boxes[j];
        if (patched && cfg.CLASS_MODE === "latest") o.cls = classes[j];
        else if (patched && cfg.CLASS_MODE !== "existing") {
          o.votes.set(classes[j], (o.votes.get(classes[j]) || 0) + 1);
          let best = o.cls, bv = -1;
          for (const [c, n] of o.votes) if (n > bv) { best = c; bv = n; }
          o.cls = best;
        }
        disappeared.set(id, 0);
        pushTraj(id, centers[j]);
        vel.set(id, velocity(traj.get(id)!, 5));
        usedD.add(j); usedO.add(i);
      }
    }
    ids.forEach((id, i) => {
      if (!usedO.has(i)) {
        disappeared.set(id, disappeared.get(id)! + 1);
        if (disappeared.get(id)! > MAXD) deregister(id);
      }
    });
    for (let j = 0; j < nD; j++) if (!usedD.has(j)) register(boxes[j], classes[j]);
  }

  function counterUpdate(k: number) {
    let fx = 0, fy = 0, n = 0;
    for (const t of traj.values()) if (t.length > 2) {
      fx += t[t.length - 1][0] - t[t.length - 2][0]; fy += t[t.length - 1][1] - t[t.length - 2][1]; n++;
    }
    const flow = n ? [fx / n, fy / n] : [0, 0];
    for (const [id, c] of [...counted]) if (k - c > cfg.ANTI_BOUNCE_COOLDOWN) counted.delete(id);

    for (const [id, o] of objects) {
      const b = o.bbox;
      if ((b[2] - b[0]) * (b[3] - b[1]) < cfg.MIN_TRAY_AREA) continue;
      const t = traj.get(id);
      if (!t || t.length < cfg.MIN_TRAJECTORY_POINTS) continue;
      if (counted.has(id)) continue;
      const raw = sd(t[t.length - 1]);
      if (Math.abs(raw) > 5000) continue;
      if (!kalman.has(id)) { kalman.set(id, new Kalman1D(raw)); sideHist.set(id, []); continue; }
      const s = kalman.get(id)!.update(raw);
      if (patched) {
        // A crossing counts once the tray is clearly on the other side: a quarter of its own length past the line
        // (at least CROSSING_BUFFER_ZONE). The side it came from is remembered for as long as it is tracked, so a
        // tray that stops, wobbles or swings on the line is counted once, when it really leaves.
        const ext = Math.abs((b[2] - b[0]) * normal[0]) + Math.abs((b[3] - b[1]) * normal[1]);
        const margin = Math.max(cfg.CROSSING_BUFFER_ZONE, 0.25 * ext);
        const side = s < -margin ? -1 : s > margin ? 1 : 0;
        if (!anchor.has(id)) {
          // first sighting: take the side it is on, even if it is not yet clear of the line (a tray that is
          // already on the line when the video starts would otherwise never get a starting side)
          if (Math.abs(s) > cfg.CROSSING_BUFFER_ZONE) anchor.set(id, Math.sign(s));
          continue;
        }
        if (side === 0 || side === anchor.get(id) || t.length < 2) continue;
        const dx = t[t.length - 1][0] - t[t.length - 2][0], dy = t[t.length - 1][1] - t[t.length - 2][1];
        const vn = Math.hypot(dx, dy);
        if (!(vn > 0.5) || Math.abs(dx * normal[0] + dy * normal[1]) / vn < 0.2) continue; // retried next frame
        anchor.set(id, side);
        const dir = side > 0 ? "out" : "in";
        if (lastDir.get(id) === dir) continue;
        lastDir.set(id, dir);
        events.push({ k: k - 1, t: Math.round(((k - 1) / fps) * 100) / 100, along: crossAlong(t), dir, cls: o.cls, oid: id,
                      bbox: b.slice(), viol: dir === "out" && !valid.has(o.cls) });
        counted.set(id, k);
        continue;
      }
      const side = s < -cfg.CROSSING_BUFFER_ZONE ? -1 : s > cfg.CROSSING_BUFFER_ZONE ? 1 : 0;
      const h = sideHist.get(id)!; h.push(side); if (h.length > 15) h.shift();
      if (h.length < 2) continue;
      const rs: number[] = [];
      for (let q = h.length - 1; q >= 0; q--) if (h[q] !== 0 && (!rs.length || h[q] !== rs[rs.length - 1])) {
        rs.push(h[q]); if (rs.length === 2) break;
      }
      if (rs.length !== 2 || t.length < 2) continue;
      const dx = t[t.length - 1][0] - t[t.length - 2][0], dy = t[t.length - 1][1] - t[t.length - 2][1];
      const vn = Math.hypot(dx, dy);
      if (!(vn > 0.5)) continue;
      if (Math.abs(dx * normal[0] + dy * normal[1]) / vn < 0.2) continue;
      const dir: "in" | "out" = Math.abs(flow[0]) + Math.abs(flow[1]) > 0.5
        ? (dx * flow[0] + dy * flow[1] > 0 ? "out" : "in")
        : (dx * normal[0] + dy * normal[1] > 0 ? "out" : "in");
      events.push({ k: k - 1, t: Math.round(((k - 1) / fps) * 100) / 100, along: crossAlong(t), dir, cls: o.cls, oid: id,
                    bbox: b.slice(), viol: dir === "out" && !valid.has(o.cls) });
      counted.set(id, k);
      h.length = 0;
    }
  }

  for (let k = 0; k < D.length; k++) {
    trackerUpdate(D[k]);
    counterUpdate(k + 1); // counter.py passes frame_num = k + 1
    const snap: Snapshot[] = [];
    for (const [id, o] of objects) if (disappeared.get(id) === 0)
      snap.push([id, o.bbox[0], o.bbox[1], o.bbox[2], o.bbox[3], o.cls]);
    frames[k] = snap;
  }
  return { events, frames };
}

// Stand-in fill score -> class, same cuts as s14/detect_all.py.
const CUTS = [[0.04, 0], [0.17, 3], [0.27, 2], [0.32, 4]];
export const fillClass = (f: number) => { for (const [c, k] of CUTS) if (f < c) return k; return 1; };

// Reference crossings for any line: offline tracks, side must hold >= 4 frames.
// OUT means the tray moved downstream (belt axis U), whatever the line's point order.
// Tracks shorter than MIN_REF_FRAMES points are skipped: on s14 the only one is a 35-frame junk box the stand-in
// detector draws across a stacked tray; every real tray is tracked for 90+ frames.
export const MIN_REF_FRAMES = 60;
export function reference(refTracks: RefTrack[], line: Line, validOut: number[], U: Pt, fps = 15): RefCrossing[] {
  const [p1, p2] = line, L = Math.hypot(p2[0] - p1[0], p2[1] - p1[1]) || 1;
  const lu = [(p2[0] - p1[0]) / L, (p2[1] - p1[1]) / L];
  const nrm = [-(p2[1] - p1[1]) / L, (p2[0] - p1[0]) / L];
  const downstreamSign = Math.sign(nrm[0] * U[0] + nrm[1] * U[1]) || 1;
  const valid = new Set(validOut), out: RefCrossing[] = [];
  for (const tr of refTracks) {
    if (tr.k.length < MIN_REF_FRAMES) continue;
    const n = tr.k.length, s = new Array<number>(n);
    for (let i = 0; i < n; i++) {
      const x = tr.p[2 * i], y = tr.p[2 * i + 1];
      s[i] = Math.sign(((p2[0] - p1[0]) * (y - p1[1]) - (p2[1] - p1[1]) * (x - p1[0])) / L);
    }
    let state: number | null = null;
    for (let j = 0; j < n; j++) {
      if (s[j] === 0) continue;
      if (state === null) { state = s[j]; continue; }
      if (s[j] !== state && j + 4 <= n && s.slice(j, j + 4).every((v) => v === s[j])) {
        const dir = s[j] * downstreamSign > 0 ? "out" : "in", cls = fillClass(tr.fill);
        const x = tr.p[2 * j], y = tr.p[2 * j + 1], along = ((x - p1[0]) * lu[0] + (y - p1[1]) * lu[1]) / L;
        out.push({ t: Math.round((tr.k[j] / fps) * 100) / 100, dir, cls, viol: dir === "out" && !valid.has(cls), along });
        state = s[j];
      }
    }
  }
  // One tray seen as two overlapping tracks (the stand-in splits a stacked tray) gives two crossings at the same place
  // within a few frames; keep the first. Real trays in one lane are at least 0.5 s apart even at 4x belt speed.
  out.sort((a, b) => a.t - b.t);
  return out.filter((r, i) => !out.slice(0, i).some((o) => o.dir === r.dir && r.t - o.t < 0.3 && Math.abs(r.along - o.along) < 0.12));
}

// Match counter events to reference crossings: the count must come from one frame before to 1.5 s after the crossing
// (the counter's smoothing makes it count a few frames late, never early), within 0.2 of the line's length of where
// the tray crossed, closest pairs first. Place keeps trays crossing together in two lanes apart; the time order keeps a
// count from being paired with the next tray in the same lane.
export function score(ref: RefCrossing[], events: CountEvent[]): Score {
  const used = new Set<number>(), marks = new Map<number, Mark>(), missedRefs: RefCrossing[] = [];
  let right = 0, wrong = 0, missed = 0, vTP = 0, vFN = 0, vFP = 0;
  const pairs: [number, number, number][] = [];
  ref.forEach((r, i) => events.forEach((e, q) => {
    const dt = e.t - r.t, da = Math.abs(e.along - r.along);
    if (dt > -0.07 && dt < 1.5 && da < 0.2) pairs.push([Math.abs(dt) + 5 * da, i, q]);
  }));
  pairs.sort((a, b) => a[0] - b[0]);
  const refTo = new Map<number, number>();
  for (const [, i, q] of pairs) if (!refTo.has(i) && !used.has(q)) { refTo.set(i, q); used.add(q); }
  ref.forEach((r, i) => {
    const j = refTo.get(i) ?? -1;
    if (j < 0) { missed++; missedRefs.push(r); if (r.viol) vFN++; return; }
    const e = events[j], ok = e.dir === r.dir;
    if (ok) right++; else wrong++;
    marks.set(j, ok ? "ok" : "wrong");
    if (r.viol) { if (e.viol) vTP++; else { vFN++; if (ok) marks.set(j, "missedviol"); } }
    else if (e.viol) { vFP++; if (ok) marks.set(j, "falseviol"); }
  });
  events.forEach((e, q) => { if (!used.has(q)) { marks.set(q, "extra"); if (e.viol) vFP++; } });
  return { right, wrong, missed, extra: events.length - used.size, refN: ref.length,
           refViol: ref.filter((r) => r.viol).length, vTP, vFN, vFP, marks, missedRefs,
           matchOf: ref.map((_, i) => refTo.get(i) ?? -1) };
}

// Pair two versions' counts on the same clip (same tray: within 1 s and 0.2 of the line's length of each other)
// and say how each pair differs. Works without a reference, so it is what an uploaded video is compared on.
export type DiffKind = "same" | "direction" | "violation" | "class" | "onlyA" | "onlyB";
export interface DiffRow { t: number; kind: DiffKind; a?: CountEvent; b?: CountEvent }
export function diffRuns(a: CountEvent[], b: CountEvent[]): DiffRow[] {
  const pairs: [number, number, number][] = [];
  a.forEach((x, i) => b.forEach((y, j) => {
    const dt = Math.abs(x.t - y.t), da = Math.abs(x.along - y.along);
    if (dt < 1 && da < 0.2) pairs.push([dt + 5 * da, i, j]);
  }));
  pairs.sort((p, q) => p[0] - q[0]);
  const ua = new Map<number, number>(), ub = new Set<number>();
  for (const [, i, j] of pairs) if (!ua.has(i) && !ub.has(j)) { ua.set(i, j); ub.add(j); }
  const rows: DiffRow[] = [];
  a.forEach((x, i) => {
    const j = ua.get(i);
    if (j === undefined) { rows.push({ t: x.t, kind: "onlyA", a: x }); return; }
    const y = b[j];
    const kind: DiffKind = x.dir !== y.dir ? "direction" : x.viol !== y.viol ? "violation" : x.cls !== y.cls ? "class" : "same";
    rows.push({ t: Math.min(x.t, y.t), kind, a: x, b: y });
  });
  b.forEach((y, j) => { if (!ub.has(j)) rows.push({ t: y.t, kind: "onlyB", b: y }); });
  return rows.sort((p, q) => p.t - q.t);
}
