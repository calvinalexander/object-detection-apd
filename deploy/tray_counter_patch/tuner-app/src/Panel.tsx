import { useEffect, useRef, useState } from "react";
import type { CountEvent, Line, Snapshot } from "./engine";
import { CLASSES, FRAME_H, FRAME_W } from "./settings";

// One view of the clip: the shared <video> frame drawn onto a canvas, with that version's boxes, counts and the
// counting line on top. Two panels side by side draw the same video element, so they can never drift apart.
interface Props {
  video: HTMLVideoElement | null;
  fps: number;
  label?: string;
  frames: Snapshot[][] | null;
  events: CountEvent[];
  line: Line;
  busy: boolean;
  onDraft: (l: Line | null) => void; // live line while dragging, null when done
  onCommit: (l: Line) => void;
}

const MONO = '"JetBrains Mono", ui-monospace, Menlo, monospace';
const LINE_COLOR = "#ffb020", IN_COLOR = "#3fb8e0", DARK = "#1a1206";
const clampX = (v: number) => Math.round(Math.max(0, Math.min(FRAME_W, v)));
const clampY = (v: number) => Math.round(Math.max(0, Math.min(FRAME_H, v)));

export default function Panel({ video, fps, label, frames, events, line, busy, onDraft, onCommit }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const cvRef = useRef<HTMLCanvasElement>(null);
  const [cursor, setCursor] = useState("");

  // the rAF loop reads the latest props through this ref instead of restarting
  const live = useRef({ frames, events, label, dirty: true });
  live.current = { ...live.current, frames, events, label };
  const drag = useRef<null | { kind: "pt"; i: number } | { kind: "line"; start: [number, number]; orig: Line }>(null);
  const draftLine = useRef<Line>(line);
  useEffect(() => { if (!drag.current) { draftLine.current = line; live.current.dirty = true; } }, [line]);
  useEffect(() => { live.current.dirty = true; }, [frames, events, label]);

  useEffect(() => {
    if (!video) return;
    const wrap = wrapRef.current!, cv = cvRef.current!;
    const size = () => {
      const dpr = devicePixelRatio || 1;
      cv.width = wrap.clientWidth * dpr; cv.height = wrap.clientHeight * dpr;
      live.current.dirty = true;
    };
    const ro = new ResizeObserver(size); ro.observe(wrap); size();
    const mark = () => { live.current.dirty = true; };
    video.addEventListener("seeked", mark); video.addEventListener("loadeddata", mark);
    let raf = 0, lastK = -1;
    const loop = () => {
      const L = live.current, n = L.frames?.length ?? 1;
      const k = Math.min(n - 1, Math.floor(video.currentTime * fps + 1e-3));
      if (k !== lastK || L.dirty || !video.paused) {
        lastK = k; L.dirty = false;
        draw(cv, video, L.frames?.[k] ?? [], L.events, k, draftLine.current, L.label);
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(raf); ro.disconnect();
      video.removeEventListener("seeked", mark); video.removeEventListener("loadeddata", mark);
    };
  }, [video, fps]);

  // ---------- dragging the line ----------
  const toFrame = (e: React.PointerEvent): [number, number] => {
    const r = cvRef.current!.getBoundingClientRect();
    return [((e.clientX - r.left) / r.width) * FRAME_W, ((e.clientY - r.top) / r.height) * FRAME_H];
  };
  const hit = (p: [number, number]) => {
    const l = draftLine.current, tol = (22 * FRAME_W) / cvRef.current!.getBoundingClientRect().width;
    for (let i = 0; i < 2; i++) if (Math.hypot(p[0] - l[i][0], p[1] - l[i][1]) < tol) return { kind: "pt" as const, i };
    const [a, b] = l, dx = b[0] - a[0], dy = b[1] - a[1], L2 = dx * dx + dy * dy || 1;
    const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2));
    return Math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy) < tol * 0.6 ? { kind: "line" as const } : null;
  };
  const onDown = (e: React.PointerEvent) => {
    const p = toFrame(e), h = hit(p);
    if (!h) return;
    drag.current = h.kind === "pt" ? h : { kind: "line", start: p, orig: draftLine.current.map((q) => [...q]) as Line };
    cvRef.current!.setPointerCapture(e.pointerId); setCursor("grabbing"); e.preventDefault();
  };
  const onMove = (e: React.PointerEvent) => {
    const p = toFrame(e), d = drag.current;
    if (!d) { setCursor(hit(p) ? "grab" : ""); return; }
    const l = draftLine.current.map((q) => [...q]) as Line;
    if (d.kind === "pt") l[d.i] = [clampX(p[0]), clampY(p[1])];
    else {
      const dx = p[0] - d.start[0], dy = p[1] - d.start[1];
      d.orig.forEach((q, i) => (l[i] = [clampX(q[0] + dx), clampY(q[1] + dy)]));
    }
    draftLine.current = l; live.current.dirty = true; onDraft(l);
  };
  const onUp = () => {
    if (!drag.current) return;
    drag.current = null; setCursor(""); onDraft(null); onCommit(draftLine.current);
  };

  return (
    <div className="video-wrap" ref={wrapRef}>
      <canvas ref={cvRef} className={cursor} onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp}
        onPointerCancel={onUp} aria-label={`${label ?? "Camera"} view with the counting line. Drag the numbered handles to move it.`} />
      {busy && <div className="recount">Recounting…</div>}
    </div>
  );
}

function draw(cv: HTMLCanvasElement, video: HTMLVideoElement, snap: Snapshot[], events: CountEvent[], k: number, line: Line, label?: string) {
  const c = cv.getContext("2d")!, s = cv.width / FRAME_W, dpr = devicePixelRatio || 1;
  c.setTransform(1, 0, 0, 1, 0, 0);
  c.fillStyle = "#000"; c.fillRect(0, 0, cv.width, cv.height);
  if (video.readyState >= 2) c.drawImage(video, 0, 0, cv.width, cv.height);
  c.setTransform(s, 0, 0, s, 0, 0);
  const u = dpr / s; // one CSS pixel in frame units
  drawLine(c, line, u);                                         // first, so boxes and VIOLATION tags sit on top
  const flash = new Map<number, CountEvent>();
  for (const e of events) if (k >= e.k && k < e.k + 22) flash.set(e.oid, e);
  // boxes carry no text: colour shows fill (green full .. red empty), a tray counted OUT is drawn bold,
  // and the only label is VIOLATION
  for (const [id, x1, y1, x2, y2, cls] of snap) {
    // a tray just counted is drawn bold: OUT in its fill colour, IN in blue
    const f = flash.get(id), counted = !!f, col = f?.dir === "in" ? IN_COLOR : CLASSES[cls].color;
    if (counted) { c.strokeStyle = "rgba(0,0,0,.55)"; c.lineWidth = 6 * u; c.strokeRect(x1, y1, x2 - x1, y2 - y1); }
    c.strokeStyle = col; c.lineWidth = (counted ? 4 : 1.5) * u; c.strokeRect(x1, y1, x2 - x1, y2 - y1);
    if (f?.viol) {
      c.font = `700 ${13 * u}px ${MONO}`;
      const lbl = "VIOLATION", tw = c.measureText(lbl).width + 10 * u;
      c.fillStyle = "#e0261b"; c.fillRect(x1 - 3 * u, y1 - 22 * u, tw, 20 * u);
      c.fillStyle = "#fff"; c.fillText(lbl, x1 + 2 * u, y1 - 7 * u);
    }
  }
  // running totals up to this frame
  let o = 0, i = 0, v = 0;
  for (const e of events) { if (e.k > k) break; if (e.dir === "out") o++; else i++; if (e.viol) v++; }
  c.font = `600 ${14 * u}px ${MONO}`;
  const txt = `OUT ${o}   IN ${i}   VIOLATIONS ${v}`, tw = c.measureText(txt).width;
  c.fillStyle = "rgba(0,0,0,.7)"; c.fillRect(10 * u, 10 * u, tw + 16 * u, 26 * u);
  c.fillStyle = "#fff"; c.fillText(txt, 18 * u, 28 * u);
  if (label) {
    c.font = `700 ${15 * u}px ${MONO}`;
    const lw = c.measureText(label).width;
    c.fillStyle = label === "PATCHED" ? "rgba(10,115,131,.9)" : "rgba(40,40,40,.85)";
    c.fillRect(FRAME_W - lw - 30 * u, 10 * u, lw + 20 * u, 28 * u);
    c.fillStyle = "#fff"; c.fillText(label, FRAME_W - lw - 20 * u, 29 * u);
  }
}

function drawLine(c: CanvasRenderingContext2D, [p1, p2]: Line, u: number) {
  const dx = p2[0] - p1[0], dy = p2[1] - p1[1], L = Math.hypot(dx, dy) || 1;
  const n = [-dy / L, dx / L], m = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2];
  c.lineCap = "round";
  c.strokeStyle = "rgba(0,0,0,.5)"; c.lineWidth = 3.5 * u; c.beginPath(); c.moveTo(...p1); c.lineTo(...p2); c.stroke();
  c.strokeStyle = LINE_COLOR; c.lineWidth = 1.75 * u; c.beginPath(); c.moveTo(...p1); c.lineTo(...p2); c.stroke();
  // OUT arrow on the positive side (where the line normal points), IN label on the other
  const a = 40 * u, tip = [m[0] + n[0] * a, m[1] + n[1] * a], hx = 9 * u;
  c.lineWidth = 2 * u; c.beginPath(); c.moveTo(m[0], m[1]); c.lineTo(tip[0], tip[1]); c.stroke();
  c.fillStyle = LINE_COLOR; c.beginPath();
  c.moveTo(tip[0] + n[0] * hx, tip[1] + n[1] * hx);
  c.lineTo(tip[0] - n[1] * hx * 0.8, tip[1] + n[0] * hx * 0.8);
  c.lineTo(tip[0] + n[1] * hx * 0.8, tip[1] - n[0] * hx * 0.8); c.fill();
  pill(c, "OUT", m[0] + n[0] * (a + 24 * u), m[1] + n[1] * (a + 24 * u), LINE_COLOR, DARK, u);
  pill(c, "IN", m[0] - n[0] * 28 * u, m[1] - n[1] * 28 * u, IN_COLOR, "#06161c", u);
  [p1, p2].forEach((p, i) => {
    c.fillStyle = LINE_COLOR; c.strokeStyle = DARK; c.lineWidth = 1.5 * u;
    c.beginPath(); c.arc(p[0], p[1], 8 * u, 0, 7); c.fill(); c.stroke();
    c.fillStyle = DARK; c.font = `700 ${10 * u}px ${MONO}`; c.textAlign = "center"; c.textBaseline = "middle";
    c.fillText(String(i + 1), p[0], p[1] + 0.5 * u); c.textAlign = "start"; c.textBaseline = "alphabetic";
  });
}

function pill(c: CanvasRenderingContext2D, t: string, x: number, y: number, bg: string, fg: string, u: number) {
  c.font = `700 ${10.5 * u}px ${MONO}`;
  const w = c.measureText(t).width + 8 * u, h = 15 * u;
  c.fillStyle = bg; c.fillRect(x - w / 2, y - h / 2, w, h);
  c.fillStyle = fg; c.textAlign = "center"; c.textBaseline = "middle"; c.fillText(t, x, y + 0.5 * u);
  c.textAlign = "start"; c.textBaseline = "alphabetic";
}
