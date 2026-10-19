import { useEffect, useRef, useState } from "react";

// Play/pause, speed, time and a timeline with one row of ticks per version shown.
export interface Tick { t: number; tone: "good" | "bad" | "warn" | "out" | "in"; hollow?: boolean }
export interface TimelineRow { label?: string; ticks: Tick[] }

const cssVar = (n: string) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

export default function Transport({ video, duration, rows }: { video: HTMLVideoElement | null; duration: number; rows: TimelineRow[] }) {
  const tlRef = useRef<HTMLCanvasElement>(null);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [time, setTime] = useState(0);
  const live = useRef({ rows, duration, dirty: true });
  live.current = { rows, duration, dirty: true };

  useEffect(() => {
    if (!video) return;
    const on = () => setPlaying(!video.paused);
    const rate = () => { video.playbackRate = speed; };
    video.addEventListener("play", on); video.addEventListener("pause", on); video.addEventListener("loadedmetadata", rate);
    rate(); on();
    return () => { video.removeEventListener("play", on); video.removeEventListener("pause", on); video.removeEventListener("loadedmetadata", rate); };
  }, [video, speed]);

  useEffect(() => {
    const tl = tlRef.current!;
    const size = () => { const dpr = devicePixelRatio || 1; tl.width = tl.clientWidth * dpr; tl.height = tl.clientHeight * dpr; live.current.dirty = true; };
    const ro = new ResizeObserver(size); ro.observe(tl); size();
    let raf = 0, last = -1;
    const loop = () => {
      const t = video?.currentTime ?? 0;
      if (t !== last || live.current.dirty) { last = t; live.current.dirty = false; draw(tl, live.current.rows, live.current.duration, t); setTime(t); }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => { cancelAnimationFrame(raf); ro.disconnect(); };
  }, [video]);

  const seek = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!video || !duration) return;
    const r = e.currentTarget.getBoundingClientRect(), x0 = rows.some((w) => w.label) ? 74 : 4;
    video.currentTime = Math.max(0, Math.min(duration - 0.05, ((e.clientX - r.left - x0) / (r.width - x0 - 4)) * duration));
  };

  return (
    <div className="transport">
      <button className="primary" id="play" disabled={!video} onClick={() => { if (video!.paused) video!.play(); else video!.pause(); }}>
        {playing ? "Pause" : "Play"}
      </button>
      <select aria-label="Playback speed" value={speed} onChange={(e) => setSpeed(+e.target.value)}>
        <option value={0.5}>0.5×</option><option value={1}>1×</option><option value={2}>2×</option><option value={4}>4×</option>
      </select>
      <span className="time">{time.toFixed(1)} s</span>
      <canvas id="timeline" ref={tlRef} onClick={seek} style={{ height: rows.length > 1 ? 52 : 34 }}
        aria-label="Timeline of counted trays. Click to jump." />
    </div>
  );
}

function draw(tl: HTMLCanvasElement, rows: TimelineRow[], dur: number, t: number) {
  const dpr = devicePixelRatio || 1, w = tl.width / dpr, h = tl.height / dpr, c = tl.getContext("2d")!;
  c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, w, h);
  if (!dur) return;
  const labelled = rows.some((r) => r.label), x0 = labelled ? 74 : 4;
  const x = (s: number) => x0 + (s / dur) * (w - x0 - 4);
  const col: Record<Tick["tone"], string> = { good: cssVar("--good"), bad: cssVar("--bad"), warn: cssVar("--warn"), out: cssVar("--accent"), in: cssVar("--in") };
  const rh = (h - 4) / rows.length;
  rows.forEach((row, ri) => {
    const y0 = 2 + ri * rh;
    if (row.label) {
      c.fillStyle = cssVar("--muted"); c.font = `600 10.5px "JetBrains Mono", ui-monospace, monospace`; c.textBaseline = "middle";
      c.fillText(row.label, 6, y0 + rh / 2); c.textBaseline = "alphabetic";
    }
    for (const k of row.ticks) {
      const big = k.tone !== "good" && k.tone !== "out" && k.tone !== "in";
      const th = big ? rh - 4 : rh * 0.55, ty = y0 + (rh - th) / 2;
      if (k.hollow) { c.strokeStyle = col[k.tone]; c.lineWidth = 1.5; c.strokeRect(x(k.t) - 2, ty, 4, th); }
      else { c.fillStyle = col[k.tone]; c.fillRect(x(k.t) - 1, ty, 2, th); }
    }
  });
  c.fillStyle = cssVar("--ink"); c.fillRect(x(t) - 1, 0, 2, h);
}
