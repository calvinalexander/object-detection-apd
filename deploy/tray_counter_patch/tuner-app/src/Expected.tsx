import { useEffect, useRef, useState } from "react";

// The expected running count (from the attached expected-results spreadsheet) at the video's current time: a small
// overlay centred between existing and patched, in the same style as each panel's own count.
export interface ExpectedData {
  source: string; name: string; out: number; inn: number; viol: number;
  crossings: { t: number; dir: "in" | "out"; viol: boolean; fill: string; role: string; lane: string; tray: number }[];
}

export default function Expected({ video, data }: { video: HTMLVideoElement | null; data: ExpectedData }) {
  const [t, setT] = useState(0);
  const last = useRef(-1);
  useEffect(() => {
    let raf = 0;
    const loop = () => {
      const now = video?.currentTime ?? 0;
      if (Math.abs(now - last.current) > 0.03) { last.current = now; setT(now); }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [video]);
  const done = data.crossings.filter((c) => c.t <= t);
  const out = done.filter((c) => c.dir === "out").length, inn = done.length - out, viol = done.filter((c) => c.viol).length;
  return (
    <div className="expected-overlay" title={`Expected results from ${data.source}`}>
      EXPECTED&nbsp;&nbsp; OUT {out}&nbsp;&nbsp; IN {inn}&nbsp;&nbsp; VIOLATIONS {viol}
    </div>
  );
}
