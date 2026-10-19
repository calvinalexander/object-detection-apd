// Client for server/server.mjs. When the app is served as plain static files (no server), only the built-in run exists.
import type { Line, Pt, Version } from "./engine";
import type { Knobs } from "./settings";

export interface SavedSettings { version: Version; line: Line; valid: number[]; knobs: Knobs; saved?: string }
export interface RunInfo {
  id: string; name: string; builtin: boolean; created: string;
  files: { data: string; video: string; poster: string };
  status: "queued" | "converting" | "background" | "calibrating" | "detecting" | "ready" | "error";
  progress?: { done: number; total: number; fps?: number };
  error?: string; settings?: SavedSettings | null;
  beltDir?: Pt; defaultLine: Line; start: number; originalName?: string; size?: number; note?: string;
  expected?: string | null; // URL of the attached expected results, if any
}

export const BUILTIN_RUN: RunInfo = {
  id: "s14_x4", name: "s14 conveyor render, 4× belt speed", builtin: true, created: "2026-10-12T00:00:00Z",
  files: { data: "s14_x4_data.json", video: "s14_x4_640.mp4", poster: "poster_x4.jpg" }, status: "ready",
  beltDir: [0.918, 0.397], defaultLine: [[485, 524], [794, 210]], start: 10,
};

export async function listRuns(): Promise<{ runs: RunInfo[]; server: boolean }> {
  try {
    const r = await fetch("api/runs", { cache: "no-store" });
    if (!r.ok || !(r.headers.get("content-type") ?? "").includes("json")) throw new Error(String(r.status));
    return { runs: await r.json(), server: true };
  } catch {
    return { runs: [BUILTIN_RUN], server: false };
  }
}

async function check(r: Response) {
  if (r.ok) return r.json();
  let msg = `The server answered ${r.status}.`;
  try { msg = (await r.json()).error ?? msg; } catch { /* not JSON */ }
  throw new Error(msg);
}

export const saveSettings = (id: string, s: SavedSettings): Promise<RunInfo> =>
  fetch(`api/runs/${id}/settings`, { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify(s) }).then(check);

export const deleteRun = (id: string) => fetch(`api/runs/${id}`, { method: "DELETE" }).then(check);
export const reprocessRun = (id: string): Promise<RunInfo> => fetch(`api/runs/${id}/reprocess`, { method: "POST" }).then(check);

export interface RenderJob {
  fps: number; line: Line; version?: string;
  expected?: { out: number; inn: number; viol: number; crossings: { t: number; dir: string; viol: boolean }[] };
  panels: { label: string | null; frames: number[][][]; events: { k: number; dir: string; viol: boolean; oid: number }[] }[];
}
export const renderRun = (id: string, job: RenderJob): Promise<{ url: string; filename: string }> =>
  fetch(`api/runs/${id}/render`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(job) }).then(check);

// Save a URL or a Blob under a file name.
export function download(href: string | Blob, filename: string) {
  const a = document.createElement("a");
  const url = typeof href === "string" ? href : URL.createObjectURL(href);
  a.href = url; a.download = filename; document.body.appendChild(a); a.click(); a.remove();
  if (typeof href !== "string") setTimeout(() => URL.revokeObjectURL(url), 2000);
}

// XHR rather than fetch so the upload can report progress.
export function uploadRun(file: File, name: string, onProgress: (share: number) => void): Promise<RunInfo> {
  return new Promise((ok, bad) => {
    const x = new XMLHttpRequest();
    x.open("POST", `api/runs?name=${encodeURIComponent(name)}`);
    x.setRequestHeader("x-filename", encodeURIComponent(file.name));
    x.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    x.onload = () => {
      let body: { error?: string } & Partial<RunInfo> = {};
      try { body = JSON.parse(x.responseText); } catch { /* not JSON */ }
      if (x.status >= 200 && x.status < 300) ok(body as RunInfo);
      else bad(new Error(body.error ?? `The server answered ${x.status}.`));
    };
    x.onerror = () => bad(new Error("The upload failed. Check the connection to the server."));
    x.send(file);
  });
}
