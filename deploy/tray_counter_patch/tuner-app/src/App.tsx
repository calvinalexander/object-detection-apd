import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  diffRuns, reference, run, score, type CountEvent, type Det, type DiffKind, type Line, type Mark, type RefTrack, type Score, type Snapshot, type Version,
} from "./engine";
import Panel from "./Panel";
import Expected, { type ExpectedData } from "./Expected";
import Transport, { type Tick, type TimelineRow } from "./Transport";
import { deleteRun, download, listRuns, renderRun, reprocessRun, saveSettings, uploadRun, type RunInfo } from "./api";
import { CLASSES, DEFAULT_KNOBS, DEFAULT_VALID, FILL_ORDER, KNOBS, fmtTime, roiString, type Knobs } from "./settings";

interface ClipData { fps: number; w: number; h: number; dets: Det[][]; ref: RefTrack[]; detector?: string }
interface Settings { version: Version; line: Line; valid: number[]; knobs: Knobs }
interface Result { events: CountEvent[]; frames: Snapshot[][]; sc: Score | null }

const MARKS: Record<Mark | "missed", [tone: string, label: string]> = {
  ok: ["ok", "Correct"], wrong: ["bad", "Wrong direction"], falseviol: ["warn", "False violation"],
  missedviol: ["warn", "Violation missed"], extra: ["warn", "Extra count"], missed: ["bad", "Missed (not counted)"],
};
const DIFF: Record<DiffKind, [tone: string, label: string]> = {
  same: ["ok", "Same"], direction: ["bad", "Different direction"], violation: ["warn", "Violation differs"],
  class: ["ok", "Fill class differs"], onlyA: ["warn", "Only existing counted"], onlyB: ["warn", "Only patched counted"],
};
const VERSIONS: Version[] = ["production", "patched"];
// what the UI calls each version: "production" is the code running on the devices today
const NAME: Record<Version, string> = { production: "Existing", patched: "Patched" };
const STAGES: Record<string, string> = { queued: "Waiting", converting: "Converting", background: "Building background",
  calibrating: "Measuring tray size", detecting: "Detecting trays", ready: "Ready", error: "Failed" };

function useDebounced<T>(value: T, ms: number): [T, boolean] {
  const [v, setV] = useState(value);
  useEffect(() => { const t = setTimeout(() => setV(value), ms); return () => clearTimeout(t); }, [value, ms]);
  return [v, v !== value];
}
const defaultsFor = (r: RunInfo): Settings => ({ version: "patched", line: r.defaultLine, valid: DEFAULT_VALID, knobs: DEFAULT_KNOBS });
const fromSaved = (r: RunInfo): Settings => {
  const s = r.settings;
  return s ? { version: s.version ?? "patched", line: s.line ?? r.defaultLine, valid: s.valid ?? DEFAULT_VALID, knobs: { ...DEFAULT_KNOBS, ...s.knobs } } : defaultsFor(r);
};
const sameSettings = (a: Settings, b: Settings) => JSON.stringify(a) === JSON.stringify(b);
const nOutOf = (ev: CountEvent[]) => ev.filter((e) => e.dir === "out").length;
const evTag = (e?: CountEvent) => e ? `${e.dir.toUpperCase()}${e.viol ? " + violation" : ""} · ${CLASSES[e.cls].name}` : "—";

export default function App() {
  const [video, setVideo] = useState<HTMLVideoElement | null>(null);
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [server, setServer] = useState(true);
  const [selId, setSelId] = useState<string | null>(null);
  const [data, setData] = useState<ClipData | null>(null);
  const [expected, setExpected] = useState<ExpectedData | null>(null);
  const [loadError, setLoadError] = useState("");
  const [settings, setSettings] = useState<Settings | null>(null);
  const [draft, setDraft] = useState<Line | null>(null);
  const [compare, setCompare] = useState(false);
  const [tab, setTab] = useState<"prob" | "all">("prob");
  const [copied, setCopied] = useState("");
  const [saving, setSaving] = useState(false);
  const set = (patch: Partial<Settings>) => setSettings((s) => (s ? { ...s, ...patch } : s));
  const sel = runs.find((r) => r.id === selId) ?? null;

  // ---------- runs ----------
  const refresh = useCallback(async () => {
    const { runs, server } = await listRuns();
    setRuns(runs); setServer(server);
    setSelId((id) => id ?? runs.find((r) => r.status === "ready")?.id ?? null);
    return runs;
  }, []);
  useEffect(() => { refresh(); }, [refresh]);
  const processing = runs.some((r) => !["ready", "error"].includes(r.status));
  useEffect(() => {
    if (!processing) return;
    const t = setInterval(refresh, 2000);
    return () => clearInterval(t);
  }, [processing, refresh]);

  // load the selected run's detections once it is ready, and its saved settings
  const ready = sel?.status === "ready";
  useEffect(() => {
    if (!sel || !ready) { setData(null); return; }
    let live = true;
    setData(null); setLoadError(""); setSettings(fromSaved(sel)); setDraft(null);
    setExpected(null);
    if (sel.expected) fetch(sel.expected).then((r) => (r.ok ? r.json() : null)).then((e) => { if (live) setExpected(e); }).catch(() => {});
    fetch(sel.files.data).then((r) => { if (!r.ok) throw new Error(String(r.status)); return r.json(); })
      .then((d) => { if (live) setData(d); })
      .catch(() => live && setLoadError(`Couldn't load the detections for “${sel.name}”. Reload the page to try again.`));
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selId, ready]);

  // ---------- counting ----------
  const [counted, busy] = useDebounced(settings, 80);
  const hasRef = !!(data?.ref?.length && sel?.beltDir);
  const results = useMemo(() => {
    if (!data || !counted || !sel) return null;
    const out: Partial<Record<Version, Result>> = {};
    for (const version of compare ? VERSIONS : [counted.version]) {
      const { events, frames } = run(data.dets, { ...counted.knobs, version, LINE: counted.line, VALID_OUT_CLASSES: counted.valid, TRACK_HISTORY_LEN: 30 }, data.fps);
      const sc = hasRef ? score(reference(data.ref, counted.line, counted.valid, sel.beltDir!, data.fps), events) : null;
      out[version] = { events, frames, sc };
    }
    return out;
  }, [data, counted, compare, hasRef, sel]);
  const shown: Version[] = compare ? VERSIONS : settings ? [settings.version] : [];
  const main = settings && results ? results[compare ? "patched" : settings.version] ?? null : null;
  const diff = useMemo(() => (compare && results?.production && results?.patched ? diffRuns(results.production.events, results.patched.events) : null), [compare, results]);

  const line = draft ?? settings?.line ?? null;
  const fps = data?.fps ?? 15, duration = data ? data.dets.length / fps : 0;
  const seek = (t: number) => { if (video) { video.pause(); video.currentTime = Math.max(0, t - 1.2); } };

  // ---------- timeline ----------
  const timeline: TimelineRow[] = shown.map((v) => {
    const r = results?.[v];
    if (!r) return { label: compare ? NAME[v].toUpperCase() : undefined, ticks: [] };
    const ticks: Tick[] = r.events.map((e, i) => {
      if (!r.sc) return { t: e.t, tone: e.viol ? "warn" : e.dir === "out" ? "out" : "in" };
      const m = r.sc.marks.get(i);
      return { t: e.t, tone: m === "ok" ? "good" : m === "wrong" ? "bad" : "warn" };
    });
    if (r.sc) for (const m of r.sc.missedRefs) ticks.push({ t: m.t, tone: "bad", hollow: true });
    return { label: compare ? NAME[v].toUpperCase() : undefined, ticks };
  });

  // ---------- lists ----------
  const rows = useMemo(() => {
    if (!main) return [];
    return main.events.map((e, i) => ({ t: e.t, dir: e.dir, cls: e.cls, m: (main.sc?.marks.get(i) ?? "ok") as Mark | "missed", viol: e.viol }))
      .concat(main.sc ? main.sc.missedRefs.map((r) => ({ t: r.t, dir: r.dir, cls: r.cls, m: "missed" as const, viol: false })) : [])
      .sort((a, b) => a.t - b.t)
      .filter((r) => tab === "all" || !main.sc || r.m !== "ok");
  }, [main, tab]);
  const diffRows = useMemo(() => (diff ?? []).filter((d) => tab === "all" || (d.kind !== "same" && d.kind !== "class")), [diff, tab]);

  // patched takes IN/OUT from the line's point order, so mostly-IN on a serving line usually means a flipped line
  const flipped = (() => {
    const r = results?.patched; if (!r || hasRef) return false;
    const o = nOutOf(r.events);
    return r.events.length >= 3 && r.events.length - o > 2 * o;
  })();

  // ---------- verdict ----------
  const verdict = useMemo(() => {
    if (compare || !main || !settings) return null;
    const s = main.sc;
    if (!s) return { tone: "warn", head: "No reference for this video.", body: "The counts can't be scored. Turn on Side by side to see where the existing and patched code disagree, and check those moments by eye." };
    let tone = "good", head: string, body = "";
    if (!s.refN) { tone = "warn"; head = "No tray crosses this line."; body = "Draw it across the belt, at right angles to the way trays move."; }
    else if (s.wrong > s.refN / 2 && settings.version === "patched") { tone = "bad"; head = "OUT is on the wrong side."; body = "Most trays are counted backwards. Press Swap points."; }
    else if (s.missed > s.refN * 0.1) { tone = "bad"; head = `${s.missed} crossings missed.`; body = "The line may be too short, too near the frame edge, or where trays are hidden. Tracking limits that are too tight also lose trays."; }
    else if (s.wrong || s.missed || s.extra) { tone = "warn"; head = `${s.wrong + s.missed + s.extra} counting problems.`; body = "Open the Problems list and click a row to see it."; }
    else head = "Every crossing counted, in the right direction.";
    return { tone, head, body };
  }, [compare, main, settings]);

  // ---------- actions ----------
  const copy = (text: string, what: string) => {
    navigator.clipboard?.writeText(text).then(() => { setCopied(what); setTimeout(() => setCopied(""), 1400); }, () => {});
  };
  const envText = () => settings ? [
    `COUNTING_LINE=${roiString(settings.line)}`,
    ...KNOBS.map((k) => `${k.key}=${settings.knobs[k.key]}`),
    `# allowed out classes: ${[...settings.valid].sort().map((c) => CLASSES[c].name).join(", ") || "none"}`,
  ].join("\n") : "";
  const setPoint = (i: 0 | 1, j: 0 | 1, raw: string) => {
    if (!settings) return;
    const lim = j ? 720 : 1280, v = Math.max(0, Math.min(lim, Math.round(+raw || 0)));
    const l = settings.line.map((p) => [...p]) as Line; l[i][j] = v; set({ line: l });
  };
  const save = async () => {
    if (!sel || !settings) return;
    setSaving(true);
    try { const r = await saveSettings(sel.id, settings); setRuns((rs) => rs.map((x) => (x.id === r.id ? r : x))); }
    catch (e) { setLoadError((e as Error).message); }
    finally { setSaving(false); }
  };
  const unsaved = !!(sel && settings && !sameSettings(settings, fromSaved(sel)));

  const [rendering, setRendering] = useState(false);
  const [dlError, setDlError] = useState("");
  const baseName = sel ? (sel.originalName ?? sel.name).replace(/\.[^.]+$/, "") : "run";
  const renderVideo = async () => {
    if (!sel || !counted || !results) return;
    setRendering(true); setDlError("");
    try {
      const job = {
        fps, line: counted.line, version: compare ? "existing-vs-patched" : NAME[counted.version],
        expected: compare && expected ? { out: expected.out, inn: expected.inn, viol: expected.viol,
          crossings: expected.crossings.map((c) => ({ t: c.t, dir: c.dir, viol: c.viol })) } : undefined,
        panels: shown.map((v) => ({
          label: compare ? NAME[v].toUpperCase() : null,
          frames: results[v]!.frames,
          events: results[v]!.events.map((e) => ({ k: e.k, dir: e.dir, viol: e.viol, oid: e.oid })),
        })),
      };
      const r = await renderRun(sel.id, job);
      download(`${r.url}?download=${encodeURIComponent(r.filename)}`, r.filename);
    } catch (e) { setDlError((e as Error).message); }
    finally { setRendering(false); }
  };
  const countsCsv = () => {
    if (!results || !counted) return;
    const head = "version,time_s,frame,direction,fill_class,violation,track_id";
    const lines = shown.flatMap((v) => results[v]!.events.map((e) =>
      [NAME[v].toLowerCase(), e.t.toFixed(2), e.k, e.dir.toUpperCase(), CLASSES[e.cls].name, e.viol ? "yes" : "no", e.oid].join(",")));
    const meta = `# ${sel?.name}; line ${roiString(counted.line)}; ${KNOBS.map((k) => `${k.key}=${counted.knobs[k.key]}`).join(" ")}`;
    download(new Blob([[meta, head, ...lines].join("\n") + "\n"], { type: "text/csv" }), `${baseName} - counts.csv`);
  };

  const nOut = (r?: Result) => r?.events.filter((e) => e.dir === "out").length ?? 0;
  const resultRows: [string, (r: Result) => string | number][] = [
    ["OUT counted", (r) => nOut(r)], ["IN counted", (r) => r.events.length - nOut(r)], ["Violations", (r) => r.events.filter((e) => e.viol).length],
    ...(hasRef ? ([
      ["Right direction", (r: Result) => `${r.sc!.right} / ${r.sc!.refN}`], ["Wrong direction", (r: Result) => r.sc!.wrong],
      ["Missed crossings", (r: Result) => r.sc!.missed], ["Extra counts", (r: Result) => r.sc!.extra],
      ["Violations caught", (r: Result) => `${r.sc!.vTP} / ${r.sc!.refViol}`], ["False violations", (r: Result) => r.sc!.vFP],
    ] as [string, (r: Result) => string | number][]) : []),
  ];

  return (
    <div className="wrap">
      <header>
        <h1>Tray Counter Tuner</h1>
        <p>Pick a saved run, or upload a recording to make one. Move the counting line and change the counter's settings, and every change reruns the whole video with the real counting logic. Turn on Side by side to watch the existing and patched code on the same frames.</p>
      </header>

      {/* the one video element both panels draw from; hidden, the panels show it */}
      {sel && ready && (
        <video ref={setVideo} key={sel.id} className="source-video" src={sel.files.video} muted playsInline preload="auto"
          onLoadedMetadata={(e) => { if (!e.currentTarget.currentTime && sel.start) e.currentTarget.currentTime = sel.start; }} />
      )}

      <div className="grid">
        <section className="stage">
          {!sel || !ready ? (
            <div className="video-wrap placeholder">
              <p>{!sel ? "Loading runs…" : sel.status === "error" ? `Processing failed: ${sel.error ?? "unknown error"}` : `${STAGES[sel.status] ?? sel.status}…`}</p>
            </div>
          ) : (
            <div className={compare ? "panels two" : "panels"}>
              {shown.map((v) => (
                <Panel key={v} video={video} fps={fps} label={compare ? NAME[v].toUpperCase() : undefined} frames={results?.[v]?.frames ?? null}
                  events={results?.[v]?.events ?? []} line={line ?? sel.defaultLine} busy={busy || !results?.[v]}
                  onDraft={setDraft} onCommit={(l) => set({ line: l })} />
              ))}
              {compare && expected && <Expected video={video} data={expected} />}
            </div>
          )}
          <Transport video={ready ? video : null} duration={duration} rows={timeline} />
          <div className="legend fill-legend">
            <span className="lg-title">Box colour</span>
            {FILL_ORDER.map((c) => <span key={c}><i style={{ background: CLASSES[c].color }} />{CLASSES[c].name}</span>)}
            <span><i className="bold-box" />counted OUT</span>
            <span><i className="bold-box in-box" />counted IN</span>
            <span><i className="viol-box">VIOLATION</i></span>
          </div>
          <div className="legend">
            {hasRef ? (<>
              <span><i style={{ background: "var(--good)" }} />Counted correctly</span>
              <span><i style={{ background: "var(--bad)" }} />Wrong direction or missed</span>
              <span><i style={{ background: "var(--warn)" }} />False violation, missed violation or extra count</span>
            </>) : (<>
              <span><i style={{ background: "var(--accent)" }} />OUT</span>
              <span><i style={{ background: "var(--in)" }} />IN</span>
              <span><i style={{ background: "var(--warn)" }} />OUT with a violation</span>
            </>)}
          </div>
          <p className="hint">Drag the numbered handles to move the line, or drag the line itself to slide it. The arrow marks the OUT side. Walking from handle 1 to handle 2, OUT is on your right.</p>

          <div className="panel events">
            {compare ? (<>
              <h2>Existing vs patched <small>{diff ? `${diff.filter((d) => d.kind !== "same" && d.kind !== "class").length} differences` : ""}</small></h2>
              <div className="tabs">
                <button aria-pressed={tab === "prob"} onClick={() => setTab("prob")}>Differences</button>
                <button aria-pressed={tab === "all"} onClick={() => setTab("all")}>All</button>
              </div>
              <div className="list"><table>
                <thead><tr><th>Time</th><th>Existing</th><th>Patched</th><th /></tr></thead>
                <tbody>
                  {!diffRows.length && <tr className="empty"><td colSpan={4}>{diff ? "Both versions count the same trays the same way." : "Counting…"}</td></tr>}
                  {diffRows.map((d, i) => {
                    const [tone, label] = DIFF[d.kind];
                    return (
                      <tr key={i} onClick={() => seek(d.t)}>
                        <td className="mono">{fmtTime(d.t)}</td><td>{evTag(d.a)}</td><td>{evTag(d.b)}</td>
                        <td><span className={`tag ${tone}`}>{label}</span></td>
                      </tr>
                    );
                  })}
                </tbody>
              </table></div>
            </>) : (<>
              <h2>Counted trays <small>{main ? `${main.events.length} counted${main.sc ? ` · reference ${main.sc.refN}` : ""}` : ""}</small></h2>
              {hasRef && <div className="tabs">
                <button aria-pressed={tab === "prob"} onClick={() => setTab("prob")}>Problems</button>
                <button aria-pressed={tab === "all"} onClick={() => setTab("all")}>All</button>
              </div>}
              <div className="list"><table><tbody>
                {!rows.length && <tr className="empty"><td>{main ? (main.sc ? "No problems with these settings." : "Nothing crossed the line.") : "Counting…"}</td></tr>}
                {rows.map((r, i) => {
                  const [tone, label] = MARKS[r.m];
                  return (
                    <tr key={i} onClick={() => seek(r.t)}>
                      <td className="mono">{fmtTime(r.t)}</td>
                      <td><span className={`chip ${r.dir}`}>{r.dir.toUpperCase()}</span></td>
                      <td>{CLASSES[r.cls].name}</td>
                      <td>{main?.sc
                        ? <span className={`tag ${tone}`}>{label}{r.m === "ok" && r.viol ? " · violation" : ""}</span>
                        : r.viol ? <span className="tag warn">Violation</span> : null}</td>
                    </tr>
                  );
                })}
              </tbody></table></div>
            </>)}
          </div>
        </section>

        <aside className="side">
          <RunsPanel runs={runs} server={server} selId={selId} onSelect={setSelId} refresh={refresh} onError={setLoadError} />

          <div className="panel">
            <h2>Result <small>whole video</small></h2>
            {compare ? (
              <div className="tbl"><table className="cmp">
                <thead><tr><th /><th className="num">Existing</th><th className="num">Patched</th></tr></thead>
                <tbody>{resultRows.map(([label, f]) => (
                  <tr key={label}><td>{label}</td>
                    {VERSIONS.map((v) => <td key={v} className="num mono">{results?.[v] ? f(results[v]!) : "–"}</td>)}</tr>
                ))}</tbody>
              </table></div>
            ) : (<>
              <div className="totals">
                <div><b>{main ? nOut(main) : "–"}</b><span>Out</span></div>
                <div><b>{main ? main.events.length - nOut(main) : "–"}</b><span>In</span></div>
                <div><b>{main ? main.events.filter((e) => e.viol).length : "–"}</b><span>Violations</span></div>
              </div>
              {hasRef && <dl className="score">{resultRows.slice(3).map(([label, f]) => (
                <div key={label} style={{ display: "contents" }}><dt>{label}</dt><dd>{main ? f(main) : "–"}</dd></div>
              ))}</dl>}
            </>)}
            {loadError && <div className="verdict bad">{loadError}</div>}
            {verdict && <div className={`verdict ${verdict.tone}`}><b>{verdict.head}</b> {verdict.body}</div>}
            {flipped && <div className="verdict warn"><b>Patched counts most trays IN.</b> If the trays in this video are going out, the line's OUT side is flipped: press Swap points.</div>}
          </div>

          {server && sel && settings && (
            <div className="panel">
              <h2>Save with this run</h2>
              <p className="note">Stores the line, version, allowed classes and tuning with “{sel.name}”, so they load next time it is opened, on any computer that uses this server.</p>
              <div className="row">
                <button className="primary" disabled={!unsaved || saving} onClick={save}>{saving ? "Saving…" : unsaved ? "Save settings" : "Saved"}</button>
                <button disabled={!unsaved} onClick={() => sel && setSettings(fromSaved(sel))}>Revert to saved</button>
              </div>
              {sel.settings?.saved && <p className="note">Last saved {new Date(sel.settings.saved).toLocaleString()}.</p>}
            </div>
          )}
          {sel && ready && settings && (
            <div className="panel">
              <h2>Download</h2>
              <div className="dl">
                {!sel.builtin && server && <a className="btn" href={`runs/${sel.id}/original`}>Original video</a>}
                {server && <button onClick={renderVideo} disabled={rendering || !results || busy}>
                  {rendering ? "Rendering…" : compare ? "Annotated video, side by side" : `Annotated video, ${NAME[settings.version].toLowerCase()}`}</button>}
                <button onClick={countsCsv} disabled={!results || busy}>Counts (CSV)</button>
              </div>
              <p className="note">The annotated video and the counts use the line and settings shown now{compare ? ", for both versions" : ""}, saved or not. Rendering takes a few seconds per minute of video.</p>
              {dlError && <p className="err">{dlError}</p>}
            </div>
          )}

          {settings && (<>
            <div className="panel">
              <h2>Counter version</h2>
              <label className="switch">
                <input type="checkbox" checked={compare} onChange={(e) => { setCompare(e.target.checked); setTab("prob"); }} />
                <span>Side by side: existing and patched</span>
              </label>
              <div className="seg" role="group" aria-label="Counter version">
                {VERSIONS.slice().reverse().map((v) => (
                  <button key={v} disabled={compare} aria-pressed={!compare && settings.version === v} onClick={() => set({ version: v })}>
                    {NAME[v]}
                  </button>
                ))}
              </div>
              <p className="note">{compare
                ? "Both versions run on the same detections with the same line and settings."
                : settings.version === "patched"
                  ? "Direction comes from which side of the line the tray ends up on, so the point order decides what OUT means. The tracker matches across fill classes and ignores tracks left behind by earlier trays."
                  : "Direction comes from comparing the tray's motion with the average motion on screen. Swapping the points changes nothing here, and lone trays going against the flow are misread."}</p>
            </div>

            <div className="panel">
              <h2>Counting line <small>dashboard ROI</small></h2>
              <div className="pts">
                <span /><label>x</label><label>y</label>
                {line && ([0, 1] as const).map((i) => <PointRow key={`${i}-${line[i][0]}-${line[i][1]}`} i={i} p={line[i]} onSet={setPoint} />)}
              </div>
              <div className="row">
                <button onClick={() => set({ line: [settings.line[1], settings.line[0]] })}>Swap points (flip OUT side)</button>
                <button onClick={() => sel && set({ line: sel.defaultLine })}>Reset line</button>
              </div>
              <div className="roi"><code>{line ? roiString(line) : ""}</code><button onClick={() => copy(roiString(settings.line), "roi")}>Copy</button></div>
              <p className="note">Coordinates are in the recorder's 1280 × 720 frame. Paste this value into the camera's ROI field on the dashboard.</p>
            </div>

            <div className="panel">
              <h2>Allowed to go out</h2>
              <div className="classes">
                {CLASSES.map((c) => (
                  <label key={c.id}>
                    <input type="checkbox" checked={settings.valid.includes(c.id)} onChange={(e) => set({
                      valid: e.target.checked ? [...settings.valid, c.id] : settings.valid.filter((v) => v !== c.id),
                    })} />
                    <i style={{ background: c.color }} />{c.name}
                  </label>
                ))}
              </div>
              <p className="note">An OUT crossing by any unticked class is a violation. The existing code allows full and stack (<span className="mono">VALID_OUT_CLASSES</span>).</p>
            </div>

            <div className="panel">
              <h2>Tuning <small>.env</small></h2>
              <div className="knobs">
                {KNOBS.map((k) => {
                  const v = settings.knobs[k.key];
                  return (
                    <div className="knob" key={k.key}>
                      <div className="top">
                        <label className="name" htmlFor={`k_${k.key}`}>{k.key}</label>
                        <span className={`val${v !== k.def ? " changed" : ""}`} title={v !== k.def ? `device value: ${k.def}` : ""}>{v}{k.unit ? ` ${k.unit}` : ""}</span>
                      </div>
                      <input type="range" id={`k_${k.key}`} min={k.min} max={k.max} step={k.step} value={v}
                        onChange={(e) => set({ knobs: { ...settings.knobs, [k.key]: +e.target.value } })} />
                      <p>{k.help}</p>
                    </div>
                  );
                })}
              </div>
              <div className="row">
                <button onClick={() => set({ knobs: DEFAULT_KNOBS, valid: DEFAULT_VALID })}>Reset all to device values</button>
                <button onClick={() => copy(envText(), "env")}>Copy .env lines</button>
                {copied && <span className="copied">Copied</span>}
              </div>
            </div>

          </>)}
        </aside>
      </div>

      <footer>
        Videos are converted to what <span className="mono">recorder.py</span> writes (1280 × 720 at 15 fps) before detection. Boxes come from a background-subtraction stand-in, not the tc_2272 model: it needs a fixed camera, estimates one tray's size from each video, reads fill level from food-coloured pixels and doesn't recognise stacks. The counting code is a line-by-line port of <span className="mono">tracker.py</span> and <span className="mono">line_counter.py</span>, checked to give exactly the same events as the Python. Only the built-in s14 render has a reference to score against.
      </footer>
    </div>
  );
}

function RunsPanel({ runs, server, selId, onSelect, refresh, onError }: {
  runs: RunInfo[]; server: boolean; selId: string | null; onSelect: (id: string) => void;
  refresh: () => Promise<RunInfo[]>; onError: (m: string) => void;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [sent, setSent] = useState<number | null>(null);
  const [err, setErr] = useState("");
  const [confirm, setConfirm] = useState<string | null>(null);

  const upload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;
    setErr(""); setSent(0);
    try {
      const r = await uploadRun(file, name, setSent);
      setFile(null); setName(""); if (fileRef.current) fileRef.current.value = "";
      await refresh(); onSelect(r.id);
    } catch (x) { setErr((x as Error).message); }
    finally { setSent(null); }
  };
  const reprocess = async (id: string) => {
    try { await reprocessRun(id); await refresh(); } catch (x) { onError((x as Error).message); }
  };
  const remove = async (id: string) => {
    try { await deleteRun(id); setConfirm(null); const rs = await refresh(); if (id === selId) onSelect(rs.find((r) => r.status === "ready")?.id ?? rs[0]?.id); }
    catch (x) { onError((x as Error).message); }
  };

  return (
    <div className="panel">
      <h2>Runs <small>{runs.length} saved</small></h2>
      <ul className="runs">
        {runs.map((r) => {
          const pct = r.progress?.total ? Math.round((100 * r.progress.done) / r.progress.total) : 0;
          return (
            <li key={r.id} className={r.id === selId ? "on" : ""}>
              <button className="run" onClick={() => onSelect(r.id)} aria-pressed={r.id === selId}>
                <b>{r.name}</b>
                <span>{r.builtin ? "Built-in · has a reference" : new Date(r.created).toLocaleString()}
                  {r.status !== "ready" && <em className={r.status === "error" ? "bad" : ""}> · {STAGES[r.status] ?? r.status}{r.status === "detecting" ? ` ${pct}%` : ""}</em>}
                  {r.settings?.saved && " · settings saved"}</span>
                {r.status === "detecting" && <i className="bar"><i style={{ width: `${pct}%` }} /></i>}
              </button>
              {!r.builtin && server && (confirm === r.id
                ? <span className="confirm">Delete? <button onClick={() => remove(r.id)}>Delete</button><button onClick={() => setConfirm(null)}>Keep</button></span>
                : <span className="acts">
                    {["ready", "error"].includes(r.status) && <button className="del" title="Run detection again with the current detector; saved settings are kept"
                      aria-label={`Reprocess ${r.name}`} onClick={() => reprocess(r.id)}>Reprocess</button>}
                    <button className="del" aria-label={`Delete ${r.name}`} onClick={() => setConfirm(r.id)}>Delete</button>
                  </span>)}
            </li>
          );
        })}
      </ul>
      {server ? (
        <form className="upload" onSubmit={upload}>
          <label htmlFor="upfile">Upload a recording</label>
          <input id="upfile" ref={fileRef} type="file" accept="video/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          <input id="upname" type="text" placeholder="Name (optional), e.g. SPPG 9 cam1, 12 Oct lunch" value={name} maxLength={120}
            onChange={(e) => setName(e.target.value)} aria-label="Run name" />
          <button className="primary" type="submit" disabled={!file || sent !== null}>
            {sent !== null ? `Uploading ${Math.round(sent * 100)}%` : "Upload and process"}
          </button>
          {err && <p className="err">{err}</p>}
          <p className="note">Any length or format ffmpeg reads. Processing takes about twice the video's length; you can keep tuning other runs meanwhile. The camera must be fixed.</p>
        </form>
      ) : (
        <p className="note">This copy is served as static files, so it can't save runs. Start it with <span className="mono">npm start</span> to upload videos and save settings.</p>
      )}
    </div>
  );
}

// Uncontrolled inputs keyed on the value, so typing isn't fought by re-renders; commits on change/blur.
function PointRow({ i, p, onSet }: { i: 0 | 1; p: [number, number]; onSet: (i: 0 | 1, j: 0 | 1, v: string) => void }) {
  return (
    <>
      <label htmlFor={`p${i}x`}>Point {i + 1}</label>
      <input id={`p${i}x`} type="number" min={0} max={1280} defaultValue={p[0]} onBlur={(e) => onSet(i, 0, e.target.value)}
        onKeyDown={(e) => { if (e.key === "Enter") onSet(i, 0, e.currentTarget.value); }} />
      <input id={`p${i}y`} type="number" min={0} max={720} defaultValue={p[1]} aria-label={`Point ${i + 1} y`}
        onBlur={(e) => onSet(i, 1, e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") onSet(i, 1, e.currentTarget.value); }} />
    </>
  );
}
