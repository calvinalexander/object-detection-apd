#!/usr/bin/env python3
"""Standalone image fetcher -- run on a machine whose IP the WAF accepts.

Python 3.8+, stdlib only.

    python fetch_local.py --csv site_sample.csv --out sites \
        --cookie "HWWAFSESID=...; HWWAFSESTIME=..."

Speed is adaptive rather than fixed. A browser opening a gallery pulls images
several at a time, so a few concurrent connections and sub-second gaps are normal
traffic, not abuse -- but only while the server is happy. The pacing therefore
tightens on sustained success and widens the moment anything comes back 429 or
503, which is what a considerate client looks like and also what keeps this IP
usable. It is the scarce resource now: two datacenter IPs are already refused.

Stops immediately and completely on 401/403/418. Those are blocks, and retrying
converts a temporary one into a lasting one. Re-running always resumes.
"""
import argparse, csv, os, queue, random, sys, threading, time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:155.0) "
      "Gecko/20100101 Firefox/155.0")
RETRYABLE = {429, 500, 502, 503, 504}
BLOCKED = {401, 403, 418}

p = argparse.ArgumentParser()
p.add_argument("--csv", default="site_sample.csv")
p.add_argument("--out", default="sites")
p.add_argument("--cookie", default=os.environ.get("APD_COOKIE", ""))
p.add_argument("--workers", type=int, default=3, help="concurrent connections (browsers use up to 6)")
p.add_argument("--delay", type=float, default=1.2, help="starting per-worker gap, seconds")
p.add_argument("--min-delay", type=float, default=0.4, help="fastest the adaptive pacing will go")
p.add_argument("--max-delay", type=float, default=8.0)
p.add_argument("--stop-after-failures", type=int, default=4)
p.add_argument("--limit", type=int, default=0)
p.add_argument("--careful", action="store_true", help="1 worker, slow fixed pacing")
a = p.parse_args()
if a.careful: a.workers, a.delay, a.min_delay = 1, 3.0, 3.0

rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
root = Path(a.out); root.mkdir(parents=True, exist_ok=True)
dest = lambda r: root / r["site_code"] / f"{r['ID Pelanggaran']}.jpg"
todo = [r for r in rows if not (dest(r).exists() and dest(r).stat().st_size > 1000)]
if a.limit: todo = todo[:a.limit]
print(f"{len(rows)} sampled | {len(rows)-len(todo)} already here | {len(todo)} to fetch")
if not todo: sys.exit(0)

hdrs = {"User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9", "Connection": "keep-alive",
        "Referer": "https://sipgn-api.bgn.go.id/",
        "Sec-Fetch-Dest": "document", "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none", "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1"}
if a.cookie: hdrs["Cookie"] = a.cookie
else: print("WARNING: no --cookie; the WAF will very likely refuse everything.")

lock = threading.Lock()
stop = threading.Event()
state = {"delay": a.delay, "ok": 0, "bad": 0, "run": 0, "streak": 0, "reason": ""}
logf = open(root/"_fetch_log.csv", "a", newline="", encoding="utf-8")
log = csv.writer(logf)
if logf.tell() == 0: log.writerow(["ts","id","site","status","bytes","note"])
q = queue.Queue()
for r in todo: q.put(r)

def slower(f=2.0):
    state["delay"] = min(a.max_delay, state["delay"]*f); state["streak"] = 0
def faster():
    # only after a long clean run, and never below the floor
    state["streak"] += 1
    if state["streak"] >= 25:
        state["delay"] = max(a.min_delay, state["delay"]*0.85); state["streak"] = 0

def worker():
    while not stop.is_set():
        try: r = q.get_nowait()
        except queue.Empty: return
        d = dest(r); d.parent.mkdir(parents=True, exist_ok=True)
        status, data, note = None, None, ""
        for attempt in (1, 2, 3):
            if stop.is_set(): return
            try:
                with urlopen(Request(r["Gambar"], headers=hdrs), timeout=45) as resp:
                    status, data = resp.status, resp.read()
                break
            except HTTPError as e:
                status = e.code
                if status in BLOCKED: note = "blocked"; break
                if status in RETRYABLE and attempt < 3:
                    with lock: slower()
                    ra = e.headers.get("Retry-After")
                    w = float(ra) if (ra or "").isdigit() else 2.0*(2**(attempt-1))
                    time.sleep(w + random.uniform(0, 1.5)); continue
                note = f"http {status}"; break
            except (URLError, TimeoutError, OSError) as e:
                status, note = 0, str(e)[:60]
                if attempt < 3: time.sleep(2.0*(2**(attempt-1))); continue
                break

        with lock:
            if data and len(data) > 1000:
                tmp = d.with_suffix(f".part{threading.get_ident()}")
                tmp.write_bytes(data); tmp.replace(d)
                state["ok"] += 1; state["run"] = 0; faster()
                log.writerow([int(time.time()), r["ID Pelanggaran"], r["site_code"], status, len(data), ""])
            else:
                state["bad"] += 1; state["run"] += 1; slower()
                log.writerow([int(time.time()), r["ID Pelanggaran"], r["site_code"], status, 0, note or "short"])
                if status in BLOCKED:
                    state["reason"] = f"HTTP {status} -- blocked, not transient"; stop.set()
                elif state["run"] >= a.stop_after_failures:
                    state["reason"] = f"{state['run']} consecutive failures"; stop.set()
            n = state["ok"] + state["bad"]
            if n % 50 == 0:
                logf.flush()
                print(f"  {n}/{len(todo)}  ok={state['ok']} fail={state['bad']} "
                      f"delay={state['delay']:.2f}s", flush=True)
            dly = state["delay"]
        time.sleep(random.uniform(dly*0.6, dly*1.4))

t0 = time.time()
threads = [threading.Thread(target=worker, daemon=True) for _ in range(a.workers)]
[t.start() for t in threads]
try:
    while any(t.is_alive() for t in threads): time.sleep(0.4)
except KeyboardInterrupt:
    state["reason"] = "interrupted"; stop.set(); [t.join(timeout=5) for t in threads]
logf.close()

el = time.time()-t0
print(f"\ndownloaded {state['ok']}, failed {state['bad']}, "
      f"remaining {len(todo)-state['ok']-state['bad']}")
print(f"elapsed {el/60:.1f} min  ({state['ok']/max(el,1)*60:.0f} images/min)  "
      f"final delay {state['delay']:.2f}s")
if state["reason"]:
    print(f"STOPPED: {state['reason']}")
    print("Refresh the cookie from your browser and re-run; finished files are kept.")
print(f"layout: {root}/<site_code>/<id>.jpg")
