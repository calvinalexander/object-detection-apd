"""Fetch sampled violation images politely enough not to get the IP blocked.

The existing scraper in playwright-sppg-dashboard already gets the important
parts right -- .part files renamed atomically, resume by skipping what exists,
exponential backoff, Retry-After. Two things it lacks matter here:

  jitter          a request exactly every 0.5s is a fingerprint. Real browsing is
                  irregular, so every delay is randomised and a longer pause is
                  taken every N requests.

  circuit breaker on a soft block the server starts refusing; continuing to retry
                  each URL and moving on is what converts that into a hard ban.
                  Consecutive failures stop the whole run, and 401/403 stops it
                  immediately -- those never resolve by trying harder.

Default pacing is ~1 request per 4s, so 1400 images take roughly 1.5 hours. That
is deliberate: this is a one-off dataset build, not a latency-sensitive job.
"""
from __future__ import annotations
import argparse, csv, random, sys, time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")
RETRYABLE = {429, 500, 502, 503, 504}
FATAL = {401, 403}          # a block. Retrying makes it worse.

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="artifacts/site_sample.csv")
ap.add_argument("--out", default="data/sites")
ap.add_argument("--url-column", default="Gambar")
ap.add_argument("--rate-min", type=float, default=2.5, help="min seconds between requests")
ap.add_argument("--rate-max", type=float, default=5.5)
ap.add_argument("--pause-every", type=int, default=75, help="take a long pause every N requests")
ap.add_argument("--pause-secs", type=float, default=60.0)
ap.add_argument("--max-retries", type=int, default=3)
ap.add_argument("--stop-after-failures", type=int, default=5,
                help="consecutive failures that abort the whole run")
ap.add_argument("--limit", type=int, default=0, help="stop after N downloads (0 = all)")
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()

rows = list(csv.DictReader(open(args.csv)))
out_root = Path(args.out)
log_path = out_root / "_fetch_log.csv"
out_root.mkdir(parents=True, exist_ok=True)

def dest_for(r):
    # one directory per site: the split unit is the site, so the layout should be too
    return out_root / r["site_code"] / f"{r['ID Pelanggaran']}.jpg"

todo = [r for r in rows if not (dest_for(r).exists() and dest_for(r).stat().st_size > 0)]
print(f"{len(rows)} sampled  |  {len(rows)-len(todo)} already on disk  |  {len(todo)} to fetch")
est = len(todo)*(args.rate_min+args.rate_max)/2 + (len(todo)//args.pause_every if args.pause_every else 0)*args.pause_secs
print(f"pacing {args.rate_min}-{args.rate_max}s + {args.pause_secs:.0f}s every {args.pause_every}"
      f"  ->  about {est/3600:.1f} h")
if args.dry_run:
    print("\ndry run; nothing fetched. Sample destinations:")
    for r in todo[:3]: print("   ", dest_for(r))
    sys.exit(0)

def fetch(url):
    req = Request(url, headers={
        "User-Agent": UA, "Accept": "image/avif,image/webp,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,id;q=0.8", "Connection": "keep-alive"})
    with urlopen(req, timeout=45) as resp:
        return resp.status, resp.read()

log = open(log_path, "a", newline="")
w = csv.writer(log)
if log.tell() == 0: w.writerow(["ts","id","site_code","status","bytes","note"])

done = fails = consecutive = 0
for i, r in enumerate(todo, 1):
    if args.limit and done >= args.limit:
        print(f"\nreached --limit {args.limit}"); break
    url = r[args.url_column]; dst = dest_for(r)
    dst.parent.mkdir(parents=True, exist_ok=True)
    status, note, data = None, "", None
    for attempt in range(1, args.max_retries + 1):
        try:
            status, data = fetch(url); note = ""; break
        except HTTPError as e:
            status = e.code
            if status in FATAL:
                note = "fatal"; break
            if status in RETRYABLE and attempt < args.max_retries:
                ra = e.headers.get("Retry-After")
                wait = float(ra) if (ra or "").isdigit() else 2.0 * (2 ** (attempt-1))
                wait += random.uniform(0, 1.5)
                print(f"  [{status}] backing off {wait:.1f}s (attempt {attempt})", flush=True)
                time.sleep(wait); continue
            note = f"http {status}"; break
        except (URLError, TimeoutError, OSError) as e:
            status = 0; note = str(e)[:60]
            if attempt < args.max_retries:
                time.sleep(2.0*(2**(attempt-1)) + random.uniform(0,1.5)); continue
            break

    if data:
        tmp = dst.with_suffix(".part")
        tmp.write_bytes(data); tmp.replace(dst)          # atomic: no half files on disk
        done += 1; consecutive = 0
        w.writerow([int(time.time()), r["ID Pelanggaran"], r["site_code"], status, len(data), ""])
    else:
        fails += 1; consecutive += 1
        w.writerow([int(time.time()), r["ID Pelanggaran"], r["site_code"], status, 0, note])
        if status in FATAL:
            print(f"\nSTOP: HTTP {status} -- this is a block, not a transient error.")
            print("Retrying would make it worse. Resume later; finished files are kept.")
            break
        if consecutive >= args.stop_after_failures:
            print(f"\nSTOP: {consecutive} consecutive failures. Circuit breaker tripped.")
            break
    log.flush()

    if i % 25 == 0:
        print(f"  {i}/{len(todo)}  ok={done} fail={fails}", flush=True)
    if args.pause_every and done and done % args.pause_every == 0:
        p = args.pause_secs * random.uniform(0.8, 1.3)
        print(f"  ... long pause {p:.0f}s after {done}", flush=True)
        time.sleep(p)
    time.sleep(random.uniform(args.rate_min, args.rate_max))

log.close()
print(f"\ndownloaded {done}, failed {fails}, remaining {len(todo)-done-fails}")
print(f"images: {out_root}/<site_code>/<id>.jpg   log: {log_path}")
print("re-run to resume; existing files are skipped.")
