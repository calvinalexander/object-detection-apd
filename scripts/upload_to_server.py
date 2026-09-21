#!/usr/bin/env python3
"""Copy a local folder up to the training server. Run this on YOUR machine.

    python upload_to_server.py sites

Windows-friendly: stdlib only, uses the OpenSSH client that ships with Windows 10+
(scp / ssh on PATH). Nothing to install.

It tars first rather than doing `scp -r`. For ~1,400 small JPEGs that matters a
lot -- scp negotiates per file, so a directory copy spends most of its time on
round trips rather than bytes. One compressed archive is typically several times
faster and gives a single progress bar.
"""
import argparse, os, subprocess, sys, tarfile, tempfile, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("folder", help="local folder to upload, e.g. sites")
ap.add_argument("--host", default="ubuntu@3.138.66.83")
ap.add_argument("--dest", default="~/tryan/apd/data", help="parent dir on the server")
ap.add_argument("--key", default=None, help="path to your .pem/.ppk if needed")
ap.add_argument("--keep-archive", action="store_true")
a = ap.parse_args()

src = Path(a.folder).resolve()
if not src.is_dir(): sys.exit(f"not a folder: {src}")
files = [p for p in src.rglob("*") if p.is_file()]
total = sum(p.stat().st_size for p in files)
print(f"{src.name}: {len(files)} files, {total/1e6:.1f} MB")
if not files: sys.exit("nothing to upload")

ssh_opts = (["-i", a.key] if a.key else [])
tmp = Path(tempfile.gettempdir()) / f"{src.name}_{int(time.time())}.tar.gz"
print(f"packing -> {tmp.name}")
with tarfile.open(tmp, "w:gz") as t:
    t.add(src, arcname=src.name)
print(f"  {tmp.stat().st_size/1e6:.1f} MB compressed "
      f"({100*tmp.stat().st_size/max(total,1):.0f}% of original)")

print(f"uploading to {a.host}:{a.dest}/")
r = subprocess.run(["scp", *ssh_opts, str(tmp), f"{a.host}:{a.dest}/"])
if r.returncode: sys.exit("scp failed -- check --host / --key and that ssh works")

print("extracting on the server")
remote = f"cd {a.dest} && tar xzf {tmp.name} && rm -f {tmp.name} && " \
         f"echo EXTRACTED && find {src.name} -type f | wc -l"
r = subprocess.run(["ssh", *ssh_opts, a.host, remote])
if r.returncode: sys.exit("remote extract failed; the archive is still on the server")

if not a.keep_archive:
    try: tmp.unlink()
    except OSError: pass
print(f"\ndone -- {a.dest}/{src.name} on the server")
