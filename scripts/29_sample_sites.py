"""Build a compact, site-stratified sample from the violation export.

Three constraints, in order of importance:

1. SITE SEPARATION. The site code in the image URL is an administrative code, so
   location identity is exact -- no perceptual hashing, and none of the leakage
   that invalidated the earlier splits. Sites become the grouping unit.

2. LABEL RICHNESS. Every row is a violation, so a naive sample teaches only what
   non-compliance looks like. But the violation *type* is informative in reverse:
   a record flagged only for "Masker" implies the head cap and gloves were
   acceptable, so it carries two positives and one negative. A record flagged for
   all three carries no positives at all. Preference therefore runs
   single-violation first, and the rarer single type ahead of the common one.

3. TEMPORAL SPREAD. Consecutive violations at one site are frames of the same
   person in the same minute. A minimum gap between selections buys distinct
   people, poses and lighting instead of near-duplicates.
"""
import argparse, collections, json, re, sys
from pathlib import Path
import pandas as pd

MONTHS = {"Januari":1,"Februari":2,"Maret":3,"April":4,"Mei":5,"Juni":6,"Juli":7,
          "Agustus":8,"September":9,"Oktober":10,"November":11,"Desember":12}
TYPES = {"Penutup Kepala":"head_cap", "Masker":"mask", "Sarung Tangan":"gloves"}

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="data/Pelanggaran_APD_SPPG_Pilihan_20260908-20260908.csv")
ap.add_argument("--sites", type=int, default=60, help="how many sites to draw from")
ap.add_argument("--per-site", type=int, default=30)
ap.add_argument("--min-gap-min", type=int, default=20, help="minimum minutes between picks at one site")
ap.add_argument("--min-records", type=int, default=50, help="skip sites thinner than this")
ap.add_argument("--out", default="artifacts/site_sample.csv")
args = ap.parse_args()

df = pd.read_csv(args.csv)
df["code"] = df["Gambar"].str.extract(r"/stream/([0-9.]+)/")
df["prov"] = df["code"].str[:2]

def parse_ts(s):
    m = re.match(r"(\d+)\s+(\w+)\s+(\d{4})\s+pukul\s+(\d+):(\d+)", str(s))
    if not m: return pd.NaT
    d, mon, y, hh, mm = m.groups()
    return pd.Timestamp(int(y), MONTHS.get(mon,1), int(d), int(hh), int(mm))
df["ts"] = df["Tanggal"].map(parse_ts)
df["types"] = df["Jenis Pelanggaran"].str.split(", ")
df["n_viol"] = df["types"].str.len()
print(f"{len(df):,} records, {df['code'].nunique()} sites, {df['prov'].nunique()} provinces")
print(f"violations per record: {dict(df['n_viol'].value_counts().sort_index())}")

# Quotas per attribute, not a global preference order. Two things force this:
# a single global ranking just fills the sample with the commonest type, and
# there is no standalone "Penutup Kepala" category at all -- head_cap violations
# exist only inside multi-violation records, so they have to be asked for
# explicitly or they never appear.
df["has_cap"]  = df["types"].map(lambda t: "Penutup Kepala" in t)
df["has_mask"] = df["types"].map(lambda t: "Masker" in t)
df["has_glov"] = df["types"].map(lambda t: "Sarung Tangan" in t)
def bucket(r):
    if r["has_cap"]:  return "cap_viol"       # only source of head_cap negatives
    if r["has_mask"] and not r["has_glov"]: return "mask_viol"
    if r["has_glov"] and not r["has_mask"]: return "glov_viol"
    return "mixed"
df["bucket"] = df.apply(bucket, axis=1)
QUOTA = {"cap_viol": 0.34, "mask_viol": 0.33, "glov_viol": 0.33}
print(f"buckets: {dict(df['bucket'].value_counts())}")

# Choose sites: spread across provinces, then by record count.
counts = df.groupby("code").size()
eligible = counts[counts >= args.min_records].index
by_prov = collections.defaultdict(list)
for c in eligible:
    by_prov[c[:2]].append(c)
for p in by_prov:
    by_prov[p].sort(key=lambda c: -counts[c])
chosen, i = [], 0
while len(chosen) < args.sites:                          # round-robin over provinces
    added = False
    for p in sorted(by_prov):
        if i < len(by_prov[p]) and len(chosen) < args.sites:
            chosen.append(by_prov[p][i]); added = True
    if not added: break
    i += 1
print(f"\neligible sites (>= {args.min_records} records): {len(eligible)}")
print(f"selected {len(chosen)} sites across {len({c[:2] for c in chosen})} provinces")

rows = []
for code in chosen:
    sub = df[df["code"] == code]
    picked, times = [], []
    # Fill each bucket to quota, fewest-violations first inside the bucket so a
    # record still implies as many compliant attributes as possible.
    for b, share in QUOTA.items():
        want = max(1, round(args.per_site * share))
        cand = sub[sub["bucket"] == b].sort_values(["n_viol", "ts"])
        got = 0
        for _, r in cand.iterrows():
            if pd.isna(r["ts"]) or got >= want: continue
            if any(abs((r["ts"] - t).total_seconds()) < args.min_gap_min*60 for t in times):
                continue
            picked.append(r); times.append(r["ts"]); got += 1
    # top up from anything remaining if a bucket ran dry at this site
    if len(picked) < args.per_site:
        for _, r in sub.sort_values(["n_viol","ts"]).iterrows():
            if len(picked) >= args.per_site: break
            if pd.isna(r["ts"]) or r["ID Pelanggaran"] in {p["ID Pelanggaran"] for p in picked}:
                continue
            if any(abs((r["ts"] - t).total_seconds()) < args.min_gap_min*60 for t in times):
                continue
            picked.append(r); times.append(r["ts"])
    rows.extend(picked)

out = pd.DataFrame(rows)
out["site_code"] = out["code"]; out["site_name"] = out["Nama SPPG"]
out["violations"] = out["types"].map(lambda t: ",".join(TYPES.get(x,x) for x in t))
out["implied_ok"] = out["types"].map(
    lambda t: ",".join(v for k,v in TYPES.items() if k not in t))
keep = ["site_code","site_name","prov","ts","ID Pelanggaran","violations","implied_ok","n_viol","Gambar"]
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
out[keep].to_csv(args.out, index=False)

print(f"\nsampled {len(out)} images from {out['site_code'].nunique()} sites")
print(f"  per site: min {out.groupby('site_code').size().min()}  "
      f"median {int(out.groupby('site_code').size().median())}  max {out.groupby('site_code').size().max()}")
print(f"\nviolations per sampled record: {dict(out['n_viol'].value_counts().sort_index())}")
print(f"\nlabel yield (implied compliant attributes come free):")
v = collections.Counter(); o = collections.Counter()
for _, r in out.iterrows():
    for x in r["violations"].split(","):
        if x: v[x] += 1
    for x in r["implied_ok"].split(","):
        if x: o[x] += 1
print(f"  {'attribute':<12}{'violation':>11}{'implied ok':>12}")
for a in ("head_cap","mask","gloves"):
    print(f"  {a:<12}{v[a]:>11}{o[a]:>12}")
print(f"\nwrote {args.out}")
