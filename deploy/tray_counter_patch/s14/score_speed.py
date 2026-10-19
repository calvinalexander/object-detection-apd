"""Score a sped-up run against the 1x reference crossings, with times divided by the speed factor.
Usage: python score_speed.py <events.json> <speed> [window_s]"""
import json, sys
CUTS = [(0.04, 0), (0.17, 3), (0.27, 2), (0.32, 4)]
fc = lambda f: next((k for c, k in CUTS if f < c), 1)
R = json.load(open(sys.argv[1])); sp = float(sys.argv[2]); win = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5 / sp
ref = [dict(t=r["t"] / sp, dir=r["dir"], viol=r["dir"] == "out" and fc(r["fill"]) not in (1, 5))
       for r in json.load(open("s14_reference.json"))]
out = {}
for v in ("production", "patched"):
    E = R[v]; used = set(); ok = wrong = miss = tp = fn = fp = 0; issues = []
    for r in ref:
        j = next((j for j, e in enumerate(E) if j not in used and abs(e["t"] - r["t"]) < win), None)
        if j is None:
            miss += 1; fn += r["viol"]; issues.append(("missed", round(r["t"], 2), r["dir"])); continue
        used.add(j); e = E[j]; ev = e["dir"] == "out" and e["cls"] not in (1, 5)
        if e["dir"] == r["dir"]: ok += 1
        else: wrong += 1; issues.append(("wrong", round(r["t"], 2), r["dir"]))
        if r["viol"]: tp += ev; fn += not ev
        else: fp += ev
    extra = [(e["t"], e["dir"]) for j, e in enumerate(E) if j not in used]
    fp += sum(1 for j, e in enumerate(E) if j not in used and e["dir"] == "out" and e["cls"] not in (1, 5))
    n_out = sum(e["dir"] == "out" for e in E)
    out[v] = dict(out=n_out, inn=len(E) - n_out, right=ok, wrong=wrong, missed=miss, extra=len(extra), viol_tp=tp, viol_fn=fn, viol_fp=fp)
    print(f"{v:10s} OUT={n_out} IN={len(E)-n_out} right={ok}/{len(ref)} wrong={wrong} missed={miss} extra={len(extra)} "
          f"viol caught={tp}/{sum(r['viol'] for r in ref)} false={fp}")
    print("   issues:", issues[:20]); print("   extra:", extra[:20])
json.dump(dict(ref_n=len(ref), **out), open(sys.argv[1].replace("_events", "_score"), "w"), indent=1)
