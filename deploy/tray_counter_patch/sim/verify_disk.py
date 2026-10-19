"""Run the sim with the on-disk patched files (deploy/tray_counter_patch) and compare to the page numbers."""
import os, sys, json, random, numpy as np
sys.argv = ["x"]; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tlabcode")
sim.PATCHED_LC = sim._patched_module(f"{P}/line_counter.py", "lc_disk", [])
sim.PATCHED_TR = sim._patched_module(f"{P}/tracker.py", "tr_disk", [])
random.seed(7); T = sim.build_scenario(); gt = sim.gt_events(T)
ref = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_30runs.json")))["profiles"]
keys = ["sys_out", "sys_in", "correct", "wrong_dir", "extra", "missed", "viol_tp", "viol_fp"]
for prof in sim.PROFILES:
    rs = [sim.score(T, gt, sim.run(T, prof, s, variant="patched"))[0] for s in range(30)]
    m = {k: np.mean([r[k] for r in rs]) for k in keys}
    print(f"{prof:9s} disk   " + "  ".join(f"{k}={m[k]:.1f}" for k in keys))
    print(f"{'':9s} page   " + "  ".join(f"{k}={ref['patched/' + prof]['mean'][k]:.1f}" for k in keys))
