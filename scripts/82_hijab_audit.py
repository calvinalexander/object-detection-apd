"""Does the training data actually draw the hijab/cap boundary the rule states?

33 of 51 head_cap errors are false positives on headscarves: the model says
compliant, the rule says violation. Before changing the model, check the labels.
`absent` currently mixes bare heads and hijabs -- visually opposite things -- and
if hijabs were labelled inconsistently, or are rare, there is no boundary to learn.

Each sampled crop is categorised by covering TYPE (a question the rule can answer
unambiguously), then cross-tabulated against the present/absent label it carries.
"""
import argparse, json, random, sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import glob, os, re, subprocess
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apd.labeller import render_sheet, resolve_token, _FENCE
from apd.schema import HEAD_CAP_RULE, assert_contract

assert_contract()
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/crops_plus_sdk")
ap.add_argument("--n", type=int, default=160)
ap.add_argument("--per-sheet", type=int, default=8)
ap.add_argument("--model", default="opus")
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--out", default="artifacts/hijab_audit.jsonl")
a = ap.parse_args()

TYPES = ["food_cap", "hijab_only", "cap_over_hijab", "street_hat", "bare", "unclear"]
SYSTEM = """\
You categorise head coverings in CCTV crops from commercial kitchens.

You will be given one contact sheet with numbered crops. For each crop, say what
the person is wearing on their head. Use exactly one of these labels:

  food_cap        a food-grade covering that contains the hair: bouffant cap,
                  hairnet, surgical cap, hygiene skullcap
  hijab_only      a hijab or headscarf with NO food-grade cap over it
  cap_over_hijab  a food-grade cap worn OVER a hijab
  street_hat      a baseball cap, peaked or brimmed cap, beanie, or bandana
  bare            no head covering
  unclear         too blurred, too small, or the head is not visible

Report what you SEE. Do not judge compliance; that is decided from your answer.

Return ONLY a JSON object mapping slot number (as a string) to one label.
Example for 3 slots: {"1":"food_cap","2":"hijab_only","3":"bare"}
"""

def ask(sheet, n):
    prompt = (f"Read the image at {sheet.resolve()} and categorise all {n} numbered "
              f"crops. Return the JSON object and nothing else.")
    cmd = ["claude", "-p", "--output-format", "json", "--model", a.model,
           "--allowedTools", "Read", "--exclude-dynamic-system-prompt-sections",
           "--system-prompt-snapshot", "on", "--system-prompt", SYSTEM]
    env = dict(os.environ)
    if (t := resolve_token()): env["CLAUDE_CODE_OAUTH_TOKEN"] = t
    try:
        p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=600, env=env)
        env_json = json.loads(p.stdout)
        body = _FENCE.sub("", (env_json.get("result") or "").strip())
        m = re.search(r"\{.*\}", body, re.S)
        raw = json.loads(m.group(0)) if m else {}
        return {k: v for k, v in raw.items() if v in TYPES}, env_json.get("total_cost_usd") or 0.0
    except Exception:
        return {}, 0.0

items = []
for cls in ("present", "absent"):
    for split in ("train", "val"):
        for p in glob.glob(f"{a.data}/head_cap/{split}/{cls}/*__o.jpg"):
            items.append((p, cls))
random.Random(7).shuffle(items)
items = items[:a.n]
print(f"auditing {len(items)} crops "
      f"({sum(1 for _,c in items if c=='present')} labelled present / "
      f"{sum(1 for _,c in items if c=='absent')} absent)\n")

work = Path("artifacts/hijab_sheets"); work.mkdir(parents=True, exist_ok=True)
groups = [items[i:i+a.per_sheet] for i in range(0, len(items), a.per_sheet)]
def run(ig):
    i, grp = ig
    s = work / f"h{i:03d}.jpg"
    render_sheet([Path(p) for p, _ in grp], s, cell=300, cols=4)
    v, c = ask(s, len(grp))
    return grp, v, c

rows, cost = [], 0.0
with ThreadPoolExecutor(a.workers) as ex:
    for grp, v, c in ex.map(run, enumerate(groups)):
        cost += c
        for k, (p, cls) in enumerate(grp):
            t = v.get(str(k+1))
            if t: rows.append({"file": p, "label": cls, "type": t})
Path(a.out).write_text("".join(json.dumps(r)+"\n" for r in rows))

x = defaultdict(Counter)
for r in rows: x[r["type"]][r["label"]] += 1
print(f"{'covering type':16} {'labelled present':>17} {'labelled absent':>16}   what the RULE says")
RULE = {"food_cap":"present","cap_over_hijab":"present","hijab_only":"absent",
        "street_hat":"absent","bare":"absent","unclear":"-"}
wrong = 0
for t in TYPES:
    pr, ab = x[t]["present"], x[t]["absent"]
    if pr+ab == 0: continue
    exp = RULE[t]
    bad = ab if exp=="present" else (pr if exp=="absent" else 0)
    wrong += bad
    flag = f"  <- {bad} mislabelled" if bad else ""
    print(f"{t:16} {pr:17d} {ab:16d}   {exp}{flag}")
print(f"\n{len(rows)} audited, {wrong} contradict the rule ({wrong/max(1,len(rows)):.0%})   ${cost:.2f}")
EOF_MARK = None
