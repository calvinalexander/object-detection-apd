"""Collect every audit-driven data fix in ONE place: data/clean/. Originals are never touched.

    data/clean/mask_face/g0.8/     from data/mask_face/g0.8            (audit: artifacts/mask_face_audit/all)
    data/clean/mask_face/test_manifest.json
    data/clean/head_cap/           from data/crops_plus_sdk/head_cap   (audit: artifacts/headcap_audit)
    data/clean/det7/               from data/det7                      (audits: artifacts/det7_audit/*)
    data/clean/CHANGES.csv         one row per change, with the reviewer's note
    data/clean/README.md

Images are hard links (no extra disk; the originals are only ever read); label
files and manifests are new files. Re-running rebuilds data/clean from scratch.

Rules
  classifier sets (mask_face, head_cap), per audited face/head:
    ok -> kept;  wrong_class -> moved to the other class;  resolved_no_cap -> absent only;
    wrong_face / not_head / unclear -> dropped (with all augmented copies)
  detector set (det7), per audited box:
    wrong_class -> class flipped;  not_hand (box not on a hand/face) -> box removed;
    ok / unclear -> kept. An unclear box is NOT removed: deleting it would teach the
    detector that the object there is background.

    python scripts/115_build_clean_data.py
"""
import collections, csv, json, os, re, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"data/clean"
changes = []                                   # (dataset, item, action, detail, note)


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try: os.link(src, dst)
    except OSError: shutil.copy2(src, dst)


def read(path, key="id"):
    rows = list(csv.DictReader(open(path)))
    return rows if key is None else {r[key]: r for r in rows}


# ------------------------------------------------------------------ mask_face
def mask_face():
    A = ROOT/"artifacts/mask_face_audit/all"
    faces, ver = read(A/"faces.csv"), read(A/"verdicts.csv")
    assert set(faces) == set(ver), "mask_face audit incomplete"
    key = {(r["frame"][:60], r["face"], r["split"]): i for i, r in faces.items()}
    n = collections.Counter()
    for p in sorted((ROOT/"data/mask_face/g0.8").rglob("*.jpg")):
        split, cls = p.parts[-3], p.parts[-2]
        m = re.match(r"(.+)_(\d+)_([nx])(\d)\.jpg$", p.name)
        i = key.get((m.group(1), m.group(2), split))
        v = ver[i]["verdict"] if i else "not_audited"
        if v in ("wrong_face", "unclear", "not_audited"): n[(split, "dropped")] += 1; continue
        if v == "wrong_class": cls = "absent" if cls == "present" else "present"
        link(p, OUT/"mask_face/g0.8"/split/cls/p.name); n[(split, cls)] += 1
    for i, r in faces.items():
        v = ver[i]["verdict"]
        if v != "ok":
            act = "relabel mask->no_mask" if v == "wrong_class" else "drop"
            changes.append(("mask_face", f"face#{i} {r['split']} {r['frame']}#{r['face']}", act, v, ver[i]["note"]))
    man = json.loads((ROOT/"data/mask_face/test_manifest.json").read_text()); out = []
    for e in man:
        i = key.get((e["stem"][:60], str(e["face"]), "test")); v = ver[i]["verdict"] if i else "not_audited"
        if v in ("wrong_face", "unclear", "not_audited"): continue
        if v == "wrong_class": e = {**e, "present": not e["present"], "relabelled": True}
        out.append(e)
    (OUT/"mask_face/test_manifest.json").write_text(json.dumps(out))
    print("mask_face", dict(n), f"test manifest {len(man)} -> {len(out)}")


# ------------------------------------------------------------------ head_cap
def head_cap():
    A = ROOT/"artifacts/headcap_audit"; D = ROOT/"data/crops_plus_sdk/head_cap"
    heads, ver = read(A/"heads.csv"), read(A/"verdicts.csv")
    assert set(heads) == set(ver), "head_cap audit incomplete"
    files = collections.defaultdict(list)
    for p in D.glob("*/*/*.jpg"):
        m = re.match(r"(.+?)__([a-z0-9]+)\.jpg$", p.name)
        files[(p.parts[-3], m.group(1) if m else p.stem)].append(p)
    n = collections.Counter()
    for i, r in heads.items():
        v = ver[i]["verdict"]; ps = files[(r["split"], r["base"])]
        if v in ("not_head", "unclear"):
            n[(r["split"], "dropped")] += len(ps)
            changes.append(("head_cap", f"head#{i} {r['split']} {r['base']}", "drop", v, ver[i]["note"])); continue
        seen = set()
        for p in ps:
            if p.name in seen: continue            # a conflicting head sits in both folders under one name
            seen.add(p.name)
            cls = p.parts[-2]
            if v == "wrong_class": cls = "absent" if cls == "present" else "present"
            if v == "resolved_no_cap": cls = "absent"
            if v == "resolved_cap": cls = "present"
            link(p, OUT/"head_cap"/r["split"]/cls/p.name); n[(r["split"], cls)] += 1
        if v != "ok":
            changes.append(("head_cap", f"head#{i} {r['split']} {r['base']}",
                            {"wrong_class": "relabel " + ("cap->no_cap" if r["label"] == "cap" else "no_cap->cap"),
                             "resolved_no_cap": "conflict -> no_cap only", "resolved_cap": "conflict -> cap only"}[v],
                            v, ver[i]["note"]))
    print("head_cap", dict(n))


# ------------------------------------------------------------------ det7
def det7_extra_edits(edits):
    """Two audits that live outside artifacts/det7_audit.

    head_cap: the audited head crops (artifacts/headcap_audit) match det7 people by
              frame + person index, so the user's cap decisions carry over.
    gloves  : artifacts/dark_gloves, where reviewers checked every dark-looking glove
              box; the ones that are not gloves at all are fixed here.
    """
    import re
    D = ROOT/"data/det7"
    lines = {}                                   # (split, stem) -> [class per line]
    for sp in ("train", "val"):
        for lp in (D/"labels"/sp).glob("*.txt"):
            lines[(sp, lp.stem)] = [int(l.split()[0]) for l in lp.read_text().splitlines() if l.strip()]

    # ---- head_cap from the head-crop audit
    heads = read(ROOT/"artifacts/headcap_audit/heads.csv")
    ver = {r["id"]: r["verdict"] for r in csv.DictReader(open(ROOT/"artifacts/headcap_audit/verdicts.csv"))}
    n = collections.Counter()
    for i, r in heads.items():
        m = re.match(r"(.+)_(\d+)$", re.sub(r"^sdk_", "", r["base"]))
        if not m: continue
        stem = f"{m.group(1)}_p{m.group(2)}"
        v = ver[i]
        for sp in ("train", "val"):
            cl = lines.get((sp, stem))
            if not cl: continue
            for k, c in enumerate(cl):
                if c not in (2, 5): continue
                if v == "wrong_class":
                    edits[(sp, stem)][k] = (c, 5 if c == 2 else 2); n["head_cap relabelled"] += 1
                elif v in ("not_head", "unclear"):
                    edits[(sp, stem)][k] = (c, None); n["head_cap box dropped (" + v + ")"] += 1
                elif v == "resolved_no_cap" and c == 2:
                    edits[(sp, stem)][k] = (c, 5); n["head_cap conflict -> no_cap"] += 1
            changes.append(("det7", f"head_cap audit {stem}", {"wrong_class": "relabel", "not_head": "drop",
                            "unclear": "drop", "resolved_no_cap": "-> no_cap"}.get(v, v), v, ""))

    # ---- gloves boxes the dark-glove review found are not gloves
    boxes = read(ROOT/"artifacts/dark_gloves/boxes.csv", key=None)
    dark = [b for b in boxes if b["dark"] == "1"]
    verd = {}
    for f in sorted((ROOT/"artifacts/dark_gloves").glob("verdicts_part*.txt")):
        for l in open(f):
            t = l.split()
            if len(t) >= 2 and t[0].isdigit(): verd[int(t[0])] = t[1]
    BARE = {"bare hand", "hand"}
    for i, b in enumerate(dark):
        v = verd.get(i)
        if v != "nonglove": continue
        note = " ".join(l.split(" ", 2)[2:] for l in [""]) if False else ""
        key = (b["split"], b["image"]); k = int(b["box"])
        cl = lines.get(key)
        if not cl or k >= len(cl) or cl[k] != 1: continue
        edits[key][k] = (1, None)                 # drop: the box is on a bare hand, hair or sleeve
        n["gloves box dropped (not a glove)"] += 1
        changes.append(("det7", f"dark-glove review {b['image']} box {k}", "remove box", "nonglove", ""))
    for k, v in sorted(n.items()): print(f"  {k}: {v}")
    return edits


def det7():
    D = ROOT/"data/det7"
    edits = collections.defaultdict(dict)       # (split, stem) -> {line: new_class or None}
    for audit in sorted((ROOT/"artifacts/det7_audit").iterdir()):
        if not (audit/"verdicts.csv").exists(): continue
        attr = audit.name.split("_")[0]
        pos, neg = {"gloves": (1, 4), "mask": (3, 6), "head": (2, 5)}[attr]
        boxes, ver = read(audit/"boxes.csv"), read(audit/"verdicts.csv")
        if set(boxes) != set(ver):
            print(f"  skip {audit.name}: audit incomplete ({len(ver)}/{len(boxes)})"); continue
        for i, b in boxes.items():
            v = ver[i]["verdict"]
            if v not in ("wrong_class", "not_hand"): continue
            cls = pos if b["class"] == attr else neg
            new = None if v == "not_hand" else (neg if cls == pos else pos)
            edits[(b["split"], b["image"])][int(b["line"])] = (cls, new)
            changes.append(("det7", f"{audit.name}#{i} {b['split']}/{b['image']} line {b['line']}",
                            "remove box" if new is None else f"relabel {b['class']}->"
                            + (attr if b['class'] != attr else 'no_' + attr), v, ver[i].get("note", "")))
    det7_extra_edits(edits)
    n = collections.Counter()
    for split in ("train", "val"):
        for img in (D/"images"/split).glob("*.jpg"):
            link(img, OUT/"det7/images"/split/img.name)
        for lp in (D/"labels"/split).glob("*.txt"):
            lines = lp.read_text().splitlines(); e = edits.get((split, lp.stem), {})
            out = []
            for k, l in enumerate(lines):
                if k in e:
                    cls, new = e[k]
                    assert int(l.split()[0]) == cls, f"{lp} line {k} is not class {cls}"
                    if new is None: n["boxes_removed"] += 1; continue
                    l = f"{new} " + l.split(" ", 1)[1]; n["boxes_relabelled"] += 1
                out.append(l)
            dst = OUT/"det7/labels"/split/lp.name; dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text("\n".join(out) + ("\n" if out else ""))
    y = (D/"dataset.yaml").read_text().replace(str(D), str(OUT/"det7"))
    (OUT/"det7/dataset.yaml").write_text(y)
    print("det7", dict(n))


if __name__ == "__main__":
    if OUT.exists(): shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    mask_face(); head_cap(); det7()
    # these two write into data/clean as well, and the rebuild above wipes it
    import subprocess, sys as _sys
    for scr in ("132_export_coco_people.py", "133_det7_unclear_from_audit.py"):
        subprocess.run([_sys.executable, str(ROOT/"scripts"/scr)], check=True)
    with open(OUT/"CHANGES.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["dataset", "item", "action", "verdict", "note"]); w.writerows(changes)
    c = collections.Counter((d, a.split()[0]) for d, _, a, _, _ in changes)
    (OUT/"README.md").write_text(
        "# data/clean -- audited copies of the training data\n\n"
        "Built by `python scripts/115_build_clean_data.py` from the audit verdicts; rebuild any time.\n"
        "Originals (data/mask_face, data/crops_plus_sdk/head_cap, data/det7) are untouched;\n"
        "images here are hard links to them, label files are new.\n\n"
        "| dataset | source | audit |\n|---|---|---|\n"
        "| mask_face/g0.8 | data/mask_face/g0.8 | artifacts/mask_face_audit/all |\n"
        "| head_cap | data/crops_plus_sdk/head_cap | artifacts/headcap_audit |\n"
        "| det7 | data/det7 | artifacts/det7_audit/* (gloves site+voter3, mask site) |\n\n"
        "Every change is a row in CHANGES.csv. Totals: "
        + ", ".join(f"{d} {a} {k}" for (d, a), k in sorted(c.items())) + ".\n\n"
        "Rules: classifier sets drop wrong-person / not-a-head / unjudgeable items and flip wrong\n"
        "labels; det7 flips wrong boxes and removes boxes not on the body part, but keeps unclear\n"
        "boxes (removing them would teach the detector that the object is background).\n"
        "Head-cap rule as of 2026-09-17 (apd/schema.py): navy beret-style kitchen caps, white chef\n"
        "toques and cloth chef skullcaps (plain black or striped) are PRESENT; 16 black-skullcap heads\n"
        "labelled no_cap were relabelled to cap.\n")
    print("changes", dict(c))
