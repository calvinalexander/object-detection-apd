"""Derive the prodflow monitor from production's apd_monitor_axelera_multi.py.

    python patch_monitor.py <live apd_monitor_axelera_multi.py> apd_monitor_axelera_multi.py

Production's perception is KEPT: its apd_v3 network on the AIPU, its person
threshold, ByteTrack, assign_apd_smart, the vote window, MIN_AGE_FRAMES and the
cooldown. The only change is at the moment production's vote fires an event:
the person classifier (apd_person_filter.py) judges that person's crop from the
clean stream frame, and in filter mode

  - the classifier finds a violation -> the event is saved, reporting the
    classifier's items; the snapshot and apd_detail show its verdicts
  - the classifier clears the person -> no event row; the clean frame goes to
    <outdir>_passed/ for review. Production's cooldown and last state still
    advance, as they would have, so each person gets one judgement per
    cooldown -- the same unit the offline filter16 numbers were measured on.

Every fired event, kept or not, is one line in <APD_STORAGE_PATH>/person_filter.jsonl.
Every anchor is asserted, so a changed live file fails loudly.
"""
import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text()
out = Path(sys.argv[2])
n_before = src.count("\n")


def sub(old, new, what):
    global src
    assert src.count(old) == 1, f"anchor for [{what}] found {src.count(old)}x, expected 1"
    src = src.replace(old, new)


sub("from apd_database import Database\n",
    "from apd_database import Database\n"
    "from apd_person_filter import PersonFilter\n", "import")

sub("    class_to_group = build_class_maps(CLASS_NAMES, APD_GROUPS)\n",
    "    class_to_group = build_class_maps(CLASS_NAMES, APD_GROUPS)\n"
    "    person_filter = PersonFilter(os.path.join(config.Paths.APD_STORAGE_PATH, \"person_filter.jsonl\"))\n"
    "    logger.info(f\"Perception: PRODUCTION ({args.model}) + person filter {person_filter.describe()}\")\n",
    "filter init")

sub('''                    event_id = str(uuid.uuid4())
''', '''                    event_id = str(uuid.uuid4())

                    # ---- PERSON FILTER: production voted; the person classifier
                    # judges the same track box on the clean stream frame.
                    prod_voted = list(voted_violations)
                    try:
                        pv, pscore = person_filter.classify(frame, tr["bbox"])
                    except Exception as e:
                        logger.error(f"[stream {sid}] person filter failed, keeping production's event: {e}")
                        pv, pscore = None, None
                    keep = pv is None or any(pv[g] == "violation" for g in ENABLED_GROUPS)
                    person_filter.log({
                        "event_id": event_id, "camera_id": cam_id_map.get(sid, f"CAM{sid:02d}"),
                        "track_id": track_id, "person_bbox": [int(v) for v in tr["bbox"]],
                        "production": prod_voted, "production_apd": dict(tr["apd"]),
                        "classifier": pv, "p": None if pscore is None else [round(float(x), 4) for x in pscore],
                        "decision": "unscored" if pv is None else ("kept" if keep else "passed")})
                    if person_filter.mode == "filter" and pv is not None:
                        if not keep:
                            # production would have alerted here: advance its state the same way
                            st["last_event_apd_state"] = tuple(sorted(prod_voted))
                            st["last_event_ms"] = now_ms
                            RECENT_EVENTS[sid][track_id] = now_ms
                            pdir = os.path.join(args.outdir.rstrip("/") + "_passed", cam_id_map.get(sid, f"cam_{sid}"))
                            os.makedirs(pdir, exist_ok=True)
                            cv2.imwrite(os.path.join(pdir, f"{event_id}.jpg"), draw_table(frame.copy(), [dict(tr, apd=pv)]))
                            continue
                        voted_violations = [g for g in ENABLED_GROUPS if pv[g] == "violation"]
                        tr = dict(tr, apd=pv)          # snapshot and apd_detail show the classifier
''', "filter at event time")

sub('                    st["last_event_apd_state"] = tuple(sorted(voted_violations))\n',
    '                    st["last_event_apd_state"] = tuple(sorted(prod_voted))   # production\'s own dedupe key\n',
    "dedupe on production's vote")

out.write_text(src)
print(f"wrote {out}  ({n_before} -> {src.count(chr(10))} lines)")
