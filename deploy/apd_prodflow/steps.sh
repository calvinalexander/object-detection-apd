#!/usr/bin/env bash
# Run ON the SPPG box as root:   bash <bundle>/steps.sh <step>
# Installs the person filter INTO the running production service (apd-cctv):
# its monitor is re-derived from the live file, the filter module and models go
# beside it, and the container is recreated. `rollback` restores the original.
set -euo pipefail
B="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROD=${PROD:-apd-cctv}
insp() { docker inspect "$PROD" --format "$1"; }
IMG=$(insp '{{.Config.Image}}')
TC=$(insp '{{index .Config.Labels "com.docker.compose.project.working_dir"}}')
COMPOSE=$(insp '{{index .Config.Labels "com.docker.compose.project.config_files"}}')
PY=$(insp '{{range .Config.Cmd}}{{println .}}{{end}}' | grep -o '[^ ]*python[0-9.]*' | head -1); PY=${PY:-python3}
: "${TC:?no compose working_dir label on $PROD}"
LIVE="$TC/apd_monitor_axelera_multi.py"
ORIG="$TC/apd_monitor_axelera_multi.py.pre_person"   # production's own file, kept for rollback
# the worker log lives INSIDE the container: LOG_DIR = $PROJECT_DIR/logs, and
# production's .env sets PROJECT_DIR=/mnt/data/voyager-sdk, which is not bind-mounted
wlog() { docker exec "$PROD" sh -c 'tail -n '"${1:-5000}"' "${PROJECT_DIR:-/mnt/data/voyager-sdk}/logs/apd_monitor.log"' 2>/dev/null || true; }
installed() { grep -q "apd_person_filter" "$LIVE"; }
pristine() { if installed; then echo "$ORIG"; else echo "$LIVE"; fi; }
# patch inside production's image with production's python; files in, file out, nothing else mounted
derive() { docker run --rm -v "$(dirname "$1"):/src:ro" -v "$B:/b:ro" -v "$(dirname "$2"):/out" "$IMG" \
             $PY /b/patch_monitor.py "/src/$(basename "$1")" "/out/$(basename "$2")"; }
set_env() {   # set KEY=VAL in the compose .env (read by apd-cctv via env_file)
  [ -f "$TC/.env.pre_person" ] || cp -p "$TC/.env" "$TC/.env.pre_person"
  if grep -q "^$1=" "$TC/.env"; then sed -i "s|^$1=.*|$1=$2|" "$TC/.env"; else echo "$1=$2" >> "$TC/.env"; fi
}
recreate() {  # env_file changes need a recreate, not a restart
  cd "$TC" && docker compose -f "$(basename "$COMPOSE")" up -d --force-recreate --no-deps "$PROD"
}

case "${1:-help}" in
check)   # 0. read-only: what is on this machine, and would the patch apply?
  echo "image=$IMG"; echo "compose=$COMPOSE"; echo "project=$TC"; echo "python=$PY"; echo "bundle=$B"
  grep -E '^APD_(MODEL|PERSON_[A-Z]+)=' "$TC/.env" || true
  installed && echo "person filter: INSTALLED (original at $ORIG)" || echo "person filter: not installed"
  docker run --rm "$IMG" $PY -c "import onnxruntime as o; print('onnxruntime', o.__version__)"
  tmp=$(mktemp -d); derive "$(pristine)" "$tmp/m.py" && diff "$(pristine)" "$tmp/m.py" | head -60; rm -rf "$tmp"
  ls -la "$B"/apd_ours_person_ppe_*.onnx
  ;;
install) # 1. MODE=filter|shadow (default filter)  MODEL=v5|v3 (default v5)
  MODE=${MODE:-filter}; MODEL=${MODEL:-v5}
  [ -f "$B/apd_ours_person_ppe_$MODEL.onnx" ] || { echo "no model $MODEL in $B"; exit 1; }
  installed || cp -p "$LIVE" "$ORIG"
  tmp=$(mktemp -d); derive "$ORIG" "$tmp/m.py"
  mkdir -p "$TC/apd_person"
  cp "$B"/apd_ours_person_ppe_*.onnx "$B"/apd_ours_person_ppe_*.json "$TC/apd_person/"
  cp "$B/apd_person_filter.py" "$TC/"
  cp "$tmp/m.py" "$LIVE"; rm -rf "$tmp"
  set_env APD_PERSON_MODE "$MODE"
  set_env APD_PERSON_ONNX "/mnt/data/tray_counter/apd_person/apd_ours_person_ppe_$MODEL.onnx"
  recreate
  echo "waiting for the worker's startup line ..."; sleep 40
  wlog | grep -a "person filter" | tail -1 | grep . || echo "no startup line yet: bash $0 logs"
  ;;
mode)    # switch without reinstalling:  mode filter|shadow
  set_env APD_PERSON_MODE "${2:?filter or shadow}"; recreate
  ;;
model)   # switch without reinstalling:  model v5|v3
  [ -f "$TC/apd_person/apd_ours_person_ppe_${2:?v5 or v3}.onnx" ] || { echo "not installed: $2"; exit 1; }
  set_env APD_PERSON_ONNX "/mnt/data/tray_counter/apd_person/apd_ours_person_ppe_$2.onnx"; recreate
  ;;
rollback) # production's original monitor back; the APD_PERSON_* lines in .env are then ignored
  installed || { echo "not installed"; exit 0; }
  cp -p "$ORIG" "$LIVE"; recreate
  echo "restored $LIVE from $ORIG"
  ;;
logs)    # follow the worker log
  docker exec "$PROD" sh -c 'tail -F "${PROJECT_DIR:-/mnt/data/voyager-sdk}/logs/apd_monitor.log"'
  ;;
status)  # what is the filter deciding?
  echo "container: $(docker ps --format '{{.Status}}' -f name="^$PROD\$" | head -1 || true)"
  wlog | grep -a "person filter" | tail -1 | grep . || echo "no 'person filter' startup line"
  f="$TC/apd_storage/person_filter.jsonl"
  if [ -f "$f" ]; then
    echo "fired by the detector: $(wc -l < "$f")   kept: $(grep -c '"kept"' "$f" || true)   passed: $(grep -c '"passed"' "$f" || true)   unscored: $(grep -c '"unscored"' "$f" || true)"
    echo "last hour: $(awk -v t="$(date -u -d '-1 hour' '+%Y-%m-%d %H:%M:%S')" -F'"ts": "' '{split($2,a,"\""); if (a[1] >= t) n++} END {print n+0}' "$f") fired"
  else echo "no events judged yet"; fi
  echo "passed snapshots: $(find "$TC/apd_storage/snapshots_passed" -name '*.jpg' 2>/dev/null | wc -l || true)  ($(du -sh "$TC/apd_storage/snapshots_passed" 2>/dev/null | cut -f1 || echo 0))"
  echo "crops: $(find "$TC/apd_storage/person_crops" -name '*.png' 2>/dev/null | wc -l || true)  ($(du -sh "$TC/apd_storage/person_crops" 2>/dev/null | cut -f1 || echo 0))"
  echo "filter errors in log: $(wlog 100000 | grep -ac 'person filter failed' || true)"
  ;;
export)  # pack new events (frame + crop + log + index.csv) and push to 185.  ALL=1 resends everything
  # DEST=user@host:dir  (default infidea@185.111.159.81:/home/infidea/sppg/review/)
  DEST=${DEST:-infidea@185.111.159.81:/home/infidea/sppg/review/}
  f="$TC/apd_storage/person_filter.jsonl"; [ -f "$f" ] || { echo "no events judged yet"; exit 0; }
  mark="$B/.exported_lines"; skip=0; [ "${ALL:-}" = "1" ] || skip=$(cat "$mark" 2>/dev/null || echo 0)
  [ "$(wc -l < "$f")" -gt "$skip" ] || { echo "nothing new since the last export ($skip lines)"; exit 0; }
  name="person_review_$(hostname)_$(date +%Y%m%d_%H%M%S)"; tmp=$(mktemp -d)
  if command -v python3 >/dev/null; then
    python3 "$B/export_review.py" "$TC" "$tmp/$name" --skip "$skip" | tee "$tmp/out.txt"
  else   # no python on the host: production's image has one
    docker run --rm -v "$TC:$TC:ro" -v "$B:/b:ro" -v "$tmp:/o" "$IMG" $PY /b/export_review.py "$TC" "/o/$name" --skip "$skip" | tee "$tmp/out.txt"
  fi
  tar -C "$tmp" -czf "$tmp/$name.tar.gz" "$name"
  echo "archive $(du -h "$tmp/$name.tar.gz" | cut -f1) -> $DEST"
  ssh "${DEST%%:*}" "mkdir -p '${DEST#*:}'"
  if command -v rsync >/dev/null; then rsync -P "$tmp/$name.tar.gz" "$DEST"; else scp "$tmp/$name.tar.gz" "$DEST"; fi
  # only advance the mark once the copy succeeded (set -e stops before this otherwise)
  sed -n 's/^LINES //p' "$tmp/out.txt" > "$mark"
  rm -rf "$tmp"
  echo "sent $name.tar.gz; on 185: tar -xzf ${DEST#*:}$name.tar.gz"
  ;;
prune)   # passed snapshots and crops are not in the DB, so production's cleanup never deletes them.  DAYS=7
  find "$TC/apd_storage/snapshots_passed" "$TC/apd_storage/person_crops" \( -name '*.jpg' -o -name '*.png' \) -mtime +"${DAYS:-7}" -print -delete 2>/dev/null | wc -l
  ;;
*) grep -E '^[a-z]+\)' "$0" | sed 's/)  *#/ -/' ;;
esac
