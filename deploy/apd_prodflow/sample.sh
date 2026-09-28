#!/usr/bin/env bash
# Copy random events judged by the person filter to a desktop, for review / UAT evidence.
#
#   bash sample.sh [passed|kept|unscored|any] [N] [DEST]
#   bash sample.sh                 # 1 random passed event -> /home/sppg/Desktop
#   bash sample.sh kept 5
#
# Per event: <decision>_<cam>_<id8>.jpg (full frame, if still on disk),
# <...>_crop.png (what the model judged) and <...>.txt (production vs model).
# Events whose images were deleted (prune, or by hand) are skipped; the log keeps them.
set -euo pipefail
DEC=${1:-passed}; N=${2:-1}; DEST=${3:-/home/sppg/Desktop}
TC=${TC:-/mnt/data/tray_counter}
ST="$TC/apd_storage"; LOG="$ST/person_filter.jsonl"
[ -f "$LOG" ] || { echo "no events judged yet ($LOG)"; exit 1; }
case "$DEC" in passed|kept|unscored) pat="\"decision\": \"$DEC\"";; any) pat='"decision"';; *) echo "decision: passed|kept|unscored|any"; exit 1;; esac

mkdir -p "$DEST"
n=0; made=()
while IFS= read -r line; do
  id=$(sed -n 's/.*"event_id": "\([^"]*\)".*/\1/p' <<<"$line")
  cam=$(sed -n 's/.*"camera_id": "\([^"]*\)".*/\1/p' <<<"$line")
  d=$(sed -n 's/.*"decision": "\([^"]*\)".*/\1/p' <<<"$line")
  [ -n "$id" ] || continue
  frame=""; for f in "$ST/snapshots/$cam/$id.jpg" "$ST/snapshots_passed/$cam/$id.jpg"; do [ -f "$f" ] && { frame=$f; break; }; done
  crop="$ST/person_crops/$cam/$id.png"; [ -f "$crop" ] || crop=""
  [ -n "$frame$crop" ] || continue                 # images gone: pick another event
  out="$DEST/${d}_${cam}_${id:0:8}"
  [ -n "$frame" ] && cp "$frame" "$out.jpg"
  [ -n "$crop" ] && cp "$crop" "${out}_crop.png"
  {
    echo "event     $id"
    echo "time      $(sed -n 's/.*"ts": "\([^"]*\)".*/\1/p' <<<"$line")"
    echo "camera    $cam"
    echo "decision  $d"
    echo "device    $(sed -n 's/.*"production": \[\([^]]*\)\].*/\1/p' <<<"$line")"
    echo "model     $(sed -n 's/.*"classifier": \({[^}]*}\|null\).*/\1/p' <<<"$line")"
    echo "scores    $(sed -n 's/.*"p": \(\[[^]]*\]\|null\).*/\1/p' <<<"$line")  (present cap/mask/gloves, known cap/mask/gloves)"
  } > "$out.txt"
  for f in "$out".jpg "${out}_crop.png" "$out.txt"; do [ -f "$f" ] && { made+=("$f"); echo "  $f"; }; done
  n=$((n+1)); [ "$n" -ge "$N" ] && break
done < <(grep -a "$pat" "$LOG" | shuf)

[ "$n" -gt 0 ] || { echo "no $DEC events with images left"; exit 0; }
# files made by root on a user's desktop: give them to that user
owner=$(stat -c %U "$DEST" 2>/dev/null || echo root)
[ "$owner" != root ] && chown "$owner:" "${made[@]}" 2>/dev/null || true
echo "copied $n event(s) to $DEST"
