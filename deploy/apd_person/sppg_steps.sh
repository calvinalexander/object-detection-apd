#!/usr/bin/env bash
# Run ON SPPG-3881 as root, from anywhere:   bash /mnt/data/apd_ours/sppg_steps.sh <step>
# Steps are independent commands you were pasting by hand. Nothing here writes to
# production storage, production code, or the dashboard.
set -euo pipefail
B="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # this bundle, wherever it was copied
PROD=${PROD:-apd-cctv}                                # the running production container to mirror

# Everything below is READ from the running production container, so the same
# script works on any SPPG box: image tag, compose project dir, the compiler
# venv volume and the customers/build bind-mounts all differ per machine.
insp() { docker inspect "$PROD" --format "$1"; }
IMG=$(insp '{{.Config.Image}}')
TC=$(insp '{{index .Config.Labels "com.docker.compose.project.working_dir"}}')
COMPOSE=$(insp '{{index .Config.Labels "com.docker.compose.project.config_files"}}')
VOL=$(insp '{{range .Mounts}}{{if eq .Destination "/home/cpu-tower/.local"}}{{.Name}}{{end}}{{end}}')
CUST=$(insp '{{range .Mounts}}{{if eq .Destination "/mnt/data/voyager-sdk/customers"}}{{.Source}}{{end}}{{end}}')
BUILD=$(insp '{{range .Mounts}}{{if eq .Destination "/mnt/data/voyager-sdk/build"}}{{.Source}}{{end}}{{end}}')
# the python production itself runs (venv path differs per box); works for both "python x.py" and "bash -c '... python x.py'"
PY=$(insp '{{range .Config.Cmd}}{{println .}}{{end}}' | grep -o '[^ ]*python[0-9.]*' | head -1); PY=${PY:-python3}
SDK=$(insp '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^AXELERA_FRAMEWORK=//p'); SDK=${SDK:-/mnt/data/voyager-sdk}
: "${TC:?no compose working_dir label on $PROD}"; : "${CUST:?no customers mount on $PROD}"; : "${BUILD:?no build mount on $PROD}"

# Quantisation calibration pool. The SDK defaults to its own 400 generic COCO
# photos; CAL=ours uses the 200 SPPG kitchen frames shipped in the bundle, which
# match the cameras this model will actually see.
CALFLAG=""; [ "${CAL:-}" = "ours" ] && CALFLAG="--no-default-representative-images"
# The lowering stage prints nothing for a long time. VERBOSE=1 turns on the
# SDK's debug logging with timestamps so there is something to watch.
VFLAG=""; [ "${VERBOSE:-}" = "1" ] && VFLAG="-v -v --logtimestamp"

case "${1:-help}" in

check)   # 0. prerequisites (read-only) -- prints what was detected on THIS machine
  echo "image=$IMG"; echo "compose=$COMPOSE"; echo "project=$TC"; echo "customers=$CUST"; echo "build=$BUILD"
  echo "venv volume=${VOL:-NONE (compile will rely on the image only)}"; echo "python=$PY"; echo "sdk=$SDK"; echo "bundle=$B"
  echo "prod cmd=$(insp '{{.Config.Cmd}}')"
  docker run --rm ${VOL:+-v "$VOL:/home/cpu-tower/.local"} "$IMG" bash -c "ls -la $SDK/deploy.py; $PY -c 'import sys;print(sys.executable)'" 2>&1 | sed 's/^/  in image: /' 
  ls "$CUST" "$BUILD"; df -h "$TC" | tail -1; nproc; free -g | head -2
  ;;

place)   # 1. put the network definition where the compiler looks (bind-mounted into the container)
  mkdir -p "$CUST/apd_ours"
  cp -r "$B/config.yaml" "$B/yolo11s-pose.onnx" "$B/dataset" "$CUST/apd_ours/"
  ls -la "$CUST/apd_ours/"
  ;;

compile) # 2. compile the pose network in a throwaway container (production untouched)
  # --cpus caps CPU TIME but the compiler still sees every core and spawns a
  # worker per core, so peak memory does not fall. --cpuset-cpus limits the
  # cores it can see, which does reduce workers. MEM (e.g. MEM=6g) caps the
  # container so a runaway compile is killed instead of something production needs.
  NC=${CPUS:-4}
  docker run --rm --name apd-ours-compile \
    --cpuset-cpus="0-$((NC-1))" --cpus=$NC ${MEM:+--memory=$MEM --memory-swap=$MEM} \
    --privileged -v /dev:/dev \
    -e AXELERA_CONFIGURE_DEVICE=0 \
    -v "$CUST:/mnt/data/voyager-sdk/customers" \
    -v "$BUILD:/mnt/data/voyager-sdk/build" \
    ${VOL:+-v "$VOL:/home/cpu-tower/.local"} \
    "$IMG" \
    bash -lc "cd $SDK && $PY ./deploy.py apd_ours --model apd_ours_pose --metis pcie $CALFLAG $VFLAG" \
    2>&1 | tee "$B/compile.log"
  rc=${PIPESTATUS[0]}          # tee would otherwise mask the real exit code
  echo "deploy.py exit code: $rc"      # --rm takes the container away, so record it here
  [ $rc -ne 0 ] && echo "FAILED. Run: bash $0 diagnose"
  # quantisation is cached under quantized/, so a re-run skips ~9 minutes and
  # goes straight back to the lowering stage that failed.
  exit $rc
  echo; ls -la "$BUILD/apd_ours/apd_ours_pose/" 2>/dev/null || echo "NO BUILD OUTPUT - see $B/compile.log"
  ;;

pipeline) # 2b. OPTIONAL. Build the pipeline representation from the compiled model.
  # NOTE: the .axnet file is NOT produced here. It is written at RUNTIME by the
  # GStreamer pipe (axelera/app/pipe/gst.py::_save_axnet_files), so production's
  # build/apd_v3/apd_v3.axnet is an output of its first successful run, not an
  # input to it. A missing .axnet means the stream has never started, which is a
  # symptom. Fix the reason the stream fails, and the file appears by itself.
  docker run --rm --name apd-ours-pipeline --privileged -v /dev:/dev \
    -e AXELERA_CONFIGURE_DEVICE=0 \
    -v "$CUST:/mnt/data/voyager-sdk/customers" \
    -v "$BUILD:/mnt/data/voyager-sdk/build" \
    ${VOL:+-v "$VOL:/home/cpu-tower/.local"} \
    "$IMG" \
    bash -lc "cd $SDK && $PY ./deploy.py apd_ours --pipeline-only --metis pcie" \
    2>&1 | tee "$B/pipeline.log"
  rc=${PIPESTATUS[0]}; echo "deploy.py exit code: $rc"
  exit $rc
  ;;

watch)   # is the compile alive, and how long has it been going?
  c=apd-ours-compile
  if ! docker ps --format '{{.Names}}' | grep -qx "$c"; then
    echo "no compile container running"; tail -5 "$B/compile.log" 2>/dev/null; exit 0
  fi
  echo "started $(docker inspect -f '{{.State.StartedAt}}' $c), up $(docker ps --format '{{.Status}}' -f name=$c)"
  docker stats --no-stream --format 'cpu {{.CPUPerc}}  mem {{.MemUsage}}' $c
  echo "busy CPU means it is lowering; near 0% with no new log lines means it stopped"
  echo "--- build dir (1/ appears only at the very end)"
  ls -la "$BUILD/apd_ours/apd_ours_pose/" 2>/dev/null
  echo "--- last 3 log lines, and how long ago the log changed"
  tail -3 "$B/compile.log" 2>/dev/null
  [ -f "$B/compile.log" ] && echo "log last written $(( ($(date +%s) - $(stat -c %Y "$B/compile.log")) / 60 )) min ago"
  ;;

recalib) # force quantisation to run again on OUR frames
  # A re-run reuses build/apd_ours/apd_ours_pose/quantized/ whatever it was
  # calibrated on, so after changing the calibration set the old quantised model
  # has to go or the change never takes effect. Costs ~9 minutes, touches only
  # our build directory.
  q="$BUILD/apd_ours/apd_ours_pose/quantized"
  [ -d "$q" ] && { echo "removing $q"; rm -rf "$q"; } || echo "nothing cached at $q"
  echo "now run: bash $0 compile"
  ;;

diagnose) # why did the compile finish without producing 1/ ?
  echo "== what exists under build/apd_ours"
  find "$BUILD/apd_ours" -maxdepth 4 -printf '%y %10s  %p\n' 2>/dev/null | head -40
  echo
  echo "== how far the log got (last 25 INFO lines)"
  grep -a "^INFO" "$B/compile.log" 2>/dev/null | tail -25
  echo
  echo "== anything that looks like a failure"
  grep -anE "ERROR|Traceback|Failed|No space|Killed|MemoryError|abort" "$B/compile.log" 2>/dev/null | tail -25
  echo
  echo "== last 15 lines of the log, whatever they are"
  tail -15 "$B/compile.log" 2>/dev/null
  echo
  echo "== was it the kernel that killed it?"
  dmesg -T 2>/dev/null | grep -iE "out of memory|oom-kill|killed process" | tail -5 || echo "  (dmesg needs root; try: sudo dmesg -T | grep -i oom)"
  echo
  echo "== room to write, and memory"
  df -h "$BUILD" | tail -1
  free -g | sed -n 2p
  echo
  echo "== reading of the above"
  if [ -d "$BUILD/apd_ours/apd_ours_pose/1" ]; then
    n=$(ls -A "$BUILD/apd_ours/apd_ours_pose/1" 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] && echo "  1/ exists with $n files -- the compile DID finish; run: bash $0 verify" \
                   || echo "  1/ exists but is EMPTY. The compiler creates and clears it before lowering, so"
    [ "$n" -eq 0 ] && echo "  it started lowering and died. The cause is in the ERROR/Traceback lines above."
  elif [ -d "$BUILD/apd_ours/apd_ours_pose/quantized" ]; then
    echo "  only quantized/ exists: quantisation succeeded, lowering never wrote anything."
    echo "  Either deploy.py ran in a quantize-only mode, or it stopped before lowering."
    echo "  Re-run compile; quantisation is cached, so it resumes at the slow stage."
  else
    echo "  nothing under apd_ours_pose: deploy.py failed before quantising. See the log above."
  fi
  ;;

verify)  # 3. did the compile produce what apd_v3 has?
  ls -la "$BUILD/apd_ours/apd_ours_pose/" "$BUILD/apd_ours/apd_ours_pose/1/"
  echo "--- models this network declares (every one of them gets deployed at startup)"
  grep -E "^  apd_ours_[a-z]+:" "$CUST/apd_ours/config.yaml" 2>/dev/null || echo "  (config not placed yet)"
  echo "--- .axnet: written by the runtime on its first successful start, not by the compiler"
  ls -la "$BUILD/apd_ours/"*.axnet 2>/dev/null || echo "  none yet -- expected until the stream runs once"
  # which images the quantiser calibrated on (coco2017_repr400 = SDK default,
  # customers/apd_ours/dataset = our SPPG frames, from CAL=ours)
  f="$BUILD/apd_ours/apd_ours_pose_calibration_images.txt"
  [ -f "$f" ] && { echo "calibrated on $(wc -l < "$f") images, e.g."; head -2 "$f"; } || echo "no calibration image list"
  ;;

start)   # 4. run our model through the production flow, beside production
  # overlay is written with THIS machine's image tag and bundle path
  sed -e "s|sipgn-git.bgn.go.id:axelera-v1.5|$IMG|g" -e "s|/mnt/data/apd_ours|$B|g" -e "s|/mnt/data/tray_counter|$TC|g" \
      "$B/docker-compose.apd-ours-test.yml" > "$TC/docker-compose.apd-ours-test.yml"
  cd "$TC"
  docker compose -f "$(basename "$COMPOSE")" -f docker-compose.apd-ours-test.yml up -d apd-cctv-test
  sleep 5; docker logs --tail 30 apd-cctv-test
  ;;

logs)    # 5a. follow our container
  docker logs -f --tail 50 apd-cctv-test
  ;;

status)  # 5b. what is it deciding, and is production unaffected?
  # every count is guarded: set -e plus pipefail would otherwise abort the whole
  # step the first time a directory does not exist yet, which is normal early on
  echo "--- ours"
  echo "container: $(docker ps --format '{{.Status}}' -f name=apd-cctv-test | head -1 || true)"
  echo "snapshots: $(find "$B/apd_storage/snapshots" -name '*.jpg' 2>/dev/null | wc -l || true)"
  if command -v sqlite3 >/dev/null 2>&1 && [ -f "$B/apd_storage/apd_events.db" ]; then
    sqlite3 "$B/apd_storage/apd_events.db" \
      "select ppe_type,status,count(*) from apd_detail group by 1,2;" 2>/dev/null || echo "(db not readable yet)"
  else
    echo "(no events db yet)"
  fi
  echo "--- worker (the real signal: is OUR perception loaded and reading frames?)"
  lg="$B/logs/apd_monitor.log"
  if [ -f "$lg" ]; then
    grep -a "Perception: OURS" "$lg" | tail -1 || echo "  no 'Perception: OURS' line yet"
    tail -5 "$lg" || true
  else
    echo "  $lg not created yet; docker logs apd-cctv-test for the supervisor"
  fi
  echo "--- production (must keep rising, crashes must stay 0)"
  echo "snapshots: $(find "$TC/apd_storage/snapshots" -name '*.jpg' 2>/dev/null | wc -l || true)"
  echo "crashes last 10m: $(docker logs --since 10m "$PROD" 2>&1 | grep -ciE "crash" || true)"
  ;;

stop)    # 6. stop ours (production keeps running)
  docker stop apd-cctv-test && docker rm apd-cctv-test
  ;;

*) grep -E '^[a-z]+\)' "$0" | sed 's/)  *#/ -/' ;;
esac
