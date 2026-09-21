#!/usr/bin/env bash
# Sync this project to another server.
#
# Profiles exist because the tree has three very different cadences:
#   code       180K, changes constantly
#   state       15M, the label stores and manifests -- the expensive-to-recreate part
#   crops      185M / 26k files, regenerable from data + persons.jsonl
#   data       883M, the source dataset, effectively immutable
#
# Dry-run is the default. Nothing transfers until you pass --go.

set -euo pipefail

REMOTE="${APD_REMOTE:-<my_server>}"          # user@host  -- or set APD_REMOTE
REMOTE_DIR="${APD_REMOTE_DIR:-~/apd}"        # destination path on that host
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PROFILE=code
GO=0
DELETE=()

usage() {
  cat <<'USAGE'
usage: scripts/sync.sh [profile] [--go] [--delete] [--to user@host] [--dir PATH]

profiles
  code        apd/ scripts/                      ~180K   default
  state       code + label stores & manifests      ~15M   what you cannot cheaply rebuild
  artifacts   state + crops/ + sheets/           ~256M
  data        data/sppg/                         ~883M   source frames
  all         everything above                   ~1.3G

flags
  --go        actually transfer (default is a dry run)
  --delete    remove files on the remote that no longer exist locally
  --to        user@host          (or export APD_REMOTE)
  --dir       remote path        (or export APD_REMOTE_DIR, default ~/apd)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    code|state|artifacts|data|all) PROFILE="$1"; shift ;;
    --go)     GO=1; shift ;;
    --delete) DELETE=(--delete); shift ;;
    --to)     REMOTE="$2"; shift 2 ;;
    --dir)    REMOTE_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ "$REMOTE" == "<my_server>" ]]; then
  echo "error: destination not set." >&2
  echo "  scripts/sync.sh $PROFILE --to user@host" >&2
  echo "  or: export APD_REMOTE=user@host" >&2
  exit 2
fi

# Always excluded: caches, and anything that looks like a credential. The
# subscription token this project uses lives outside the tree, but an exclude
# costs nothing and a leaked token costs a lot.
COMMON=(
  --exclude='__pycache__/' --exclude='*.pyc' --exclude='.ipynb_checkpoints/'
  --exclude='.DS_Store'    --exclude='.env'  --exclude='*.pem'
  --exclude='*credentials*' --exclude='*token*'
)

case "$PROFILE" in
  code)      PATHS=(apd scripts) ;;
  state)     PATHS=(apd scripts artifacts)
             COMMON+=(--exclude='artifacts/crops/' --exclude='artifacts/sheets/'
                      --exclude='artifacts/*.png') ;;
  artifacts) PATHS=(apd scripts artifacts) ;;
  data)      PATHS=(data) ;;
  all)       PATHS=(apd scripts artifacts data) ;;
esac

FLAGS=(-a -h --partial --info=progress2 --human-readable "${DELETE[@]}" "${COMMON[@]}")
[[ $GO -eq 1 ]] || FLAGS+=(--dry-run)
# Compression helps the JSONL manifests and hurts on already-compressed JPEGs.
[[ "$PROFILE" == "data" || "$PROFILE" == "artifacts" || "$PROFILE" == "all" ]] || FLAGS+=(-z)

echo "profile   : $PROFILE  (${PATHS[*]})"
echo "source    : $SRC"
echo "target    : $REMOTE:$REMOTE_DIR"
echo "mode      : $([[ $GO -eq 1 ]] && echo TRANSFER || echo 'DRY RUN -- pass --go to transfer')"
[[ ${#DELETE[@]} -gt 0 ]] && echo "delete    : ON -- remote files absent locally will be removed"
echo

ssh -o BatchMode=yes "$REMOTE" "mkdir -p $REMOTE_DIR" 2>/dev/null \
  || echo "note: could not pre-create $REMOTE_DIR (rsync will try anyway)"

cd "$SRC"
rsync "${FLAGS[@]}" "${PATHS[@]}" "$REMOTE:$REMOTE_DIR/"

echo
if [[ $GO -eq 1 ]]; then
  echo "done. On the remote:"
  echo "  cd $REMOTE_DIR && python scripts/05_status.py"
else
  echo "dry run only. Re-run with --go to transfer."
fi
