#!/usr/bin/env bash
# Sync the shared Hugging Face model cache to a target machine so the
# model does not have to be downloaded again.
#
# Usage:
#   ./sync-model-cache.sh TARGET_HOST [SOURCE_CACHE] [TARGET_CACHE]
#
#   TARGET_HOST   ssh target, e.g. user@host
#   SOURCE_CACHE  default /var/lib/huggingface/hub
#   TARGET_CACHE  default /var/lib/huggingface/hub
#
# The target must already have the gufo package installed (it creates
# the gufo user and the llm-servers group). The cache is transferred
# with root on both sides (some cache metadata files are 0600), then
# ownership is fixed so the gufo service can read it.
#
# Environment:
#   IDENTITY_FILE  ssh private key for the target (default
#                  $HOME/.ssh/id_ed25519; root's ssh does not read your
#                  ssh config, so pass the key explicitly when running
#                  the transfer as root, e.g.
#                  IDENTITY_FILE="$HOME/.ssh/id_ed25519")

set -euo pipefail

TARGET_HOST="${1:-}"
SOURCE_CACHE="${2:-/var/lib/huggingface/hub}"
TARGET_CACHE="${3:-/var/lib/huggingface/hub}"
IDENTITY_FILE="${IDENTITY_FILE:-$HOME/.ssh/id_ed25519}"

if [ -z "$TARGET_HOST" ]; then
  echo "error: usage: $0 TARGET_HOST [SOURCE_CACHE] [TARGET_CACHE]" >&2
  exit 1
fi

if [ ! -d "$SOURCE_CACHE" ]; then
  echo "error: source cache $SOURCE_CACHE does not exist" >&2
  exit 1
fi

SSH_OPTS="-i $IDENTITY_FILE -o StrictHostKeyChecking=accept-new"

# Space, checked on the TARGET, before transferring anything. This is ~150 GB onto a host whose
# free space is not ours to assume: filling a serving host's disk takes down every model that has
# to write there, not only ours.
src_bytes=$(sudo du -sb "$SOURCE_CACHE" 2>/dev/null | cut -f1)
free_bytes=$(ssh -i "$IDENTITY_FILE" -o StrictHostKeyChecking=accept-new "$TARGET_HOST" \
  "df -P -B1 '$TARGET_CACHE' 2>/dev/null || df -P -B1 \"\$(dirname '$TARGET_CACHE')\"" | awk 'NR==2{print $4}')
if [ -z "${src_bytes:-}" ] || [ -z "${free_bytes:-}" ]; then
  echo "error: cannot size the source or the target's free space -- refusing to transfer blind" >&2
  exit 1
fi
need_bytes=$(( src_bytes * 105 / 100 ))   # 5% headroom
if [ "$need_bytes" -gt "$free_bytes" ]; then
  printf 'error: target has %s GiB free, this needs %s GiB (source + 5%%) -- refusing\n' \
    "$((free_bytes/1073741824))" "$((need_bytes/1073741824))" >&2
  exit 1
fi
printf '==> Space ok on %s: %s GiB free, %s GiB needed\n' "$TARGET_HOST" \
  "$((free_bytes/1073741824))" "$((need_bytes/1073741824))"

# A running instance reads these files while we write them, and the recursive chown below
# briefly removes read access from a process that is serving traffic. Refuse unless the operator
# says the window is deliberate.
live=$(ssh -i "$IDENTITY_FILE" -o StrictHostKeyChecking=accept-new "$TARGET_HOST" \
  "systemctl list-units --state=active --no-legend --plain 'gufo*' 2>/dev/null | awk '{print \$1}' | tr '\n' ' '" || true)
if [ -n "${live// /}" ] && [ "${ALLOW_LIVE:-0}" != "1" ]; then
  printf 'error: %s is serving with: %s\n' "$TARGET_HOST" "$live" >&2
  echo "       a cache sync changes files and ownership under a running server. Stop it first," >&2
  echo "       or re-run with ALLOW_LIVE=1 only in a window you own." >&2
  exit 1
fi

echo "==> Syncing $SOURCE_CACHE to $TARGET_HOST:$TARGET_CACHE"
# --no-owner/--no-group: ownership is fixed afterwards; the target's gufo uid may differ.
# --out-format records what actually moved, so ownership is repaired on those paths and not on
# every byte in a shared cache the target may hold for other services (queue 081).
transferred=$(mktemp)
 trap 'rm -f "$transferred"' EXIT INT TERM
sudo rsync -a --no-owner --no-group -e "ssh $SSH_OPTS" \
  --rsync-path="sudo rsync" \
  --out-format='%n' \
  "$SOURCE_CACHE/" "$TARGET_HOST:$TARGET_CACHE/" | tee "$transferred"

# Repair ownership on the paths that moved (top-level entries, so the argument list stays small),
# plus the cache root so future writes inherit the group through its setgid bit.
top=$(cut -d/ -f1 "$transferred" | grep -vE '^$|^\./$' | sort -u | tr '\n' ' ')
if [ -n "${top// /}" ]; then
  printf '==> Repairing ownership on %s top-level path(s) that moved\n' "$(printf '%s' "$top" | wc -w)"
  quoted=""
  for p in $top; do quoted="$quoted '$TARGET_CACHE/$p'"; done
  ssh -i "$IDENTITY_FILE" -o StrictHostKeyChecking=accept-new "$TARGET_HOST" \
    "sudo chown gufo:llm-servers $quoted && sudo chmod g+s $quoted"
else
  echo "==> Nothing transferred; no ownership change"
fi
ssh -i "$IDENTITY_FILE" -o StrictHostKeyChecking=accept-new "$TARGET_HOST" \
  "sudo chown gufo:llm-servers '$TARGET_CACHE' && sudo chmod g+s '$TARGET_CACHE'"

echo "==> Verifying"
ssh "$TARGET_HOST" "sudo du -sh '$TARGET_CACHE'"

echo "==> Done. Configure /etc/gufo/gufo.env on the target and start with:"
echo "    sudo systemctl enable --now gufo"