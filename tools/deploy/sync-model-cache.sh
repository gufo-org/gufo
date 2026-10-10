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
#                  IDENTITY_FILE=/home/user/.ssh/id_ed25519)

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

echo "==> Syncing $SOURCE_CACHE to $TARGET_HOST:$TARGET_CACHE"
# --no-owner/--no-group: ownership is fixed afterwards; the target's
# gufo uid may differ from the source's.
sudo rsync -a --no-owner --no-group -e "ssh $SSH_OPTS" \
  --rsync-path="sudo rsync" \
  "$SOURCE_CACHE/" "$TARGET_HOST:$TARGET_CACHE/"

echo "==> Fixing ownership on $TARGET_HOST"
ssh "$TARGET_HOST" "sudo chown -R gufo:llm-servers '$TARGET_CACHE' \
  && sudo chmod g+s '$TARGET_CACHE'"

echo "==> Verifying"
ssh "$TARGET_HOST" "sudo du -sh '$TARGET_CACHE'"

echo "==> Done. Configure /etc/gufo/gufo.env on the target and start with:"
echo "    sudo systemctl enable --now gufo"