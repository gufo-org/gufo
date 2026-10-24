#!/bin/bash
# Launch the gufo inference server from /etc/gufo/gufo.env.
#
# Every tuning knob lives in the environment file, so changing the model, the
# speculative backend, the draft length, the context, the session count or the
# row-cache size is an edit to /etc/gufo/gufo.env plus `systemctl restart
# gufo`. The unit file stays untouched.
#
# Run it by hand as the gufo user to test a configuration without restarting
# the service:
#
#   sudo -u gufo GUFO_ENV_FILE=~/try.env /usr/lib/gufo/gufo-launch.sh
set -euo pipefail

env_file=${GUFO_ENV_FILE:-/etc/gufo/gufo.env}
if [[ -r "$env_file" ]]; then
  # shellcheck disable=SC1090
  source "$env_file"
fi

# Instantiated units (gufo@<name>) export GUFO_INSTANCE=%i, and systemd has
# already loaded /etc/gufo/instance.d/%i.env. Sourcing it again here keeps the
# launcher honest when it is run by hand, so a manual run reproduces exactly
# what the service will do - the assembled command line is the contract.
if [[ -n "${GUFO_INSTANCE:-}" && -r "/etc/gufo/instance.d/${GUFO_INSTANCE}.env" ]]; then
  # shellcheck disable=SC1090
  source "/etc/gufo/instance.d/${GUFO_INSTANCE}.env"
fi

: "${GUFO_BIND:=0.0.0.0}"
: "${GUFO_PORT:=8080}"

# Each instance gets a private disk-cache directory. The server refuses a cache
# directory it does not consider private ("continuation disk directory is not a
# private owned directory"), so instances must not share one, and it is created
# here because the launcher runs as the service user that will own it.
if [[ -n "${GUFO_INSTANCE:-}" ]]; then
  : "${GUFO_CACHE_DISK_DIR:=/var/lib/gufo/cache/instances/${GUFO_INSTANCE}}"
  mkdir -p "$GUFO_CACHE_DISK_DIR" \
    || { echo "gufo-launch: cannot create cache dir $GUFO_CACHE_DISK_DIR" >&2; exit 2; }
fi

if [[ -z "${GUFO_MODEL:-}" ]]; then
  echo "gufo-launch: GUFO_MODEL is not set in $env_file${GUFO_INSTANCE:+ (instance $GUFO_INSTANCE)}" >&2
  exit 2
fi

args=(serve --host "$GUFO_BIND" --port "$GUFO_PORT" llm --model "$GUFO_MODEL")

# Append "<flag> <value>" only when the variable is set, so an unset knob
# keeps the server's own default instead of passing an empty argument. The
# explicit if matters: under `set -e` a trailing `[[ ... ]] && ...` would make
# this function return 1 for every unset knob and abort the launch.
append() {
  local flag=$1 value=${2:-}
  if [[ -n "$value" ]]; then
    args+=("$flag" "$value")
  fi
}

append -c  "${GUFO_CONTEXT:-}"
append -j  "${GUFO_SESSIONS:-}"
append -d  "${GUFO_DRAFT_TOKENS:-}"
append -n  "${GUFO_MAX_TOKENS:-}"
append --min-draft-tokens "${GUFO_MIN_DRAFT_TOKENS:-}"
append --speculative "${GUFO_SPECULATIVE:-}"
# Pass the draft artifact only for the backend that consumes it, so the
# assembled command line states what the server will actually do.
case "${GUFO_SPECULATIVE:-}" in
  dflash2)
    append --dflash-model "${GUFO_DFLASH_MODEL:-}"
    if [[ -z "${GUFO_DFLASH_MODEL:-}" ]]; then
      echo "gufo-launch: GUFO_SPECULATIVE=dflash2 needs GUFO_DFLASH_MODEL" >&2
    fi
    # Draft length and policy are DFlash2 knobs: the server rejects
    # --draft-policy for any other backend ("--draft-policy requires
    # DFlash2"), so they are appended only here.
    append --draft-policy "${GUFO_DRAFT_POLICY:-}"
    ;;
  mtp)
    # Qwen3.6-35B-A3B carries its nextn block inside the target GGUF, so the
    # target path doubles as the draft path. Qwen3.8-Flash-Next keeps it in a
    # separate sidecar, so there GUFO_MTP_MODEL must be set explicitly.
    append --mtp-model "${GUFO_MTP_MODEL:-$GUFO_MODEL}"
    ;;
esac
append --ngram-cache-bytes "${GUFO_NGRAM_CACHE_BYTES:-}"
append --prefill-chunk "${GUFO_PREFILL_CHUNK:-}"
append --cache-ram-bytes "${GUFO_CACHE_RAM_BYTES:-}"
append --cache-disk "${GUFO_CACHE_DISK_DIR:-}"
append --cache-disk-bytes "${GUFO_CACHE_DISK_BYTES:-}"
append --cache-disk-staging-bytes "${GUFO_CACHE_DISK_STAGING_BYTES:-}"
append --log-level "${GUFO_LOG_LEVEL:-}"

# Last resort for a flag this script does not know yet. It is word-split on
# purpose so several flags fit in one variable; values must not contain
# spaces.
if [[ -n "${GUFO_EXTRA_ARGS:-}" ]]; then
  read -r -a extra <<<"$GUFO_EXTRA_ARGS"
  args+=("${extra[@]}")
fi

printf 'gufo-launch: %s\n' "${args[*]}"
exec "${GUFO_BIN:-/usr/bin/gufo}" "${args[@]}"
