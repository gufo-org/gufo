#!/usr/bin/env bash
# Install the Gufo Debian packages on a target machine without pulling
# compilers or development packages.
#
# The AMD ROCm runtime packages (hipblas, hipblaslt, rocblas) recommend
# their -dev counterparts, which depend on the ROCm compiler and the
# GCC toolchain. apt is therefore invoked with --no-install-recommends;
# the Gufo package itself only depends on runtime libraries.
#
# Notes:
# - The final verification step warns if compiler/toolchain packages are
#   present. A pre-existing toolchain (e.g. dkms for custom kernel
#   headers) is expected and harmless; the warning is informational.
# - Never remove ROCm -dev packages with apt on a machine that also has
#   xrt or dkms installed: AMD's dependency graph is entangled
#   (rocsolver depends on rocsolver-dev, hipblas recommends hipblas-dev)
#   and removal cascades into unrelated packages.
# - The service is not started by the package; configure
#   /etc/gufo/gufo.env first, then run 'sudo systemctl enable --now gufo'.
#
# Usage:
#   ./install-gufo-deb.sh [GUFO_DEB] [GUFO_ROCM_REPO_DEB]
#
# Package paths default to the newest ./gufo_*.deb and ./gufo-rocm-repo_*.deb in
# the current directory, so new versions do not require script changes.
# Override with environment variables or positional arguments:
#
#   GUFO_DEB=/path/to/gufo_1.2.3_amd64.deb \
#   GUFO_ROCM_REPO_DEB=/path/to/gufo-rocm-repo_1.2.3_all.deb \
#   ./install-gufo-deb.sh

set -euo pipefail

GUFO_DEB="${GUFO_DEB:-${1:-}}"
GUFO_ROCM_REPO_DEB="${GUFO_ROCM_REPO_DEB:-${2:-}}"

# Select the NEWEST version present, not the first name `ls` returns: a plain
# `ls gufo_*.deb | head -1` picks 0.10.0 over 0.11.0 because '0' sorts before
# '1', which silently re-installs the previous release instead of upgrading.
if [ -z "$GUFO_DEB" ]; then
  GUFO_DEB=$(ls -1 gufo_*.deb 2>/dev/null | sort -V | tail -1 || true)
fi
if [ -z "$GUFO_ROCM_REPO_DEB" ]; then
  GUFO_ROCM_REPO_DEB=$(ls -1 gufo-rocm-repo_*.deb 2>/dev/null | sort -V | tail -1 || true)
fi

if [ -z "$GUFO_DEB" ] || [ -z "$GUFO_ROCM_REPO_DEB" ]; then
  echo "error: gufo .deb files not found; pass paths as arguments or run" >&2
  echo "       from the directory containing them" >&2
  exit 1
fi

# apt treats bare names as package names; make local paths explicit.
for var in GUFO_DEB GUFO_ROCM_REPO_DEB; do
  case "${!var}" in
    /*|./*) ;;
    *) eval "$var=\"./${!var}\"" ;;
  esac
done

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  SUDO="sudo"
fi

# Never let this script depend on an interactive answer. A modified conffile asks a question;
# over ssh without a TTY that question hits EOF and dpkg leaves the package half-configured
# (state `iU'), which is the failure this project already recorded and was bitten by.
# confold keeps the operator's file and answers the question up front.
export DEBIAN_FRONTEND=noninteractive
DPKG_OPTS="-o Dpkg::Options::=--force-confold"

apt_install() { # apt_install <deb> <strict: yes|no>
  local deb="$1" strict="${2:-yes}" out rc
  out=$($SUDO apt install -y --no-install-recommends $DPKG_OPTS "$deb" 2>&1); rc=$?
  printf '%s\n' "$out"
  if [ "$rc" -ne 0 ]; then
    echo "error: apt failed on $deb (exit $rc)" >&2
    return "$rc"
  fi
  # "0 upgraded, 0 newly installed" means NOTHING was installed: the version string did not
  # move, so the host keeps running the old binary while every log line says it was updated.
  if grep -qE '^[0-9]+ upgraded, 0 newly installed' <<< "$out"; then
    if [ "$strict" = yes ]; then
      echo "error: apt installed nothing for $deb -- the package version is already current." >&2
      echo "       bump the revision (see debian/changelog) or this deploy is a no-op." >&2
      return 1
    fi
    echo "note: $deb already present at this version" >&2
  fi
  return 0
}

echo "==> Installing $GUFO_ROCM_REPO_DEB (registers the AMD ROCm repository)"
apt_install "$GUFO_ROCM_REPO_DEB" no

echo "==> Updating package lists"
$SUDO apt update

echo "==> Installing $GUFO_DEB (runtime dependencies only)"
apt_install "$GUFO_DEB" yes

echo "==> Verifying no compiler or development toolchain was installed"
COMPILERS=$($SUDO dpkg-query -W -f='${Package}\n' 2>/dev/null \
  | grep -E '^(gcc(-[0-9]+)?|g\+\+(-[0-9]+)?|clang(-[0-9]+)?|cmake|ninja-build|make|build-essential|dpkg-dev|binutils(-.+)?)$' || true)
COMPILER_BINS=""
for tool in gcc g++ cc c++ clang clang++ cmake ninja make; do
  if command -v "$tool" >/dev/null 2>&1; then
    COMPILER_BINS="$COMPILER_BINS $tool"
  fi
done
if [ -n "$COMPILERS$COMPILER_BINS" ]; then
  echo "warning: compiler toolchain present:" >&2
  [ -n "$COMPILERS" ] && echo "  packages:$COMPILERS" >&2
  [ -n "$COMPILER_BINS" ] && echo "  executables:$COMPILER_BINS" >&2
else
  echo "OK: no compiler toolchain packages or executables present"
fi

echo "==> Verifying the installed VERSION against the artifact"
want_v=$(dpkg-deb -f "$GUFO_DEB" Version 2>/dev/null || echo "?")
got_v=$($SUDO dpkg-query -W -f='${Version}' gufo 2>/dev/null || echo "missing")
if [ "$want_v" != "$got_v" ]; then
  echo "error: dpkg holds '$got_v' but the artifact is '$want_v'" >&2
  exit 1
fi
echo "OK: dpkg reports $got_v, matching $GUFO_DEB"

# The process, not the package database: an upgrade can install correctly and leave every unit
# down -- correctly after an operator stop, defectively after a prerm stop -- and `dpkg -l'
# cannot tell the two apart, or either from "running the deleted inode".
state_dir_note=''
for unit in $(systemctl list-unit-files --no-legend --plain 'gufo*' 2>/dev/null | awk '{print $1}'); do
  case "$unit" in gufo@.service) continue ;; esac
  st=$(systemctl is-active "$unit" 2>/dev/null || true)
  en=$(systemctl is-enabled "$unit" 2>/dev/null || true)
  if [ "$st" = active ]; then
    pid=$(systemctl show -p MainPID --value "$unit" 2>/dev/null || echo 0)
    exe=$(readlink "/proc/$pid/exe" 2>/dev/null || echo "unreadable (needs root)")
    case "$exe" in
      *"(deleted)") echo "  FAIL $unit pid $pid runs a DELETED binary: $exe -> systemctl restart $unit" >&2; state_dir_note=1 ;;
      *) printf '  ok   %-24s active, enabled=%s, exe=%s\n' "$unit" "$en" "$exe" ;;
    esac
  else
    printf '  note %-24s %s (enabled=%s) -- start it only when /etc/gufo/gufo.env is right\n' "$unit" "$st" "$en"
  fi
done
[ -z "$state_dir_note" ] || exit 1

echo "==> Done. Configure /etc/gufo/gufo.env and start with:"
echo "    sudo systemctl enable --now gufo"