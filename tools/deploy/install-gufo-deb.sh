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
# Package paths default to ./gufo_*.deb and ./gufo-rocm-repo_*.deb in
# the current directory, so new versions do not require script changes.
# Override with environment variables or positional arguments:
#
#   GUFO_DEB=/path/to/gufo_1.2.3_amd64.deb \
#   GUFO_ROCM_REPO_DEB=/path/to/gufo-rocm-repo_1.2.3_all.deb \
#   ./install-gufo-deb.sh

set -euo pipefail

GUFO_DEB="${GUFO_DEB:-${1:-}}"
GUFO_ROCM_REPO_DEB="${GUFO_ROCM_REPO_DEB:-${2:-}}"

if [ -z "$GUFO_DEB" ]; then
  GUFO_DEB=$(ls gufo_*.deb 2>/dev/null | head -1 || true)
fi
if [ -z "$GUFO_ROCM_REPO_DEB" ]; then
  GUFO_ROCM_REPO_DEB=$(ls gufo-rocm-repo_*.deb 2>/dev/null | head -1 || true)
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

echo "==> Installing $GUFO_ROCM_REPO_DEB (registers the AMD ROCm repository)"
$SUDO apt install -y --no-install-recommends "$GUFO_ROCM_REPO_DEB"

echo "==> Updating package lists"
$SUDO apt update

echo "==> Installing $GUFO_DEB (runtime dependencies only)"
$SUDO apt install -y --no-install-recommends "$GUFO_DEB"

echo "==> Verifying no compiler or development packages were installed"
COMPILERS=$($SUDO dpkg-query -W -f='${Package}\n' 2>/dev/null \
  | grep -E '^(gcc|g\+\+|clang|cmake|ninja|make|build-essential|dpkg-dev|binutils)' || true)
if [ -n "$COMPILERS" ]; then
  echo "warning: compiler/toolchain packages present:" >&2
  echo "$COMPILERS" >&2
else
  echo "OK: no compiler or toolchain packages installed"
fi

echo "==> Done. Configure /etc/gufo/gufo.env and start with:"
echo "    sudo systemctl enable --now gufo"