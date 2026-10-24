#!/bin/sh
# get-revision.sh — source revision stamp for the package build.
#
# The package version comes from debian/changelog (deb-changelog(5): "the
# format of the debian/changelog allows the package building tools to discover
# which version of the package is being built"). This script supplies only the
# additional provenance stamp compiled into the binary, and it must never make
# a package lie about its own tree.
#
# Resolution order:
#   1. GUFO_REVISION in the environment - the packager's explicit answer, and
#      the only way to build from a tree this script cannot judge. Matches the
#      established overrides elsewhere: the kernel's LOCALVERSION=
#      (scripts/setlocalversion: "If the variable LOCALVERSION is set
#      (including being set to an empty string), we don't want to append a plus
#      sign") and setuptools_scm's SETUPTOOLS_SCM_PRETEND_VERSION. Pass it to
#      dpkg-buildpackage with -eGUFO_REVISION=<rev>.
#   2. git describe in a git checkout, plus -dirty when the tree really is
#      modified (see the two rules below).
#   3. "unknown" - a source tarball has no git, and the binary's own default
#      is "unknown" (src/cli/main.cpp).
#
# Two rules about the stamp, both deliberate:
#   * No `git describe --dirty`. A build writes generated artifacts into the
#     source tree (debian/gen-changelog.sh, debhelper's substvars), so
#     --dirty would stamp every packager's tree as dirty for a dirtiness they
#     never caused. The kernel removed the same construct for the same class
#     of reason: "This script must avoid any write attempt to the source tree,
#     which might be read-only. You cannot use 'git describe --dirty' because
#     it tries to create .git/index.lock." Dirt is computed read-only here,
#     with the fallback chain the kernel uses.
#   * Dirt means tracked modifications, excluding the artifacts the packaging
#     generates: debian/changelog (regenerated from version.txt) and the
#     release manifest. Untracked build droppings do not count either, as in
#     the kernel's `git status -uno`.
#
# DEB_BUILD_OPTIONS:
#   no-gufo-dirty   never append -dirty (release builders that want a stamp
#                   that is purely the commit; mirrors the kernel's --no-dirty
#                   and setuptools_scm's local_scheme=no-local-version).
#
# Tags are lightweight in this repository (git for-each-ref reports
# `objecttype commit`), and git-describe(1) considers annotated tags unless
# --tags is given, so --tags is required to see the release tag at all.
# Without it the stamp collapses to a bare hash and cannot show that the build
# is N commits past the tag it claims to be.

set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
root=$(CDPATH= cd -- "${script_dir}/.." && pwd)

# Tracked paths whose modification is produced by the build, not by a person:
# debian/changelog (regenerated from version.txt). Inlined because /bin/sh is
# dash here - no arrays.

if [ -n "${GUFO_REVISION:-}" ]; then
  printf '%s' "${GUFO_REVISION}"
  exit 0
fi

cd "${root}"

# Pin every git call to THIS source tree. Without it, a source tarball unpacked
# inside another repository (a build directory under a project checkout, a
# nested clone) resolves to that enclosing repository and stamps a commit that
# has nothing to do with this code - observed while testing, where an extracted
# tarball reported the parent project's HEAD. A tree with no .git of its own is
# not a checkout, so the answer is "unknown", the binary's own default.
if [ ! -e "${root}/.git" ]; then
  printf 'unknown'
  exit 0
fi

git_here() {
  git --git-dir="${root}/.git" --work-tree="${root}" "$@"
}

if ! git_here rev-parse -q --verify HEAD > /dev/null 2>&1; then
  printf 'unknown'
  exit 0
fi

# Only release tags may name a build. This repository carries non-release tags such as
# backup/packaging-pre0150, and `git describe --tags` happily picks the *nearest* tag of any
# kind: sandbox-proven, a checkout one commit past such a tag stamps
# `backup/packaging-pre0150-1-g<sha>`, and sitting on the tag it prints
# `backup/packaging-pre0150` -- a shipped binary whose provenance names nothing real, which is
# precisely what this script exists to prevent. Exclude every non-release namespace rather than
# whitelisting one spelling, so the next backup tag cannot reintroduce it.
revision=$(git_here describe --tags --always \
  --exclude 'backup/*' --exclude 'ci/*' --exclude 'wip/*' --exclude 'tmp/*' \
  2>/dev/null || git_here rev-parse --short HEAD)
# v1.2.3-4-g0123456 -> 1.2.3-4-g0123456, so `gufo --version' does not
# print the same number twice as "1.2.3 (v1.2.3-4-...)".
revision=$(printf '%s' "${revision}" | sed -e 's/^v\([0-9]\)/\1/')

case " ${DEB_BUILD_OPTIONS:-} " in
  *" no-gufo-dirty "*)
    printf '%s' "${revision}"
    exit 0
    ;;
esac

# Read-only dirt check. --no-optional-locks avoids creating .git/index.lock
# (git >= 2.14); if it is unsupported the command fails and git-diff-index,
# which never writes either, takes over. The test is on the command's own exit
# status plus non-empty output - not on grep's, which cannot tell "clean" from
# "git refused to answer".
dirty=false
if {
  git_here --no-optional-locks status -uno --porcelain -- . ':(exclude)debian/changelog' 2>/dev/null ||
  git_here diff-index --name-only HEAD -- . ':(exclude)debian/changelog' 2>/dev/null
} | grep -q .; then
  dirty=true
fi

if ${dirty}; then
  revision="${revision}-dirty"
fi

printf '%s' "${revision}"
