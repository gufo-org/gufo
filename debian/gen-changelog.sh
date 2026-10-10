#!/bin/sh
# gen-changelog.sh — keep debian/changelog in sync with the release version.
#
# The canonical release version is version.txt, bumped by the release
# automation (Release Please) together with CHANGELOG.md. Debian packaging takes
# its package version from the top entry of debian/changelog, so this script
# regenerates that entry from version.txt and the matching CHANGELOG.md section
# rather than letting the two drift.
#
# Usage: debian/gen-changelog.sh [UPSTREAM_VERSION]
#
# Environment:
#   DEBIAN_REVISION  force the Debian revision. Defaults to the current revision
#                    when the upstream version is unchanged (so re-runs are
#                    idempotent), otherwise 1.
#
# The script is idempotent: re-running it for the same version and CHANGELOG
# content does not modify the file.

set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
root=$(CDPATH= cd -- "${script_dir}/.." && pwd)

changelog="${script_dir}/changelog"
control="${script_dir}/control"
upstream_changelog="${root}/CHANGELOG.md"

version="${1:-$(cat "${root}/version.txt")}"
if [ -z "${version}" ]; then
  echo "gen-changelog: empty upstream version" >&2
  exit 1
fi

package=$(sed -n 's/^Source:[[:space:]]*//p' "${control}" | head -n 1)
if [ -z "${package}" ]; then
  echo "gen-changelog: no Source field in debian/control" >&2
  exit 1
fi

existing_version=$(dpkg-parsechangelog -l "${changelog}" -SVersion 2>/dev/null || true)
existing_upstream="${existing_version}"
existing_revision=""
case "${existing_version}" in
  *-*)
    existing_upstream="${existing_version%-*}"
    existing_revision="${existing_version##*-}"
    ;;
esac

if [ -n "${DEBIAN_REVISION:-}" ]; then
  revision="${DEBIAN_REVISION}"
elif [ "${existing_upstream}" = "${version}" ] && [ -n "${existing_revision}" ]; then
  revision="${existing_revision}"
else
  revision=1
fi
new_version="${version}-${revision}"

# Body taken from the matching CHANGELOG.md section, reduced to its bullets.
changes=$(
  awk -v ver="${version}" '
    /^## \[/ {
      if (capture) exit
      if (index($0, "[" ver "]") > 0) { capture = 1; next }
    }
    capture && /^\* / { print }
  ' "${upstream_changelog}" 2>/dev/null \
  | sed -e 's/\[\([^]]*\)\](https:\/\/[^)]*)/\1/g' \
        -e 's/ ([0-9a-f]\{7,\})//g' \
        -e 's/[[:space:]]*$//' \
        -e 's/^\* /  * /'
)
if [ -z "${changes}" ]; then
  changes="  * Upstream release ${version}."
fi

# Trailer: keep the existing maintainer and, for the same upstream version, the
# existing date so repeated runs are idempotent.
previous_trailer=$(sed -n 's/^ -- \(.*\)$/\1/p' "${changelog}" | head -n 1)
maintainer=$(printf '%s\n' "${previous_trailer}" \
  | sed -n 's/^\(.*\)  [A-Z][a-z][a-z], [0-9].*$/\1/p')
if [ -z "${maintainer}" ]; then
  maintainer="${DEBFULLNAME:-gufo release} <${DEBEMAIL:-release@gufo.invalid}>"
fi
if [ "${existing_upstream}" = "${version}" ] && [ -n "${previous_trailer}" ]; then
  trailer="${previous_trailer}"
else
  trailer="${maintainer}  $(date -Ru)"
fi

entry="${package} (${new_version}) unstable; urgency=medium

${changes}

 -- ${trailer}"

if [ "${existing_upstream}" = "${version}" ]; then
  # Replace the existing entry for this version instead of duplicating it.
  remainder=$(awk '
    !dropped { if ($0 ~ /^ -- /) dropped = 1; next }
    dropped  {
      if (!started && $0 ~ /^[[:space:]]*$/) next
      started = 1
      print
    }
  ' "${changelog}")
else
  remainder=$(cat "${changelog}")
fi

tmp="$(mktemp "${changelog}.XXXXXX")"
trap 'rm -f "${tmp}"' EXIT INT TERM
printf '%s\n\n' "${entry}" > "${tmp}"
if [ -n "${remainder}" ]; then
  printf '%s\n' "${remainder}" >> "${tmp}"
fi
mv "${tmp}" "${changelog}"
trap - EXIT INT TERM

echo "debian/changelog: ${package} ${new_version}"
