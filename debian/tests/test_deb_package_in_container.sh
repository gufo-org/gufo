#!/usr/bin/env bash
# test_deb_package_in_container.sh - install, upgrade and purge the gufo .deb in a clean Debian
# container with real systemd, and assert what the maintainer scripts promise.
#
# WHAT THIS COVERS THAT A SHELL MATRIX CANNOT
# debian/tests/upgrade-units stubs systemctl and feeds the decision script hand-written input. That
# proves the decision logic, not the packaging: only real dpkg runs prerm before unpack and postinst
# after, and only real systemd holds the enablement state and the marker files the two exchange. A
# matrix tests a model of the transaction against a model of it, and cannot produce the inputs an
# older release actually writes - which is how an upgrade that left a running instance down for the
# whole boot cleared seventeen green scenarios.
#
# WHAT IT DOES NOT COVER
# Inference. ExecStart is replaced by a stub INSIDE THESE CONTAINERS ONLY: no weights are loaded and
# no accelerator throughput is claimed. `gufo diagnose` does run, and only enumerates the device.
#
# USAGE
#   debian/tests/test_deb_package_in_container.sh [options] [package.deb ...]
#
#   package.deb ...   built artifacts; the first is the package under test.
#                     Default: newest gufo_*_amd64.deb in ./deb, . or ..
#   -i, --image NAME  image; default $IMAGE, else built on demand from the Containerfile written here
#   -p, --port N      loopback port for the stub; default $PORT, else a free loopback port picked
#                     below the kernel's ephemeral range (never the packaged default)
#   -c, --cache DIR   host model cache, mounted READ-ONLY; default $CACHE, else
#                     /var/lib/huggingface/hub if present, else not mounted
#       --legacy-deb D  an older gufo package, to replay an upgrade from a release whose prerm wrote
#                     empty markers. Without it the same input is produced by patching the INSTALLED
#                     prerm, so no binary artifact is needed (see scenario E).
#       --gpu | --no-gpu  pass /dev/kfd and /dev/dri when present (default auto). With them the
#                     diagnose assertion runs; without it the packaging path is still fully tested and
#                     the hardware assertion is reported as skipped, not silently dropped.
#       --keep        leave the containers up (default: removed)
#       --podman CMD  runtime; default $PODMAN, else podman, else docker (no --systemd mode there,
#                     so the systemd assertions may not hold - it is reported, not assumed)
#
# ENV equivalents: IMAGE PORT CACHE LEGACY_DEB GPU KEEP PODMAN IMG_NAME
# Exits 0 only when every assertion passed.
set -uo pipefail

TMPD="$(mktemp -d)"; trap 'rm -rf "$TMPD"' EXIT

IMAGE="${IMAGE:-}"
PORT="${PORT:-}"
# The port the packaged default binds (`GUFO_PORT=8080` in debian/gufo.env), used only to refuse
# picking it for the stub. Override with DEFAULT_PORT if your host already binds it elsewhere.
DEFAULT_PORT="${DEFAULT_PORT:-8080}"
CACHE="${CACHE:-}"
LEGACY_DEB="${LEGACY_DEB:-}"
GPU="${GPU:-auto}"
KEEP="${KEEP:-no}"
PODMAN="${PODMAN:-}"
IMG_NAME="${IMG_NAME:-localhost/gufo-package-test}"
DEBS=()

while [ $# -gt 0 ]; do
  case "$1" in
    -i|--image)    IMAGE="$2"; shift 2 ;;
    -p|--port)     PORT="$2"; shift 2 ;;
    -c|--cache)    CACHE="$2"; shift 2 ;;
    --legacy-deb) LEGACY_DEB="$2"; shift 2 ;;
    --gpu)         GPU=yes; shift ;;
    --no-gpu)      GPU=no; shift ;;
    --keep)        KEEP=yes; shift ;;
    --podman)      PODMAN="$2"; shift 2 ;;
    -h|--help)     sed -n '2,36p' "$0"; exit 0 ;;
    -*)            echo "unknown option: $1" >&2; exit 2 ;;
    *)             DEBS+=("$1"); shift ;;
  esac
done

# ---- preflight ------------------------------------------------------------------------------
if [ "$(id -u)" -ne 0 ]; then
  echo "note: not root. /dev passthrough and systemd as PID 1 need a rootful runtime (sudo podman ...)." >&2
fi
if [ -z "$PODMAN" ]; then
  PODMAN="$(command -v podman || true)"
  [ -n "$PODMAN" ] || PODMAN="$(command -v docker || true)"
fi
[ -n "$PODMAN" ] || { echo "REFUSED: no container runtime found. Set PODMAN=/path/to/podman." >&2; exit 2; }
# A name that resolves to nothing is not a runtime: PODMAN=/wrong/path/podman used to pass the
# check above and then fail on every later call with a message that named no cause.
RT="${PODMAN%% *}"
if [ "${RT#/*}" = "$RT" ]; then RT="$(command -v "$RT" || true)"; fi
[ -x "$RT" ] || { echo "REFUSED: runtime $PODMAN does not resolve to an executable ($RT)." >&2; exit 2; }

if [ "${#DEBS[@]}" -eq 0 ]; then
  while IFS= read -r f; do [ -n "$f" ] && DEBS+=("$f"); done < <(ls -t ./deb/gufo_*_amd64.deb ./gufo_*_amd64.deb ../gufo_*_amd64.deb 2>/dev/null)
fi
[ "${#DEBS[@]}" -gt 0 ] || { echo "REFUSED: no .deb given and none found in ./deb, . or .. Build one first (dpkg-buildpackage writes to the parent directory)." >&2; exit 2; }
# podman reads a relative -v source as a NAMED VOLUME, so every path that reaches the runtime is
# absolutised here, once, instead of at each use site (that is how this broke twice).
MAIN_DEB="$(readlink -f "${DEBS[0]}")"
MAIN_VER="$(dpkg-deb -f "$MAIN_DEB" Version)"
COMPANION="${COMPANION:-}"
for i in "${!DEBS[@]}"; do DEBS[i]="$(readlink -f "${DEBS[i]}")"; done
if [ -z "$COMPANION" ]; then
  COMPANION="$(ls -t ./deb/gufo-rocm-repo_*_all.deb ./gufo-rocm-repo_*_all.deb ../gufo-rocm-repo_*_all.deb 2>/dev/null | head -1 || true)"
  [ -n "$COMPANION" ] && COMPANION="$(readlink -f "$COMPANION")"
fi

run_in() { $PODMAN exec "$1" bash -lc "$2" 2>&1; }

# The repo's own installer, mounted in: `dpkg -i` on a bare dependency list is not how this
# package reaches a host. It failed once here with "dependency problems", which left no `gufo`
# user, which made every unit exit 217/USER -- a cascade that reads like broken packaging but is
# a missing first step.
SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INSTALLER="${INSTALLER:-$SRC_ROOT/tools/deploy/install-gufo-deb.sh}"

free_port() {
  # A loopback port outside the kernel's ephemeral range, so the stub cannot collide with an
  # outbound connection and is not the packaged default. Nothing here names a host or a deployment:
  # the range is read from the kernel, and 8080 is refused because it is the port the shipped units
  # bind.
  local p start lo hi
  read -r lo hi </proc/sys/net/ipv4/ip_local_port_range 2>/dev/null || { lo=32768; hi=60999; }
  start=$((lo - 1))
  for p in $(seq "$start" -1 "$((start - 199))"); do
    [ "$p" -ge 1024 ] || break
    [ "$p" != "$DEFAULT_PORT" ] || continue
    if ! ss -ltnH "sport = :$p" 2>/dev/null | grep -q .; then printf '%s' "$p"; return 0; fi
  done
  return 1
}
[ -n "$PORT" ] || PORT="$(free_port)" || true
[ -n "$PORT" ] || { echo "REFUSED: no free loopback port." >&2; exit 2; }
[ "$PORT" != "$DEFAULT_PORT" ] || { echo "REFUSED: $DEFAULT_PORT is the port the shipped units bind; pick another." >&2; exit 2; }

if [ -z "$CACHE" ] && [ -d /var/lib/huggingface/hub ]; then CACHE=/var/lib/huggingface/hub; fi
[ -n "$CACHE" ] && CACHE="$(readlink -f "$CACHE")"
if [ -n "$LEGACY_DEB" ]; then
  [ -r "$LEGACY_DEB" ] || { echo "REFUSED: --legacy-deb not readable: $LEGACY_DEB" >&2; exit 2; }
  LEGACY_DEB="$(readlink -f "$LEGACY_DEB")"
fi
if [ "$GPU" = auto ]; then
  if [ -e /dev/kfd ] && [ -e /dev/dri ]; then GPU=yes; else GPU=no; fi
fi

pass=0; fail=0
ck()     { if [ "$2" = "$3" ]; then printf '  ok   %-46s %s\n' "$1" "$3"; pass=$((pass+1)); else printf '  FAIL %-46s want[%s] got[%s]\n' "$1" "$2" "$3"; fail=$((fail+1)); fi; }
ck_has() { if printf '%s' "$3" | grep -qF -- "$2"; then printf '  ok   %-46s found: %s\n' "$1" "$2"; pass=$((pass+1)); else printf '  FAIL %-46s missing [%s]\n' "$1" "$2"; fail=$((fail+1)); fi; }
ck_not() { if printf '%s' "$3" | grep -qF -- "$2"; then printf '  FAIL %-46s must NOT contain [%s]\n' "$1" "$2"; fail=$((fail+1)); else printf '  ok   %-46s absent: %s\n' "$1" "$2"; pass=$((pass+1)); fi; }
# An empty needle matches everything, so ck_not with one reports a pass it never earned.
ck_not_s() {
  if [ -z "$2" ]; then printf '  FAIL %-46s EMPTY NEEDLE - asserts nothing\n' "$1"; fail=$((fail+1)); return 0; fi
  ck_not "$1" "$2" "$3"
}

write_stub() { # the drop-in that keeps the unit observably up without loading anything
  cat > "$TMPD/stub.conf" <<'EOS'
# written by debian/tests/test_deb_package_in_container.sh
# ExecStart is replaced so the lifecycle can be exercised without loading weights: this test proves
# dpkg, systemd and the maintainer scripts, NOT model load or token rate. Delete this drop-in to run
# the real service.
[Service]
ExecStart=
ExecStart=/bin/sh -c 'echo gufo-test-stub; exec sleep infinity'
EOS
}

build_image() {
  cat > "$TMPD/Containerfile" <<'EOS'
# Minimal Debian with a working systemd as PID 1, for package lifecycle tests.
# policy-rc.d is removed ON PURPOSE: its default exit 101 makes invoke-rc.d and deb-systemd-invoke
# refuse every service action, so a container that keeps it can "pass" an upgrade while restarting
# nothing - the container equivalent of a bare host, which is what we want to test against.
FROM docker.io/library/debian:trixie
# The explicit CMD is load-bearing, not decoration: without it the base image keeps its `bash` CMD
# and podman leaves it there, so the container exits at once and every assertion downstream fails
# with something that reads like a packaging problem.
CMD ["/lib/systemd/systemd"]
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends systemd systemd-sysv dbus ca-certificates \
 && rm -f /usr/sbin/policy-rc.d \
 && apt-get clean && rm -rf /var/lib/apt/lists/*
EOS
  echo "-- building $IMG_NAME"
  $PODMAN build -t "$IMG_NAME" -f "$TMPD/Containerfile" "$TMPD" >/dev/null
}

spawn() { # spawn <name>; prints readiness
  $PODMAN rm -f "$1" >/dev/null 2>&1 || true
  local args=(run -d --name "$1" --systemd=always --network=host --ulimit memlock=-1)
  if [ "$GPU" = yes ]; then args+=(--device /dev/kfd --device /dev/dri); fi
  args+=(-v "$MAIN_DEB":/root/main.deb:ro)
  if [ -n "$COMPANION" ]; then args+=(-v "$COMPANION":/root/companion.deb:ro); fi
  if [ -n "$CACHE" ]; then args+=(-v "$CACHE":/var/lib/huggingface/hub:ro); fi
  if [ -f "$INSTALLER" ]; then args+=(-v "$INSTALLER":/root/installer.sh:ro); fi
  case "$1" in
    gufo-deb-B) if [ -n "$LEGACY_DEB" ]; then args+=(-v "$LEGACY_DEB":/root/legacy.deb:ro); fi ;;
  esac
  local out
  # never swallow the runtime's stderr: a silent `|| return 1` cost a debugging round-trip when
  # a bad bind source looked like "the container just would not start".
  out="$($PODMAN "${args[@]}" "$IMG_NAME" 2>&1 >/dev/null)" || { printf '  podman run failed: %s\n' "$out" >&2; return 1; }
  # poll for readiness instead of sleeping blind, and if it dies, say why: a dead container whose
  # logs nobody printed looks exactly like a broken package.
  local i state
  for i in $(seq 1 40); do
    state="$($PODMAN inspect -f '{{.State.Status}}' "$1" 2>/dev/null || echo gone)"
    [ "$state" = running ] && break
    case "$state" in
      exited|dead|gone|created|paused)
        printf '  container %s never reached running (status=%s, exit=%s); its log:\n' "$1" "$state" \
          "$($PODMAN inspect -f '{{.State.ExitCode}}' "$1" 2>/dev/null || echo '?')" >&2
        $PODMAN logs --tail 15 "$1" 2>&1 | sed 's/^/    /' >&2
        return 1 ;;
    esac
    sleep 1
  done
  if [ "$state" != running ]; then
    printf '  container %s still not running after 40s\n' "$1" >&2
    return 1
  fi
  run_in "$1" 'if [ -e /usr/sbin/policy-rc.d ]; then echo BLOCKED; else echo open; fi'
}

start_instance() { # start_instance <container> <instance>
  write_stub
  $PODMAN exec "$1" mkdir -p "/etc/systemd/system/gufo@$2.service.d"
  $PODMAN cp "$TMPD/stub.conf" "$1:/etc/systemd/system/gufo@$2.service.d/zz-test-stub.conf"
  run_in "$1" 'systemctl daemon-reload >/dev/null 2>&1; systemctl enable --now gufo@'"$2"' >/dev/null 2>&1; sleep 2; systemctl is-active gufo@'"$2"
}

pid_of() { run_in "$1" 'systemctl show -p MainPID --value gufo@'"$2"'.service'; }

echo "=== gufo .deb lifecycle in a container ==="
echo "  runtime   : $PODMAN"
echo "  image     : $IMG_NAME"
echo "  package   : $MAIN_DEB (Version $MAIN_VER)"
[ -n "$COMPANION" ] && echo "  companion : $COMPANION"
echo "  port      : 127.0.0.1:$PORT (stub only - nothing is served)"
echo "  cache     : ${CACHE:-none} (read-only when set)"
echo "  devices   : $GPU"

# Built on every run: podman reuses the layer cache, so this costs about a second, whereas an
# image "skipped because it exists" is a stale one that may not match the Containerfile in the tree.
build_image || { echo "REFUSED: image build failed" >&2; exit 2; }

echo
echo "== A. first deployment into a clean system =="
R="$(spawn gufo-deb-A)" || { echo "REFUSED: container A did not start (see the podman error above)"; exit 2; }
ck "container up, no policy-rc.d blocking lifecycle" "open" "$R"
EXTRA_IN=""
[ -n "$COMPANION" ] && EXTRA_IN="/root/companion.deb"
if [ -f "$INSTALLER" ]; then
  echo "  installing via the shipped installer: $(basename "$INSTALLER")"
  OUT="$(run_in gufo-deb-A "bash /root/installer.sh /root/main.deb $EXTRA_IN 2>&1 | tail -30")"
else
  printf '  note: installer not found at %s, using bare dpkg (deps must already be present)\n' "$INSTALLER"
  OUT="$(run_in gufo-deb-A "dpkg -i /root/main.deb $EXTRA_IN 2>&1 | tail -30")"
fi
printf '%s\n' "$OUT" | grep -E '^gufo:' | sed 's/^/    /'
INSTALLED="$(run_in gufo-deb-A 'dpkg-query -W gufo | cut -f2')"
ck "dpkg holds the version of the artifact" "$MAIN_VER" "$INSTALLED"
ck_has "the binary reports that version" "${MAIN_VER%%-*}" "$(run_in gufo-deb-A 'gufo --version')"
NOTFOUND="$(run_in gufo-deb-A 'ldd /usr/bin/gufo | grep -c "not found" || true')"
ck "no unresolved shared libraries" "0" "$NOTFOUND"
ck "ROCm runtime resolvable through the loader cache" "yes" "$(run_in gufo-deb-A 'ldconfig -p | grep -q libhsa && echo yes || echo no')"
# The apt pin, asserted and not assumed. A preferences stanza is inert if it carries more than one
# `Package:` field -- apt reads one per stanza, installs the file without complaint, and leaves the
# distro priority in force. The shape check needs no network; the priority check is honest about the
# index it needs and reports a skip rather than a pass when the sandbox fetched nothing.
ck "the pin stanza carries exactly one Package: field" "1" \
  "$(run_in gufo-deb-A 'grep -c "^Package:" /etc/apt/preferences.d/gufo-rocm-pin 2>/dev/null || echo 0')"
PINPRIO="$(run_in gufo-deb-A 'apt-cache policy hipblaslt 2>/dev/null | awk "/^ \*\*\*/{print \$3; exit}"')"
if [ -z "$PINPRIO" ]; then
  printf '  skip %-46s ROCm index not fetched here; --network or a fetched index enables it\n' "apt pin resolves the ROCm family to 600"
else
  ck "apt pin resolves the ROCm family to 600" "600" "$PINPRIO"
fi
PKGS="$(run_in gufo-deb-A 'dpkg-query -W -f="\${Package} "')"
ck_not "no compiler toolchain was pulled in by a runtime install" "g++" "$PKGS"
ck "first install does NOT enable the classic unit" "disabled" "$(run_in gufo-deb-A 'systemctl is-enabled gufo.service 2>&1')"
ck "first install does NOT start the classic unit" "inactive" "$(run_in gufo-deb-A 'systemctl is-active gufo.service')"
ck "the instance template is installed" "present" "$(run_in gufo-deb-A 'test -f /lib/systemd/system/gufo@.service && echo present || echo missing')"
if [ "$GPU" = yes ]; then
  ck "diagnose sees the accelerator in the container" "supported" "$(run_in gufo-deb-A 'gufo diagnose 2>&1 | grep -oE "\[supported\]" | tr -d "[]" | head -1')"
else
  printf '  skip %-46s no devices passed; --gpu enables it\n' "diagnose on the accelerator"
fi

echo
echo "== B. an instance the operator stopped stays stopped across an upgrade =="
ck "instance started for the test" "active" "$(start_instance gufo-deb-A demo)"
run_in gufo-deb-A 'systemctl stop gufo@demo >/dev/null 2>&1; sleep 1' >/dev/null
UP="$(run_in gufo-deb-A 'dpkg -i /root/main.deb 2>&1 | grep -E "^ *gufo:|nothing was stopped|nothing to restart"')"
printf '%s\n' "$UP" | sed 's/^/    /'
ck "the stopped unit is still inactive afterwards" "inactive" "$(run_in gufo-deb-A 'systemctl is-active gufo@demo')"
ck_has "postinst says it left it alone" "left as the admin left it" "$UP"
ck_not "no start was attempted on it" "start gufo@demo" "$UP"
ck_has "prerm reports only its own action" "nothing was stopped" "$UP"
ck_not "prerm makes no claim about the whole transaction" "nothing to restart" "$UP"

echo
echo "== C. an active instance is restarted onto the new package =="
run_in gufo-deb-A 'systemctl start gufo@demo >/dev/null 2>&1; sleep 2' >/dev/null
P1="$(pid_of gufo-deb-A demo)"
UP2="$(run_in gufo-deb-A 'dpkg -i /root/main.deb 2>&1 | grep -E "^ *gufo:"')"
printf '%s\n' "$UP2" | sed 's/^/    /'
sleep 2
P2="$(pid_of gufo-deb-A demo)"
ck "unit active after the reinstall" "active" "$(run_in gufo-deb-A 'systemctl is-active gufo@demo')"
ck_has "postinst names it as a restart" "restart gufo@demo" "$UP2"
if [ -n "$P2" ] && [ "$P2" != "0" ] && [ "$P1" != "$P2" ]; then
  printf '  ok   %-46s pid %s -> %s\n' "the process was really replaced" "$P1" "$P2"; pass=$((pass+1))
else
  printf '  FAIL %-46s pid %s -> %s\n' "the process was really replaced" "$P1" "$P2"; fail=$((fail+1))
fi

echo
echo "== D. purge: kept data handed to root, caches left alone, account removed =="
# numeric ownership only: a read-only bind mount resolves no names inside the container, so a
# "%U:%G" comparison there reports the same unknown word forever and can never disagree.
CACHE_OWN="$(run_in gufo-deb-A 'stat -c "%u:%g" /var/lib/huggingface/hub 2>/dev/null || echo none')"
PUR="$(run_in gufo-deb-A 'dpkg --purge gufo 2>&1 | tail -3')"; printf '%s\n' "$PUR" | sed 's/^/    /'
ck "classic unit gone from the manager" "inactive" "$(run_in gufo-deb-A 'systemctl is-active gufo.service 2>&1')"
ACCOUNT="$(run_in gufo-deb-A 'getent passwd gufo | cut -d: -f1 || true')"
ck "the service account is removed on purge" "" "$ACCOUNT"
KEPT="$(run_in gufo-deb-A 'test -d /var/lib/gufo && stat -c "%U:%G" /var/lib/gufo || echo gone')"
ck "kept /var/lib/gufo handed to root, not deleted with the package" "root:root" "$KEPT"
if [ -n "$CACHE" ]; then
  ck "model cache ownership untouched (numeric)" "$CACHE_OWN" "$(run_in gufo-deb-A 'stat -c "%u:%g" /var/lib/huggingface/hub')"
fi
ck "the conffile is removed on purge" "absent" "$(run_in gufo-deb-A 'test -e /etc/gufo/gufo.env && echo present || echo absent')"

echo
echo "== E. upgrade from a release whose prerm wrote EMPTY markers =="
# Self-contained by design: the older release's prerm is reproduced by patching the INSTALLED one,
# so no binary artifact has to be committed. dpkg runs the prerm of the version being upgraded FROM,
# so what executes after the patch is exactly the legacy shape - an empty marker, plus a boot record
# naming this boot. A consumer that refused empty markers would strand the instance right here, which
# is the failure this scenario exists to catch.
$PODMAN rm -f gufo-deb-B >/dev/null 2>&1 || true
R2="$(spawn gufo-deb-B)" || { echo "  FAIL container B did not start"; fail=$((fail+1)); }
# What B starts FROM: the older artifact when one was given, otherwise the package under test with
# its prerm patched below. Either way it is installed through the shipped installer -- a bare
# `dpkg -i` here leaves the package unconfigured, no `gufo` user, and a unit cycling at
# `activating`, which looks like a packaging failure but is a missing first step.
FROM_IN=/root/main.deb
[ -n "$LEGACY_DEB" ] && FROM_IN=/root/legacy.deb
ck "the package B starts from is mounted in the container" "present" "$(run_in gufo-deb-B "test -f $FROM_IN && echo present || echo missing")"
OUTB="$(run_in gufo-deb-B "bash /root/installer.sh $FROM_IN $EXTRA_IN 2>&1 | tail -3")"
printf '%s\n' "$OUTB" | grep -E '^(OK|FAIL)' | sed 's/^/    /'
ck "gufo is configured in container B" "present" "$(run_in gufo-deb-B 'test -f /var/lib/dpkg/info/gufo.prerm && echo present || echo missing')"
ck "the service account exists, so the unit can start" "present" "$(run_in gufo-deb-B 'getent passwd gufo >/dev/null && echo present || echo missing')"
printf '  B starts from version: %s\n' "$(run_in gufo-deb-B 'dpkg-query -W gufo | cut -f2')"
if [ -z "$LEGACY_DEB" ]; then
  # No older artifact available: reproduce the older release's prerm in place. This is the shape
  # 0.15.1-1 and 0.15.1-2 actually shipped -- verified from their installed prerm, quoted here:
  #     : > "/run/gufo/${unit}.was-active"
  cat > "$TMPD/legacy-prerm.sh" <<'EOS'
set -eu
f=/var/lib/dpkg/info/gufo.prerm
cp "$f" "$f.testbak"
awk '{
  if ($0 ~ /\.was-active/ && $0 ~ />/) print "        : > \"/run/gufo/${unit}.was-active\""
  else print
}' "$f.testbak" > "$f"
EOS
  $PODMAN cp "$TMPD/legacy-prerm.sh" gufo-deb-B:/tmp/legacy-prerm.sh
  run_in gufo-deb-B 'bash /tmp/legacy-prerm.sh'
  PRERM_SRC="$(run_in gufo-deb-B 'cat /var/lib/dpkg/info/gufo.prerm')"
  ck "installed prerm now writes the legacy empty marker" "empty-marker" "$(printf '%s' "$PRERM_SRC" | grep -qF ': > "/run/gufo/' && echo empty-marker || echo unchanged)"
  ck_not_s "and no longer writes the boot id into that file" '"$boot_id" > "/run/gufo/' "$PRERM_SRC"
fi
ck "instance active before the upgrade" "active" "$(start_instance gufo-deb-B legacy)"
PID1="$(pid_of gufo-deb-B legacy)"
SHA1="$(run_in gufo-deb-B 'sha256sum /usr/bin/gufo | cut -c1-16')"
UP3="$(run_in gufo-deb-B 'dpkg -i /root/main.deb 2>&1 | grep -E "^ *gufo:"')"
printf '%s\n' "$UP3" | sed 's/^/    /'
sleep 2
PID2="$(pid_of gufo-deb-B legacy)"
ck_not "no refusal about an earlier boot" "earlier boot" "$UP3"
ck_has "empty marker honoured, boot provenance from the state record" "older prerm" "$UP3"
ck "instance NOT stranded by the upgrade" "active" "$(run_in gufo-deb-B 'systemctl is-active gufo@legacy')"
if [ -n "$PID2" ] && [ "$PID2" != "0" ] && [ "$PID1" != "$PID2" ]; then
  printf '  ok   %-46s pid %s -> %s\n' "restarted onto the new binary" "$PID1" "$PID2"; pass=$((pass+1))
else
  printf '  FAIL %-46s pid %s -> %s\n' "restarted onto the new binary" "$PID1" "$PID2"; fail=$((fail+1))
fi
SHA2="$(run_in gufo-deb-B 'sha256sum /usr/bin/gufo | cut -c1-16')"
# Only a real version change replaces bytes. The self-contained path reinstalls the SAME package
# with a patched prerm, so identical bytes there are correct, and claiming otherwise once made the
# suite red for a reason that had nothing to do with the packaging.
if [ -n "$LEGACY_DEB" ]; then
  ck "the binary bytes changed (older package to newer)" "yes" "$([ "$SHA2" != "$SHA1" ] && echo yes || echo no)"
else
  printf '  skip %-46s same version reinstalled: identical bytes are correct\n' "binary bytes replaced"
fi

echo
[ "$KEEP" = yes ] || $PODMAN rm -f gufo-deb-A gufo-deb-B >/dev/null 2>&1
echo "  $pass passed, $fail failed"
echo "  inference NOT covered: ExecStart stubbed in these containers, no weights loaded."
[ "$fail" -eq 0 ]
