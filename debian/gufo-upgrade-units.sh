#!/bin/sh
# Decide and apply the systemd action for every gufo unit after dpkg configure.
#
#   gufo-upgrade-units.sh [--dry-run] <version-configured-from>
#
#     <version-configured-from> is dpkg's postinst argument $2: empty on a first
#     install, the previous version on an upgrade or reinstall.
#
# Why this exists as a shipped script instead of inline shell in postinst: the
# decision below is the part that has to be right for every upgrade path, and it
# is now testable without touching users, groups, mounts or the real service
# manager. See the harness at the bottom of this file's usage note.
#
# Why the decision is here and not in prerm: dpkg runs the prerm of the package
# being upgraded FROM, so logic placed there cannot cover the upgrade that first
# delivers it, and `start' on an already-active unit is a no-op - the unit would
# keep executing the inode this package just replaced, while `dpkg -l' and
# `gufo --version' both report the new one.
#
# Rules, in priority order. Exactly one line of evidence is printed per unit:
#
#   restart  the unit is active. Its binary was just replaced under it, whether or
#            not prerm stopped it, so it must be restarted. Covers: the normal
#            upgrade; a same-version reinstall; a unit started without enabling.
#   restart  a /run marker names it (prerm stopped it, so it is inactive now, and
#            restart on an inactive unit starts it).
#   start    the unit is inactive but enabled - a wants symlink exists. The admin
#            asked for it at boot; /run may have been wiped between prerm and
#            configure. Enabling is never done here - only starting what is
#            already enabled.
#   nothing  inactive and not enabled. The admin's choice, never overridden.
#
# Environment overrides exist for testing only, and are documented as such:
#   GUFO_RUN_DIR     directory holding *.was-active markers   (default /run/gufo)
#   GUFO_WANTS_DIRS  space-separated wants dirs to enumerate  (default the real two)
#   GUFO_SYSTEMCTL   the systemctl binary                     (default systemctl)
#   GUFO_STATE_DIR   package state dir surviving /run wipes     (default /var/lib/gufo)
#   GUFO_BOOT_ID     current boot id                          (default /proc/.../boot_id)
#
# Why `enabled' alone must not start a unit: enablement is a statement about BOOT, not
# about right now. An operator who runs `systemctl stop gufo@x' on a machine that stays
# up leaves the unit enabled and inactive, and that is their live intent - here it means
# "keep the ~90 GiB device free for other work". An upgrade that starts the unit takes
# the device and puts the host back in the load balancer's path. The two situations are
# separable only by time: if the machine rebooted between prerm and configure, /run was
# wiped, the markers went with it, and starting what is enabled is the recovery the admin
# asked for at boot. prerm therefore records the boot id under $GUFO_STATE_DIR, which
# survives a reboot; `start' fires only when that record differs from the current boot id.
# No record proves nothing, so the package leaves the unit down.

set -u

dry=no
if [ "${1:-}" = "--dry-run" ]; then dry=yes; shift; fi
from="${1:-}"

run_dir="${GUFO_RUN_DIR:-/run/gufo}"
wants_dirs="${GUFO_WANTS_DIRS:-/etc/systemd/system/multi-user.target.wants /etc/systemd/system/*.wants /run/systemd/system/*.wants}"
systemctl_bin="${GUFO_SYSTEMCTL:-systemctl}"
state_dir="${GUFO_STATE_DIR:-/var/lib/gufo}"
boot_id_file="$state_dir/upgrade-boot.id"
current_boot_id="${GUFO_BOOT_ID:-$(cat /proc/sys/kernel/random/boot_id 2>/dev/null || echo unknown)}"

# True only when prerm recorded a boot id AND the machine booted since. An absent file is
# deliberately NOT read as "rebooted": it proves only that prerm did not run.
rebooted_since_prerm() {
  [ -f "$boot_id_file" ] || return 1
  [ "$(cat "$boot_id_file" 2>/dev/null || echo unknown)" != "$current_boot_id" ]
}

# Only names this package may act on: the classic unit and instantiated ones;
# the bare template gufo@.service is rejected by systemctl itself.
actionable() {
  for name in $@; do
    case "$name" in
      gufo.service) printf '%s\n' "$name" ;;
      gufo@.service) ;;
      gufo@*.service) printf '%s\n' "$name" ;;
    esac
  done
}

is_active() { "$systemctl_bin" is-active --quiet "$1" 2>/dev/null; }
is_enabled() { "$systemctl_bin" is-enabled --quiet "$1" 2>/dev/null; }

units=""
# Markers: the units prerm found active and stopped for us. Expand in the shell
# and existence-check each candidate; never `ls' these paths (a glob that matches
# nothing is passed through literally and would become a bogus unit name).
for marker in "$run_dir"/*.was-active; do
  [ -e "$marker" ] || continue
  units="$units $(basename "$marker" .was-active)"
done
# Enabled units, whatever the admin named them, from every wants directory.
for dir in $wants_dirs; do
  [ -d "$dir" ] || continue
  for link in "$dir"/gufo.service "$dir"/gufo@*.service; do
    [ -e "$link" ] || continue
    units="$units $(basename "$link")"
  done
done
# Loaded units: covers started-without-enabled, and the active-unit case that the
# marker scheme structurally cannot see. Never let a query abort the enumeration.
units="$units $($systemctl_bin list-units --all --type=service --no-legend --plain 'gufo*' 2>/dev/null | awk '{print $1}' || true)"

failures=0
acted=0
for unit in $(actionable $units | LC_ALL=C sort -u); do
  marker="$run_dir/$unit.was-active"
  if is_active "$unit"; then
    action=restart
    if [ -e "$marker" ]; then why="active, stopped by prerm and restarted onto the new binary"; else why="ACTIVE while its binary was replaced - start would be a no-op"; fi
  elif [ -e "$marker" ]; then
    # A marker belongs to the transaction that wrote it, and to that boot only. A
    # transaction that died between prerm and configure leaves one behind; if the
    # operator then stops the unit, honouring the stale marker would start it again.
    if [ "$(cat "$marker" 2>/dev/null)" = "$current_boot_id" ]; then
      action=restart; why="prerm marked it as running on this boot"
    else
      action=none; why="marker left by an earlier boot's aborted transaction - ignored"
    fi
  elif is_enabled "$unit" && rebooted_since_prerm; then
    action=start; why="inactive, enabled, rebooted since prerm (/run wiped)"
  elif is_enabled "$unit"; then
    action=none; why="inactive, enabled, but stopped by the operator on this boot - left down"
  else
    printf 'gufo: none   %-24s (inactive, not enabled - left as the admin left it)\n' "$unit"
    continue
  fi

  # A decision to touch nothing is not an action: it must not count, and it must not
  # reach the invoke below.
  if [ "$action" = none ]; then
    printf 'gufo: leave   %-24s %s\n' "$unit" "$why"
    continue
  fi

  acted=$((acted + 1))
  if [ "$dry" = yes ]; then
    printf 'gufo: dry    %-24s %s\n' "$unit" "$why"
    continue
  fi
  if command -v deb-systemd-invoke > /dev/null 2>&1; then
    if ! deb-systemd-invoke "$action" "$unit" > /dev/null 2>&1; then
      printf 'gufo: WARNING could not %s %s; run: sudo systemctl %s %s\n' "$action" "$unit" "$action" "$unit" >&2
      failures=$((failures + 1))
      continue
    fi
  else
    if ! "$systemctl_bin" "$action" "$unit" > /dev/null 2>&1; then
      printf 'gufo: WARNING could not %s %s; run: sudo systemctl %s %s\n' "$action" "$unit" "$action" "$unit" >&2
      failures=$((failures + 1))
      continue
    fi
  fi
  printf 'gufo: %-6s %-24s %s\n' "$action" "$unit" "$why"
  rm -f "$marker"
done

# Loud, not silent: an upgrade that touched no unit is how a host ends up serving
# the previous binary under a new package version.
if [ "$acted" -eq 0 ] && [ -n "$from" ]; then
  printf 'gufo: note: no running or enabled gufo unit found during upgrade from %s; nothing was restarted\n' "$from" >&2
fi

# The boot-id proof is single-use: once configure has decided, the record is consumed, so
# a later upgrade cannot start something on the strength of a reboot that predates it.
# Must run before the exits below - after them it is unreachable.
if [ "$dry" != yes ] && [ -n "${from:-}" ]; then rm -f "$boot_id_file" 2>/dev/null || true; fi

[ "$failures" -eq 0 ] || exit 1
exit 0