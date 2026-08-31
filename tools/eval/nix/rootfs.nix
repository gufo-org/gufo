# Shared task-rootfs builder for gufo-agent-eval.
#
# Produces a directory tree that the runner copies per attempt and binds as
# `/` inside a bubblewrap jail.
#
# The tree is copied rather than overlay-mounted. Nix store contents are
# always mode 0555 and owned by host root, which is unmapped inside the
# jail's user namespace, so `CAP_DAC_OVERRIDE` does not apply to them: an
# overlayfs whose lower layer is the store can never accept a file create in
# the task workdir. The tree is cheap to copy because the userland is a
# single symlink into the store rather than a materialized closure -- this
# task's rootfs is 28K across 15 entries.
#
# Copying also drops the unprivileged-overlayfs kernel requirement, which
# would otherwise have limited the harness to kernels 5.11 and newer.
#
# The tasks were ported from Dockerfiles and their instructions, fixtures, and
# verifiers assume an FHS layout (`/app/...`, `/usr/bin/...`, `/bin/sh`). Nix
# is not FHS, so this builder materializes an FHS-shaped tree from a Nix
# closure: a `buildEnv` provides `bin`/`lib`/`share`, and the usual FHS
# entry points are symlinked onto it.
#
# The tree contains symlinks into /nix/store, so the runner must bind
# /nix/store read-only into the jail for it to resolve.
{ pkgs }:

{
  # Task name, used for the derivation name only.
  name,
  # Packages visible to the agent inside the jail.
  packages ? [ ],
  # Working directory the agent starts in. Created if absent.
  workdir ? "/app",
  # Attrset of absolute in-rootfs path -> source path. Staged verbatim.
  files ? { },
  # Extra shell run after staging, for tasks needing setup that Dockerfiles
  # expressed as a RUN step (the upstream `setup.sh` pattern).
  postBuild ? "",
}:

let
  inherit (pkgs) lib;

  userland = pkgs.buildEnv {
    name = "gufo-agent-eval-userland-${name}";
    paths = packages;
    pathsToLink = [
      "/bin"
      "/lib"
      "/libexec"
      "/share"
      "/etc"
    ];
    # Tasks reference tools by absolute path; broken links must fail the
    # build rather than surface as a confusing "not found" inside the jail.
    ignoreCollisions = false;
  };

  stageFiles = lib.concatStringsSep "\n" (
    lib.mapAttrsToList (dest: src: ''
      mkdir -p "$out$(dirname ${lib.escapeShellArg dest})"
      cp -r ${src} "$out${dest}"
      chmod -R u+w "$out$(dirname ${lib.escapeShellArg dest})"
    '') files
  );
in

pkgs.runCommand "gufo-agent-eval-rootfs-${name}"
  {
    passthru = { inherit name workdir userland; };
  }
  ''
    mkdir -p "$out"

    # FHS skeleton. /proc, /dev, /tmp, and the home directory are provided by
    # the jail at run time, not staged here.
    mkdir -p "$out"/{app,etc,root,run,usr,var/tmp}

    # Userland. /bin and /usr/bin both resolve, since ported tasks and their
    # verifiers use each interchangeably.
    ln -s ${userland}/bin "$out/bin"
    ln -s ${userland}/lib "$out/lib"
    ln -s ${userland}/bin "$out/usr/bin"
    ln -s ${userland}/lib "$out/usr/lib"
    if [ -e ${userland}/share ]; then ln -s ${userland}/share "$out/usr/share"; fi

    # Minimal /etc. The jail supplies passwd/group, so only static files land
    # here.
    cat > "$out/etc/os-release" <<'EOF'
    NAME="gufo-agent-eval"
    ID=gufo-agent-eval
    PRETTY_NAME="gufo-agent-eval task rootfs"
    EOF

    mkdir -p "$out${workdir}"

    ${stageFiles}

    ${postBuild}
  ''
