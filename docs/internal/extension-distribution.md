# Extension distribution and support

How extensions reach users, what each support state promises, and how to build,
package and audit an installed Gufo with extensions. The user-facing side is in
[extensions](../extensions.md); writing one is in
[writing an extension](../writing-extensions.md).

## Three ways an extension ships

| Route | Who builds | Selection | Review |
| --- | --- | --- | --- |
| Official build | Gufo releases (Nix `#release`, containers, Debian package, Windows ZIP) | `extensions.lock` | Core extensions, maintained in-tree |
| Source build | The user, from a Gufo checkout | Discovery plus `GUFO_EXTENSION_DIRS`, or Nix `override { extensions = [...]; }` | None implied |
| Module | `gufo extension install` on the user's machine | `installed.json` in the extension store | Registry releases are reviewed; directories and Git URLs are not |

A directory being discovered never makes it shipped: only the lock selects
official builds, and the lock is validated in every configure.

## Support states

| State | Location and ownership | Required evidence |
| --- | --- | --- |
| Core | In-tree and in `extensions.lock`, with a named maintainer | Repository numerical, functional, benchmark and release gates; API changes fix every in-tree user |
| Reviewed community | Out of tree; a release listed in the [extension registry](https://github.com/gufo-org/gufo-extension) | Source review at an exact commit following the registry's `REVIEW.md`; the extension documents its own quality, speed and limits |
| Community | Out of tree, owned by its authors | [Capability conformance](extension-conformance.md), documented limitations and reproducible build information; never a Gufo release gate |
| Abandoned | Needs a new maintainer or an explicit retirement | Keep the last passing revision and open failures visible; removal from the lock is a reviewed release decision |

A registry entry promises reviewed source, not quality: the registry checks that
`digest` matches the tree at the listed commit and that `version`,
`api_version` and `platforms` match the manifest there. Releases are immutable;
a fix is a new release and a withdrawal is its own reviewed pull request.

Promotion to core needs a fit with Gufo's model-selection goals, a named
maintainer, licensing and dependency review, public-header isolation, platform
evidence and the core gates. Passing a suite, a registry listing or a
third-party image doesn't promote an extension. A core extension that loses its
maintainer stays under core checks while a replacement is sought; if none is
found, review lock removal, migration guidance and release impact together.

Diagnostic `available: true` means a provider built with its requirements
satisfied. It says nothing about review, quality or maintenance.

## Trust and provenance

Extension CMake and C++/HIP run arbitrary native code at configure, build and
inference time. The include allowlist enforces the API boundary, not security,
and a Nix build sandbox does not sandbox the server that later runs the model.

- `gufo extension install <name>[@<version>]` fetches
  `https://raw.githubusercontent.com/gufo-org/gufo-extension/main/index.json`,
  picks the newest release compatible with the host platform and API series,
  installs `<source>@<revision>` and aborts on a digest mismatch.
  `GUFO_EXTENSION_REGISTRY` overrides the index URL or file.
- A directory or `<git-url>@<revision>` installs only after a warning and
  confirmation (`--yes` without a terminal). The store records the exact commit,
  or a SHA-256 digest of a copied directory.
- `diagnose --section extensions` reports a module's source as
  `module:<path>`.

For any published artifact, retain the Gufo and extension revisions, toolchain,
build options, `extensions.lock`, model artifact identities, install inventory,
capability results and the image digest or ZIP checksum. Record the installed
binary's own output, not another `gufo` from `PATH`:

```sh
/path/to/installed/gufo --version
/path/to/installed/gufo diagnose --section extensions --json
```

## Modules

A module is the same extension source built as a loadable library against the
SDK installed with a Gufo. A module loads only into a Gufo with the same
**build identity**: a hash of the extension API headers and version, compilers,
ROCm, global flags, GPU architecture, build type and sanitizers. Modules are
stored per identity under `$GUFO_EXTENSIONS_DIR` (default
`~/.local/share/gufo/extensions`):

```text
installed.json                       name, version, source, revision or digest
<name>/source/                       the pinned source
<name>/modules/<build-identity>.so   one module per Gufo build identity
```

After a Gufo upgrade the old identity no longer matches: `gufo extension list`
shows `needs update`, the command skips the module with a warning, and
`gufo extension update` rebuilds from the pinned source. Modules are never
unloaded from a running process. Conflicts with built-in or installed
extensions are refused; `--replace-builtin` replaces a built-in for testing.

Module builds need the SDK and its toolchain: `<build>/sdk` or
`<prefix>/lib/gufo/sdk` for CMake installs, the `sdk` output of the Nix
packages (`#sdk`, `#release-sdk`), or the `gufo-runtime` image, which carries
both. `GUFO_SDK_DIR` points elsewhere. In a build tree, `GUFO_EXTENSION_MODULES`
builds listed extensions as `extension-modules/gufo-ext-<name>.so` instead of
linking them. Modules may use only the public `<gufo/ext/...>` surface, cannot
have `depends`, and cannot be `kind: library`.

On Windows, `gufo.exe` exports its symbols through a generated `.def` file and
the SDK ships `lib/gufo.lib`; modules are DLLs that load only into `gufo.exe`.
Exported data symbols need `__declspec(dllimport)`, so an extension using such
data doesn't link on Windows yet.

## Linux source and CMake recipe

From a Gufo checkout with the [build dependencies](../install.md#build-from-source)
or inside `nix develop`:

```sh
cmake --preset release -DGUFO_EXTENSION_DIRS=/absolute/path/to/extension
cmake --build --preset release --target gufo --parallel 4
cmake --install build/release --prefix /absolute/path/to/install
/absolute/path/to/install/bin/gufo diagnose --section extensions --json
```

`-DGUFO_EXTENSIONS=<names>` selects an exact set; include every `depends`.
Official builds use `-DGUFO_EXTENSIONS_FROM_LOCK=ON`, which excludes anything
not in the lock. A redistributor wanting its own curated set maintains its own
lock and source revision and names the result as its own distribution.

`extension.cmake` installs the extension's own assets and notices:

```cmake
install(FILES "${GUFO_EXTENSION_BUILD_ROOT}/LICENSE"
  DESTINATION share/licenses/gufo/extensions/my_extension)
install(FILES "${GUFO_EXTENSION_BUILD_ROOT}/data/config.json"
  DESTINATION share/gufo/models/my_extension)
```

The manifest's `licenses`, `install_data` and `runtime_libraries` inventory
must match what the recipe installs; declaring them copies nothing. Namespace
asset paths, and test the installed layout without access to the build tree.

## Nix recipe

The [template](../../examples/extension_template/README.md) flake pins Gufo and
builds `gufo.packages.x86_64-linux.default.override { extensions = [ self.outPath ]; }`.
Extra Nix dependencies come from the author's package (`overrideAttrs`,
`buildInputs`); manifest package names are an inventory, not an installer. Use
the `release` package for lock-mode builds. `gufo.lib.<system>.withExtensionModules`
builds a Gufo with a read-only store of pinned modules instead. See
[Nix serving](nix-serving.md) for metadata-aware service rendering.

## Native Windows and ZIP recipe

Follow [BUILD-WINDOWS.md](../../BUILD-WINDOWS.md) for the TheRock SDK, MSVC and
vcpkg setup. Check vcpkg packages before `project()` runs:

```powershell
python tools/windows/check_extension_packages.py --extension-dir C:/src/my-extension
tools/windows/vs-env.cmd cmake --preset windows-release -DGUFO_EXTENSION_DIRS=C:/src/my-extension
tools/windows/vs-env.cmd cmake --build --preset windows-release --target gufo
cmake --install build/windows-release --prefix C:/dist/gufo
C:/dist/gufo/bin/gufo.exe diagnose --section extensions --json
```

The check matches each selected extension's `packages.windows` against
`vcpkg.json`; for extra ports, pass a combined manifest with `--manifest` and
configure with `-DVCPKG_MANIFEST_DIR`. The reusable Windows workflow accepts
`extension-repo` and `extension-ref` (pin a full commit); a release-tag build
stays lock-selected. Before zipping a runtime, run
`python tools/windows/check_bundle.py C:/dist/gufo-runtime` (with
`pefile==2024.8.26`) to check imports and the video runtime manifest, then
extract the final ZIP into a fresh directory and repeat the checks there.
Windows support claims need native Windows execution evidence.

## Install audit

`tools/extensions/check_install.py` hashes the binary and required files,
records version and diagnostic output, and fails on missing assets:

```sh
python3 tools/extensions/check_install.py \
  --binary /absolute/path/to/install/bin/gufo \
  --mode discovery --extension my_extension \
  --license /absolute/path/to/install/share/licenses/gufo/LICENSE \
  --license /absolute/path/to/install/share/licenses/gufo/extensions/my_extension/LICENSE \
  --data /absolute/path/to/install/share/gufo/models/my_extension/config.json \
  --report artifacts/my-extension-install.json
```

`--runtime-manifest PATH` also requires every file a line-oriented runtime
manifest names. Pair it with dynamic-dependency inspection and real capability
runs; packaging alone qualifies nothing.

## Toolboxes

[Toolboxes](https://github.com/gufo-org/toolboxes) publishes the
`gufo-runtime` and `gufo-dev` images. `gufo-runtime` includes the SDK and its
toolchain, so `gufo extension install` works inside it with a volume mounted for
the store. As of November 2026 the images build Gufo's default package
(discovery) with the release version, not the lock-mode `release` package. Today
the in-tree extensions are exactly the locked ones, so the set is the same, but
nothing enforces it; switching Toolboxes to lock mode is open.
