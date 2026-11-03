# Source extension distribution and support

A **source extension** is native code compiled into a Gufo executable. Adding
one requires rebuilding Gufo. Official binaries and containers include the
reviewed extensions selected by `extensions.lock`. The ordinary development
package uses discovery; a directory being discovered does not make it shipped.

For the architecture, a creation walkthrough, the LFM3 example and maintenance,
start with the [extension author guide](../extensions.md).

## Core, community, promotion and abandoned maintainers

| State | Location and ownership | Required evidence |
| --- | --- | --- |
| Core | In-tree, with a named maintainer accountable for updates and review | Repository numerical, functional, benchmark and release gates; interface changes fix all affected in-tree users |
| Community | Out-of-tree, owned and released by its authors | Capability conformance, independent quality evidence, documented limitations and reproducible build/provenance information; no Gufo release gate on its speed |
| Promotion | Explicit review before a community extension enters the tree or official lock | Fit with Gufo's model-selection goals, named maintainer, licensing and dependency review, public-header isolation, supported-platform evidence and the core gates |
| Abandoned | Maintainer replacement or an explicit retirement decision | Keep the last passing revision and open failures visible; removal from official selection is a reviewed lock/release decision |

Niche and experimental models stay out-of-tree until promotion. Passing a suite,
shipping a third-party image or using a public API does not automatically promote
an extension. A core extension that loses its maintainer remains subject to core
checks while a replacement is sought. If maintenance cannot be restored, review
removal from `extensions.lock`, migration guidance and release impact together.
An external continuation requires an identified new owner and an explicit source
revision. Do not silently disable a failing gate or call an abandoned build
supported. Historical evidence remains available after retirement.

The distinction is a support policy, not a new registry field. Diagnostic
`available: true` means a provider was built with its requirements satisfied;
it does not establish review, quality, maintenance status or promotion.

## Trust and provenance

Extension CMake and C++/HIP can execute arbitrary native code during configuration,
build and inference, with the permissions and resources of those processes.
The include allowlist enforces API boundaries; it is not a security sandbox.
A Nix build sandbox does not sandbox a subsequently running model server.
Review and pin source revisions before building or running third-party code.
Official selection includes only reviewed core extensions.

Retain the Gufo and extension source revisions, toolchain, build options,
`extensions.lock`, model artifact identities, install inventory and capability
results. For a published image or ZIP, retain its digest/checksum and the actual
runtime output. Source repository links and successful build jobs alone do not
prove an artifact's selected providers or runtime readiness.

```sh
/path/to/installed/gufo --version
/path/to/installed/gufo diagnose --section extensions --json
```

The output records API identity, selection mode, extension name/version/source,
provider availability and excluded-extension reasons. Inspect the installed
binary, not a different executable from `PATH`.

## Linux source and CMake recipe

Use the dependencies in [Build from source](../../README.md#build-from-source),
or enter the pinned `nix develop` shell. Check out the Gufo and extension commits
recorded for the build, then run from the Gufo source root:

```sh
cmake --preset release \
  -DGUFO_EXTENSION_DIRS=/absolute/path/to/extension
cmake --build --preset release --target gufo --parallel 4
cmake --install build/release --prefix /absolute/path/to/install
/absolute/path/to/install/bin/gufo diagnose --section extensions --json
```

Discovery includes the external directory. For a focused development build,
`-DGUFO_EXTENSIONS=template_model` selects exactly the template; include any
manifest `depends` in an explicit selection. The weight-free template can be
built with `ENGINE_ENABLE_HIP=OFF` in a separate build directory to check source
API registration and installation. That does not qualify a real model.

Official builds use `-DGUFO_EXTENSIONS_FROM_LOCK=ON` with the default selector.
An external extension absent from the lock remains excluded. A third-party
redistributor wanting a curated set must maintain its own reviewed lock/source
revision and identify the result as its own distribution.

Extension `extension.cmake` owns installation of additional assets and notices:

```cmake
install(FILES "${GUFO_EXTENSION_BUILD_ROOT}/LICENSE"
  DESTINATION share/licenses/gufo/extensions/my_extension)
install(FILES "${GUFO_EXTENSION_BUILD_ROOT}/data/config.json"
  DESTINATION share/gufo/models/my_extension)
```

The manifest's `licenses`, `install_data` and `runtime_libraries` inventory must
agree with what the recipe ships. Declaring those fields does not copy files or
collect arbitrary runtime dependencies automatically. Avoid basename collisions
by namespacing extension assets. Test the installed layout after removing access
to the build tree, especially for lazily loaded libraries and model defaults.
The H3 manifest and DeepSeek license are installed by their owning extensions,
only when selected, at their existing package paths.

## Nix recipe

The [template](../../examples/extension_template/README.md) includes a flake and
lock file pinning Gufo and its transitive toolchain inputs. Copy the template
into an independent source repository, retain its license, and build:

```sh
nix build
./result/bin/gufo diagnose --section extensions --json
```

Its package uses `gufo.packages.x86_64-linux.default.override` with
`extensions = [ self.outPath ]`. To use an independently packaged source,
replace that entry with a pinned source derivation. Extra Nix dependencies must
be supplied by the author's package, for example through `overrideAttrs` and
`buildInputs`; metadata package names are an inventory, not an automatic package
installer. `cmake` installation rules also run in the Nix package.

Pin source and lock files for reproducibility. Use the explicit `release` package
for official lock-mode builds; a version override alone does not select the lock.
See [Nix serving](serving.md) for package-aware argument validation.

## Native Windows and ZIP recipe

Follow [BUILD-WINDOWS.md](../../BUILD-WINDOWS.md) for the pinned TheRock SDK,
MSVC and vcpkg setup. Windows uses native CMake and vcpkg, not the Nix override.
Run the dependency preflight before `project()` invokes vcpkg:

```powershell
python tools/windows/check_extension_packages.py --extension-dir C:/src/my-extension
tools/windows/vs-env.cmd cmake --preset windows-release -DGUFO_EXTENSION_DIRS=C:/src/my-extension
tools/windows/vs-env.cmd cmake --build --preset windows-release --target gufo
cmake --install build/windows-release --prefix C:/dist/gufo
C:/dist/gufo/bin/gufo.exe diagnose --section extensions --json
```

The check matches each selected extension's `packages.windows` ports/features
against `vcpkg.json`. For an additional port, supply a combined custom manifest
with `--manifest C:/manifests/vcpkg.json` and configure with
`-DVCPKG_MANIFEST_DIR=C:/manifests`; supply overlays as needed. An overlay supplies
port definitions and does not itself add a missing dependency to the manifest.
Keep the Gufo dependencies and pinned baseline in the custom manifest.

The reusable workflow accepts `extension-repo` and `extension-ref`; pin a full
extension commit in a third-party caller. A release-tag build stays lock-selected
and does not include an unlocked community extension. The workflow packages
`cmake --install` output, including extension-owned data and licenses, alongside
HIP/vcpkg/CRT DLLs. A source ZIP of the extension is a separate artifact and does
not replace the executable's runtime bundle.

A redistributor must stage the full runtime closure, required HIP kernel data,
licenses and lazy-loaded library manifests before creating a ZIP. With the
pinned `pefile==2024.8.26` installed, run:

```powershell
python tools/windows/check_bundle.py C:/dist/gufo-runtime
```

This checks normal/delay imports and the video runtime manifest. It does not
execute the model or discover arbitrary libraries loaded by name; explicitly
inventory and check those too. Extract the final ZIP into a fresh directory and
repeat the import, install and capability checks there. Retain a SHA-256 checksum
for that exact archive. Native Windows execution evidence is required for a
Windows support claim; a Linux test of the checker is insufficient.

## Install audit and conformance

The install auditor hashes the binary and explicitly required files, records
actual version/diagnostic output, checks expected selection and extension
identity, and fails on missing assets. Use a new report path for each attempt:

```sh
python3 tools/extensions/check_install.py \
  --binary /absolute/path/to/install/bin/gufo \
  --mode discovery --extension my_extension \
  --license /absolute/path/to/install/share/licenses/gufo/LICENSE \
  --license /absolute/path/to/install/share/licenses/gufo/extensions/my_extension/LICENSE \
  --data /absolute/path/to/install/share/gufo/models/my_extension/config.json \
  --report artifacts/my-extension-install.json
```

List every required notice/data file. For a line-oriented runtime manifest,
`--runtime-manifest PATH` also requires each named file beside the executable.
Pair this inventory check with the platform's dynamic dependency inspection and
actual capability execution. Follow [capability conformance](CONFORMANCE.md);
a packaging pass alone is not numerical, API or performance qualification.

## Toolboxes external deliverable

Toolboxes owns the published builder/runtime images and rolling edge integration.
The required extension-author builder must pin the toolchain and Gufo sources,
provide their build dependencies, and demonstrate an external extension build.
Its matching runtime must contain the built executable and its dependency/data/
license closure. Retain published image digests, source/engine revisions, build
logs, and actual lock-mode diagnostic/readiness/conformance output.

The existing `gufo-dev` image is a development image. Its existence does not prove
that builder contract. The inspected Toolboxes revision derives edge from the
ordinary default package and changes only version identity for releases; it does
not explicitly select `extensionsFromLock = true`. This external work and actual
runtime checks remain pending. See [Card 26 evidence](evidence/card26/README.md)
for inspected revisions, registry digests, completed checks and open gates.
