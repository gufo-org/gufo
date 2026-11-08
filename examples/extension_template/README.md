# Gufo Extension Template

This repository template provides the blueprint for authoring an out-of-tree model extension for Gufo.
Start with the [extension author guide](../../docs/writing-extensions.md), including the
LFM3 example and ongoing maintenance workflow.

## Overview

Gufo uses build-time source extensions. An out-of-tree extension provides:
1. **Metadata Manifest (`extension.json`)**: Declares model identity, version, capabilities, supported platforms, dependency constraints, artifact keys, options, and entry symbols.
2. **Build Manifest (`extension.cmake`)**: Configures CMake targets via `gufo_extension_target(...)` and links against the public extension runtime (`gufo_ext_runtime`).
3. **Source Implementation (`src/...`)**: Implements extension registration and capability providers using only public headers under `<gufo/ext/...>`.

## Directory Structure

```text
extension_template/
├── README.md             # This document
├── LICENSE               # Preserve the source license
├── flake.nix             # Pinned Gufo package with this external source
├── flake.lock            # Pinned transitive inputs
├── extension.json        # Extension metadata manifest
├── extension.cmake       # Gufo build manifest
└── src/
    └── template_model.cpp # Extension registration entrypoint
```

The template registers metadata only. Implement an actual capability runner and
independent model tests before claiming inference support or conformance.
Community and niche models stay out-of-tree until promotion; see
[distribution/support policy](../../docs/internal/extension-distribution.md) and
[capability conformance](../../docs/internal/extension-conformance.md).

## Include Boundary Policy

Out-of-tree extensions are strictly checked at CMake configure time:
- Extensions may only include their own directories and Gufo's public headers under `<gufo/ext/...>`.
- Repository-root and private `src/` include directories are rejected. The runtime publishes only the forwarded headers listed in `include/gufo/ext/SURFACE` through a separate build include tree.
- Validation covers target, source, dependency and inherited language/configuration settings. `BUILD_INTERFACE`, `LINK_ONLY` and simple configuration/language/platform conditions are inspected recursively; other generator expressions are rejected when they cannot be validated.
- Build manifests are trusted CMake code. These checks enforce the source API contract; they are not a security sandbox for untrusted manifests.
- External extensions cannot link private internal libraries (`gufo_core`, `gufo_serving`, etc.); they link against `gufo_ext_runtime`.

## Building with Gufo

### Using CMake

Pass the path to one or more external extension directories via `GUFO_EXTENSION_DIRS`:

```sh
cmake -B build -DGUFO_EXTENSION_DIRS=/path/to/extension_template
cmake --build build --target gufo
```

### Using Nix

Copy this template into an independent repository and run `nix build`. Its
`flake.nix` and `flake.lock` pin Gufo and the toolchain inputs; update that pin
intentionally after conformance checks. In another package expression:

```nix
gufo.override {
  extensions = [
    /path/to/extension_template
    # or a remote flake / fetchFromGitHub derivation:
    # (pkgs.fetchFromGitHub { owner = "..."; repo = "..."; rev = "..."; hash = "..."; })
  ];
}
```

The serving helper reads the overridden package's manifests automatically; see
[Nix serving from extension metadata](../../docs/internal/nix-serving.md)
for provider validation, draft paths, voices and lock selection.

## Windows Packaging (ZIP Recipe)

Third-party Windows extension distributions package:
1. `extension.json` and `extension.cmake` at the root of the archive.
2. Source files and headers under `src/` and `include/`.
3. `THIRD_PARTY_NOTICES.md` or `LICENSE`.
4. Precompiled dependencies (if using vcpkg or prebuilt binaries).

This is a source ZIP. For an executable ZIP, follow the dependency, installation,
runtime closure and fresh-extraction checks in the distribution guide above.
To package the source ZIP from PowerShell:
```powershell
Compress-Archive -Path extension.json, extension.cmake, src, README.md, LICENSE, flake.nix, flake.lock -DestinationPath template_extension-windows.zip
```
