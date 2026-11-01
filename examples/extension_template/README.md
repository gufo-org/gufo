# Gufo Extension Template

This repository template provides the blueprint for authoring an out-of-tree model extension for Gufo.

## Overview

Gufo uses build-time source extensions. An out-of-tree extension provides:
1. **Metadata Manifest (`extension.json`)**: Declares model identity, version, capabilities, supported platforms, dependency constraints, artifact keys, options, and entry symbols.
2. **Build Manifest (`extension.cmake`)**: Configures CMake targets via `gufo_extension_target(...)` and links against the public extension runtime (`gufo_ext_runtime`).
3. **Source Implementation (`src/...`)**: Implements extension registration and capability providers using only public headers under `<gufo/ext/...>`.

## Directory Structure

```text
extension_template/
├── README.md             # This document
├── extension.json        # Extension metadata manifest
├── extension.cmake       # Gufo build manifest
└── src/
    └── template_model.cpp # Extension registration entrypoint
```

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

In `flake.nix` or an expression:

```nix
gufo.override {
  extensions = [
    /path/to/extension_template
    # or a remote flake / fetchFromGitHub derivation:
    # (pkgs.fetchFromGitHub { owner = "..."; repo = "..."; rev = "..."; hash = "..."; })
  ];
}
```

## Windows Packaging (ZIP Recipe)

Third-party Windows extension distributions package:
1. `extension.json` and `extension.cmake` at the root of the archive.
2. Source files and headers under `src/` and `include/`.
3. `THIRD_PARTY_NOTICES.md` or `LICENSE`.
4. Precompiled dependencies (if using vcpkg or prebuilt binaries).

To package a ZIP from PowerShell:
```powershell
Compress-Archive -Path extension.json, extension.cmake, src, README.md -DestinationPath template_extension-windows.zip
```
