# Extensions built as loadable modules

Experimental contracts for building an extension as a module that an installed
`gufo` executable loads at runtime, instead of linking it in.

Public CMake (`cmake/GufoExtension.cmake`, `cmake/GufoExtensionDiscovery.cmake`,
`cmake/GufoExtensionSdk.cmake`):
- `GUFO_EXTENSION_MODULES`: selected extensions built as
  `extension-modules/gufo-ext-<name>.so`. Their non-test libraries are
  position-independent and take only the usage requirements of
  `gufo_ext_runtime`, `gufo_ext_core`, `gufo_ext_cache` and `gufo_ext_hip`;
  the host services resolve from the loading executable. Test targets keep
  linking every implementation they list. Linux only; standalone extensions
  only (no extension dependencies, not `kind: library`).
- `gufo_extension_modules(EXTENSIONS ...)` creates the module targets.
  A module exports `gufo_ext_module_build_id_v1`, `gufo_ext_module_manifest_v1`
  and `gufo_ext_module_register_v1`.
- `gufo_extension_build_id(<out>)`: the build identity a module must match to
  load, from the audited `SURFACE` headers, API version, toolchain, ROCm (HIP
  builds) and global compile flags. `GUFO_SURFACE_SOURCE_ROOT` locates the
  non-`include/` surface headers.
- `gufo_target_extensions(<target> LOAD_MODULES ...)`: the generated registry
  skips a built-in extension that an installed module replaces, and registers
  installed modules after the built-in extensions.
- `gufo_extension_sdk(<target>)` stages the extension SDK (`<build>/sdk`) and
  installs it into `GUFO_SDK_INSTALL_DIR` (default `lib/gufo/sdk`). The SDK
  is a standalone CMake project configured with `-C <sdk>/profile.cmake`; it
  defines header-only `gufo_ext_*` link contracts, with OpenSSL headers on
  `gufo_ext_runtime` for `sha256.hpp`.
- External extension validation accepts `$<COMPILE_ONLY:...>` link items.

Platform leaves in the surface (`src/core/platform/`):
- `fs.hpp`: `UserDataDirectory()`.
- `os_info.hpp`: `StdinIsTerminal()`, `ExecutablePath()`.
- `process.hpp`: `RunInherited()`, `FindExecutable()`.

Compatibility: additive. Extensions that are not listed in
`GUFO_EXTENSION_MODULES` build and register as before.
