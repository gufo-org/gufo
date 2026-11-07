# Extensions built as loadable modules

Experimental public CMake contracts (`cmake/GufoExtension.cmake`,
`cmake/GufoExtensionDiscovery.cmake`) for building a selected extension as a
module that a `gufo` executable loads at runtime, instead of linking it in:
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
  load, from the audited `SURFACE` headers, API version, toolchain, ROCm and
  global compile flags.
- `gufo_target_extensions(<target> LOAD_MODULES ...)`: the generated registry
  also registers loaded modules after built-in extensions.
- External extension validation accepts `$<COMPILE_ONLY:...>` link items.

Compatibility: additive. Extensions that are not listed in
`GUFO_EXTENSION_MODULES` build and register as before.
