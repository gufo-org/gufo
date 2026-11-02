# External extension source trees

Support external extension source trees and build sandboxing (Card 23):
- Out-of-tree discovery: `GUFO_EXTENSION_DIRS` CMake discovery and Nix `extensions` argument.
- Configure-time include allowlist: external extension targets are isolated to public headers under `<gufo/ext/...>` and prohibited from linking internal private libraries or including private repository paths.
- Runtime usage requirements expose only public headers and their audited `SURFACE` closure, without a repository-root include. External validation also checks inherited language/configuration flags, source settings, dependency interfaces, aliases and supported generator expressions; unvalidated expressions are rejected.
- Template blueprint: `examples/extension_template/` provides reference manifests and source layout.
