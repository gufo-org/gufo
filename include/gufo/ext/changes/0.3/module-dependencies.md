# Module dependencies

Public CMake contract: a loadable module (`GUFO_EXTENSION_MODULES`, or the SDK
that `gufo extension install` drives) may list in `depends` the in-tree
extensions built into its gufo executable, on Linux. In module mode a
dependency's `gufo::` export links as a stand-in: an `INTERFACE` target with
the export's recorded usage requirements (include paths outside Gufo's trees,
definitions, compile options and features, link options and imported links)
and no code. The same stand-in serves source-tree and SDK modules.
`gufo::qwen_family` links as `$<COMPILE_ONLY:...>`. The executable keeps every
object behind both and exports their `gufo::` symbols.

The SDK records the host's in-tree extensions in `host-extensions.cmake`
(`gufo_host_extension`, `gufo_host_export`) and their names in `sdk.json`
under `host_extensions`. It skips an extension being built to replace a
built-in. On Linux only, it also defines `gufo::qwen_family`, which has no
usage requirements; the export stand-ins carry theirs.

New rejections, all in module mode:

- a dependency that is not an in-tree extension built into the host;
- linking a dependency's target by name instead of its export;
- an enabled host export whose usage requirements cannot be recorded: a
  generator expression other than `$<BUILD_INTERFACE:...>`,
  `$<INSTALL_INTERFACE:...>` or link-only edges, `INTERFACE_SOURCES` and
  similar properties, or a non-imported link other than Gufo's public link
  contracts;
- an export closure reaching a static or object library that is neither a
  `gufo_*` nor a manifest target;
- on Windows, any `depends` or `gufo::qwen_family` link. The Windows SDK lists
  no `host_extensions` and gives the reason in `host_extensions_unavailable`.

`gufo extension install` and `update` refuse a `depends` entry the SDK
doesn't list before building, and gufo skips, before loading, a module whose
pinned manifest depends on an extension it doesn't build in.

A module with such dependencies is built for its own identity: the SHA-256 of
the host's identity and one `host_extension=<name>:<version>:<digest>` line
per dependency that exports anything. The digest covers the dependency's
version, its exports' records and its own dependencies' identities. A
standalone module keeps the host's identity unchanged.

Compatibility: additive under the release policy's revision rule. Modules
with `depends`, and modules linking `gufo::qwen_family`, failed to configure
before. Standalone modules build and load unchanged. Exports stay outside the
versioned API boundary.
