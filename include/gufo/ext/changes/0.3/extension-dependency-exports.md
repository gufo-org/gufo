# Extension dependency exports

Public CMake contract: `gufo_extension_export(TARGET <target> NAME <name>)`
aliases one of the calling extension's manifest targets as `gufo::<name>`.
`<name>` is the extension's name or starts with it and `_`, and may not fall in
a longer extension name's namespace. Only the owning extension's
`extension.cmake`, identified by the directory extension discovery runs it in,
can export; test targets and duplicate names are rejected. A target disabled by
its `requires` exports nothing.

New rejection: after all extensions are built, configure fails when any target
an extension directory defines links another extension's library, through its
export or its target name (generator-expression wrappers and `INTERFACE` links
included), and the linking manifest doesn't list that extension in `depends`.
It also fails on a `gufo::` name in an extension's namespace that the extension
doesn't export, or an alias of an extension target that the export function
didn't create. For external extensions, every transitive link must stay within
the extension's dependency closure.

Selected extensions, in-tree and from `GUFO_EXTENSION_DIRS`, build
dependency-first in every selection mode, so a dependency's exports exist
before its dependents' `extension.cmake` runs. A cyclic `depends` fails with
the whole cycle named.

For external extensions, the include and link allowlist treats an export of an
in-tree extension in the dependency closure like `gufo_ext_core`: it accepts
the export and its usage requirements whole. An external extension's export is
checked like any other external target. Exports are the exporting extension's
code, outside the versioned API boundary.

Compatibility: additive under the release policy's revision rule. The new
rejection affects only links between extensions that don't declare each other
in `depends`; no extension in Gufo's tree or known outside it makes one, and an
external extension linking an in-tree extension's target already failed the
allowlist, which rejects private include paths and libraries. Modules still
reject `depends`.
