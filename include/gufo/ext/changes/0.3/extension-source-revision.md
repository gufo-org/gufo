# External extension source revision

Registry (`include/gufo/ext/registry.hpp`): an external extension records where
its source came from.
- `ExtensionInfo::revision`: the commit of the extension's Git work tree, else
  the revision the build declared for its directory, else `"unknown"`. Empty
  for an in-tree extension and for a module.
- `ExtensionInfo::dirty`: whether that work tree had uncommitted changes
  (modified, staged or untracked files) when the build was configured.

Public CMake contract: `GUFO_EXTENSION_DIR_REVISIONS` declares one revision per
`GUFO_EXTENSION_DIRS` entry, in the same order, for directories without Git
metadata such as a Nix store path; `unknown` declares none and a trailing
`-dirty` sets `dirty`. Git metadata, when present, takes precedence. Configure
depends on the work tree's `HEAD`, branch and index, so a later commit is
recorded by the next build.

`gufo --version` and `gufo diagnose --section extensions` report both; the JSON
entries of external extensions gain `revision` and `dirty`.

Compatibility: additive. Both fields have defaults, and the generated registry
fills them, so existing extensions need no change.
