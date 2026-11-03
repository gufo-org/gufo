# Nix serving metadata

Publish Nix serving metadata (`schemas/extension.schema.json`):
- Optional provider `serving.voices` support for named voice references.
- Optional `serving.draft_options`, mapping an advertised speculative mode to
  its required provider-owned path option.
- CMake validation of draft-option ownership, type and serving command.
- Package `extensionMetadata` for Nix serving argument generation and validation,
  following discovery or lock selection and external extension sources.

CMake recognizes `baseline.md` as the non-counting baseline record; descriptive
record names and headings do not assign API revisions. Guidance symlinks are
excluded from the Python record inventory.

Existing manifests remain valid. No runtime C++ declarations or existing
canonical draft option names change.
