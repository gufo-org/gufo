# Disk manifests stored and charged once

Additive: `cache::DurableEntry`, `cache::DiskDescription` and
`cache::DiskSnapshot` (`src/cache/disk.hpp`) share one manifest instead of
each holding its own copy. Only their private members change, and every
`Manifest()` still returns `const DiskManifest&`.

- A description or snapshot shares the entry's manifest and its metadata
  charge. The charge is released with the last holder, after the manifest.
- `DiskStore::Describe` charges only the description itself, and
  `DiskStore::Open` charges no metadata, so neither needs room for another
  copy of the manifest.
- Each index slot also charges the shared manifest's header and control
  block, which the entry no longer holds inline.
