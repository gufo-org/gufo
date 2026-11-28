# Disk checkpoint tokens held apart from shared manifests

Additive (`src/cache/disk.hpp`): `cache::DiskTokenStorage` selects where a
`cache::DiskStore` keeps decoded checkpoint tokens. The new constructor
arguments default to `kManifest`, which keeps every existing behavior.

- A `kSeparate` store keeps `Manifest().tokens` of its entries, descriptions
  and snapshots empty, and holds the tokens beside the manifest, charged to
  the RAM ledger separately, until `DiskStore::ReleaseTokens`. In-flight
  publications are never released.
- `cache::DiskRecoveredTokens`, passed to a `kSeparate` store's constructor,
  receives each recovered entry's tokens as its manifest is decoded; the store
  then holds none of them.
- `DurableEntry`, `DiskDescription` and `DiskSnapshot::TokenCount()` report
  the checkpoint's token count in both modes.
- `DiskStore::Tokens` and `DiskDescription::Tokens` return the held tokens,
  or an empty span once released or retired; spans are valid until then.
  `DiskStore::TokenStorage` reports the mode.
- `OwnedManifestBytes` counts separately held tokens while they are held.

Persisted manifests and recovery validation are unchanged. Source compatible
after rebuilding.
