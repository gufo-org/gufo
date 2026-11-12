# Recovered manifest storage accounting

`cache::DiskStore::OwnedManifestBytes()` reports the initialized, retained
manifest vector elements included in the RAM ledger: input bytes, tokens,
components and chunk references. It excludes spare vector capacity, entry-table
storage, estimated metadata overhead, temporary validation, payloads and pending
reservations. Callers use it at quiescence with the post-recovery host-memory
observation to compare remaining allocation obligations with remaining headroom.
It is not a process RSS measurement or a general credit for ledger charges.

The addition is source-compatible after rebuilding. Existing persisted cache
representations and recovery validation are unchanged.
