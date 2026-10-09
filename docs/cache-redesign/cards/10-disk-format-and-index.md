# 10 · Disk format, directory lock and startup index

**Milestone:** Disk store · **Depends on:** 04, D4 · **Size:** M ·
**Affects:** nothing at runtime until card 19 · **Status:** agreed

## Goal

Define the versioned on-disk layout, give one process exclusive ownership of
the directory, and build the startup index from metadata only.

## Scope

- **Layout:** `<dir>/v2/` with `manifests/`, `chunks/`, `private/`, `tmp/`,
  plus a `LOCK` file. Format and component layout versions are recorded
  separately from model display names.
- **Manifest:**
  - format version and compatibility identity (hashed);
  - exact tokens, needed for exact verification;
  - component positions;
  - chunk references with sizes and checksums;
  - private-state and tail files with sizes and checksums.
- **Exclusive lock:** a second process fails at startup with a clear message
  naming the directory.
- **Startup:** read manifests (each bounded in size) and stat their
  dependencies. Read no payload bytes. Mark entries "durable, unverified".
- **Legacy files:** recognize the current store's managed files (`*.kvc` with
  the `GUFO` magic, `.tmp-*`) and remove them within the disk budget. Never
  touch unknown files and never recurse into unrelated directories.

## Not in this PR

Writing and publication (card 11). Streaming payloads (card 12). Lookup
across tiers (card 13).

## Test first

CPU tests:

- a second store on the same directory fails with the ownership error;
- bytes read at startup grow with the number of manifests, not with payload
  size (assert through an I/O counter);
- unknown files and directories are untouched;
- legacy managed files are removed and the disk budget is respected
  throughout.

## Step baseline

Startup index time and bytes read for 100 and 1,000 manifests, and manifest
size at 100k tokens.

## Done when

- [ ] Tests above pass.

## Review focus

- Manifest encoding: binary with a checksum, or JSON? Tokens make manifests
  large at 100k+ tokens. Consider a separate tokens file per lineage.
- File count per checkpoint: the research measured about 13% overhead for
  chunk files against one large file.

## References

Material to consider during implementation. Before reusing code from a
project, check its license and add it to `THIRD_PARTY_NOTICES.md` when
required (see the README).

- [Cost-Efficient LLM Serving for Multi-turn Conversations with CachedAttention](https://www.usenix.org/conference/atc24/presentation/gao-bin-cost) (USENIX ATC 2024): a host and disk hierarchy for multi-turn KV state.
- [LMCache hybrid model support](https://docs.lmcache.ai/mp/hybrid_models.html): storage backends for hybrid-model state.
- [llama.cpp server slot save and restore](https://github.com/ggml-org/llama.cpp/blob/41abbfd59/tools/server/README.md): a simple per-slot file format and its limits.

## RFC

[Disk representation and bounded transfers](../RFC.md#disk-representation-and-bounded-transfers) ·
[Publication, crash recovery and format upgrades](../RFC.md#publication-crash-recovery-and-format-upgrades)

## Review notes

