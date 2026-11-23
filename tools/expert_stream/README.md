# Expert-stream trace tools

The experimental expert store (`<gufo/ext/experimental/expert_stream.hpp>`)
writes a binary routing trace when `Policy::trace_path` is set. Extensions
expose it through their own switch (the deepseek41 extension:
`DS41_EXPERT_TRACE=PATH`). These tools replay a trace offline, so
cache and prefetch rules can be compared before a live A/B. They need Python 3
and numpy; `trace_compare.py` needs only the standard library.

| Tool | Purpose |
| --- | --- |
| `expert_trace.py` | Reader: `records(path)` (stdlib) and `read_trace(path)` (numpy) |
| `trace_compare.py` | Check that two traces record the same cache decisions |
| `prefetch_sim.py` | Compare prefetch policies, and check the cache model against the recorded hits |
| `evict_sim.py` | Compare eviction rules, LRU and Belady's offline optimum |

```sh
python3 tools/expert_stream/trace_compare.py A.bin B.bin
python3 tools/expert_stream/prefetch_sim.py TRACE
python3 tools/expert_stream/evict_sim.py TRACE current,lru,belady
```

`prefetch_sim.py` and `evict_sim.py` model the store's rule before frequency
classes (decayed score); `evict_sim.py`'s `classlru` rule approximates the
current one. `prefetch_sim.py`'s predefined policies are the ones compared on
DeepSeek V4.1 Flash (top-6 predictions); edit them for a model with another
top-k. Prefetch policies need router predictions in the trace
(`Store::TracePrediction`); without them only the cache figures apply.

## Format, version 2

Little-endian, packed. The file starts with the 8 bytes `GUFOXTRC` and a
`u32` version (2). Records follow, each a one-byte tag:

`S`, before the first prepare: the cache as seeded. Another follows before
the next prepare whenever `Fit`, `Release` or `Restore` changed the cache;
the replay tools start from the last one and skip the prepares before it.

| Field | Type |
| --- | --- |
| layers, experts, top_k, pieces | 4 × `u32` |
| read_margin_bytes, tick, cache_limit, cache_bytes, available_bytes, chunk_bytes | 6 × `u64` |
| use_half_life, use_cap, max_class, clock_rows, seed_uses, recent_rows, sweep_tokens | `f64 u32 f64 u32 f64 u32 u32` (Policy) |
| size classes, then each class's slot bytes | `u32`, classes × `u64` |
| per layer: size class, token clock, each piece's bytes per expert | `u32 u64`, pieces × `u64` |
| slots, then (size class, layer, expert) per slot; free slots hold `0xFFFFFFFF` | `u32`, slots × 3 × `u32` |
| per layer, per expert: uses, tick of the last use | layers × experts × (`f64 u64`) |

`P`, per prepare (seeding is not traced):

| Field | Type |
| --- | --- |
| start, routes read, end (steady-clock ns) | 3 × `u64` |
| layer, tokens | 2 × `u32` |
| read wait ns, demand bytes, adopted prefetch bytes | 3 × `u64` |
| routes: expert per row, -1 for a dropped row | tokens × top_k × `i32` |
| hits: 1 when the row's expert was cached before admission | tokens × top_k × `u8` |

`Q`, a caller's router prediction, right after its target layer's `P`:

| Field | Type |
| --- | --- |
| target layer, lead (layers ahead), tokens, k | 4 × `u32` |
| predicted experts | tokens × k × `i32` |

Version 1 (deepseek41 traces before the store moved here) has no header; its
`S` starts with a `u32` version 1 and records neither pieces, margin, chunk
size, policy nor clocks, and each layer holds `u32 class, u64 expert bytes`.
`expert_trace.py` reads both.
