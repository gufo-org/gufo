# Second-GPU prompt prefill (`--prefill-device`)

Gufo targets one AMD Strix Halo GPU (gfx1151). This fork adds an opt-in
mode that offloads **prompt processing (prefill)** to a second GPU while
decode stays on the primary APU. Prefill is compute-bound and benefits
directly from a discrete GPU's extra compute, and the fork's transfer path
keeps the hand-off cheap: after each prefill only the recurrent state and
the newly produced KV rows cross the bus, not the whole cache.

## Usage

```bash
gufo serve llm \
  --model models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q8_K_XL.gguf \
  --prefill-device 1
```

`--prefill-device N` (default `0` = disabled) names the HIP device that
runs prompt prefill. Device 0 remains the decode device and the device
that owns HTTP sessions.

## How it works

1. At load time a second `QwenGpuModel` is created on device `N`. Host
   registered unified-memory pages are not addressable by a second GPU,
   so the second copy is placed device-resident (`hipMalloc` + upload)
   instead of using the Linux mapped-registration fast path.
2. Each session slot gets a prefill executor on device `N` sharing the
   second weight copy. `QwenGpuExecutor::ForwardPromptBatch` detects the
   attached executor and runs the whole prompt batch there.
3. State moves across devices around each offloaded prefill:
   - **Before:** if the prefill arena is not already synced at the
     request's start position (cold start, restored branch, decode in
     between), the prefix KV rows and recurrent state are exported to
     device `N`.
   - **After:** the recurrent state, the new KV rows `[start, end)` and
     the final logit row are imported back into the primary arena.
     Chunked prefill keeps the mirror synced, so a multi-chunk prompt
     exports the prefix once and imports deltas per chunk.
4. Decode, speculative verification, snapshots and the continuation cache
   continue to run on device 0 unchanged. Any operation that mutates the
   primary arena outside the offload path (decode, verification,
   snapshot restore, rollback) invalidates the sync marker.

Cross-device copies use PCIe peer-to-peer when the device pair supports
it and fall back to chunked pinned-host staging otherwise
(`src/core/hip/device.cpp`).

## Requirements and limits

- **Model:** Qwen3.8-27B (`QwenGpuExecutor`). DeepSeek V4 Flash and
  Qwen3.8-Flash-Next use separate engines and reject the flag.
- **Memory:** device `N` must hold a full weight copy plus one arena per
  session slot (KV cache for `--context`). For 27B Q4_K_XL at 8k context
  plan for roughly the model size plus ~0.5 GiB per session.
- **Image prompts** run on the primary device: the vision encoder is
  resident there, so requests with images skip the offload.
- **Bandwidth:** without PCIe P2P the delta copies stage through host
  memory. The round trip costs roughly
  `(new-KV-bytes + recurrent-bytes) / effective PCIe bandwidth` per
  prefill chunk; on a 16-lane Gen4 link a 2k-token chunk of a 27B FP16-KV
  model moves a few hundred MiB, i.e. tens of milliseconds. Prefill
  throughput gains on a strong second GPU usually dominate this.
- The second device must be visible to the same process (`gufo diagnose`
  lists HIP devices); `--prefill-device` validates the index at load.

## Where the code lives

| Area | File |
| --- | --- |
| Device guard, P2P/staged cross-device copies | `src/core/hip/device.{hpp,cpp}` |
| Device-indexed arena/model/executor | `src/models/qwen/hip/{arena,model_loader,executor}.{hpp,cpp}` |
| Offloaded prefill + state export/import | `src/models/qwen/hip/prefill.cpp` |
| Per-slot prefill executor attach | `src/models/qwen/continuation_adapter.cpp` |
| CLI flag + backend wiring | `src/cli/serve/serve.cpp`, `src/cli/serve/inference_backend.cpp` |
