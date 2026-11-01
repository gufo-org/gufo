# Extension API changes

## 0.1.6

Publish capability-neutral contracts and provider registries:
- ASR (`include/gufo/ext/asr.hpp`): `AsrSegment`, `AsrProgress`, `AsrRunner`, `AsrProvider`,
  `RegisterAsrProvider`, `FindAsrProvider`, `ProbeAsrProvider`, `Registry::AddAsrProvider`.
- TTS (`include/gufo/ext/tts.hpp`): `TtsAudioFormat`, `TtsAudioChunk`, `TtsProgress`,
  `TtsRunner`, `TtsProvider`, `RegisterTtsProvider`, `FindTtsProvider`, `ProbeTtsProvider`,
  `Registry::AddTtsProvider`.
- Video (`include/gufo/ext/video.hpp`): `VideoParameters`, `VideoRequest`, `VideoTelemetry`,
  `VideoRunner`, `VideoProvider`, `RegisterVideoProvider`, `FindVideoProvider`, `ProbeVideoProvider`,
  `Registry::AddVideoProvider`.
- Image (`include/gufo/ext/image.hpp`): Provider registration and discovery functions
  `RegisterImageProvider`, `FindImageProvider`, `ProbeImageProvider`, `Registry::AddImageProvider`.
- Platform Process (`include/gufo/ext/platform/process.hpp`): Subprocess execution contracts
  `RunMediaMux`, `RunCapture`.
- Extension migrations for Cards 15–18: Qwen3-ASR (`qwen3_asr`), Qwen3-TTS (`qwen3_tts`),
  Qwen-Image-2.1 (`qwen_image_21`), MiniMax H3 (`minimax_h3`).

## 0.1.5

Accept every GGML block format (`gguf_reader.hpp`, `experimental/ggml_dequant.hpp`):
- `GgmlType` gains `kIQ2_XS`, `kIQ3_XXS`, `kIQ1_S`, `kIQ2_S` and `kIQ1_M`, so
  a GGUF with these tensors opens and the model binder decides; exhaustive
  switches over `GgmlType` in extensions must handle them.
- CPU dequantizers for Q4_0, Q4_1, Q5_0, Q5_1, Q2_K and the IQ1/IQ2/IQ3
  formats, and `DequantizeRow`, which dispatches every block format plus
  F32/F16/BF16.

## 0.1.4

Discover metadata-only extensions without selecting them for builds when
`extension.cmake` is absent. Explicit selection still requires that file;
excluded extension provenance reports `metadata only`.
## 0.1.3

Publish capability-neutral embedding contracts (`include/gufo/ext/embeddings.hpp`):
- `EmbeddingBatch`, `EmbeddingInput`, `EmbeddingPart`, `EmbeddingStatus`,
  `EmbeddingVideoOptions`, `EmbeddingRequestOptions`.
- `EmbeddingRunner` abstract execution interface.
- `EmbeddingProvider` registration and discovery (`RegisterEmbeddingProvider`,
  `FindEmbeddingProvider`, `ProbeEmbeddingProvider`, `Registry::AddEmbeddingProvider`).
- Pilot model migration for EmbeddingGemma 2 (Card 13).

## 0.1.2

Publish public forwarding headers under `include/gufo/ext/` per the model-used
header inventory (Card 10 of model-extensions RFC):
- Stable public API contracts (`adapter.hpp`, `cache.hpp`, `ledger.hpp`, `types.hpp`,
  `cancellable_gate.hpp`, `crypto.hpp`, `sha256.hpp`, `gguf_reader.hpp`,
  `image.hpp`, `json.hpp`, `mapped_prefetch.hpp`, `model_config.hpp`,
  `reasoning.hpp`, `sampling.hpp`, `session_mode.hpp`, `utf8.hpp`,
  `draft_backend.hpp`, `prompt_lookup.hpp`, `speculative_verifier.hpp`,
  `continuation.hpp`, `device_probe.hpp`, `host_budget.hpp`, `text_contracts.hpp`).
- Platform public API contracts under `include/gufo/ext/platform/`
  (`fs.hpp`, `library.hpp`, `mapped_file.hpp`, `os_info.hpp`, `pages.hpp`,
  `prefault.hpp`, `publish.hpp`).
- Experimental forwarding contracts under `include/gufo/ext/experimental/`
  (`hip_utils.hpp`, `heap_placement.hpp`, `host_allocations.hpp`,
  `managed_alloc.hpp`, `snapshot_copy.hpp`, `snapshot_transfer.hpp`,
  `transfer_pool.hpp`, `weight_upload.hpp`, `dispatch_telemetry.hpp`,
  `gemv_dispatcher.hpp`, `hipblaslt_plan_database.hpp`,
  `hip_graph_decode_executor.hpp`, `ggml_dequant.hpp`, `ggml_gemm.hpp`).
- Transitive source closure in `SURFACE` expanded and governed by the extension
  API diff gate. Existing registration and runtime interfaces remain source
  compatible; no binary ABI is promised.

## 0.1.1

Add excluded extension metadata and artifact-key diagnostics, and generated
selection-mode provenance. Existing registration and provider contracts remain
source compatible. Discovery and exact lock selection are pure CMake; families
also appear in built-extension provenance. Incompatible unselected metadata
can describe a future API series without forcing lock builds to adopt it; only
selected extensions enforce the current major/minor/revision floor. The metadata
schema therefore accepts an `api_version` floor of any major, and it now also
requires the key-kind prefix (`gguf:`, `safetensors:`, `manifest:`, `diffusers:`)
that configure already enforces, so schema validation cannot pass a manifest
that cannot build. Cross-tier key/option/model-ID diagnostics name the first
recorded holder of the colliding value.

## 0.1.0

Initial development API: metadata/registration, core capability kinds, typed
provider options and a borrowed logging Host handle. Compatibility requires
`api_version: ">= 0.1.0"`; the minor series must match and the requested
revision must be no newer than the runtime. An additive revision keeps older
extensions compatible within this series. No stable model/session ABI is
promised by this initial source API.
