# Public forwarding headers

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
