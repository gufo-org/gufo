# Nix check inventory

`nix flake check` is the canonical CPU/repository gate, locally and in
[GitHub Actions](../.github/workflows/checks.yml). It validates flake outputs and
builds six check derivations. [flake.nix](../flake.nix) owns this selection.
There is no separate PR-only Nix aggregate or duplicate CPU test derivation.

## Default checks

| Check | Work |
| --- | --- |
| `format` | Pinned clang-format on C/C++ headers and sources under `include`, `src`, and `tests`, with fixture/vendor exclusions. |
| `dependency-inventory` | Dependency declaration/notice checks and their Python regressions. |
| `docs` | Repository-wide Markdown links, anchors and fenced JSON validation. |
| `tests` | CPU discovery build, 168 CTest tests, then a separate lock-mode CLI build and executable validation. |
| `mk-serve` | 41 metadata evaluation cases and rendered command strings for text, TTS/voices, ASR, image and speculative modes. Uses a placeholder executable. |
| `mk-serve-nixos` | NixOS service/command evaluation using the default and release package paths, without booting a VM or realizing GPU packages. |

Nix may reuse a locally cached or substituted output. Independent checks can
build concurrently. The CPU build uses compiler-cache reuse when the writable
shared cache is available; this does not skip test execution when that derivation
needs rebuilding.

## CPU check phases

1. Copy filtered test sources to a stable sandbox path to avoid changing Nix
   store hashes in compiler-cache keys.
2. Configure Ninja with `CMAKE_BUILD_TYPE=RelWithDebInfo`, `BUILD_TESTING=ON`
   and `ENGINE_ENABLE_HIP=OFF`, then build the full CPU tree.
3. Run CTest with up to four slots, bounded by the build-core setting. Nested
   extension compilation tests reserve two slots. Long tests receive scheduling
   priority; the boundary gates share an inventory refreshed by a CTest fixture.
   The default timeout is 120 seconds; per-test timeout properties override it.
4. Configure and build a separate CPU CLI with `BUILD_TESTING=OFF` and
   `GUFO_EXTENSIONS_FROM_LOCK=ON`. Run `gufo diagnose --section extensions --json`
   and verify that `gufo --version` reports `Extension selection: lock`.
5. Report ccache statistics when the shared cache is exposed.

This selects 168 of the 175 registered CPU tests. The six H3 qualification scripts
and `serving_bench_test` are explicitly excluded. Model loaders and reference
oracles can skip when fixtures are absent; those skips do not establish model
quality. The CMake `check-pr` target remains a focused 102-test subset for local
iteration and Windows validation; it is not a second Nix gate.

### Exact selected CTest names

This list is a snapshot collected from the configured CPU inventory; it is not
a cap on future coverage.

```text
allocation_benchmark_test
arg_parser_test
audio_websocket_test
bandwidth_test
bench_cli_test
cache_adapter_lifecycle_test
cache_adapter_test
cache_boundary_test
cache_checkpoint_test
cache_disk_faults_test
cache_disk_publication_test
cache_disk_test
cache_idle_spill_test
cache_ledger_test
cache_prefix_index_test
cache_retention_test
cache_slot_test
cache_streaming_test
cache_tiered_test
cache_workloads_test
check_docs_test
check_extension_isolation_test
check_pr_title_test
check_target_graph_test
component_text_cache_test
continuation_cache_test
continuation_disk_store_test
deepseek_v4_flash_continuation_identity_test
ds4.cli
ds4.dataset
ds4.eval
ds4.sampling
ds4.template
embedding_gemma2_audio_decoder_test
embedding_gemma2_audio_processor_test
embedding_gemma2_batch_scheduler_test
embedding_gemma2_config_test
embedding_gemma2_embeddings_api_test
embedding_gemma2_image_processor_test
embedding_gemma2_loader_test
embedding_gemma2_tokenizer_test
embedding_gemma2_video_processor_test
eval_http_client_test
eval_http_test
eval_regrade_test
exact_token_test
ext_runtime_neutral_contracts_test
extension_api_policy_test
extension_api_surface
extension_cli_provenance_test
extension_core_model_includes_gate
extension_discovery_test
extension_distribution_test
extension_family_graph_gate
extension_isolation_report
extension_link_graph_gate
extension_metadata_test
extension_model_core_includes_gate
extension_role_test
extension_runtime_test
extension_schema_test
extension_serving_boundary_gate
extension_target_ownership_gate
functional_runner_test
generate_infrastructure_test
ggml_dequant_test
ggml_gemm_parity_test
gguf_identity_test
gguf_reader_test
gpu_queues_test
gufo_cpu_probe
gufo_help
gufo_probe
gufo_serve_asr_help
gufo_serve_embedding_help
gufo_serve_tts_help
gufo_transcribe_help
gufo_version
heap_placement_test
hipblaslt_plan_cache_test
http_server_signals_test
http_server_test
json_constraint_test
json_test
kernel_bench_test
kernel_resources_parser_test
logit_sampler_test
logprobs_test
minimax_h3_generation_test
minimax_h3_media_test
minimax_h3_runtime_test
minimax_h3_sampling_test
minimax_h3_tokenizer_test
minimax_h3_video_vae_plan
openai_chat_test
platform_fs_test
platform_library_relocated_test
platform_library_test
platform_mapped_file_test
platform_pages_test
platform_prefault_test
platform_publish_test
prof_test
prompt_cli_test
quote_tracker_test
qwen27b.tools
qwen27b_continuation_identity_test
qwen27b_continuation_layout_test
qwen27b_vision_test
qwen38_flash_next.config
qwen38_flash_next.continuation_layout
qwen38_flash_next.draft_vocab
qwen38_flash_next.mtp_sampling
qwen38_flash_next.ngram
qwen38_flash_next.prompt_lookup
qwen38_flash_next.rope_scaling
qwen38_flash_next.rope_scaling_oracle
qwen38_flash_next.runner_options
qwen38_flash_next_continuation_identity_test
qwen3_asr_audio_api_test
qwen3_asr_audio_test
qwen3_asr_config_test
qwen3_asr_loader_test
qwen3_asr_prompt_test
qwen3_tts_audio_api_test
qwen3_tts_audio_test
qwen3_tts_config_test
qwen3_tts_loader_test
qwen3_tts_prompt_test
qwen3_tts_reference_artifacts_test
qwen3_tts_tokenizer_test
qwen_attention_policy_test
qwen_chat_template_test
qwen_config_test
qwen_cpu_oracles_test
qwen_forward_test
qwen_fusion_route_test
qwen_gemm_route_test
qwen_generator_test
qwen_image_21_test
qwen_image_21_weights_test
qwen_l1_ffn_module_test
qwen_l1_norm_module_test
qwen_l1_ssm_module_test
qwen_mtp_reference_test
qwen_oracle_test
qwen_ssm_test
qwen_tokenizer_test
qwen_vision_cache_test
sampling_test
serve_cli_test
speculative_corpus_test
strix_diagnose_help
strix_diagnose_json
strix_prompt_help
strix_prompt_smoke
strix_serve_help
strix_video_help
text_cli_dispatch_bench
text_cli_dispatch_chat
text_cli_dispatch_prompt
text_generation_scheduler_test
text_model_runner_test
trace_test
transcribe_cli_test
video_api_test
video_cli_test
video_jobs_test
```

## Explicit qualification

These packages are outside `checks`, so `nix flake check` does not build/run them:

| Command | Work |
| --- | --- |
| `nix build .#static-analysis` | Whole-tree CPU clang-tidy using the static analyzer and selected correctness checks. |
| `nix build .#h3-manifest` | H3 synthetic source-manifest validation. |
| `nix build .#h3-quality` | Seven offline suites covering artifacts, RNG, cache controls, latent metrics, presets and profiles. |
| `nix build .#h3-ml-quality` | Pinned denoiser teacher and offline LPIPS numerical contracts; larger Torch/tool dependency closure. |

The broad style/guidelines/performance clang-tidy audit is retained in
`.clang-tidy-audit`; see [testing](TESTING.md). Benchmark-client/reporting unit
coverage is retained as an explicit command:

```sh
nix develop .#checks.x86_64-linux.tests -c python3 tests/tools/serving_bench_test.py
```

No numerical qualification tests or standard speed benchmark tools were deleted.
These optional checks must be selected when relevant to a change.

## Workflow-only work

Before Nix, the Repository checks workflow also runs `test_impact.py`,
`test_report.py`, and `test_devlab.py` from `tests/functional`, restores caches,
and installs Nix/cache actions. On main/manual dispatch it also warms the Linux
package cache. PR title/API-diff policy, Windows builds and Devlab GPU/model
checks remain separate workflows.

## Refreshing this inventory

Configure and list tests without compiling executables or executing model tests:

```sh
nix develop .#checks.x86_64-linux.tests -c cmake --preset cpu-test
nix develop .#checks.x86_64-linux.tests -c ctest \
  --test-dir build/cpu-test --show-only=json-v1 \
  --label-exclude h3-qualification --exclude-regex '^serving_bench_test$'
```

Nix source builds follow the same exclusions. The returned JSON includes labels
and properties; a successful check log records executed names, skips and timings.
