# Nix check inventory

This inventory describes the Linux CPU configuration after the hosted-CI
optimizations in this branch. The test names below were collected from CTest's
configured inventory and a successful `checks.x86_64-linux.pr` run; counts are a
snapshot, not a limit on future coverage. [flake.nix](../flake.nix) and
[cmake/Checks.cmake](../cmake/Checks.cmake) own the selection.

## Two entrypoints

| Command | Work selected |
| --- | --- |
| `nix build .#checks.x86_64-linux.pr` | Six checks listed below, plus a summary derivation. |
| `nix flake check` | Flake-output validation and all 12 declared checks for this system, including the PR checks and five additional substantive checks. |

Nix reuses identical derivations. `pr` depends on `pr-tests`, `format`, etc.;
listing those outputs separately in `checks` does **not** make them run twice.
A valid substituted or locally cached output may avoid executing its commands.
Neither command realizes every package or development shell just because it
validates the corresponding flake outputs.

## What the PR target runs

| Check output | Commands and scope |
| --- | --- |
| `format` | `python3 tools/ci/check-format.py`: pinned clang-format on C/C++ headers and sources under `include`, `src`, and `tests`, with the checker's fixture/vendor exclusions. |
| `dependency-inventory` | `python3 tools/ci/check-dependencies.py --json-report …` and `python3 -B tests/tools/check_dependencies_test.py`: dependency declarations, notices, and checker regressions. |
| `docs` | `python3 tools/ci/check-docs.py --root …`: local Markdown links, anchors, and fenced JSON throughout the source. |
| `pr-tests` | CPU discovery build, selected test executables and supporting libraries, the 102 CTest tests below, then a second lock-mode CLI build and executable checks. |
| `mk-serve` | Evaluate the 41 cases in `.devops/nix/mk-serve-tests.nix` and validate rendered CLI strings for text, TTS/voices, ASR, image, and speculative modes. Uses a placeholder executable; does not build or launch GPU inference. |
| `mk-serve-nixos` | Evaluate real NixOS service definitions using the default and release package paths and assert service/command contracts. Does not boot a VM or realize the GPU packages. |

The aggregate `pr` derivation concatenates their `result.txt` files into
`result/pr-summary.txt`. Independent derivations may build concurrently; the
aggregate waits for all six to succeed.

### CPU build and execution phases

1. Copy the filtered test source to a stable sandbox path. Configure Ninja with
   `CMAKE_BUILD_TYPE=RelWithDebInfo`, `BUILD_TESTING=ON`, and
   `ENGINE_ENABLE_HIP=OFF`. Extension selection is discovery.
2. Build `check-pr`: the real `gufo` executable, its transitive CPU libraries,
   the selected C++ test targets in `cmake/Checks.cmake`, and the disk-fault
   injection library. The isolated extension project also builds its three
   contract executables; `ext_runtime_neutral_contracts_test` is built as a
   dependency but is not selected for execution by the current PR allowlist.
3. Run the allowlisted CTest names with `--output-on-failure --no-tests=error`
   and a default 180-second timeout. Per-test timeout properties override that
   default. Up to four slots are used, bounded by detected CPUs; discovery and
   metadata tests reserve two slots each for their nested two-job builds.
   Inventory generation is a CTest fixture that precedes its boundary gates.
4. Configure a separate `lock-build` tree with `BUILD_TESTING=OFF`,
   `ENGINE_ENABLE_HIP=OFF`, and `GUFO_EXTENSIONS_FROM_LOCK=ON`. Build the real
   `gufo` target again, run `gufo diagnose --section extensions --json`, and
   require `gufo --version` to contain `Extension selection: lock`.
5. Report ccache statistics when the shared writable compiler cache is exposed.

This does not compile HIP kernels, download inference model weights, execute the
HTTP model-functional matrix, or run the standalone speed benchmarks. Some
selected tests exercise model-owned numerical contracts with synthetic fixtures;
these remain part of the hosted suite.

### Exact PR CTest selection: 102 tests

#### CLI, HTTP, audio and video contracts: 18

```text
arg_parser_test
audio_websocket_test
bench_cli_test
eval_http_client_test
eval_http_test
gufo_help
gufo_version
http_server_signals_test
http_server_test
openai_chat_test
prompt_cli_test
serve_cli_test
text_cli_dispatch_bench
text_cli_dispatch_chat
text_cli_dispatch_prompt
trace_test
transcribe_cli_test
video_api_test
```

#### Cache, continuation and scheduling: 19

```text
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
component_text_cache_test
continuation_disk_store_test
text_generation_scheduler_test
text_model_runner_test
```

#### Core, formats, platform and sampling: 16

```text
ggml_dequant_test
gguf_identity_test
gguf_reader_test
heap_placement_test
json_constraint_test
json_test
logit_sampler_test
logprobs_test
platform_fs_test
platform_library_relocated_test
platform_library_test
platform_mapped_file_test
platform_prefault_test
platform_publish_test
quote_tracker_test
sampling_test
```

#### Extension contracts and architecture: 16

```text
check_extension_isolation_test
check_target_graph_test
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
extension_role_test
extension_runtime_test
extension_schema_test
extension_serving_boundary_gate
```

#### Model and family CPU contracts: 28

```text
deepseek_v4_flash_continuation_identity_test
ds4.cli
ds4.sampling
ds4.template
embedding_gemma2_batch_scheduler_test
embedding_gemma2_config_test
embedding_gemma2_embeddings_api_test
embedding_gemma2_image_processor_test
embedding_gemma2_video_processor_test
minimax_h3_runtime_test
minimax_h3_sampling_test
qwen27b_continuation_identity_test
qwen27b_continuation_layout_test
qwen38_flash_next.config
qwen38_flash_next.continuation_layout
qwen38_flash_next.mtp_sampling
qwen38_flash_next.rope_scaling
qwen38_flash_next.runner_options
qwen38_flash_next_continuation_identity_test
qwen3_asr_audio_api_test
qwen3_asr_config_test
qwen3_tts_audio_api_test
qwen3_tts_config_test
qwen_chat_template_test
qwen_config_test
qwen_image_21_test
qwen_image_21_weights_test
qwen_tokenizer_test
```

#### Repository tools and functional-harness unit tests: 5

```text
check_docs_test
check_pr_title_test
functional_runner_test
generate_infrastructure_test
kernel_resources_parser_test
```

## What `nix flake check` adds

In addition to the seven outputs making up the PR aggregate and its dependencies,
five checks run:

| Additional check | Work |
| --- | --- |
| `tests` | Full CPU build in RelWithDebInfo, all registered CPU CTest tests with a default 120-second timeout, then the same separate lock-mode CLI build/diagnostics. Adds NumPy to the Python environment. Currently 175 registered tests: the 102 PR names plus the 73 names below. |
| `static-analysis` | Configure a CPU build with tests off, generate/deduplicate the compilation database, and run clang-tidy on every compiled `.cpp` under `src`, with up to eight workers. No HIP compilation. |
| `h3-manifest` | `tests/tools/test_h3_manifest.py`: synthetic source-manifest validation, including revision, shard, tensor, dtype, and layout rejection cases. |
| `h3-quality` | Seven Python entrypoints: quality artifact/contracts, RNG, cache controls, latent metrics, presets, profiles, and profile reports. Uses NumPy and synthetic/offline fixtures. |
| `h3-ml-quality` | `h3_denoiser_golden_test.py` and `h3_lpips_test.py`: pinned teacher and offline LPIPS contracts, using the larger Python/Torch tool environment and pinned AlexNet data. These are CPU quality tests, not full video generation. |

The seven `h3-quality` scripts are:

```text
tests/tools/test_h3_quality.py
tests/tools/h3_rng_test.py
tests/tools/h3_cache_control_test.py
tests/tools/h3_latent_quality_test.py
tests/tools/h3_preset_quality_test.py
tests/tools/h3_profile_test.py
tests/tools/h3_profile_report_test.py
```

### Additional full-CPU CTest selection: 73 tests

```text
allocation_benchmark_test
bandwidth_test
continuation_cache_test
ds4.dataset
ds4.eval
embedding_gemma2_audio_decoder_test
embedding_gemma2_audio_processor_test
embedding_gemma2_loader_test
embedding_gemma2_tokenizer_test
eval_regrade_test
exact_token_test
ext_runtime_neutral_contracts_test
extension_model_core_includes_gate
extension_target_ownership_gate
ggml_gemm_parity_test
gpu_queues_test
gufo_cpu_probe
gufo_probe
gufo_serve_asr_help
gufo_serve_embedding_help
gufo_serve_tts_help
gufo_transcribe_help
h3_cache_control_test
h3_latent_quality_test
h3_preset_quality_test
h3_profile_report_test
h3_profile_test
h3_rng_test
hipblaslt_plan_cache_test
kernel_bench_test
minimax_h3_generation_test
minimax_h3_media_test
minimax_h3_tokenizer_test
minimax_h3_video_vae_plan
platform_pages_test
prof_test
qwen27b.tools
qwen27b_vision_test
qwen38_flash_next.draft_vocab
qwen38_flash_next.ngram
qwen38_flash_next.prompt_lookup
qwen38_flash_next.rope_scaling_oracle
qwen3_asr_audio_test
qwen3_asr_loader_test
qwen3_asr_prompt_test
qwen3_tts_audio_test
qwen3_tts_loader_test
qwen3_tts_prompt_test
qwen3_tts_reference_artifacts_test
qwen3_tts_tokenizer_test
qwen_attention_policy_test
qwen_cpu_oracles_test
qwen_forward_test
qwen_fusion_route_test
qwen_gemm_route_test
qwen_generator_test
qwen_l1_ffn_module_test
qwen_l1_norm_module_test
qwen_l1_ssm_module_test
qwen_mtp_reference_test
qwen_oracle_test
qwen_ssm_test
qwen_vision_cache_test
serving_bench_test
speculative_corpus_test
strix_diagnose_help
strix_diagnose_json
strix_prompt_help
strix_prompt_smoke
strix_serve_help
strix_video_help
video_cli_test
video_jobs_test
```

Registration does not guarantee a numerical quality pass: some model loaders and
oracles declare skip return codes or need local fixtures. Missing-model skips
must stay visible and do not establish model quality.

## Work outside the Nix PR target

The GitHub [Repository checks workflow](../.github/workflows/checks.yml) also runs
`test_impact.py`, `test_report.py`, and `test_devlab.py` from `tests/functional`
before installing Nix. It restores/saves caches and installs the Nix/cache actions.
On main and manual dispatch, it additionally invokes the Linux package-cache job.
Those steps are not part of `nix build .#checks.x86_64-linux.pr`.

The PR title/API-diff policy, Windows builds/contracts, Linux package builds, and
Devlab GPU/model workflows are separate Actions jobs/workflows. Their elapsed
times cannot be attributed to this Nix target.

## Overlap and removal candidates

| Work | Actual overlap | Candidate action |
| --- | --- | --- |
| `pr-tests` versus `tests` | Separate derivations compile overlapping targets, execute the same 102 tests, and each builds a lock-mode CLI. Nix does not deduplicate these different derivations. | Split common contract execution and lock-mode qualification into shared derivations; let full CPU checks add only the extra coverage. Preserve both documented entrypoints. |
| `h3-quality` versus `tests` | Six scripts (RNG, cache control, latent quality, preset quality, profile, profile report) also run as CTest tests in the full CPU suite. The quality-artifact unit suite is unique to `h3-quality`. | Share the six checks between these entrypoints rather than execute them twice during `nix flake check`. |
| Whole-tree clang-tidy | Additional analysis, not duplicate test execution. Potentially expensive even with no inference or weights. | Consider focused/dedicated qualification if the default flake check needs a smaller scope; measure before changing the policy. |
| ML quality environment | Unique numerical coverage, with a larger dependency closure than the hosted PR target. | Keep the tests and pinned quality contracts; consider an explicitly invoked qualification tier rather than deleting coverage. |
| Shared PR subchecks | `pr`, `pr-tests`, `format`, docs, dependencies, and serve checks refer to shared derivations. | Removing aliases from the attribute set does not save repeated execution. |
| Docs/dependency/isolation checker unit tests | Unit tests validate the checker on fixtures; the repository scans validate the actual checkout. | Keep both; these are different assertions. |

This is an inventory and candidate list; it does not remove tests. The current
PR already removes repeated isolation scans and caches nested fixture builds.
Changing or removing documented flake attribute paths would change the public
flake interface and must follow the [release policy](RELEASING.md).

## Refreshing the inventory

Configure the CPU tree without building executables, using the check's CPU
dependencies rather than the GPU development shell:

```sh
nix develop .#checks.x86_64-linux.pr-tests -c cmake --preset cpu-test
nix develop .#checks.x86_64-linux.pr-tests -c ctest \
  --test-dir build/cpu-test --show-only=json-v1
```

The JSON lists all registered CPU tests, their labels and properties. The
`gufo_pr_tests` list in `cmake/Checks.cmake` determines the PR subset after the
model-name overrides and platform exclusions. A successful `pr-tests` build log
records the exact selected names and timings. These commands only configure and
list tests; they do not run the full CPU/model suite.
