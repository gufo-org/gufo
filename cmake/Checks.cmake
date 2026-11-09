# Hosted CPU regression suite. Other tests remain available through CTest and
# explicit targets; adding a model oracle must not make every PR expensive.
set(gufo_pr_targets
  cache_adapter_test
  cache_adapter_lifecycle_test
  cache_ledger_test
  cache_checkpoint_test
  cache_prefix_index_test
  cache_slot_test
  cache_idle_spill_test
  cache_disk_test
  cache_disk_publication_test
  cache_streaming_test
  cache_tiered_test
  cache_retention_test
  sched_policy_test
  extension_runtime_test extension_role_test
  arg_parser_test transcribe_cli_test json_test json_constraint_test gguf_reader_test qwen_config_test qwen_draft_vocab_test
  gguf_identity_test
  png_decode_test wav_decode_test
  heap_placement_test
  admission_memory_test
  platform_publish_test platform_prefault_test platform_mapped_file_test
  platform_fs_test platform_library_test
  logit_sampler_test ggml_dequant_test
  quote_tracker_test logprobs_test sampling_test image_api_test
  openai_chat_test http_server_test http_server_signals_test trace_test
  audio_websocket_test
  text_generation_scheduler_test layered_prefill_conformance_test
  component_text_cache_test text_model_runner_test
  bench_cli_test prompt_cli_test text_cli_dispatch_test eval_http_client_test
  # BEGIN GENERATED MODEL PR TARGETS
  deepseek_v4_flash_continuation_identity_test ds4_chat_template_test ds4_cli_test
  ds4_sampling_test embedding_gemma2_batch_scheduler_test embedding_gemma2_config_test
  embedding_gemma2_embeddings_api_test embedding_gemma2_image_processor_test
  embedding_gemma2_video_processor_test minimax_h3_runtime_test minimax_h3_sampling_test
  qwen27b_continuation_identity_test qwen27b_continuation_layout_test qwen38_flash_next_config_test
  qwen38_flash_next_continuation_identity_test qwen38_flash_next_continuation_layout_test
  qwen38_flash_next_mtp_sampling_test qwen38_flash_next_probe_test qwen38_flash_next_rope_scaling_test
  qwen38_flash_next_runner_options_test qwen3_asr_audio_api_test qwen3_asr_config_test
  qwen3_tts_audio_api_test qwen3_tts_config_test qwen_chat_template_test qwen_image_21_test
  qwen_image_21_weights_test qwen_tokenizer_test video_api_test
  # END GENERATED MODEL PR TARGETS
)
set(gufo_pr_tests ${gufo_pr_targets})
list(REMOVE_ITEM gufo_pr_tests text_cli_dispatch_test)
list(APPEND gufo_pr_tests text_cli_dispatch_prompt text_cli_dispatch_chat text_cli_dispatch_bench)
if(NOT WIN32)
  list(APPEND gufo_pr_tests platform_library_relocated_test extension_schema_test)
endif()
# BEGIN GENERATED MODEL CTEST OVERRIDES
list(REMOVE_ITEM gufo_pr_tests
  ds4_chat_template_test ds4_cli_test ds4_sampling_test qwen38_flash_next_config_test
  qwen38_flash_next_continuation_layout_test qwen38_flash_next_mtp_sampling_test
  qwen38_flash_next_probe_test qwen38_flash_next_rope_scaling_test qwen38_flash_next_runner_options_test)
list(APPEND gufo_pr_tests
  "ds4\.template" "ds4\.cli" "ds4\.sampling" "qwen38_flash_next\.config"
  "qwen38_flash_next\.continuation_layout" "qwen38_flash_next\.mtp_sampling"
  "qwen38_flash_next\.probe" "qwen38_flash_next\.rope_scaling" "qwen38_flash_next\.runner_options")
# END GENERATED MODEL CTEST OVERRIDES
list(APPEND gufo_pr_tests
  gufo_version gufo_help serve_cli_test eval_http_test functional_runner_test
  cache_workloads_test cache_disk_faults_test
  cache_boundary_test sched_boundary_test
  kernel_resources_parser_test check_docs_test check_pr_title_test
  generate_infrastructure_test extension_distribution_test
  check_extension_isolation_test extension_isolation_report
  extension_core_model_includes_gate extension_serving_boundary_gate
  extension_family_graph_gate extension_metadata_test
  check_target_graph_test extension_link_graph_gate
  extension_discovery_test extension_api_policy_test extension_api_surface
  extension_cli_provenance_test extension_module_cli_test)
if (WIN32)
  # Linux-only hosted suites: RLIMIT_FSIZE and fork()-based tests, the
  # dup2/mmap/socket fixture tests, the permission-fault disk-store cases,
  # the cache tests that fork or wrap libc symbols, and the 1,250-chunk
  # per-file fsync stress in component_text_cache_test.
  set(gufo_pr_linux_only
    gguf_reader_test gguf_identity_test platform_publish_test
    trace_test http_server_signals_test
    http_server_test audio_websocket_test
    component_text_cache_test eval_http_client_test cache_disk_faults_test
    cache_boundary_test sched_boundary_test cache_adapter_test cache_slot_test
    cache_disk_test
    cache_streaming_test cache_tiered_test cache_disk_publication_test)
  list(REMOVE_ITEM gufo_pr_targets ${gufo_pr_linux_only})
  list(REMOVE_ITEM gufo_pr_tests ${gufo_pr_linux_only})
endif()

# Keep hosted tests within the runner's CPU budget. Nested compilation tests
# consume two slots each; ordinary contracts can fill the remaining slots.
include(ProcessorCount)
ProcessorCount(gufo_pr_jobs)
if(gufo_pr_jobs LESS 1)
  set(gufo_pr_jobs 1)
elseif(gufo_pr_jobs GREATER 4)
  set(gufo_pr_jobs 4)
endif()

list(JOIN gufo_pr_tests "|" gufo_pr_pattern)
add_custom_target(check-pr
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --no-tests=error
    --parallel ${gufo_pr_jobs} --timeout 180 -R "^(${gufo_pr_pattern})$"
  DEPENDS gufo ${gufo_pr_targets}
    $<$<NOT:$<BOOL:${WIN32}>>:cache_disk_faults>
  USES_TERMINAL VERBATIM)
