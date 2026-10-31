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
  extension_runtime_test extension_role_test
  arg_parser_test json_test json_constraint_test gguf_reader_test qwen_config_test
  gguf_identity_test
  heap_placement_test
  platform_publish_test platform_prefault_test platform_mapped_file_test
  platform_fs_test platform_library_test
  logit_sampler_test ggml_dequant_test
  quote_tracker_test logprobs_test sampling_test
  openai_chat_test http_server_test http_server_signals_test trace_test
  audio_websocket_test
  text_generation_scheduler_test
  component_text_cache_test text_model_runner_test continuation_disk_store_test
  bench_cli_test prompt_cli_test eval_http_client_test
  qwen_tokenizer_test qwen_chat_template_test
  qwen27b_continuation_identity_test qwen38_flash_next_continuation_identity_test
  deepseek_v4_flash_continuation_identity_test
  qwen27b_continuation_layout_test
  qwen38_flash_next_config_test qwen38_flash_next_mtp_sampling_test
  qwen38_flash_next_rope_scaling_test
  qwen38_flash_next_continuation_layout_test
  ds4_sampling_test ds4_chat_template_test ds4_cli_test
  qwen3_asr_config_test qwen3_asr_audio_api_test
  embedding_gemma2_config_test embedding_gemma2_embeddings_api_test
  embedding_gemma2_video_processor_test
  embedding_gemma2_image_processor_test
  embedding_gemma2_batch_scheduler_test
  qwen3_tts_config_test qwen3_tts_audio_api_test
  qwen_image_21_test qwen_image_21_weights_test
  video_api_test minimax_h3_sampling_test minimax_h3_runtime_test)
set(gufo_pr_tests ${gufo_pr_targets})
if(NOT WIN32)
  list(APPEND gufo_pr_tests platform_library_relocated_test extension_schema_test)
endif()
list(REMOVE_ITEM gufo_pr_tests
  qwen38_flash_next_config_test qwen38_flash_next_mtp_sampling_test
  qwen38_flash_next_rope_scaling_test
  qwen38_flash_next_continuation_layout_test
  ds4_sampling_test ds4_chat_template_test ds4_cli_test)
list(APPEND gufo_pr_tests
  "qwen38_flash_next\\.config" "qwen38_flash_next\\.mtp_sampling"
  "qwen38_flash_next\\.rope_scaling" "qwen38_flash_next\\.continuation_layout"
  "ds4\\.sampling" "ds4\\.template" "ds4\\.cli"
  gufo_version gufo_help serve_cli_test eval_http_test functional_runner_test
  cache_workloads_test cache_disk_faults_test
  cache_boundary_test
  kernel_resources_parser_test check_docs_test check_pr_title_test
  check_extension_isolation_test extension_isolation_report
  extension_core_model_includes_gate extension_serving_boundary_gate
  extension_family_graph_gate extension_metadata_test
  check_target_graph_test extension_link_graph_gate
  extension_discovery_test extension_api_policy_test extension_api_surface
  extension_cli_provenance_test)
if (WIN32)
  # Linux-only hosted suites: RLIMIT_FSIZE and fork()-based tests, the
  # dup2/mmap/socket fixture tests, the permission-fault disk-store cases,
  # the cache tests that fork or wrap libc symbols, and the 1,250-chunk
  # per-file fsync stress in component_text_cache_test.
  set(gufo_pr_linux_only
    gguf_reader_test gguf_identity_test platform_publish_test
    platform_prefault_test trace_test http_server_signals_test
    http_server_test audio_websocket_test continuation_disk_store_test
    component_text_cache_test eval_http_client_test cache_disk_faults_test
    cache_boundary_test cache_adapter_test cache_slot_test cache_disk_test
    cache_streaming_test cache_tiered_test cache_disk_publication_test)
  list(REMOVE_ITEM gufo_pr_targets ${gufo_pr_linux_only})
  list(REMOVE_ITEM gufo_pr_tests ${gufo_pr_linux_only})
endif()

list(JOIN gufo_pr_tests "|" gufo_pr_pattern)
add_custom_target(check-pr
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --no-tests=error
    --timeout 180 -R "^(${gufo_pr_pattern})$"
  DEPENDS gufo ${gufo_pr_targets}
    $<$<NOT:$<BOOL:${WIN32}>>:cache_disk_faults>
  USES_TERMINAL VERBATIM)
