# Hosted CPU regression suite. Other tests remain available through CTest and
# explicit targets; adding a model oracle must not make every PR expensive.
set(gufo_pr_targets
  arg_parser_test json_test json_constraint_test gguf_reader_test gguf_identity_test
  logit_sampler_test ggml_dequant_test
  openai_chat_test http_server_test audio_websocket_test text_generation_scheduler_test
  text_model_runner_test continuation_disk_store_test
  bench_cli_test prompt_cli_test eval_http_client_test
  qwen_tokenizer_test qwen_chat_template_test
  qwen38_flash_next_config_test qwen38_flash_next_mtp_sampling_test
  ds4_sampling_test ds4_chat_template_test ds4_cli_test
  qwen3_asr_config_test qwen3_asr_audio_api_test
  qwen3_tts_config_test qwen3_tts_audio_api_test
  qwen_image_21_test
  video_api_test minimax_h3_sampling_test minimax_h3_runtime_test)
set(gufo_pr_tests ${gufo_pr_targets})
list(REMOVE_ITEM gufo_pr_tests
  qwen38_flash_next_config_test qwen38_flash_next_mtp_sampling_test
  ds4_sampling_test ds4_chat_template_test ds4_cli_test)
list(APPEND gufo_pr_tests
  "qwen38_flash_next\\.config" "qwen38_flash_next\\.mtp_sampling"
  "ds4\\.sampling" "ds4\\.template" "ds4\\.cli"
  gufo_version gufo_help serve_cli_test eval_http_test)
list(JOIN gufo_pr_tests "|" gufo_pr_pattern)
add_custom_target(check-pr
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --no-tests=error
    --timeout 60 -R "^(${gufo_pr_pattern})$"
  DEPENDS gufo ${gufo_pr_targets}
  USES_TERMINAL VERBATIM)
