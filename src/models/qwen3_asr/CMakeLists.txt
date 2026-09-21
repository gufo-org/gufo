# Qwen3-ASR 1.7B model-private implementation.

add_library(gufo_qwen3_asr STATIC
  audio.cpp
  bpe.cpp
  config.cpp
  hip/transcription_runtime.cpp
  loader.cpp
  prompt.cpp
  tensor.cpp
  tokenizer.cpp
)

target_include_directories(gufo_qwen3_asr PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_qwen3_asr PUBLIC ICU::uc Threads::Threads)

if (ENGINE_ENABLE_HIP)
  target_sources(gufo_qwen3_asr PRIVATE
    hip/audio_encoder_runtime.cpp
    hip/audio_ops.hip
    hip/convolution.hip
    hip/blas.cpp
    hip/text_decoder_runtime.cpp
    hip/text_ops.hip)
  target_compile_definitions(gufo_qwen3_asr PRIVATE ENGINE_ENABLE_HIP=1)
  target_link_libraries(gufo_qwen3_asr PUBLIC
    hip::host
    roc::hipblas
    roc::hipblaslt)
endif()

function(gufo_qwen3_asr_register_tests)
  add_executable(qwen3_asr_config_test
    ${PROJECT_SOURCE_DIR}/src/models/qwen3_asr/tests/config_test.cpp)
  target_link_libraries(qwen3_asr_config_test PRIVATE gufo_qwen3_asr)
  add_test(NAME qwen3_asr_config_test COMMAND qwen3_asr_config_test)
  set_tests_properties(qwen3_asr_config_test PROPERTIES
    LABELS "cpu;models;qwen3-asr;config;quality")

  add_executable(qwen3_asr_loader_test
    ${PROJECT_SOURCE_DIR}/src/models/qwen3_asr/tests/loader_test.cpp)
  target_link_libraries(qwen3_asr_loader_test PRIVATE gufo_qwen3_asr)
  add_test(NAME qwen3_asr_loader_test COMMAND qwen3_asr_loader_test)
  set_tests_properties(qwen3_asr_loader_test PROPERTIES
    LABELS "cpu;models;qwen3-asr;loader;safetensors;external-model"
    SKIP_RETURN_CODE 77)

  add_executable(qwen3_asr_audio_test
    ${PROJECT_SOURCE_DIR}/src/models/qwen3_asr/tests/audio_test.cpp)
  target_link_libraries(qwen3_asr_audio_test PRIVATE gufo_qwen3_asr)
  add_test(NAME qwen3_asr_audio_test COMMAND qwen3_asr_audio_test)
  set_tests_properties(qwen3_asr_audio_test PROPERTIES
    LABELS "cpu;models;qwen3-asr;audio;quality;external-model"
    SKIP_RETURN_CODE 77)

  add_executable(qwen3_asr_prompt_test
    ${PROJECT_SOURCE_DIR}/src/models/qwen3_asr/tests/prompt_test.cpp)
  target_link_libraries(qwen3_asr_prompt_test PRIVATE gufo_qwen3_asr)
  add_test(NAME qwen3_asr_prompt_test COMMAND qwen3_asr_prompt_test)
  set_tests_properties(qwen3_asr_prompt_test PROPERTIES
    LABELS "cpu;models;qwen3-asr;tokenizer;prompt;quality;external-model"
    SKIP_RETURN_CODE 77)

  add_executable(qwen3_asr_audio_api_test
    ${PROJECT_SOURCE_DIR}/src/models/qwen3_asr/tests/audio_api_test.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/audio_asr_api.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/asr_service.cpp)
  target_link_libraries(qwen3_asr_audio_api_test PRIVATE gufo_qwen3_asr)
  add_test(NAME qwen3_asr_audio_api_test COMMAND qwen3_asr_audio_api_test)
  set_tests_properties(qwen3_asr_audio_api_test PROPERTIES
    LABELS "cpu;models;qwen3-asr;server;api;audio;quality")

  if (ENGINE_ENABLE_HIP)
    add_executable(qwen3_asr_audio_encoder_hip_test
      ${PROJECT_SOURCE_DIR}/src/models/qwen3_asr/tests/audio_encoder_hip_test.cpp)
    target_link_libraries(qwen3_asr_audio_encoder_hip_test PRIVATE
      gufo_qwen3_asr hip::host)
    target_compile_definitions(qwen3_asr_audio_encoder_hip_test PRIVATE
      ENGINE_ENABLE_HIP=1)
    add_test(NAME qwen3_asr_audio_encoder_hip_test
      COMMAND qwen3_asr_audio_encoder_hip_test)
    set_tests_properties(qwen3_asr_audio_encoder_hip_test PROPERTIES
      LABELS "gpu;models;qwen3-asr;audio-encoder;quality;external-model"
      SKIP_RETURN_CODE 77
      RUN_SERIAL TRUE)

    add_executable(qwen3_asr_text_decoder_hip_test
      ${PROJECT_SOURCE_DIR}/src/models/qwen3_asr/tests/text_decoder_hip_test.cpp)
    target_link_libraries(qwen3_asr_text_decoder_hip_test PRIVATE
      gufo_qwen3_asr hip::host)
    target_compile_definitions(qwen3_asr_text_decoder_hip_test PRIVATE
      ENGINE_ENABLE_HIP=1)
    add_test(NAME qwen3_asr_text_decoder_hip_test
      COMMAND qwen3_asr_text_decoder_hip_test)
    set_tests_properties(qwen3_asr_text_decoder_hip_test PROPERTIES
      LABELS "gpu;models;qwen3-asr;text-decoder;quality;external-model"
      SKIP_RETURN_CODE 77
      RUN_SERIAL TRUE)

    add_executable(qwen3_asr_transcription_hip_test
      ${PROJECT_SOURCE_DIR}/src/models/qwen3_asr/tests/transcription_hip_test.cpp)
    target_link_libraries(qwen3_asr_transcription_hip_test PRIVATE
      gufo_qwen3_asr hip::host)
    add_test(NAME qwen3_asr_transcription_hip_test
      COMMAND qwen3_asr_transcription_hip_test)
    set_tests_properties(qwen3_asr_transcription_hip_test PROPERTIES
      LABELS "gpu;models;qwen3-asr;transcription;quality;exact;external-model"
      SKIP_RETURN_CODE 77
      RUN_SERIAL TRUE)
  endif()
endfunction()
