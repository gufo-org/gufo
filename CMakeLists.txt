cmake_minimum_required(VERSION 3.21)
project(gufo LANGUAGES C CXX)
if (NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
  message(FATAL_ERROR "Gufo requires Linux x86-64 with an AMD gfx1151 GPU")
endif()
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
option(BUILD_TESTING "Build correctness tests" OFF)
option(GUFO_BUILD_TOOLS "Build development benchmarks and tuning tools" OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(GUFO_VERSION "development" CACHE STRING "Gufo build revision")
include_directories(${CMAKE_SOURCE_DIR})
find_package(ICU REQUIRED COMPONENTS uc i18n)
find_package(Threads REQUIRED)
find_package(CURL REQUIRED)
find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)
set(GUFO_FFMPEG_EXECUTABLE "ffmpeg" CACHE STRING
  "FFmpeg executable used for generated media")
set(GUFO_FFPROBE_EXECUTABLE "ffprobe" CACHE STRING
  "FFprobe executable used to validate generated media")

# CPU-only builds exist for host checks; production targets gfx1151 only.
option(ENGINE_ENABLE_HIP "Build the gfx1151 HIP backend" ON)
set(CMAKE_HIP_ARCHITECTURES gfx1151 CACHE STRING "HIP architecture")
if (ENGINE_ENABLE_HIP AND NOT CMAKE_HIP_ARCHITECTURES STREQUAL "gfx1151")
  message(FATAL_ERROR "Gufo kernels require CMAKE_HIP_ARCHITECTURES=gfx1151")
endif()
# Standard ROCm installations; custom prefixes use CMAKE_PREFIX_PATH.
if (EXISTS /opt/rocm)
  list(APPEND CMAKE_PREFIX_PATH /opt/rocm)
endif()

if (ENGINE_ENABLE_HIP)
  enable_language(HIP)
  find_package(hip REQUIRED)
  find_package(hipblas REQUIRED)
  find_package(hipblaslt REQUIRED)
  find_package(rocblas REQUIRED)
  find_path(HIPCUB_INCLUDE_DIR hipcub/hipcub.hpp REQUIRED)
  find_path(ROCPRIM_INCLUDE_DIR rocprim/rocprim.hpp REQUIRED)
  find_path(ROCWMMA_INCLUDE_DIR rocwmma/rocwmma.hpp REQUIRED)
  message(STATUS "gufo: HIP target ${CMAKE_HIP_ARCHITECTURES}")
endif()

if (ENGINE_ENABLE_HIP)
  add_subdirectory(src/models/deepseek_v4_flash)
endif()

# Qwen3.8-Flash-Next — model-private CPU reference plus ROCm runtime.
add_subdirectory(src/models/qwen38_flash_next)

# Qwen3-TTS — CPU reference + HIP kernels (CustomVoice 12Hz).
add_subdirectory(src/models/qwen3_tts)

# Qwen3-ASR — model-private 1.7B speech recognition runtime.
add_subdirectory(src/models/qwen3_asr)

# Compiler warnings and sanitizers
option(GUFO_ENABLE_WARNINGS "Enable compiler warnings" ON)
option(GUFO_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

if (GUFO_ENABLE_WARNINGS)
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
  endif()
endif()

if (GUFO_ENABLE_SANITIZERS)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

add_library(gufo_diagnostics STATIC
  src/core/diagnostics/linux_sysfs.cpp
  src/core/diagnostics/system_inventory.cpp
  src/core/diagnostics/compatibility.cpp
  src/core/diagnostics/fingerprint.cpp
  src/core/diagnostics/artifact_validator.cpp
  src/core/diagnostics/allocation_benchmark.cpp
  src/core/diagnostics/bandwidth.cpp
  src/core/hip/bandwidth.cpp
  src/core/diagnostics/report.cpp
  src/cli/diagnose/diagnose.cpp
)
target_include_directories(gufo_diagnostics PUBLIC ${CMAKE_SOURCE_DIR})
target_compile_definitions(gufo_diagnostics PRIVATE
  GUFO_VERSION="${GUFO_VERSION}")

if (ENGINE_ENABLE_HIP)
  target_sources(gufo_diagnostics PRIVATE
    src/core/hip/allocation_benchmark.hip)
  target_compile_definitions(gufo_diagnostics PRIVATE ENGINE_ENABLE_HIP=1)
  target_link_libraries(gufo_diagnostics PRIVATE hip::host roc::hipblas roc::rocblas)
endif()

add_library(gufo_core STATIC
  src/core/image.cpp
  src/core/crypto/sha256.cpp
  src/core/gguf_reader.cpp
  src/core/gguf_identity.cpp
  src/core/json_constraint.cpp
  src/core/json_schema_lexeme.cpp
  src/core/sampling.cpp
  src/models/minimax_h3/audio_vae.cpp
  src/models/minimax_h3/denoiser.cpp
  src/models/minimax_h3/dit.cpp
  src/models/minimax_h3/generation.cpp
  src/models/minimax_h3/json.cpp
  src/models/minimax_h3/media.cpp
  src/models/minimax_h3/prompt_encoder.cpp
  src/models/minimax_h3/runtime.cpp
  src/models/minimax_h3/sampling.cpp
  src/models/minimax_h3/sha256.cpp
  src/models/minimax_h3/tokenizer.cpp
  src/models/minimax_h3/video_vae.cpp
  src/testing/compare/logit_comparator.cpp
  src/eval/dataset.cpp
  src/eval/extract.cpp
  src/eval/regrade.cpp
  src/core/hip/hipblaslt_plan_database.cpp
  src/core/quant/ggml_dequant.cpp
  src/core/quant/ggml_gemm.cpp
)
target_include_directories(gufo_core PUBLIC ${CMAKE_SOURCE_DIR})
target_compile_definitions(gufo_core PRIVATE
  GUFO_FFMPEG_EXECUTABLE="${GUFO_FFMPEG_EXECUTABLE}"
  GUFO_FFPROBE_EXECUTABLE="${GUFO_FFPROBE_EXECUTABLE}")
target_link_libraries(gufo_core PUBLIC ICU::uc ICU::i18n Threads::Threads
  PNG::PNG JPEG::JPEG CURL::libcurl OpenSSL::Crypto)

add_subdirectory(src/models/qwen)
add_subdirectory(src/models/qwen_image_21)

target_link_libraries(gufo_qwen3_tts PUBLIC gufo_core)
if (ENGINE_ENABLE_HIP)
  target_link_libraries(gufo_deepseek_v4_flash PUBLIC gufo_core)
endif()

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
  target_link_libraries(gufo_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if (ENGINE_ENABLE_HIP)
  set_property(SOURCE src/models/minimax_h3/dit.hip
    APPEND PROPERTY COMPILE_OPTIONS -O2)
  target_sources(gufo_core PRIVATE
    src/models/minimax_h3/audio_vae.hip
    src/models/minimax_h3/denoiser.hip
    src/models/minimax_h3/dit.hip
    src/models/minimax_h3/dense.hip
    src/models/minimax_h3/runtime_hip.cpp
    src/models/minimax_h3/prompt_encoder.hip
    src/models/minimax_h3/sampling.hip
    src/models/minimax_h3/video_vae.hip
    src/models/minimax_h3/attention.hip
    src/core/hip/weight_upload.cpp
    src/core/speculative/speculative_verifier.cpp
  )
  target_compile_definitions(gufo_core PUBLIC ENGINE_ENABLE_HIP=1)
  target_include_directories(gufo_core SYSTEM PRIVATE
    ${HIPCUB_INCLUDE_DIR}
    ${ROCPRIM_INCLUDE_DIR})
  target_link_libraries(gufo_core PUBLIC hip::host roc::hipblas
    roc::hipblaslt roc::rocblas)
endif()

add_library(gufo_llm_cli STATIC
  src/cli/bench/bench.cpp
  src/cli/prompt/prompt.cpp)
target_link_libraries(gufo_llm_cli PUBLIC gufo_core)
if(ENGINE_ENABLE_HIP)
  target_link_libraries(gufo_llm_cli PUBLIC gufo_deepseek_v4_flash
    gufo_qwen38_flash_next)
endif()

add_library(gufo_logging STATIC src/cli/serve/logging.cpp)
target_include_directories(gufo_logging PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_features(gufo_logging PUBLIC cxx_std_20)

add_library(gufo_http STATIC
  src/cli/serve/image_api.cpp
  src/cli/serve/audio_asr_api.cpp
  src/cli/serve/asr_service.cpp
  src/cli/serve/audio_tts_api.cpp
  src/cli/serve/tts_service.cpp
  src/cli/serve/video_api.cpp
  src/cli/serve/video_jobs.cpp
  src/cli/serve/http_server.cpp
  src/cli/serve/websocket.cpp
  src/cli/serve/asr_websocket.cpp
  src/cli/serve/tts_websocket.cpp
  src/cli/serve/openai_chat.cpp
)
target_link_libraries(gufo_http PUBLIC gufo_logging gufo_core gufo_qwen3_tts gufo_qwen3_asr gufo_qwen_image_21)

add_executable(gufo
  src/cli/main.cpp
  src/cli/eval/eval.cpp
  src/cli/transcribe/transcribe.cpp
  src/cli/video/video.cpp
  src/cli/serve/serve.cpp
  src/cli/serve/continuation_cache.cpp
  src/cli/serve/continuation_disk_store.cpp
  src/cli/serve/text_model_runner.cpp
  src/cli/serve/text_generation_scheduler.cpp
  src/cli/serve/inference_backend.cpp
  src/eval/http_client.cpp
)
target_link_libraries(gufo PRIVATE gufo_diagnostics gufo_core gufo_llm_cli gufo_http CURL::libcurl)
target_compile_definitions(gufo PRIVATE
  GUFO_VERSION="${GUFO_VERSION}"
  GUFO_EVAL_SOURCE_DATA="${CMAKE_CURRENT_SOURCE_DIR}/tests/models/deepseek_v4_flash/fixtures/antirez-ds4.json"
  GUFO_EVAL_INSTALLED_DATA="${CMAKE_INSTALL_PREFIX}/share/gufo/eval/antirez-ds4.json"
  GUFO_H3_SOURCE_MANIFEST="${CMAKE_CURRENT_SOURCE_DIR}/src/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json"
  GUFO_H3_INSTALLED_SOURCE_MANIFEST="${CMAKE_INSTALL_PREFIX}/share/gufo/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json")

if (ENGINE_ENABLE_HIP)
  target_link_libraries(gufo PRIVATE gufo_deepseek_v4_flash)
endif()
target_link_libraries(gufo PRIVATE gufo_qwen3_tts gufo_qwen3_asr)

add_custom_target(gufo-server ALL
  COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE_NAME:gufo> ${CMAKE_CURRENT_BINARY_DIR}/gufo-server
  DEPENDS gufo
)
if (BUILD_TESTING OR GUFO_BUILD_TOOLS)
  add_library(gufo_kernel_bench_support STATIC
    src/cli/bench/kernel_bench.cpp)
  target_include_directories(gufo_kernel_bench_support PUBLIC ${CMAKE_SOURCE_DIR})
endif()

if (ENGINE_ENABLE_HIP AND GUFO_BUILD_TOOLS)
  find_path(ROCTX_INCLUDE_DIR rocprofiler-sdk-roctx/roctx.h REQUIRED)
  find_library(ROCTX_LIBRARY NAMES rocprofiler-sdk-roctx REQUIRED)
  add_executable(gufo-kernel-bench
    src/cli/bench/kernel_bench_main.cpp
  )
  target_compile_definitions(gufo-kernel-bench PRIVATE
    GUFO_VERSION="${GUFO_VERSION}")
  target_include_directories(gufo-kernel-bench SYSTEM PRIVATE
    ${ROCTX_INCLUDE_DIR})
  target_link_libraries(gufo-kernel-bench PRIVATE
    gufo_kernel_bench_support
    gufo_diagnostics
    gufo_core
    ${ROCTX_LIBRARY})

  add_executable(tune_hipblaslt tools/bench/tune_hipblaslt.cpp)
  target_link_libraries(tune_hipblaslt PRIVATE gufo_core hip::host)

  add_executable(wmma_layout_test tools/bench/wmma_layout_test.hip)
  target_link_libraries(wmma_layout_test PRIVATE gufo_core hip::device)
endif()

install(TARGETS gufo DESTINATION bin)
install(FILES
  src/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json
  DESTINATION share/gufo/models/minimax_h3)
install(FILES
  tests/models/deepseek_v4_flash/fixtures/antirez-ds4.json
  DESTINATION share/gufo/eval)
install(FILES LICENSE NOTICE THIRD_PARTY_NOTICES.md DESTINATION share/licenses/gufo)
install(DIRECTORY licenses/ DESTINATION share/licenses/gufo/third-party)
install(FILES src/models/deepseek_v4_flash/LICENSE.ds4 DESTINATION share/licenses/gufo/third-party)
install(CODE "execute_process(COMMAND \"${CMAKE_COMMAND}\" -E create_symlink gufo \"\$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gufo-server\")")
if (ENGINE_ENABLE_HIP AND GUFO_BUILD_TOOLS)
  install(TARGETS gufo-kernel-bench tune_hipblaslt
    DESTINATION bin)
endif()

# CTest integration
include(CTest)

# Test assertions remain active when model code uses production optimization.
if (BUILD_TESTING)
  add_compile_options(-UNDEBUG)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  add_subdirectory(tests/models/qwen)
  add_subdirectory(tests/models/qwen27b)
  gufo_qwen3_asr_register_tests()
  gufo_qwen_image_21_register_tests()

  add_test(NAME gufo_version COMMAND gufo --version)
  set_tests_properties(gufo_version PROPERTIES LABELS "cpu")

  add_test(NAME gufo_help COMMAND gufo --help)
  set_tests_properties(gufo_help PROPERTIES LABELS "cpu")

  add_test(NAME gufo_probe COMMAND gufo probe)
  set_tests_properties(gufo_probe PROPERTIES LABELS "cpu")

  add_test(NAME strix_diagnose_help COMMAND gufo diagnose --help)
  set_tests_properties(strix_diagnose_help PROPERTIES LABELS "cpu")

  add_test(NAME strix_diagnose_json COMMAND gufo diagnose --json --section platform)
  set_tests_properties(strix_diagnose_json PROPERTIES LABELS "cpu")

  add_test(NAME strix_prompt_help COMMAND gufo prompt --help)
  set_tests_properties(strix_prompt_help PROPERTIES LABELS "cpu;server;cli")

  add_test(NAME strix_video_help COMMAND gufo video --help)
  set_tests_properties(strix_video_help PROPERTIES
    LABELS "cpu;server;cli;minimax-h3")

  add_test(NAME gufo_transcribe_help COMMAND gufo transcribe --help)
  set_tests_properties(gufo_transcribe_help PROPERTIES
    LABELS "cpu;server;cli;qwen3-asr")

  add_test(NAME gufo_serve_asr_help COMMAND gufo serve asr --help)
  set_tests_properties(gufo_serve_asr_help PROPERTIES
    LABELS "cpu;server;cli;qwen3-asr")
  add_test(NAME gufo_serve_tts_help COMMAND gufo serve tts --help)
  set_tests_properties(gufo_serve_tts_help PROPERTIES
    LABELS "cpu;server;cli;qwen3-tts")

  add_test(NAME strix_serve_help COMMAND gufo serve --help)
  set_tests_properties(strix_serve_help PROPERTIES LABELS "cpu;server;cli")

  add_test(NAME strix_prompt_smoke COMMAND gufo prompt "Hello world")
  set_tests_properties(strix_prompt_smoke PROPERTIES LABELS "cpu;server;cli")

  add_executable(arg_parser_test
    tests/cli/arg_parser_test.cpp
  )
  target_link_libraries(arg_parser_test PRIVATE gufo_core)
  add_test(NAME arg_parser_test COMMAND arg_parser_test)
  set_tests_properties(arg_parser_test PROPERTIES LABELS "cpu;cli")

  add_executable(json_constraint_test tests/core/json_constraint_test.cpp)
  target_link_libraries(json_constraint_test PRIVATE gufo_core)
  add_test(NAME json_constraint_test COMMAND json_constraint_test)
  set_tests_properties(json_constraint_test PROPERTIES LABELS "cpu;json;sampling")

  add_executable(json_test tests/core/json_test.cpp)
  target_link_libraries(json_test PRIVATE gufo_core)
  add_test(NAME json_test COMMAND json_test)
  set_tests_properties(json_test PROPERTIES LABELS "cpu;json")

  add_executable(openai_chat_test
    tests/cli/openai_chat_test.cpp
    src/cli/serve/openai_chat.cpp
  )
  target_link_libraries(openai_chat_test PRIVATE gufo_core)
  add_test(NAME openai_chat_test COMMAND openai_chat_test)
  set_tests_properties(openai_chat_test PROPERTIES
    LABELS "cpu;server;api;openai;streaming;tools")

  add_executable(http_server_test tests/cli/http_server_test.cpp)
  target_link_libraries(http_server_test PRIVATE gufo_http)
  add_test(NAME http_server_test COMMAND http_server_test)
  set_tests_properties(http_server_test PROPERTIES LABELS "cpu;server;http" TIMEOUT 30)
  add_executable(audio_websocket_test tests/cli/audio_websocket_test.cpp)
  target_link_libraries(audio_websocket_test PRIVATE gufo_http)
  add_test(NAME audio_websocket_test COMMAND audio_websocket_test)
  set_tests_properties(audio_websocket_test PROPERTIES LABELS "cpu;server;audio" TIMEOUT 30)

  add_test(NAME serve_cli_test
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli/serve_test.py
      $<TARGET_FILE:gufo>)
  set_tests_properties(serve_cli_test PROPERTIES LABELS "cpu;server;cli" TIMEOUT 30)

  add_subdirectory(tests/models/deepseek_v4_flash)
  add_subdirectory(tests/models/qwen38_flash_next)

  if(ENGINE_ENABLE_HIP)
    add_executable(chat_template_hf_token_golden_test
      tests/models/chat_template_hf_token_golden_test.cpp)
    target_link_libraries(chat_template_hf_token_golden_test PRIVATE gufo_core)
    target_compile_definitions(chat_template_hf_token_golden_test PRIVATE
      GUFO_CHAT_TEMPLATE_HF_GOLDENS="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/chat_template_hf_goldens.json")
    add_test(NAME chat_template_hf_token_golden_test
      COMMAND chat_template_hf_token_golden_test)
    set_tests_properties(chat_template_hf_token_golden_test PROPERTIES
      LABELS "gpu;hip;tokenization;chat_template;qwen;external-model"
      SKIP_RETURN_CODE 77)
  endif()

  add_executable(continuation_cache_test
    tests/cli/continuation_cache_test.cpp
    src/cli/serve/continuation_cache.cpp)
  target_link_libraries(continuation_cache_test PRIVATE gufo_core)
  add_test(NAME continuation_cache_test COMMAND continuation_cache_test)
  set_tests_properties(continuation_cache_test PROPERTIES
    LABELS "cpu;server;cache;state;lifecycle")

  add_executable(text_model_runner_test
    tests/cli/text_model_runner_test.cpp
    src/cli/serve/continuation_cache.cpp
    src/cli/serve/continuation_disk_store.cpp
    src/cli/serve/text_model_runner.cpp)
  target_link_libraries(text_model_runner_test PRIVATE gufo_logging gufo_core)
  add_test(NAME text_model_runner_test COMMAND text_model_runner_test)
  set_tests_properties(text_model_runner_test PROPERTIES
    LABELS "cpu;server;runner;cache;state;lifecycle")

  add_executable(continuation_disk_store_test
    tests/cli/continuation_disk_store_test.cpp
    src/cli/serve/continuation_disk_store.cpp
    src/cli/serve/text_model_runner.cpp
    src/cli/serve/continuation_cache.cpp)
  target_link_libraries(continuation_disk_store_test PRIVATE gufo_logging gufo_core)
  add_test(NAME continuation_disk_store_test
    COMMAND continuation_disk_store_test)
  set_tests_properties(continuation_disk_store_test PROPERTIES
    LABELS "cpu;server;cache;disk;state;lifecycle")

  add_executable(logit_sampler_test tests/cli/logit_sampler_test.cpp)
  target_link_libraries(logit_sampler_test PRIVATE gufo_core)
  add_test(NAME logit_sampler_test COMMAND logit_sampler_test)
  set_tests_properties(logit_sampler_test PROPERTIES
    LABELS "cpu;server;sampling")

  add_executable(text_generation_scheduler_test
    tests/cli/text_generation_scheduler_test.cpp
    src/cli/serve/continuation_cache.cpp
    src/cli/serve/continuation_disk_store.cpp
    src/cli/serve/text_model_runner.cpp
    src/cli/serve/text_generation_scheduler.cpp)
  target_link_libraries(text_generation_scheduler_test PRIVATE gufo_logging gufo_core)
  add_test(NAME text_generation_scheduler_test
    COMMAND text_generation_scheduler_test)
  set_tests_properties(text_generation_scheduler_test PROPERTIES
    LABELS "cpu;server;scheduler;queue;state;lifecycle;cancellation")

  add_executable(prompt_cli_test
    tests/cli/prompt_test.cpp
  )
  target_link_libraries(prompt_cli_test PRIVATE gufo_llm_cli)

  add_test(NAME prompt_cli_test COMMAND prompt_cli_test)
  set_tests_properties(prompt_cli_test PROPERTIES LABELS "cpu;server;cli")

  add_executable(bench_cli_test
    tests/cli/bench_test.cpp
  )
  target_link_libraries(bench_cli_test PRIVATE gufo_llm_cli)

  add_test(NAME bench_cli_test COMMAND bench_cli_test)
  set_tests_properties(bench_cli_test PROPERTIES LABELS "cpu;server;cli")

  add_executable(video_cli_test
    tests/cli/video_test.cpp
    src/cli/video/video.cpp
  )
  target_link_libraries(video_cli_test PRIVATE gufo_core)
  add_test(NAME video_cli_test COMMAND video_cli_test)
  set_tests_properties(video_cli_test PROPERTIES
    LABELS "cpu;server;cli;minimax-h3;quality")

  add_executable(video_jobs_test
    tests/cli/video_jobs_test.cpp
    src/cli/serve/video_jobs.cpp
  )
  target_link_libraries(video_jobs_test PRIVATE gufo_logging gufo_core)
  add_test(NAME video_jobs_test COMMAND video_jobs_test)
  set_tests_properties(video_jobs_test PROPERTIES
    LABELS "cpu;server;video;minimax-h3;queue;cancellation;quality")

  add_executable(video_api_test
    tests/cli/video_api_test.cpp
    src/cli/serve/video_api.cpp
    src/cli/serve/video_jobs.cpp
  )
  target_link_libraries(video_api_test PRIVATE gufo_logging gufo_core)
  add_test(NAME video_api_test COMMAND video_api_test)
  set_tests_properties(video_api_test PROPERTIES
    LABELS "cpu;server;api;video;minimax-h3;quality")

  add_test(NAME h3_profile_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/h3_profile_test.py)
  set_tests_properties(h3_profile_test PROPERTIES
    LABELS "cpu;profiling;video;minimax-h3;quality")

  add_test(NAME h3_preset_quality_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/h3_preset_quality_test.py)
  set_tests_properties(h3_preset_quality_test PROPERTIES
    LABELS "cpu;tools;models;minimax-h3;quality;video;audio")

  add_test(NAME h3_rng_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/h3_rng_test.py)
  set_tests_properties(h3_rng_test PROPERTIES
    LABELS "cpu;tools;models;minimax-h3;quality")

  add_test(NAME h3_latent_quality_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/h3_latent_quality_test.py)
  set_tests_properties(h3_latent_quality_test PROPERTIES
    LABELS "cpu;tools;models;minimax-h3;quality")

  add_test(NAME h3_cache_control_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/h3_cache_control_test.py)
  set_tests_properties(h3_cache_control_test PROPERTIES
    LABELS "cpu;tools;models;minimax-h3;profiling")

  add_test(NAME h3_profile_report_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/h3_profile_report_test.py)
  set_tests_properties(h3_profile_report_test PROPERTIES
    LABELS "cpu;tools;models;minimax-h3;profiling")

  add_test(NAME serving_bench_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/serving_bench_test.py)
  set_tests_properties(serving_bench_test PROPERTIES
    LABELS "cpu;tools;server;benchmarks;statistics;json")

  add_test(NAME speculative_corpus_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/speculative_corpus_test.py)
  set_tests_properties(speculative_corpus_test PROPERTIES
    LABELS "cpu;tools;benchmarks;speculative;ds4;qwen")

  add_test(NAME prof_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/prof_test.py)
  set_tests_properties(prof_test PROPERTIES
    LABELS "cpu;tools;profiling")

  add_executable(bandwidth_test tests/diagnostics/bandwidth_test.cpp)
  target_compile_definitions(bandwidth_test PRIVATE TEST_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/diagnostics")
  target_link_libraries(bandwidth_test PRIVATE gufo_diagnostics)

  add_test(NAME bandwidth_test COMMAND bandwidth_test)
  set_tests_properties(bandwidth_test PROPERTIES LABELS "cpu;diagnostics;bandwidth;bandwidth_schema;bandwidth_correctness")

  add_executable(allocation_benchmark_test
    tests/diagnostics/allocation_benchmark_test.cpp)
  target_compile_definitions(allocation_benchmark_test PRIVATE
    TEST_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/diagnostics")
  target_link_libraries(allocation_benchmark_test PRIVATE gufo_diagnostics)
  add_test(NAME allocation_benchmark_test COMMAND allocation_benchmark_test)
  set_tests_properties(allocation_benchmark_test PROPERTIES
    LABELS "gpu;hip;gfx1151;diagnostics;allocation;opt-c091")

  add_executable(gguf_reader_test tests/core/gguf_reader_test.cpp)
  target_link_libraries(gguf_reader_test PRIVATE gufo_core)

  add_test(NAME gguf_reader_test COMMAND gguf_reader_test)
  set_tests_properties(gguf_reader_test PROPERTIES LABELS "cpu;core;gguf;model_config")

  add_executable(gguf_identity_test tests/core/gguf_identity_test.cpp)
  target_link_libraries(gguf_identity_test PRIVATE gufo_core)

  add_test(NAME gguf_identity_test COMMAND gguf_identity_test)
  set_tests_properties(gguf_identity_test PROPERTIES LABELS "cpu;core;gguf")

  add_executable(minimax_h3_runtime_test
    tests/models/minimax_h3_runtime_test.cpp)
  target_link_libraries(minimax_h3_runtime_test PRIVATE gufo_core)
  add_test(NAME minimax_h3_runtime_test COMMAND minimax_h3_runtime_test)
  set_tests_properties(minimax_h3_runtime_test PROPERTIES
    LABELS "cpu;models;minimax-h3;safetensors;residency;failure-injection")

  add_executable(qwen3_tts_loader_test
    tests/models/qwen3_tts/loader_test.cpp)
  target_link_libraries(qwen3_tts_loader_test PRIVATE gufo_qwen3_tts)
  add_test(NAME qwen3_tts_loader_test COMMAND qwen3_tts_loader_test)
  set_tests_properties(qwen3_tts_loader_test PROPERTIES
    LABELS "cpu;models;qwen3-tts;loader;safetensors"
    SKIP_RETURN_CODE 77)

  add_executable(qwen3_tts_audio_api_test
    tests/models/qwen3_tts/audio_api_test.cpp
    src/cli/serve/audio_tts_api.cpp
    src/cli/serve/tts_service.cpp)
  target_link_libraries(qwen3_tts_audio_api_test PRIVATE gufo_qwen3_tts)
  add_test(NAME qwen3_tts_audio_api_test COMMAND qwen3_tts_audio_api_test)
  set_tests_properties(qwen3_tts_audio_api_test PROPERTIES
    LABELS "cpu;models;qwen3-tts;server;api;audio;quality")

  add_executable(qwen3_tts_audio_test
    tests/models/qwen3_tts/audio_test.cpp)
  target_link_libraries(qwen3_tts_audio_test PRIVATE gufo_qwen3_tts)
  add_test(NAME qwen3_tts_audio_test COMMAND qwen3_tts_audio_test)
  set_tests_properties(qwen3_tts_audio_test PROPERTIES
    LABELS "cpu;models;qwen3-tts;audio;quality;resampling")

  add_executable(qwen3_tts_config_test
    tests/models/qwen3_tts/config_test.cpp)
  target_link_libraries(qwen3_tts_config_test PRIVATE gufo_qwen3_tts)
  add_test(NAME qwen3_tts_config_test COMMAND qwen3_tts_config_test)

  add_executable(qwen3_tts_tokenizer_test
    tests/models/qwen3_tts/tokenizer_test.cpp)
  target_link_libraries(qwen3_tts_tokenizer_test PRIVATE gufo_qwen3_tts)
  add_test(NAME qwen3_tts_tokenizer_test COMMAND qwen3_tts_tokenizer_test)
  set_tests_properties(qwen3_tts_tokenizer_test PROPERTIES
    SKIP_RETURN_CODE 77)
  set_tests_properties(qwen3_tts_config_test PROPERTIES
    LABELS "cpu;models;qwen3-tts;config;quality")

  add_executable(qwen3_tts_prompt_test
    tests/models/qwen3_tts/prompt_test.cpp)
  target_link_libraries(qwen3_tts_prompt_test PRIVATE gufo_qwen3_tts)
  add_test(NAME qwen3_tts_prompt_test COMMAND qwen3_tts_prompt_test)
  set_tests_properties(qwen3_tts_prompt_test PROPERTIES
    LABELS "cpu;models;qwen3-tts;prompt;quality;exact")

  add_test(NAME qwen3_tts_reference_artifacts_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/models/qwen3_tts/reference/verify_artifacts.py
      --contract
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/models/qwen3_tts/reference_contract.json
      --artifacts
      ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/qwen3_tts)
  set_tests_properties(qwen3_tts_reference_artifacts_test PROPERTIES
    LABELS "cpu;models;qwen3-tts;reference;quality;exact"
    SKIP_RETURN_CODE 77)

  add_executable(minimax_h3_tokenizer_test
    tests/models/minimax_h3_tokenizer_test.cpp)
  target_link_libraries(minimax_h3_tokenizer_test PRIVATE gufo_core)
  add_test(NAME minimax_h3_tokenizer_test COMMAND minimax_h3_tokenizer_test)
  set_tests_properties(minimax_h3_tokenizer_test PROPERTIES
    LABELS "cpu;models;minimax-h3;tokenizer;quality")

  add_executable(minimax_h3_sampling_test
    tests/models/minimax_h3_sampling_test.cpp)
  target_link_libraries(minimax_h3_sampling_test PRIVATE gufo_core)
  add_test(NAME minimax_h3_sampling_test COMMAND minimax_h3_sampling_test)
  set_tests_properties(minimax_h3_sampling_test PROPERTIES
    LABELS "cpu;models;minimax-h3;layout;schedule;rng;quality")

  add_executable(minimax_h3_generation_test
    tests/models/minimax_h3_generation_test.cpp)
  target_link_libraries(minimax_h3_generation_test PRIVATE gufo_core)
  add_test(NAME minimax_h3_generation_test
    COMMAND minimax_h3_generation_test)
  set_tests_properties(minimax_h3_generation_test PROPERTIES
    LABELS "cpu;models;minimax-h3;generation;cli;quality;cancellation")

  add_executable(minimax_h3_video_vae_test
    tests/models/minimax_h3_video_vae_test.cpp)
  target_link_libraries(minimax_h3_video_vae_test PRIVATE gufo_core)
  add_test(NAME minimax_h3_video_vae_plan
    COMMAND minimax_h3_video_vae_test)
  set_tests_properties(minimax_h3_video_vae_plan PROPERTIES
    LABELS "cpu;models;minimax-h3;video;vae;tiling;chunking")

  add_executable(minimax_h3_media_test
    tests/models/minimax_h3_media_test.cpp)
  target_link_libraries(minimax_h3_media_test PRIVATE gufo_core)
  add_test(NAME minimax_h3_media_test COMMAND minimax_h3_media_test)
  set_tests_properties(minimax_h3_media_test PROPERTIES
    LABELS "cpu;models;minimax-h3;video;audio;ffmpeg;mux;quality"
    SKIP_RETURN_CODE 77)

  add_executable(ggml_dequant_test tests/core/ggml_dequant_test.cpp)
  target_link_libraries(ggml_dequant_test PRIVATE gufo_core)

  add_test(NAME ggml_dequant_test COMMAND ggml_dequant_test)
  set_tests_properties(ggml_dequant_test PROPERTIES LABELS "cpu;core;quant")

  add_executable(ggml_gemm_parity_test tests/core/ggml_gemm_parity_test.cpp)
  target_link_libraries(ggml_gemm_parity_test PRIVATE gufo_core)

  add_test(NAME ggml_gemm_parity_test COMMAND ggml_gemm_parity_test)
  set_tests_properties(ggml_gemm_parity_test PROPERTIES LABELS "cpu;core;quant")

  add_executable(exact_token_test tests/quality/exact_token_test.cpp)
  target_link_libraries(exact_token_test PRIVATE gufo_core)
  add_test(NAME exact_token_test COMMAND exact_token_test)
  set_tests_properties(exact_token_test PROPERTIES LABELS "cpu;quality;tokens")

  add_executable(eval_regrade_test tests/quality/eval_regrade_test.cpp)
  target_link_libraries(eval_regrade_test PRIVATE gufo_core)
  add_test(NAME eval_regrade_test COMMAND eval_regrade_test)
  set_tests_properties(eval_regrade_test PROPERTIES LABELS "cpu;quality;eval")

  add_test(NAME eval_http_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/test_eval_http.py
      --gufo $<TARGET_FILE:gufo>
      --data ${CMAKE_CURRENT_SOURCE_DIR}/tests/models/deepseek_v4_flash/fixtures/antirez-ds4.json)
  set_tests_properties(eval_http_test PROPERTIES
    LABELS "cpu;quality;eval;http;cli;server")

  add_executable(eval_http_client_test
    tests/quality/eval_http_client_test.cpp src/eval/http_client.cpp)
  target_include_directories(eval_http_client_test PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(eval_http_client_test PRIVATE CURL::libcurl Threads::Threads)
  add_test(NAME eval_http_client_test COMMAND eval_http_client_test)
  set_tests_properties(eval_http_client_test PROPERTIES LABELS "cpu;quality;eval;http")

  add_executable(kernel_bench_test tests/benchmarks/kernel_bench_test.cpp)
  target_link_libraries(kernel_bench_test PRIVATE gufo_kernel_bench_support)
  add_test(NAME kernel_bench_test COMMAND kernel_bench_test)
  set_tests_properties(kernel_bench_test PROPERTIES
    LABELS "cpu;benchmarks;statistics;json")

  add_executable(hipblaslt_plan_cache_test
    tests/core/hipblaslt_plan_cache_test.cpp)
  target_link_libraries(hipblaslt_plan_cache_test PRIVATE gufo_core)
  add_test(NAME hipblaslt_plan_cache_test COMMAND hipblaslt_plan_cache_test)
  set_tests_properties(hipblaslt_plan_cache_test PROPERTIES
    LABELS "cpu;hipblaslt;cache")

  if (ENGINE_ENABLE_HIP)
    add_executable(qwen3_tts_synthesis_hip_test
      tests/models/qwen3_tts/synthesis_hip_test.cpp)
    target_link_libraries(qwen3_tts_synthesis_hip_test PRIVATE gufo_qwen3_tts)
    add_test(NAME qwen3_tts_synthesis_hip_test COMMAND qwen3_tts_synthesis_hip_test)
    set_tests_properties(qwen3_tts_synthesis_hip_test PROPERTIES
      LABELS "hip;gpu;models;qwen3-tts;quality;external-model;slow" SKIP_RETURN_CODE 77)
    add_executable(qwen3_tts_speech_decoder_hip_test
      tests/models/qwen3_tts/speech_decoder_hip_test.cpp)
    target_link_libraries(qwen3_tts_speech_decoder_hip_test PRIVATE
      gufo_qwen3_tts hip::host)
    add_test(NAME qwen3_tts_speech_decoder_hip_test
      COMMAND qwen3_tts_speech_decoder_hip_test)
    set_tests_properties(qwen3_tts_speech_decoder_hip_test PROPERTIES
      LABELS "hip;gpu;gfx1151;models;qwen3-tts;speech-decoder;audio;quality;external-model;slow;opt-qwen3-tts-device-copy"
      SKIP_RETURN_CODE 77)
    add_executable(qwen3_tts_talker_hip_prefill_test
      tests/models/qwen3_tts/talker_hip_prefill_test.cpp)
    target_link_libraries(qwen3_tts_talker_hip_prefill_test PRIVATE
      gufo_qwen3_tts hip::host)
    add_test(NAME qwen3_tts_talker_hip_prefill_test
      COMMAND qwen3_tts_talker_hip_prefill_test)
    set_tests_properties(qwen3_tts_talker_hip_prefill_test PROPERTIES
      LABELS "hip;gpu;gfx1151;models;qwen3-tts;talker;prefill;quality;external-model;slow"
      SKIP_RETURN_CODE 77)

    add_executable(qwen3_tts_variant_talker_hip_test
      tests/models/qwen3_tts/variant_talker_hip_test.cpp)
    target_link_libraries(qwen3_tts_variant_talker_hip_test PRIVATE
      gufo_qwen3_tts hip::host)
    add_test(NAME qwen3_tts_variant_talker_hip_test
      COMMAND qwen3_tts_variant_talker_hip_test)
    set_tests_properties(qwen3_tts_variant_talker_hip_test PROPERTIES
      LABELS "hip;gpu;gfx1151;models;qwen3-tts;talker;voice-design;base;quality;external-model;slow"
      SKIP_RETURN_CODE 77)

    add_executable(qwen3_tts_base_encoders_hip_test
      tests/models/qwen3_tts/base_encoders_hip_test.cpp)
    target_link_libraries(qwen3_tts_base_encoders_hip_test PRIVATE
      gufo_qwen3_tts hip::host)
    add_test(NAME qwen3_tts_base_encoders_hip_test
      COMMAND qwen3_tts_base_encoders_hip_test)
    set_tests_properties(qwen3_tts_base_encoders_hip_test PROPERTIES
      LABELS "hip;gpu;gfx1151;models;qwen3-tts;base;speech-encoder;speaker-encoder;quality;external-model;slow"
      SKIP_RETURN_CODE 77)

    add_executable(minimax_h3_runtime_hip_test
      tests/models/minimax_h3_runtime_hip_test.cpp)
    target_link_libraries(minimax_h3_runtime_hip_test PRIVATE
      gufo_core hip::host)
    add_test(NAME minimax_h3_runtime_hip_test
      COMMAND minimax_h3_runtime_hip_test)
    set_tests_properties(minimax_h3_runtime_hip_test PROPERTIES
      LABELS "gpu;hip;gfx1151;minimax-h3;external-model;slow"
      SKIP_RETURN_CODE 77)

    add_executable(minimax_h3_prompt_encoder_hip_test
      tests/models/minimax_h3_prompt_encoder_hip_test.hip)
    target_compile_definitions(minimax_h3_prompt_encoder_hip_test PRIVATE
      GUFO_H3_SOURCE_MANIFEST="${CMAKE_CURRENT_SOURCE_DIR}/src/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json")
    target_link_libraries(minimax_h3_prompt_encoder_hip_test PRIVATE
      gufo_core hip::host)
    add_test(NAME minimax_h3_prompt_encoder_analytic
      COMMAND minimax_h3_prompt_encoder_hip_test --analytic)
    set_tests_properties(minimax_h3_prompt_encoder_analytic PROPERTIES
      LABELS "gpu;hip;gfx1151;minimax-h3;prompt-encoder;analytic")
    add_test(NAME minimax_h3_prompt_encoder_real
      COMMAND minimax_h3_prompt_encoder_hip_test --real)
    set_tests_properties(minimax_h3_prompt_encoder_real PROPERTIES
      LABELS "gpu;hip;gfx1151;minimax-h3;prompt-encoder;external-model;slow"
      SKIP_RETURN_CODE 77)

    add_executable(minimax_h3_sampling_hip_test
      tests/models/minimax_h3_sampling_hip_test.hip)
    target_link_libraries(minimax_h3_sampling_hip_test PRIVATE
      gufo_core hip::host)
    add_test(NAME minimax_h3_sampling_hip_test
      COMMAND minimax_h3_sampling_hip_test)
    set_tests_properties(minimax_h3_sampling_hip_test PROPERTIES
      LABELS "gpu;hip;gfx1151;minimax-h3;sampling;euler;analytic"
      SKIP_RETURN_CODE 77)

    add_executable(minimax_h3_dit_hip_test
      tests/models/minimax_h3_dit_hip_test.hip)
    target_compile_definitions(minimax_h3_dit_hip_test PRIVATE
      GUFO_H3_SOURCE_MANIFEST="${CMAKE_CURRENT_SOURCE_DIR}/src/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json")
    target_link_libraries(minimax_h3_dit_hip_test PRIVATE
      gufo_core hip::host)
    add_test(NAME minimax_h3_dit_analytic
      COMMAND minimax_h3_dit_hip_test --analytic)
    set_tests_properties(minimax_h3_dit_analytic PROPERTIES
      LABELS "gpu;hip;gfx1151;minimax-h3;dit;analytic")
    add_test(NAME minimax_h3_dit_block0_real
      COMMAND minimax_h3_dit_hip_test --real)
    set_tests_properties(minimax_h3_dit_block0_real PROPERTIES
      LABELS "gpu;hip;gfx1151;minimax-h3;dit;external-model;slow"
      SKIP_RETURN_CODE 77)

    add_executable(minimax_h3_denoiser_hip_test
      tests/models/minimax_h3_denoiser_hip_test.hip)
    target_compile_definitions(minimax_h3_denoiser_hip_test PRIVATE
      GUFO_H3_SOURCE_MANIFEST="${CMAKE_CURRENT_SOURCE_DIR}/src/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json")
    target_link_libraries(minimax_h3_denoiser_hip_test PRIVATE
      gufo_core hip::host)
    add_test(NAME minimax_h3_denoiser_smoke
      COMMAND minimax_h3_denoiser_hip_test --smoke)
    set_tests_properties(minimax_h3_denoiser_smoke PROPERTIES
      LABELS "gpu;hip;gfx1151;minimax-h3;dit;denoiser;external-model;slow"
      SKIP_RETURN_CODE 77)
    add_test(NAME minimax_h3_denoiser_oracle
      COMMAND minimax_h3_denoiser_hip_test --oracle-forward)
    set_tests_properties(minimax_h3_denoiser_oracle PROPERTIES
      LABELS "gpu;hip;gfx1151;minimax-h3;dit;denoiser;quality;external-model;slow"
      SKIP_RETURN_CODE 77)
    add_test(NAME minimax_h3_denoiser_production_shape_oracle
      COMMAND minimax_h3_denoiser_hip_test --oracle-512-forward)
    set_tests_properties(minimax_h3_denoiser_production_shape_oracle PROPERTIES
      LABELS "gpu;hip;gfx1151;minimax-h3;dit;denoiser;quality;external-model;slow"
      SKIP_RETURN_CODE 77)
    add_test(NAME minimax_h3_denoiser_reuse_smoke
      COMMAND minimax_h3_denoiser_hip_test --reuse-smoke)
    set_tests_properties(minimax_h3_denoiser_reuse_smoke PROPERTIES
      LABELS "gpu;hip;gfx1151;minimax-h3;dit;denoiser;reuse;quality;external-model;slow"
      SKIP_RETURN_CODE 77)

    add_executable(minimax_h3_video_vae_hip_test
      tests/models/minimax_h3_video_vae_hip_test.hip)
    target_compile_definitions(minimax_h3_video_vae_hip_test PRIVATE
      GUFO_H3_SOURCE_MANIFEST="${CMAKE_CURRENT_SOURCE_DIR}/src/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json"
      GUFO_H3_VIDEO_VAE_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/minimax_h3/video-vae-oracle-v1.json")
    target_link_libraries(minimax_h3_video_vae_hip_test PRIVATE
      gufo_core hip::host)
    add_test(NAME minimax_h3_video_vae_real
      COMMAND minimax_h3_video_vae_hip_test)
    set_tests_properties(minimax_h3_video_vae_real PROPERTIES
      LABELS "gpu;hip;gfx1151;minimax-h3;video;vae;quality;external-model;very-slow"
      SKIP_RETURN_CODE 77)

    add_executable(minimax_h3_audio_vae_hip_test
      tests/models/minimax_h3_audio_vae_hip_test.hip)
    target_compile_definitions(minimax_h3_audio_vae_hip_test PRIVATE
      GUFO_H3_SOURCE_MANIFEST="${CMAKE_CURRENT_SOURCE_DIR}/src/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json"
      GUFO_H3_AUDIO_VAE_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/minimax_h3/audio-vae-oracle-v1.json")
    target_link_libraries(minimax_h3_audio_vae_hip_test PRIVATE
      gufo_core hip::host)
    add_test(NAME minimax_h3_audio_vae_primitives
      COMMAND minimax_h3_audio_vae_hip_test --primitives)
    set_tests_properties(minimax_h3_audio_vae_primitives PROPERTIES
      LABELS "gpu;hip;gfx1151;minimax-h3;audio;vae;analytic")
    add_test(NAME minimax_h3_audio_vae_real
      COMMAND minimax_h3_audio_vae_hip_test --real)
    set_tests_properties(minimax_h3_audio_vae_real PROPERTIES
      LABELS "gpu;hip;gfx1151;minimax-h3;audio;vae;quality;external-model;very-slow"
      SKIP_RETURN_CODE 77)

    add_executable(hip_weight_upload_test tests/core/hip_weight_upload_test.cpp)
    target_link_libraries(hip_weight_upload_test PRIVATE gufo_core)
    add_test(NAME hip_weight_upload_test COMMAND hip_weight_upload_test)
    set_tests_properties(hip_weight_upload_test PROPERTIES LABELS "hip;gpu;gfx1151")

    add_executable(hipblaslt_runtime_cache_test
      tests/kernels/hipblaslt_runtime_cache_test.cpp)
    target_link_libraries(hipblaslt_runtime_cache_test PRIVATE
      gufo_core hip::host)
    add_test(NAME hipblaslt_runtime_cache_test
      COMMAND hipblaslt_runtime_cache_test)
    set_tests_properties(hipblaslt_runtime_cache_test PROPERTIES
      LABELS "hip;gpu;hipblaslt;cache")

    add_executable(persistent_deltanet_test
      tests/kernels/persistent_deltanet_test.cpp)
    target_link_libraries(persistent_deltanet_test PRIVATE gufo_core hip::host)
    add_test(NAME persistent_deltanet_test COMMAND persistent_deltanet_test)
    set_tests_properties(persistent_deltanet_test PROPERTIES
      LABELS "hip;gpu;kernels;deltanet;speculative")

    add_executable(gfx1151_acceptance_test tests/benchmarks/gfx1151_acceptance_test.cpp)
    target_link_libraries(gfx1151_acceptance_test PRIVATE gufo_core hip::host roc::hipblas roc::rocblas)
    add_test(NAME gfx1151_acceptance_test COMMAND gfx1151_acceptance_test)
    set_tests_properties(gfx1151_acceptance_test PROPERTIES LABELS "hip;gpu;benchmarks")

    add_executable(speculative_verification_test tests/core/test_speculative_verification.cpp)
    target_link_libraries(speculative_verification_test PRIVATE gufo_core hip::host roc::hipblas roc::rocblas)
    add_test(NAME speculative_verification_test COMMAND speculative_verification_test)
    set_tests_properties(speculative_verification_test PROPERTIES LABELS "hip;gpu;speculative")
  endif()

  if (ENGINE_ENABLE_HIP)
    add_test(NAME gufo_gpu_probe COMMAND gufo)
    set_tests_properties(gufo_gpu_probe PROPERTIES
      LABELS "platform;probe;hip;gpu")
  else()
    add_test(NAME gufo_cpu_probe COMMAND gufo)
    set_tests_properties(gufo_cpu_probe PROPERTIES
      LABELS "platform;probe;cpu")
  endif()
endif()

if (BUILD_TESTING)
  include(cmake/Checks.cmake)
endif()
