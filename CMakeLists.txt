cmake_minimum_required(VERSION 3.21)
project(gufo LANGUAGES C CXX)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(GUFO_VERSION "development" CACHE STRING "Gufo build revision")
include_directories(${CMAKE_SOURCE_DIR})
find_package(ICU REQUIRED COMPONENTS uc)
find_package(Threads REQUIRED)
find_package(CURL REQUIRED)
set(GUFO_FFMPEG_EXECUTABLE "ffmpeg" CACHE STRING
  "FFmpeg executable used for generated media")
set(GUFO_FFPROBE_EXECUTABLE "ffprobe" CACHE STRING
  "FFprobe executable used to validate generated media")

# ROCm/HIP GPU backend — compiled natively for the Strix Halo gfx1151 target.
option(ENGINE_ENABLE_HIP "Build gufo with ROCm/HIP GPU backend" OFF)

# ROCm/HIP GPU discovery — compiled natively for the Strix Halo gfx1151 target.
# Dependencies are supplied by the Nix environment via $ROCM_PATH.
# Host ambient /usr fallbacks are rejected.
if (NOT DEFINED ENV{ROCM_PATH})
  if (EXISTS /opt/rocm)
    set(ENV{ROCM_PATH} /opt/rocm)
  endif()
endif()

if (DEFINED ENV{ROCM_PATH})
  list(APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH})
  list(APPEND CMAKE_PREFIX_PATH "$ENV{ROCM_PATH}/lib64/cmake")
endif()

# Forward GPU_TARGETS (set by the nix package) to CMAKE_HIP_ARCHITECTURES.
# e.g. -DGPU_TARGETS=gfx1151
if (GPU_TARGETS AND NOT CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES "${GPU_TARGETS}" CACHE STRING "HIP architectures")
endif()

if (ENGINE_ENABLE_HIP)
  enable_language(HIP)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  find_package(hip REQUIRED)
  find_package(hipblas REQUIRED)
  find_package(hipblaslt REQUIRED)
  find_package(rocblas REQUIRED)
  find_package(aotriton CONFIG REQUIRED)
  find_path(HIPCUB_INCLUDE_DIR hipcub/hipcub.hpp
    HINTS "$ENV{GUFO_HIPCUB_ROOT}/include"
    REQUIRED)
  find_path(ROCPRIM_INCLUDE_DIR rocprim/rocprim.hpp
    HINTS "$ENV{GUFO_ROCPRIM_ROOT}/include"
    REQUIRED)
  find_path(ROCWMMA_INCLUDE_DIR rocwmma/rocwmma.hpp
    HINTS "$ENV{GUFO_ROCWMMA_ROOT}/include"
    REQUIRED)
  find_path(ROCTX_INCLUDE_DIR rocprofiler-sdk-roctx/roctx.h REQUIRED)
  find_library(ROCTX_LIBRARY NAMES rocprofiler-sdk-roctx REQUIRED)
  find_path(COMPOSABLE_KERNEL_INCLUDE_DIR ck/ck.hpp REQUIRED)

  message(STATUS "gufo: ROCm/HIP enabled, architectures=${CMAKE_HIP_ARCHITECTURES}")
  message(STATUS "gufo: ROCM_PATH=$ENV{ROCM_PATH}")

  set(GUFO_H3_ATTENTION_GENERATED_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/generated")
  set(GUFO_H3_ATTENTION_GENERATED_HEADER
    "${GUFO_H3_ATTENTION_GENERATED_DIR}/h3_attention_gfx1151.hpp")
  add_custom_command(
    OUTPUT "${GUFO_H3_ATTENTION_GENERATED_HEADER}"
    COMMAND "${CMAKE_COMMAND}" -E env
      "HOME=${CMAKE_CURRENT_BINARY_DIR}/triton-home"
      "XDG_CACHE_HOME=${CMAKE_CURRENT_BINARY_DIR}/triton-cache"
      "TRITON_CACHE_DIR=${CMAKE_CURRENT_BINARY_DIR}/triton-cache"
      "TRITON_F32_DEFAULT=ieee"
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/gufo/compile_h3_attention.py"
      --output "${GUFO_H3_ATTENTION_GENERATED_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/gufo/compile_h3_attention.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/gufo/h3_attention_kernel.py"
    COMMENT "Compiling spill-free gfx1151 MiniMax H3 attention"
    VERBATIM)
endif()

if (ENGINE_ENABLE_HIP)
  add_subdirectory(src/models/deepseek_v4_flash)
endif()

# Qwen3-TTS — CPU reference + HIP kernels (CustomVoice 12Hz).
add_subdirectory(src/models/qwen3_tts)

# XRT NPU shim — AMD XDNA2 userspace runtime over the amdxdna kernel driver.
option(ENGINE_ENABLE_XRT "Build gufo with the XRT NPU shim" OFF)

if (ENGINE_ENABLE_XRT)
  # XRT install prefix: $XRT_PATH or the default /opt/xilinx/xrt layout.
  find_path(XRT_INCLUDE_DIR xrt/experimental/xrt_system.h
    PATHS "$ENV{XRT_PATH}/include" "/opt/xilinx/xrt/include"
    NO_DEFAULT_PATH)
  find_library(XRT_COREUTIL_LIB xrt_coreutil
    PATHS "$ENV{XRT_PATH}/lib" "/opt/xilinx/xrt/lib"
    NO_DEFAULT_PATH)
  find_library(UUID_LIBRARY uuid REQUIRED)

  if (NOT XRT_INCLUDE_DIR OR NOT XRT_COREUTIL_LIB)
    message(FATAL_ERROR "XRT not found (ENGINE_ENABLE_XRT=ON). Please enter the Nix development environment (nix develop)")
  endif()
  if (DEFINED ENV{GUFO_AIE_SMOKE_PROGRAM_DIR}
      AND NOT "$ENV{GUFO_AIE_SMOKE_PROGRAM_DIR}" STREQUAL "")
    set(GUFO_AIE_SMOKE_PROGRAM_DIR "$ENV{GUFO_AIE_SMOKE_PROGRAM_DIR}"
      CACHE PATH "Directory containing the reviewed XDNA2 smoke artifacts"
      FORCE)
  else()
    set(GUFO_AIE_SMOKE_PROGRAM_DIR "${GUFO_AIE_SMOKE_PROGRAM_DIR}"
      CACHE PATH "Directory containing the reviewed XDNA2 smoke artifacts")
  endif()
  set(GUFO_AIE_SMOKE_INCLUDE_DIR "$ENV{GUFO_AIE_SMOKE_ROOT}/include")
  if (DEFINED ENV{GUFO_AIE_QWEN_MTP_RMSNORM_PROGRAM_DIR}
      AND NOT "$ENV{GUFO_AIE_QWEN_MTP_RMSNORM_PROGRAM_DIR}" STREQUAL "")
    set(GUFO_AIE_QWEN_MTP_RMSNORM_PROGRAM_DIR
      "$ENV{GUFO_AIE_QWEN_MTP_RMSNORM_PROGRAM_DIR}"
      CACHE PATH "Directory containing the reviewed Qwen MTP RMSNorm artifacts"
      FORCE)
  else()
    set(GUFO_AIE_QWEN_MTP_RMSNORM_PROGRAM_DIR
      "${GUFO_AIE_QWEN_MTP_RMSNORM_PROGRAM_DIR}"
      CACHE PATH "Directory containing the reviewed Qwen MTP RMSNorm artifacts")
  endif()
  set(GUFO_AIE_QWEN_MTP_RMSNORM_INCLUDE_DIR
    "$ENV{GUFO_AIE_QWEN_MTP_RMSNORM_ROOT}/include")
  if (DEFINED ENV{GUFO_AIE_QWEN_MTP_EH_PROJ_PROGRAM_DIR}
      AND NOT "$ENV{GUFO_AIE_QWEN_MTP_EH_PROJ_PROGRAM_DIR}" STREQUAL "")
    set(GUFO_AIE_QWEN_MTP_EH_PROJ_PROGRAM_DIR
      "$ENV{GUFO_AIE_QWEN_MTP_EH_PROJ_PROGRAM_DIR}"
      CACHE PATH "Directory containing the reviewed Qwen MTP eh_proj artifacts"
      FORCE)
  else()
    set(GUFO_AIE_QWEN_MTP_EH_PROJ_PROGRAM_DIR
      "${GUFO_AIE_QWEN_MTP_EH_PROJ_PROGRAM_DIR}"
      CACHE PATH "Directory containing the reviewed Qwen MTP eh_proj artifacts")
  endif()
  set(GUFO_AIE_QWEN_MTP_EH_PROJ_INCLUDE_DIR
    "$ENV{GUFO_AIE_QWEN_MTP_EH_PROJ_ROOT}/include")
  if (NOT GUFO_AIE_SMOKE_PROGRAM_DIR OR NOT GUFO_AIE_SMOKE_INCLUDE_DIR
      OR NOT GUFO_AIE_QWEN_MTP_RMSNORM_PROGRAM_DIR
      OR NOT GUFO_AIE_QWEN_MTP_RMSNORM_INCLUDE_DIR
      OR NOT GUFO_AIE_QWEN_MTP_EH_PROJ_PROGRAM_DIR
      OR NOT GUFO_AIE_QWEN_MTP_EH_PROJ_INCLUDE_DIR)
    message(FATAL_ERROR "XDNA2 program artifacts not found. Use the Nix development environment")
  endif()
  message(STATUS "gufo: XRT NPU shim enabled, include=${XRT_INCLUDE_DIR}")
endif()

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
  src/core/diagnostics/bandwidth.cpp
  src/core/hip/bandwidth.cpp
  src/core/xdna2/bandwidth.cpp
  src/core/xdna2/device.cpp
  src/core/xdna2/smoke.cpp
  src/core/diagnostics/report.cpp
  src/cli/diagnose/diagnose.cpp
)
target_include_directories(gufo_diagnostics PUBLIC ${CMAKE_SOURCE_DIR})

if (ENGINE_ENABLE_HIP)
  target_compile_definitions(gufo_diagnostics PRIVATE ENGINE_ENABLE_HIP=1)
  target_link_libraries(gufo_diagnostics PRIVATE hip::host roc::hipblas roc::rocblas)
endif()

if (ENGINE_ENABLE_XRT)
  target_compile_definitions(gufo_diagnostics PRIVATE
    ENGINE_ENABLE_XRT=1
    GUFO_AIE_SMOKE_PROGRAM_DIR="${GUFO_AIE_SMOKE_PROGRAM_DIR}")
  target_include_directories(gufo_diagnostics SYSTEM PRIVATE
    ${XRT_INCLUDE_DIR}
    ${GUFO_AIE_SMOKE_INCLUDE_DIR})
  target_link_libraries(gufo_diagnostics PRIVATE
    ${XRT_COREUTIL_LIB}
    ${UUID_LIBRARY})
endif()

add_library(gufo_core STATIC
  src/core/model_registry.cpp
  src/core/gguf_reader.cpp
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
target_link_libraries(gufo_core PUBLIC ICU::uc Threads::Threads)

add_subdirectory(src/models/qwen)

target_link_libraries(gufo_qwen3_tts PUBLIC gufo_core)

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
    src/models/minimax_h3/runtime_hip.cpp
    src/models/minimax_h3/prompt_encoder.hip
    src/models/minimax_h3/sampling.hip
    src/models/minimax_h3/video_vae.hip
    "${GUFO_H3_ATTENTION_GENERATED_HEADER}"
    src/core/speculative/draft_heads.cpp
    src/core/speculative/self_speculative.cpp
    src/core/speculative/prompt_lookup_backend.cpp
    src/core/speculative/speculative_verifier.cpp
    src/core/heterogeneous/npu_drafter.cpp
  )
  target_compile_definitions(gufo_core PUBLIC ENGINE_ENABLE_HIP=1)
  target_compile_definitions(gufo_core PRIVATE CK_ENABLE_BF16=1)
  target_include_directories(gufo_core SYSTEM PRIVATE
    ${COMPOSABLE_KERNEL_INCLUDE_DIR})
  target_include_directories(gufo_core PRIVATE
    "${GUFO_H3_ATTENTION_GENERATED_DIR}")
  target_link_libraries(gufo_core PUBLIC hip::host roc::hipblas
    roc::hipblaslt roc::rocblas aotriton_v2)
endif()

add_executable(gufo
  src/cli/main.cpp
  src/cli/bench/bench.cpp
  src/cli/eval/eval.cpp
  src/cli/prompt/prompt.cpp
  src/cli/video/video.cpp
  src/cli/serve/serve.cpp
  src/cli/serve/audio_tts_api.cpp
  src/cli/serve/tts_service.cpp
  src/cli/serve/video_api.cpp
  src/cli/serve/video_jobs.cpp
  src/cli/serve/continuation_cache.cpp
  src/cli/serve/text_model_runner.cpp
  src/cli/serve/text_generation_scheduler.cpp
  src/cli/serve/inference_backend.cpp
  src/cli/serve/http_server.cpp
  src/cli/serve/logging.cpp
  src/cli/serve/openai_chat.cpp
  src/eval/http_client.cpp
)
target_link_libraries(gufo PRIVATE gufo_diagnostics gufo_core CURL::libcurl)
target_compile_definitions(gufo PRIVATE
  GUFO_VERSION="${GUFO_VERSION}"
  GUFO_EVAL_SOURCE_DATA="${CMAKE_CURRENT_SOURCE_DIR}/tests/quality/antirez-ds4.json"
  GUFO_EVAL_INSTALLED_DATA="${CMAKE_INSTALL_PREFIX}/share/gufo/eval/antirez-ds4.json"
  GUFO_H3_SOURCE_MANIFEST="${CMAKE_CURRENT_SOURCE_DIR}/src/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json"
  GUFO_H3_INSTALLED_SOURCE_MANIFEST="${CMAKE_INSTALL_PREFIX}/share/gufo/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json"
  GUFO_QWEN3_TTS_REFERENCE_RUNNER="${CMAKE_CURRENT_SOURCE_DIR}/src/models/qwen3_tts/reference/run_official.py"
  GUFO_QWEN3_TTS_INSTALLED_REFERENCE_RUNNER="${CMAKE_INSTALL_PREFIX}/share/gufo/models/qwen3_tts/run_official.py")

if (ENGINE_ENABLE_HIP)
  target_link_libraries(gufo PRIVATE gufo_deepseek_v4_flash)
endif()
target_link_libraries(gufo PRIVATE gufo_qwen3_tts)

add_custom_target(gufo-server ALL
  COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE_NAME:gufo> ${CMAKE_CURRENT_BINARY_DIR}/gufo-server
  DEPENDS gufo
)
add_library(gufo_kernel_bench_support STATIC
  src/cli/bench/kernel_bench.cpp
)
target_include_directories(gufo_kernel_bench_support PUBLIC ${CMAKE_SOURCE_DIR})

if (ENGINE_ENABLE_HIP)
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

  add_executable(benchmark_ssm_replay tools/bench/benchmark_ssm_replay.cpp)
  target_link_libraries(benchmark_ssm_replay PRIVATE gufo_core hip::host)

  option(GUFO_BUILD_DIAG_TOOLS "Build standalone diagnostic tools" OFF)
  if(GUFO_BUILD_DIAG_TOOLS)
    add_executable(gguf_dump_types tools/quant/gguf_dump_types.cpp)
    target_link_libraries(gguf_dump_types PRIVATE gufo_core)

    add_executable(wmma_layout_test tools/bench/wmma_layout_test.hip)
    target_link_libraries(wmma_layout_test PRIVATE gufo_core hip::device)
  endif()
endif()

install(TARGETS gufo DESTINATION bin)
install(FILES
  src/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json
  DESTINATION share/gufo/models/minimax_h3)
install(FILES
  src/models/qwen3_tts/reference/run_official.py
  DESTINATION share/gufo/models/qwen3_tts)
install(FILES
  tests/quality/antirez-ds4.json
  DESTINATION share/gufo/eval)
if (ENGINE_ENABLE_HIP)
  install(TARGETS gufo-kernel-bench tune_hipblaslt benchmark_ssm_replay
    DESTINATION bin)
endif()

# CTest integration
include(CTest)

if (BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  add_subdirectory(tests/models/qwen)

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

  add_executable(openai_chat_test
    tests/cli/openai_chat_test.cpp
    src/cli/serve/openai_chat.cpp
  )
  target_link_libraries(openai_chat_test PRIVATE gufo_core)
  add_test(NAME openai_chat_test COMMAND openai_chat_test)
  set_tests_properties(openai_chat_test PROPERTIES
    LABELS "cpu;server;api;openai;streaming;tools")

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
    src/cli/serve/text_model_runner.cpp)
  target_link_libraries(text_model_runner_test PRIVATE gufo_core)
  add_test(NAME text_model_runner_test COMMAND text_model_runner_test)
  set_tests_properties(text_model_runner_test PROPERTIES
    LABELS "cpu;server;runner;cache;state;lifecycle")

  add_executable(text_generation_scheduler_test
    tests/cli/text_generation_scheduler_test.cpp
    src/cli/serve/continuation_cache.cpp
    src/cli/serve/text_model_runner.cpp
    src/cli/serve/text_generation_scheduler.cpp)
  target_link_libraries(text_generation_scheduler_test PRIVATE gufo_core)
  add_test(NAME text_generation_scheduler_test
    COMMAND text_generation_scheduler_test)
  set_tests_properties(text_generation_scheduler_test PROPERTIES
    LABELS "cpu;server;scheduler;queue;state;lifecycle;cancellation")

  add_executable(prompt_cli_test
    tests/cli/prompt_test.cpp
    src/cli/prompt/prompt.cpp
  )
  target_link_libraries(prompt_cli_test PRIVATE gufo_core)
  if (ENGINE_ENABLE_HIP)
    target_link_libraries(prompt_cli_test PRIVATE gufo_deepseek_v4_flash)
  endif()

  add_test(NAME prompt_cli_test COMMAND prompt_cli_test)
  set_tests_properties(prompt_cli_test PROPERTIES LABELS "cpu;server;cli")

  add_executable(bench_cli_test
    tests/cli/bench_test.cpp
    src/cli/bench/bench.cpp
  )
  target_link_libraries(bench_cli_test PRIVATE gufo_core)
  if (ENGINE_ENABLE_HIP)
    target_link_libraries(bench_cli_test PRIVATE gufo_deepseek_v4_flash)
  endif()

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
  target_link_libraries(video_jobs_test PRIVATE gufo_core)
  add_test(NAME video_jobs_test COMMAND video_jobs_test)
  set_tests_properties(video_jobs_test PROPERTIES
    LABELS "cpu;server;video;minimax-h3;queue;cancellation;quality")

  add_executable(video_api_test
    tests/cli/video_api_test.cpp
    src/cli/serve/video_api.cpp
    src/cli/serve/video_jobs.cpp
  )
  target_link_libraries(video_api_test PRIVATE gufo_core)
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

  add_executable(bandwidth_test tests/diagnostics/bandwidth_test.cpp)
  target_compile_definitions(bandwidth_test PRIVATE TEST_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/diagnostics")
  target_link_libraries(bandwidth_test PRIVATE gufo_diagnostics)

  add_test(NAME bandwidth_test COMMAND bandwidth_test)
  set_tests_properties(bandwidth_test PROPERTIES LABELS "cpu;diagnostics;bandwidth;bandwidth_schema;bandwidth_correctness")

  if (ENGINE_ENABLE_XRT)
    add_executable(xdna2_xrt_program_test
      tests/device/xdna2_xrt_program_test.cpp)
    target_compile_definitions(xdna2_xrt_program_test PRIVATE
      GUFO_AIE_SMOKE_PROGRAM_DIR="${GUFO_AIE_SMOKE_PROGRAM_DIR}")
    target_link_libraries(xdna2_xrt_program_test PRIVATE gufo_diagnostics)
    add_test(NAME xdna2_xrt_program_test COMMAND xdna2_xrt_program_test)
    set_tests_properties(xdna2_xrt_program_test PROPERTIES
      LABELS "npu;xdna2;xrt;xrt-hardware")

  endif()

  add_executable(gguf_reader_test tests/core/gguf_reader_test.cpp)
  target_link_libraries(gguf_reader_test PRIVATE gufo_core)

  add_test(NAME gguf_reader_test COMMAND gguf_reader_test)
  set_tests_properties(gguf_reader_test PROPERTIES LABELS "cpu;core;gguf;model_config")

  add_executable(model_registry_test tests/core/model_registry_test.cpp)
  target_link_libraries(model_registry_test PRIVATE gufo_core)

  add_test(NAME model_registry_test COMMAND model_registry_test)
  set_tests_properties(model_registry_test PROPERTIES
    LABELS "cpu;core;model_registry;minimax-h3")

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

  add_executable(eval_dataset_test tests/quality/eval_dataset_test.cpp)
  target_compile_definitions(eval_dataset_test PRIVATE
    GUFO_EVAL_TEST_DATA="${CMAKE_CURRENT_SOURCE_DIR}/tests/quality/antirez-ds4.json")
  target_link_libraries(eval_dataset_test PRIVATE gufo_core)
  add_test(NAME eval_dataset_test COMMAND eval_dataset_test)
  set_tests_properties(eval_dataset_test PROPERTIES
    LABELS "cpu;quality;eval;dataset;extractor")

  add_test(NAME eval_dataset_audit_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/test_eval_dataset.py)
  set_tests_properties(eval_dataset_audit_test PROPERTIES
    LABELS "cpu;quality;eval;dataset;provenance;license")

  add_test(NAME eval_http_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/test_eval_http.py
      --gufo $<TARGET_FILE:gufo>
      --data ${CMAKE_CURRENT_SOURCE_DIR}/tests/quality/antirez-ds4.json)
  set_tests_properties(eval_http_test PROPERTIES
    LABELS "cpu;quality;eval;http;cli;server")

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
    add_executable(qwen3_tts_speech_decoder_hip_test
      tests/models/qwen3_tts/speech_decoder_hip_test.cpp)
    target_link_libraries(qwen3_tts_speech_decoder_hip_test PRIVATE
      gufo_qwen3_tts hip::host)
    add_test(NAME qwen3_tts_speech_decoder_hip_test
      COMMAND qwen3_tts_speech_decoder_hip_test)
    set_tests_properties(qwen3_tts_speech_decoder_hip_test PROPERTIES
      LABELS "hip;gpu;gfx1151;models;qwen3-tts;speech-decoder;audio;quality;external-model;slow;opt-qwen3-tts-device-copy"
      SKIP_RETURN_CODE 77)
    add_test(NAME qwen3_tts_speech_decoder_mapped_hip_test
      COMMAND qwen3_tts_speech_decoder_hip_test)
    set_tests_properties(qwen3_tts_speech_decoder_mapped_hip_test PROPERTIES
      ENVIRONMENT "GUFO_QWEN3_TTS_WEIGHT_MODE=mapped"
      LABELS "hip;gpu;gfx1151;models;qwen3-tts;speech-decoder;audio;quality;external-model;slow;opt-qwen3-tts-device-copy;reference"
      SKIP_RETURN_CODE 77)
    add_test(NAME qwen3_tts_speech_decoder_precomputed_snake_hip_test
      COMMAND qwen3_tts_speech_decoder_hip_test)
    set_tests_properties(
      qwen3_tts_speech_decoder_precomputed_snake_hip_test PROPERTIES
      ENVIRONMENT "GUFO_QWEN3_TTS_PRECOMPUTE_SNAKE=1"
      LABELS "hip;gpu;gfx1151;models;qwen3-tts;speech-decoder;audio;quality;external-model;slow;opt-qwen3-tts-snake;rejected"
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

    add_executable(deepseek_v4_flash_engine_test
      tests/models/deepseek_v4_flash_engine_test.cpp)
    target_link_libraries(deepseek_v4_flash_engine_test PRIVATE
      gufo_deepseek_v4_flash)
    add_test(NAME deepseek_v4_flash_engine_test
      COMMAND deepseek_v4_flash_engine_test)
    set_tests_properties(deepseek_v4_flash_engine_test PROPERTIES
      LABELS "gpu;hip;gfx1151;deepseek-v4-flash;external-model;slow"
      SKIP_RETURN_CODE 77)

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

    add_executable(inference_backend_gpu_test
      tests/cli/inference_backend_gpu_test.cpp
      src/cli/serve/continuation_cache.cpp
      src/cli/serve/text_model_runner.cpp
      src/cli/serve/text_generation_scheduler.cpp
      src/cli/serve/inference_backend.cpp)
    target_link_libraries(inference_backend_gpu_test PRIVATE
      gufo_core gufo_deepseek_v4_flash hip::host)
    add_test(NAME inference_backend_gpu_test COMMAND inference_backend_gpu_test)
    set_tests_properties(inference_backend_gpu_test PROPERTIES
      LABELS "hip;gpu;server;correctness"
      SKIP_RETURN_CODE 77)

    add_executable(deepseek_v4_flash_backend_test
      tests/cli/deepseek_v4_flash_backend_test.cpp
      src/cli/serve/continuation_cache.cpp
      src/cli/serve/text_model_runner.cpp
      src/cli/serve/text_generation_scheduler.cpp
      src/cli/serve/inference_backend.cpp)
    target_link_libraries(deepseek_v4_flash_backend_test PRIVATE
      gufo_core gufo_deepseek_v4_flash hip::host)
    add_test(NAME deepseek_v4_flash_backend_test
      COMMAND deepseek_v4_flash_backend_test)
    set_tests_properties(deepseek_v4_flash_backend_test PROPERTIES
      LABELS "hip;gpu;server;correctness;deepseek-v4-flash;external-model;slow"
      SKIP_RETURN_CODE 77)

    add_executable(gfx1151_acceptance_test tests/benchmarks/gfx1151_acceptance_test.cpp)
    target_link_libraries(gfx1151_acceptance_test PRIVATE gufo_core hip::host roc::hipblas roc::rocblas)
    add_test(NAME gfx1151_acceptance_test COMMAND gfx1151_acceptance_test)
    set_tests_properties(gfx1151_acceptance_test PROPERTIES LABELS "hip;gpu;benchmarks")

    add_executable(speculative_verification_test tests/core/test_speculative_verification.cpp)
    target_link_libraries(speculative_verification_test PRIVATE gufo_core hip::host roc::hipblas roc::rocblas)
    add_test(NAME speculative_verification_test COMMAND speculative_verification_test)
    set_tests_properties(speculative_verification_test PROPERTIES LABELS "hip;gpu;speculative")

    add_executable(draft_heads_test tests/core/draft_heads_test.cpp)
    target_link_libraries(draft_heads_test PRIVATE gufo_core hip::host roc::hipblas roc::rocblas)
    add_test(NAME draft_heads_test COMMAND draft_heads_test)
    set_tests_properties(draft_heads_test PROPERTIES LABELS "hip;gpu;draft_heads;speculative")

    add_executable(npu_draft_fallback_test
      tests/core/npu_draft_fallback_test.cpp)
    target_link_libraries(npu_draft_fallback_test PRIVATE gufo_core hip::host roc::hipblas roc::rocblas)
    if (ENGINE_ENABLE_XRT)
      target_link_libraries(npu_draft_fallback_test PRIVATE ${XRT_COREUTIL_LIB})
    endif()
    add_test(NAME npu_draft_fallback_test COMMAND npu_draft_fallback_test)
    set_tests_properties(npu_draft_fallback_test PROPERTIES
      LABELS "host;speculative;lookup;npu-fallback")

    add_executable(prompt_lookup_test tests/core/prompt_lookup_test.cpp)
    target_link_libraries(prompt_lookup_test PRIVATE gufo_core hip::host roc::hipblas roc::rocblas)
    add_test(NAME prompt_lookup_test COMMAND prompt_lookup_test)
    set_tests_properties(prompt_lookup_test PROPERTIES LABELS "speculative;lookup;pld")

  endif()

  if (ENGINE_ENABLE_HIP AND ENGINE_ENABLE_XRT)
    add_test(NAME gufo_gpu_xdna2_probe COMMAND gufo)
    set_tests_properties(gufo_gpu_xdna2_probe PROPERTIES
      LABELS "platform;probe;xdna2;xrt;hip;gpu")
  elseif (ENGINE_ENABLE_HIP)
    add_test(NAME gufo_gpu_probe COMMAND gufo)
    set_tests_properties(gufo_gpu_probe PROPERTIES
      LABELS "platform;probe;hip;gpu")
  elseif (ENGINE_ENABLE_XRT)
    add_test(NAME gufo_xdna2_probe COMMAND gufo)
    set_tests_properties(gufo_xdna2_probe PROPERTIES
      LABELS "platform;probe;xdna2;xrt")
  else()
    add_test(NAME gufo_cpu_probe COMMAND gufo)
    set_tests_properties(gufo_cpu_probe PROPERTIES
      LABELS "platform;probe;cpu")
  endif()
endif()

# Static analysis and formatting targets
include(cmake/StaticAnalysis.cmake)
