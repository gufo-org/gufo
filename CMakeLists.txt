cmake_minimum_required(VERSION 3.21)
project(strix LANGUAGES C CXX)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(STRIX_VERSION "development" CACHE STRING "Strix build revision")
include_directories(${CMAKE_SOURCE_DIR})
find_package(ICU REQUIRED COMPONENTS uc)
find_package(Threads REQUIRED)
set(STRIX_FFMPEG_EXECUTABLE "ffmpeg" CACHE STRING
  "FFmpeg executable used for generated media")
set(STRIX_FFPROBE_EXECUTABLE "ffprobe" CACHE STRING
  "FFprobe executable used to validate generated media")

# ROCm/HIP GPU backend — compiled natively for the Strix Halo gfx1151 target.
option(ENGINE_ENABLE_HIP "Build strix with ROCm/HIP GPU backend" OFF)

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
    HINTS "$ENV{STRIX_HIPCUB_ROOT}/include"
    REQUIRED)
  find_path(ROCPRIM_INCLUDE_DIR rocprim/rocprim.hpp
    HINTS "$ENV{STRIX_ROCPRIM_ROOT}/include"
    REQUIRED)
  find_path(ROCWMMA_INCLUDE_DIR rocwmma/rocwmma.hpp
    HINTS "$ENV{STRIX_ROCWMMA_ROOT}/include"
    REQUIRED)
  find_path(ROCTX_INCLUDE_DIR rocprofiler-sdk-roctx/roctx.h REQUIRED)
  find_library(ROCTX_LIBRARY NAMES rocprofiler-sdk-roctx REQUIRED)
  find_path(COMPOSABLE_KERNEL_INCLUDE_DIR ck/ck.hpp REQUIRED)

  message(STATUS "strix: ROCm/HIP enabled, architectures=${CMAKE_HIP_ARCHITECTURES}")
  message(STATUS "strix: ROCM_PATH=$ENV{ROCM_PATH}")

  set(STRIX_H3_ATTENTION_GENERATED_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/generated")
  set(STRIX_H3_ATTENTION_GENERATED_HEADER
    "${STRIX_H3_ATTENTION_GENERATED_DIR}/h3_attention_gfx1151.hpp")
  add_custom_command(
    OUTPUT "${STRIX_H3_ATTENTION_GENERATED_HEADER}"
    COMMAND "${CMAKE_COMMAND}" -E env
      "HOME=${CMAKE_CURRENT_BINARY_DIR}/triton-home"
      "XDG_CACHE_HOME=${CMAKE_CURRENT_BINARY_DIR}/triton-cache"
      "TRITON_CACHE_DIR=${CMAKE_CURRENT_BINARY_DIR}/triton-cache"
      "TRITON_F32_DEFAULT=ieee"
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/strix/compile_h3_attention.py"
      --output "${STRIX_H3_ATTENTION_GENERATED_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/strix/compile_h3_attention.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/strix/h3_attention_kernel.py"
    COMMENT "Compiling spill-free gfx1151 MiniMax H3 attention"
    VERBATIM)
endif()

if (ENGINE_ENABLE_HIP)
  add_subdirectory(src/models/deepseek_v4_flash)
endif()

# Qwen3-TTS — CPU reference + HIP kernels (CustomVoice 12Hz).
add_subdirectory(src/models/qwen3_tts)

# XRT NPU shim — AMD XDNA2 userspace runtime over the amdxdna kernel driver.
option(ENGINE_ENABLE_XRT "Build strix with the XRT NPU shim" OFF)

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
  if (DEFINED ENV{STRIX_AIE_SMOKE_PROGRAM_DIR}
      AND NOT "$ENV{STRIX_AIE_SMOKE_PROGRAM_DIR}" STREQUAL "")
    set(STRIX_AIE_SMOKE_PROGRAM_DIR "$ENV{STRIX_AIE_SMOKE_PROGRAM_DIR}"
      CACHE PATH "Directory containing the reviewed XDNA2 smoke artifacts"
      FORCE)
  else()
    set(STRIX_AIE_SMOKE_PROGRAM_DIR "${STRIX_AIE_SMOKE_PROGRAM_DIR}"
      CACHE PATH "Directory containing the reviewed XDNA2 smoke artifacts")
  endif()
  set(STRIX_AIE_SMOKE_INCLUDE_DIR "$ENV{STRIX_AIE_SMOKE_ROOT}/include")
  if (DEFINED ENV{STRIX_AIE_QWEN_MTP_RMSNORM_PROGRAM_DIR}
      AND NOT "$ENV{STRIX_AIE_QWEN_MTP_RMSNORM_PROGRAM_DIR}" STREQUAL "")
    set(STRIX_AIE_QWEN_MTP_RMSNORM_PROGRAM_DIR
      "$ENV{STRIX_AIE_QWEN_MTP_RMSNORM_PROGRAM_DIR}"
      CACHE PATH "Directory containing the reviewed Qwen MTP RMSNorm artifacts"
      FORCE)
  else()
    set(STRIX_AIE_QWEN_MTP_RMSNORM_PROGRAM_DIR
      "${STRIX_AIE_QWEN_MTP_RMSNORM_PROGRAM_DIR}"
      CACHE PATH "Directory containing the reviewed Qwen MTP RMSNorm artifacts")
  endif()
  set(STRIX_AIE_QWEN_MTP_RMSNORM_INCLUDE_DIR
    "$ENV{STRIX_AIE_QWEN_MTP_RMSNORM_ROOT}/include")
  if (DEFINED ENV{STRIX_AIE_QWEN_MTP_EH_PROJ_PROGRAM_DIR}
      AND NOT "$ENV{STRIX_AIE_QWEN_MTP_EH_PROJ_PROGRAM_DIR}" STREQUAL "")
    set(STRIX_AIE_QWEN_MTP_EH_PROJ_PROGRAM_DIR
      "$ENV{STRIX_AIE_QWEN_MTP_EH_PROJ_PROGRAM_DIR}"
      CACHE PATH "Directory containing the reviewed Qwen MTP eh_proj artifacts"
      FORCE)
  else()
    set(STRIX_AIE_QWEN_MTP_EH_PROJ_PROGRAM_DIR
      "${STRIX_AIE_QWEN_MTP_EH_PROJ_PROGRAM_DIR}"
      CACHE PATH "Directory containing the reviewed Qwen MTP eh_proj artifacts")
  endif()
  set(STRIX_AIE_QWEN_MTP_EH_PROJ_INCLUDE_DIR
    "$ENV{STRIX_AIE_QWEN_MTP_EH_PROJ_ROOT}/include")
  if (NOT STRIX_AIE_SMOKE_PROGRAM_DIR OR NOT STRIX_AIE_SMOKE_INCLUDE_DIR
      OR NOT STRIX_AIE_QWEN_MTP_RMSNORM_PROGRAM_DIR
      OR NOT STRIX_AIE_QWEN_MTP_RMSNORM_INCLUDE_DIR
      OR NOT STRIX_AIE_QWEN_MTP_EH_PROJ_PROGRAM_DIR
      OR NOT STRIX_AIE_QWEN_MTP_EH_PROJ_INCLUDE_DIR)
    message(FATAL_ERROR "XDNA2 program artifacts not found. Use the Nix development environment")
  endif()
  message(STATUS "strix: XRT NPU shim enabled, include=${XRT_INCLUDE_DIR}")
endif()

# Compiler warnings and sanitizers
option(STRIX_ENABLE_WARNINGS "Enable compiler warnings" ON)
option(STRIX_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

if (STRIX_ENABLE_WARNINGS)
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
  endif()
endif()

if (STRIX_ENABLE_SANITIZERS)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

add_executable(strix src/main.cpp)

if (ENGINE_ENABLE_HIP)
  target_compile_definitions(strix PRIVATE ENGINE_ENABLE_HIP=1)
  target_link_libraries(strix PRIVATE hip::host roc::hipblas roc::rocblas)
endif()

if (ENGINE_ENABLE_XRT)
  target_compile_definitions(strix PRIVATE ENGINE_ENABLE_XRT=1)
  target_include_directories(strix SYSTEM PRIVATE ${XRT_INCLUDE_DIR})
  target_link_libraries(strix PRIVATE ${XRT_COREUTIL_LIB} ${UUID_LIBRARY})
endif()

add_library(strix_diagnostics STATIC
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
  src/cli/diagnose.cpp
)
target_include_directories(strix_diagnostics PUBLIC ${CMAKE_SOURCE_DIR})

if (ENGINE_ENABLE_HIP)
  target_compile_definitions(strix_diagnostics PRIVATE ENGINE_ENABLE_HIP=1)
  target_link_libraries(strix_diagnostics PRIVATE hip::host roc::hipblas roc::rocblas)
endif()

if (ENGINE_ENABLE_XRT)
  target_compile_definitions(strix_diagnostics PRIVATE
    ENGINE_ENABLE_XRT=1
    STRIX_AIE_SMOKE_PROGRAM_DIR="${STRIX_AIE_SMOKE_PROGRAM_DIR}")
  target_include_directories(strix_diagnostics SYSTEM PRIVATE
    ${XRT_INCLUDE_DIR}
    ${STRIX_AIE_SMOKE_INCLUDE_DIR})
  target_link_libraries(strix_diagnostics PRIVATE
    ${XRT_COREUTIL_LIB}
    ${UUID_LIBRARY})
endif()

add_library(strix_core STATIC
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
  src/models/qwen/qwen_tokenizer.cpp
  src/models/qwen/qwen_chat_template.cpp
  src/models/qwen/qwen_oracles.cpp
  src/models/qwen/qwen_state.cpp
  src/models/qwen/qwen_ssm.cpp
  src/models/qwen/qwen_forward.cpp
  src/models/qwen/modules/attention.cpp
  src/models/qwen/modules/embed.cpp
  src/models/qwen/modules/ffn.cpp
  src/models/qwen/modules/norm.cpp
  src/models/qwen/modules/quant_gemm.cpp
  src/models/qwen/modules/residual.cpp
  src/models/qwen/modules/rope.cpp
  src/models/qwen/modules/sample.cpp
  src/models/qwen/modules/ssm.cpp
  src/models/qwen/modules/unembed.cpp
  src/models/qwen/qwen_generator.cpp
  src/models/qwen/qwen_mtp_reference.cpp
  src/testing/compare/logit_comparator.cpp
  src/eval/regrade.cpp
  src/core/hip/hipblaslt_plan_database.cpp
  src/core/quant/ggml_dequant.cpp
  src/core/quant/ggml_gemm.cpp
)
target_include_directories(strix_core PUBLIC ${CMAKE_SOURCE_DIR})
target_compile_definitions(strix_core PRIVATE
  STRIX_FFMPEG_EXECUTABLE="${STRIX_FFMPEG_EXECUTABLE}"
  STRIX_FFPROBE_EXECUTABLE="${STRIX_FFPROBE_EXECUTABLE}")
target_link_libraries(strix_core PUBLIC ICU::uc Threads::Threads)
target_link_libraries(strix_qwen3_tts PUBLIC strix_core)

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
  target_link_libraries(strix_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if (ENGINE_ENABLE_HIP)
  set_property(SOURCE src/models/minimax_h3/dit.hip
    APPEND PROPERTY COMPILE_OPTIONS -O2)
  target_sources(strix_core PRIVATE
    src/models/qwen/hip/qwen_gpu_attention.hip
    src/models/qwen/hip/qwen_gpu_batched_ssm.hip
    src/models/qwen/hip/qwen_gpu_decode_ops.hip
    src/models/qwen/hip/qwen_gpu_prefill_ops.hip
    src/models/qwen/hip/kernels/embed.hip
    src/models/qwen/hip/kernels/residual.hip
    src/models/qwen/hip/kernels/unpack.hip
    src/models/qwen/hip/kernels/rope.hip
    src/models/qwen/hip/kernels/sample.hip
    src/models/qwen/hip/kernels/norm.hip
    src/models/qwen/hip/kernels/gemv.hip
    src/models/qwen/hip/kernels/fused.hip
    src/models/qwen/hip/kernels/common.hip
    src/models/qwen/hip/kernels/qkv.hip
    src/models/qwen/hip/kernels/ssm.hip
    src/models/qwen/hip/kernels/swiglu.hip
    src/models/qwen/hip/kernels/attention_batched.hip
    src/models/qwen/hip/kernels/attention_ck.hip
    src/models/qwen/hip/kernels/attention_decode.hip
    src/models/qwen/hip/kernels/attention_tile.hip
    src/models/qwen/hip/kernels/hipblas_gemm.hip
    src/models/qwen/hip/kernels/hipblaslt_gemm.hip
    src/models/qwen/hip/qwen_gpu_arena.cpp
    src/models/qwen/hip/qwen_gpu_decode.cpp
    src/models/qwen/hip/qwen_gpu_executor.cpp
    src/models/qwen/hip/qwen_gpu_model_loader.cpp
    src/models/qwen/hip/qwen_gpu_prefill.cpp
    src/models/qwen/hip/mtp/model.cpp
    src/models/qwen/hip/mtp/executor.cpp
    src/models/qwen/hip/mtp/backend.cpp
    src/models/minimax_h3/audio_vae.hip
    src/models/minimax_h3/denoiser.hip
    src/models/minimax_h3/dit.hip
    src/models/minimax_h3/runtime_hip.cpp
    src/models/minimax_h3/prompt_encoder.hip
    src/models/minimax_h3/sampling.hip
    src/models/minimax_h3/video_vae.hip
    "${STRIX_H3_ATTENTION_GENERATED_HEADER}"
    src/core/speculative/draft_heads.cpp
    src/core/speculative/self_speculative.cpp
    src/core/speculative/prompt_lookup_backend.cpp
    src/core/speculative/speculative_verifier.cpp
    src/core/heterogeneous/npu_drafter.cpp
  )
  target_compile_definitions(strix_core PUBLIC ENGINE_ENABLE_HIP=1)
  target_compile_definitions(strix_core PRIVATE CK_ENABLE_BF16=1)
  target_include_directories(strix_core SYSTEM PRIVATE
    ${COMPOSABLE_KERNEL_INCLUDE_DIR})
  target_include_directories(strix_core PRIVATE
    "${STRIX_H3_ATTENTION_GENERATED_DIR}")
  target_link_libraries(strix_core PUBLIC hip::host roc::hipblas
    roc::hipblaslt roc::rocblas aotriton_v2)
endif()

if (ENGINE_ENABLE_XRT)
  target_sources(strix_core PRIVATE
    src/models/qwen/xdna2/qwen_mtp_eh_proj.cpp
    src/models/qwen/xdna2/qwen_mtp_rmsnorm.cpp)
  target_compile_definitions(strix_core PUBLIC ENGINE_ENABLE_XRT=1)
  target_compile_definitions(strix_core PRIVATE
    STRIX_AIE_QWEN_MTP_EH_PROJ_PROGRAM_DIR="${STRIX_AIE_QWEN_MTP_EH_PROJ_PROGRAM_DIR}"
    STRIX_AIE_QWEN_MTP_RMSNORM_PROGRAM_DIR="${STRIX_AIE_QWEN_MTP_RMSNORM_PROGRAM_DIR}")
  target_include_directories(strix_core SYSTEM PUBLIC
    ${XRT_INCLUDE_DIR}
    ${STRIX_AIE_QWEN_MTP_EH_PROJ_INCLUDE_DIR}
    ${STRIX_AIE_QWEN_MTP_RMSNORM_INCLUDE_DIR})
  target_link_libraries(strix_core PUBLIC
    strix_diagnostics
    ${XRT_COREUTIL_LIB}
    ${UUID_LIBRARY})
endif()

add_executable(strix-server
  src/server/main.cpp
  src/server/bench_cli.cpp
  src/server/prompt_cli.cpp
  src/server/audio_tts_api.cpp
  src/server/tts_service.cpp
  src/server/video_api.cpp
  src/server/video_jobs.cpp
  src/server/video_cli.cpp
  src/server/inference_backend.cpp
  src/server/http_server.cpp
)
target_link_libraries(strix-server PRIVATE strix_diagnostics strix_core)
target_compile_definitions(strix-server PRIVATE
  STRIX_H3_SOURCE_MANIFEST="${CMAKE_CURRENT_SOURCE_DIR}/src/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json"
  STRIX_H3_INSTALLED_SOURCE_MANIFEST="${CMAKE_INSTALL_PREFIX}/share/strix/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json"
  STRIX_QWEN3_TTS_REFERENCE_RUNNER="${CMAKE_CURRENT_SOURCE_DIR}/src/models/qwen3_tts/reference/run_official.py"
  STRIX_QWEN3_TTS_INSTALLED_REFERENCE_RUNNER="${CMAKE_INSTALL_PREFIX}/share/strix/models/qwen3_tts/run_official.py")

if (ENGINE_ENABLE_HIP)
  target_link_libraries(strix-server PRIVATE strix_deepseek_v4_flash)
endif()
target_link_libraries(strix-server PRIVATE strix_qwen3_tts)
add_library(strix_kernel_bench_support STATIC
  src/bench/kernel_bench.cpp
)
target_include_directories(strix_kernel_bench_support PUBLIC ${CMAKE_SOURCE_DIR})

if (ENGINE_ENABLE_HIP)
  add_executable(strix-kernel-bench
    src/bench/kernel_bench_main.cpp
  )
  target_compile_definitions(strix-kernel-bench PRIVATE
    STRIX_VERSION="${STRIX_VERSION}")
  target_include_directories(strix-kernel-bench SYSTEM PRIVATE
    ${ROCTX_INCLUDE_DIR})
  target_link_libraries(strix-kernel-bench PRIVATE
    strix_kernel_bench_support
    strix_diagnostics
    strix_core
    ${ROCTX_LIBRARY})

  add_executable(tune_hipblaslt tools/tune_hipblaslt.cpp)
  target_link_libraries(tune_hipblaslt PRIVATE strix_core hip::host)

  add_executable(benchmark_ssm_replay tools/benchmark_ssm_replay.cpp)
  target_link_libraries(benchmark_ssm_replay PRIVATE strix_core hip::host)
endif()

install(TARGETS strix strix-server DESTINATION bin)
install(FILES
  src/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json
  DESTINATION share/strix/models/minimax_h3)
install(FILES
  src/models/qwen3_tts/reference/run_official.py
  DESTINATION share/strix/models/qwen3_tts)
if (ENGINE_ENABLE_HIP)
  install(TARGETS strix-kernel-bench tune_hipblaslt benchmark_ssm_replay
    DESTINATION bin)
endif()

# CTest integration
include(CTest)

if (BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  add_test(NAME strix_server_version COMMAND strix-server --version)
  set_tests_properties(strix_server_version PROPERTIES LABELS "cpu")

  add_test(NAME strix_server_help COMMAND strix-server --help)
  set_tests_properties(strix_server_help PROPERTIES LABELS "cpu")

  add_test(NAME strix_server_diagnose_help COMMAND strix-server diagnose --help)
  set_tests_properties(strix_server_diagnose_help PROPERTIES LABELS "cpu")

  add_test(NAME strix_server_diagnose_json COMMAND strix-server diagnose --json --section platform)
  set_tests_properties(strix_server_diagnose_json PROPERTIES LABELS "cpu")

  add_test(NAME strix_server_prompt_help COMMAND strix-server prompt --help)
  set_tests_properties(strix_server_prompt_help PROPERTIES LABELS "cpu;server;cli")

  add_test(NAME strix_server_video_help COMMAND strix-server video --help)
  set_tests_properties(strix_server_video_help PROPERTIES
    LABELS "cpu;server;cli;minimax-h3")

  add_test(NAME strix_server_serve_help COMMAND strix-server serve --help)
  set_tests_properties(strix_server_serve_help PROPERTIES LABELS "cpu;server;cli")

  add_test(NAME strix_server_prompt_smoke COMMAND strix-server prompt "Hello world")
  set_tests_properties(strix_server_prompt_smoke PROPERTIES LABELS "cpu;server;cli")

  add_executable(prompt_cli_test
    tests/server/prompt_cli_test.cpp
    src/server/prompt_cli.cpp
  )
  target_link_libraries(prompt_cli_test PRIVATE strix_core)
  if (ENGINE_ENABLE_HIP)
    target_link_libraries(prompt_cli_test PRIVATE strix_deepseek_v4_flash)
  endif()

  add_test(NAME prompt_cli_test COMMAND prompt_cli_test)
  set_tests_properties(prompt_cli_test PROPERTIES LABELS "cpu;server;cli")

  add_executable(bench_cli_test
    tests/server/bench_cli_test.cpp
    src/server/bench_cli.cpp
  )
  target_link_libraries(bench_cli_test PRIVATE strix_core)
  if (ENGINE_ENABLE_HIP)
    target_link_libraries(bench_cli_test PRIVATE strix_deepseek_v4_flash)
  endif()

  add_test(NAME bench_cli_test COMMAND bench_cli_test)
  set_tests_properties(bench_cli_test PROPERTIES LABELS "cpu;server;cli")

  add_executable(video_cli_test
    tests/server/video_cli_test.cpp
    src/server/video_cli.cpp
  )
  target_link_libraries(video_cli_test PRIVATE strix_core)
  add_test(NAME video_cli_test COMMAND video_cli_test)
  set_tests_properties(video_cli_test PROPERTIES
    LABELS "cpu;server;cli;minimax-h3;quality")

  add_executable(video_jobs_test
    tests/server/video_jobs_test.cpp
    src/server/video_jobs.cpp
  )
  target_link_libraries(video_jobs_test PRIVATE strix_core)
  add_test(NAME video_jobs_test COMMAND video_jobs_test)
  set_tests_properties(video_jobs_test PROPERTIES
    LABELS "cpu;server;video;minimax-h3;queue;cancellation;quality")

  add_executable(video_api_test
    tests/server/video_api_test.cpp
    src/server/video_api.cpp
    src/server/video_jobs.cpp
  )
  target_link_libraries(video_api_test PRIVATE strix_core)
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

  add_executable(bandwidth_test tests/diagnostics/bandwidth_test.cpp)
  target_compile_definitions(bandwidth_test PRIVATE TEST_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/diagnostics")
  target_link_libraries(bandwidth_test PRIVATE strix_diagnostics)

  add_test(NAME bandwidth_test COMMAND bandwidth_test)
  set_tests_properties(bandwidth_test PROPERTIES LABELS "cpu;diagnostics;bandwidth;bandwidth_schema;bandwidth_correctness")

  if (ENGINE_ENABLE_XRT)
    add_executable(xdna2_xrt_program_test
      tests/device/xdna2_xrt_program_test.cpp)
    target_compile_definitions(xdna2_xrt_program_test PRIVATE
      STRIX_AIE_SMOKE_PROGRAM_DIR="${STRIX_AIE_SMOKE_PROGRAM_DIR}")
    target_link_libraries(xdna2_xrt_program_test PRIVATE strix_diagnostics)
    add_test(NAME xdna2_xrt_program_test COMMAND xdna2_xrt_program_test)
    set_tests_properties(xdna2_xrt_program_test PROPERTIES
      LABELS "npu;xdna2;xrt;xrt-hardware")

    add_executable(qwen_mtp_xdna2_rmsnorm_test
      tests/device/qwen_mtp_xdna2_rmsnorm_test.cpp)
    target_compile_definitions(qwen_mtp_xdna2_rmsnorm_test PRIVATE
      STRIX_AIE_QWEN_MTP_RMSNORM_PROGRAM_DIR="${STRIX_AIE_QWEN_MTP_RMSNORM_PROGRAM_DIR}")
    target_link_libraries(qwen_mtp_xdna2_rmsnorm_test PRIVATE
      strix_core
      strix_diagnostics)
    add_test(NAME qwen_mtp_xdna2_rmsnorm_test
      COMMAND qwen_mtp_xdna2_rmsnorm_test)
    set_tests_properties(qwen_mtp_xdna2_rmsnorm_test PROPERTIES
      LABELS "npu;xdna2;xrt;xrt-hardware;qwen;mtp;rmsnorm")

    add_executable(qwen_mtp_xdna2_eh_proj_test
      tests/device/qwen_mtp_xdna2_eh_proj_test.cpp)
    target_compile_definitions(qwen_mtp_xdna2_eh_proj_test PRIVATE
      STRIX_AIE_QWEN_MTP_EH_PROJ_PROGRAM_DIR="${STRIX_AIE_QWEN_MTP_EH_PROJ_PROGRAM_DIR}")
    target_link_libraries(qwen_mtp_xdna2_eh_proj_test PRIVATE
      strix_core
      strix_diagnostics)
    add_test(NAME qwen_mtp_xdna2_eh_proj_test
      COMMAND qwen_mtp_xdna2_eh_proj_test)
    set_tests_properties(qwen_mtp_xdna2_eh_proj_test PROPERTIES
      LABELS "npu;xdna2;xrt;xrt-hardware;qwen;mtp;gemv")
  endif()

  add_executable(qwen_aie2p_w4a8_pack_test
    tests/kernels/qwen38_27b/test_aie2p_w4a8.cpp)
  target_link_libraries(qwen_aie2p_w4a8_pack_test PRIVATE strix_core)
  add_test(NAME qwen_aie2p_w4a8_pack_test
    COMMAND qwen_aie2p_w4a8_pack_test)
  set_tests_properties(qwen_aie2p_w4a8_pack_test PROPERTIES
    LABELS "cpu;qwen;mtp;w4a8;quantization")

  add_executable(gguf_reader_test tests/core/gguf_reader_test.cpp)
  target_link_libraries(gguf_reader_test PRIVATE strix_core)

  add_test(NAME gguf_reader_test COMMAND gguf_reader_test)
  set_tests_properties(gguf_reader_test PROPERTIES LABELS "cpu;core;gguf;model_config")

  add_executable(model_registry_test tests/core/model_registry_test.cpp)
  target_link_libraries(model_registry_test PRIVATE strix_core)

  add_test(NAME model_registry_test COMMAND model_registry_test)
  set_tests_properties(model_registry_test PROPERTIES
    LABELS "cpu;core;model_registry;minimax-h3")

  add_executable(minimax_h3_runtime_test
    tests/models/minimax_h3_runtime_test.cpp)
  target_link_libraries(minimax_h3_runtime_test PRIVATE strix_core)
  add_test(NAME minimax_h3_runtime_test COMMAND minimax_h3_runtime_test)
  set_tests_properties(minimax_h3_runtime_test PROPERTIES
    LABELS "cpu;models;minimax-h3;safetensors;residency;failure-injection")

  add_executable(qwen3_tts_loader_test
    tests/models/qwen3_tts/loader_test.cpp)
  target_link_libraries(qwen3_tts_loader_test PRIVATE strix_qwen3_tts)
  add_test(NAME qwen3_tts_loader_test COMMAND qwen3_tts_loader_test)
  set_tests_properties(qwen3_tts_loader_test PROPERTIES
    LABELS "cpu;models;qwen3-tts;loader;safetensors"
    SKIP_RETURN_CODE 77)

  add_executable(qwen3_tts_audio_api_test
    tests/models/qwen3_tts/audio_api_test.cpp
    src/server/audio_tts_api.cpp
    src/server/tts_service.cpp)
  target_link_libraries(qwen3_tts_audio_api_test PRIVATE strix_qwen3_tts)
  add_test(NAME qwen3_tts_audio_api_test COMMAND qwen3_tts_audio_api_test)
  set_tests_properties(qwen3_tts_audio_api_test PROPERTIES
    LABELS "cpu;models;qwen3-tts;server;api;audio;quality")

  add_executable(qwen3_tts_config_test
    tests/models/qwen3_tts/config_test.cpp)
  target_link_libraries(qwen3_tts_config_test PRIVATE strix_qwen3_tts)
  add_test(NAME qwen3_tts_config_test COMMAND qwen3_tts_config_test)

  add_executable(qwen3_tts_tokenizer_test
    tests/models/qwen3_tts/tokenizer_test.cpp)
  target_link_libraries(qwen3_tts_tokenizer_test PRIVATE strix_qwen3_tts)
  add_test(NAME qwen3_tts_tokenizer_test COMMAND qwen3_tts_tokenizer_test)
  set_tests_properties(qwen3_tts_tokenizer_test PROPERTIES
    SKIP_RETURN_CODE 77)
  set_tests_properties(qwen3_tts_config_test PROPERTIES
    LABELS "cpu;models;qwen3-tts;config;quality")

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
  target_link_libraries(minimax_h3_tokenizer_test PRIVATE strix_core)
  add_test(NAME minimax_h3_tokenizer_test COMMAND minimax_h3_tokenizer_test)
  set_tests_properties(minimax_h3_tokenizer_test PROPERTIES
    LABELS "cpu;models;minimax-h3;tokenizer;quality")

  add_executable(minimax_h3_sampling_test
    tests/models/minimax_h3_sampling_test.cpp)
  target_link_libraries(minimax_h3_sampling_test PRIVATE strix_core)
  add_test(NAME minimax_h3_sampling_test COMMAND minimax_h3_sampling_test)
  set_tests_properties(minimax_h3_sampling_test PROPERTIES
    LABELS "cpu;models;minimax-h3;layout;schedule;rng;quality")

  add_executable(minimax_h3_generation_test
    tests/models/minimax_h3_generation_test.cpp)
  target_link_libraries(minimax_h3_generation_test PRIVATE strix_core)
  add_test(NAME minimax_h3_generation_test
    COMMAND minimax_h3_generation_test)
  set_tests_properties(minimax_h3_generation_test PROPERTIES
    LABELS "cpu;models;minimax-h3;generation;cli;quality;cancellation")

  add_executable(minimax_h3_video_vae_test
    tests/models/minimax_h3_video_vae_test.cpp)
  target_link_libraries(minimax_h3_video_vae_test PRIVATE strix_core)
  add_test(NAME minimax_h3_video_vae_plan
    COMMAND minimax_h3_video_vae_test)
  set_tests_properties(minimax_h3_video_vae_plan PROPERTIES
    LABELS "cpu;models;minimax-h3;video;vae;tiling;chunking")

  add_executable(minimax_h3_media_test
    tests/models/minimax_h3_media_test.cpp)
  target_link_libraries(minimax_h3_media_test PRIVATE strix_core)
  add_test(NAME minimax_h3_media_test COMMAND minimax_h3_media_test)
  set_tests_properties(minimax_h3_media_test PROPERTIES
    LABELS "cpu;models;minimax-h3;video;audio;ffmpeg;mux;quality"
    SKIP_RETURN_CODE 77)

  add_executable(ggml_dequant_test tests/core/ggml_dequant_test.cpp)
  target_link_libraries(ggml_dequant_test PRIVATE strix_core)

  add_test(NAME ggml_dequant_test COMMAND ggml_dequant_test)
  set_tests_properties(ggml_dequant_test PROPERTIES LABELS "cpu;core;quant")

  add_executable(ggml_gemm_parity_test tests/core/ggml_gemm_parity_test.cpp)
  target_link_libraries(ggml_gemm_parity_test PRIVATE strix_core)

  add_test(NAME ggml_gemm_parity_test COMMAND ggml_gemm_parity_test)
  set_tests_properties(ggml_gemm_parity_test PROPERTIES LABELS "cpu;core;quant")

  add_executable(qwen_mtp_reference_test
    tests/core/qwen_mtp_reference_test.cpp)
  target_link_libraries(qwen_mtp_reference_test PRIVATE strix_core)

  add_test(NAME qwen_mtp_reference_test COMMAND qwen_mtp_reference_test)
  set_tests_properties(qwen_mtp_reference_test PROPERTIES
    LABELS "cpu;core;qwen;mtp;external-model"
    SKIP_RETURN_CODE 77)

  add_executable(qwen_tokenizer_test tests/tokenization/qwen_tokenizer_test.cpp)
  target_link_libraries(qwen_tokenizer_test PRIVATE strix_core)

  add_test(NAME qwen_tokenizer_test COMMAND qwen_tokenizer_test)
  set_tests_properties(qwen_tokenizer_test PROPERTIES LABELS "cpu;tokenization;qwen")

  add_executable(qwen_chat_template_test tests/tokenization/qwen_chat_template_test.cpp)
  target_link_libraries(qwen_chat_template_test PRIVATE strix_core)

  add_test(NAME qwen_chat_template_test COMMAND qwen_chat_template_test)
  set_tests_properties(qwen_chat_template_test PROPERTIES LABELS "cpu;tokenization;chat_template")

  add_executable(qwen_cpu_oracles_test tests/models/qwen_cpu_oracles_test.cpp)
  target_link_libraries(qwen_cpu_oracles_test PRIVATE strix_core)

  add_test(NAME qwen_cpu_oracles_test COMMAND qwen_cpu_oracles_test)
  set_tests_properties(qwen_cpu_oracles_test PROPERTIES LABELS "cpu;models;oracles")

  add_executable(qwen_forward_test tests/models/qwen_forward_test.cpp)
  target_link_libraries(qwen_forward_test PRIVATE strix_core)

  add_test(NAME qwen_forward_test COMMAND qwen_forward_test)
  set_tests_properties(qwen_forward_test PROPERTIES LABELS "cpu;models;forward;reference")

  add_executable(qwen_generator_test tests/models/qwen_generator_test.cpp)
  target_link_libraries(qwen_generator_test PRIVATE strix_core)

  add_test(NAME qwen_generator_test COMMAND qwen_generator_test)
  set_tests_properties(qwen_generator_test PROPERTIES LABELS "cpu;models;generator")

  add_executable(qwen_ssm_test tests/models/qwen_ssm_test.cpp)
  target_link_libraries(qwen_ssm_test PRIVATE strix_core)

  add_test(NAME qwen_ssm_test COMMAND qwen_ssm_test)
  set_tests_properties(qwen_ssm_test PROPERTIES LABELS "cpu;models;ssm;reference")
  add_executable(qwen_l1_ssm_module_test tests/models/qwen_l1_ssm_module_test.cpp)
  target_link_libraries(qwen_l1_ssm_module_test PRIVATE strix_core)
  add_test(NAME qwen_l1_ssm_module_test COMMAND qwen_l1_ssm_module_test)
  set_tests_properties(qwen_l1_ssm_module_test PROPERTIES LABELS "cpu;module;ssm")
  add_executable(qwen_l1_norm_module_test tests/models/qwen_l1_norm_module_test.cpp)
  target_link_libraries(qwen_l1_norm_module_test PRIVATE strix_core)
  add_test(NAME qwen_l1_norm_module_test COMMAND qwen_l1_norm_module_test)
  set_tests_properties(qwen_l1_norm_module_test PROPERTIES LABELS "cpu;module;norm")

  add_executable(qwen_l1_ffn_module_test tests/models/qwen_l1_ffn_module_test.cpp)
  target_link_libraries(qwen_l1_ffn_module_test PRIVATE strix_core)
  add_test(NAME qwen_l1_ffn_module_test COMMAND qwen_l1_ffn_module_test)
  set_tests_properties(qwen_l1_ffn_module_test PROPERTIES LABELS "cpu;module;ffn")

  add_executable(qwen_oracle_test tests/models/qwen_oracle_test.cpp)
  target_link_libraries(qwen_oracle_test PRIVATE strix_core)
  add_test(NAME qwen_oracle_test COMMAND qwen_oracle_test)
  set_tests_properties(qwen_oracle_test PROPERTIES LABELS "cpu;models;oracles")

  add_executable(exact_token_test tests/quality/exact_token_test.cpp)
  target_link_libraries(exact_token_test PRIVATE strix_core)
  add_test(NAME exact_token_test COMMAND exact_token_test)
  set_tests_properties(exact_token_test PROPERTIES LABELS "cpu;quality;tokens")

  add_executable(eval_regrade_test tests/quality/eval_regrade_test.cpp)
  target_link_libraries(eval_regrade_test PRIVATE strix_core)
  add_test(NAME eval_regrade_test COMMAND eval_regrade_test)
  set_tests_properties(eval_regrade_test PROPERTIES LABELS "cpu;quality;eval")

  add_executable(qwen_attention_policy_test
    tests/kernels/qwen_attention_policy_test.cpp)
  target_link_libraries(qwen_attention_policy_test PRIVATE strix_core)
  add_test(NAME qwen_attention_policy_test COMMAND qwen_attention_policy_test)
  set_tests_properties(qwen_attention_policy_test PROPERTIES
    LABELS "cpu;models;attention;dispatch")

  add_executable(kernel_bench_test tests/benchmarks/kernel_bench_test.cpp)
  target_link_libraries(kernel_bench_test PRIVATE strix_kernel_bench_support)
  add_test(NAME kernel_bench_test COMMAND kernel_bench_test)
  set_tests_properties(kernel_bench_test PROPERTIES
    LABELS "cpu;benchmarks;statistics;json")

  add_executable(hipblaslt_plan_cache_test
    tests/core/hipblaslt_plan_cache_test.cpp)
  target_link_libraries(hipblaslt_plan_cache_test PRIVATE strix_core)
  add_test(NAME hipblaslt_plan_cache_test COMMAND hipblaslt_plan_cache_test)
  set_tests_properties(hipblaslt_plan_cache_test PROPERTIES
    LABELS "cpu;hipblaslt;cache")

  if (ENGINE_ENABLE_HIP)
    add_executable(qwen3_tts_speech_decoder_hip_test
      tests/models/qwen3_tts/speech_decoder_hip_test.cpp)
    target_link_libraries(qwen3_tts_speech_decoder_hip_test PRIVATE
      strix_qwen3_tts hip::host)
    add_test(NAME qwen3_tts_speech_decoder_hip_test
      COMMAND qwen3_tts_speech_decoder_hip_test)
    set_tests_properties(qwen3_tts_speech_decoder_hip_test PROPERTIES
      LABELS "hip;gpu;gfx1151;models;qwen3-tts;speech-decoder;audio;quality;external-model;slow"
      SKIP_RETURN_CODE 77)

    add_executable(qwen3_tts_talker_hip_prefill_test
      tests/models/qwen3_tts/talker_hip_prefill_test.cpp)
    target_link_libraries(qwen3_tts_talker_hip_prefill_test PRIVATE
      strix_qwen3_tts hip::host)
    add_test(NAME qwen3_tts_talker_hip_prefill_test
      COMMAND qwen3_tts_talker_hip_prefill_test)
    set_tests_properties(qwen3_tts_talker_hip_prefill_test PROPERTIES
      LABELS "hip;gpu;gfx1151;models;qwen3-tts;talker;prefill;quality;external-model;slow"
      SKIP_RETURN_CODE 77)

    add_executable(minimax_h3_runtime_hip_test
      tests/models/minimax_h3_runtime_hip_test.cpp)
    target_link_libraries(minimax_h3_runtime_hip_test PRIVATE
      strix_core hip::host)
    add_test(NAME minimax_h3_runtime_hip_test
      COMMAND minimax_h3_runtime_hip_test)
    set_tests_properties(minimax_h3_runtime_hip_test PROPERTIES
      LABELS "gpu;hip;gfx1151;minimax-h3;external-model;slow"
      SKIP_RETURN_CODE 77)

    add_executable(minimax_h3_prompt_encoder_hip_test
      tests/models/minimax_h3_prompt_encoder_hip_test.hip)
    target_compile_definitions(minimax_h3_prompt_encoder_hip_test PRIVATE
      STRIX_H3_SOURCE_MANIFEST="${CMAKE_CURRENT_SOURCE_DIR}/src/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json")
    target_link_libraries(minimax_h3_prompt_encoder_hip_test PRIVATE
      strix_core hip::host)
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
      strix_core hip::host)
    add_test(NAME minimax_h3_sampling_hip_test
      COMMAND minimax_h3_sampling_hip_test)
    set_tests_properties(minimax_h3_sampling_hip_test PROPERTIES
      LABELS "gpu;hip;gfx1151;minimax-h3;sampling;euler;analytic"
      SKIP_RETURN_CODE 77)

    add_executable(minimax_h3_dit_hip_test
      tests/models/minimax_h3_dit_hip_test.hip)
    target_compile_definitions(minimax_h3_dit_hip_test PRIVATE
      STRIX_H3_SOURCE_MANIFEST="${CMAKE_CURRENT_SOURCE_DIR}/src/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json")
    target_link_libraries(minimax_h3_dit_hip_test PRIVATE
      strix_core hip::host)
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
      STRIX_H3_SOURCE_MANIFEST="${CMAKE_CURRENT_SOURCE_DIR}/src/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json")
    target_link_libraries(minimax_h3_denoiser_hip_test PRIVATE
      strix_core hip::host)
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
      STRIX_H3_SOURCE_MANIFEST="${CMAKE_CURRENT_SOURCE_DIR}/src/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json"
      STRIX_H3_VIDEO_VAE_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/minimax_h3/video-vae-oracle-v1.json")
    target_link_libraries(minimax_h3_video_vae_hip_test PRIVATE
      strix_core hip::host)
    add_test(NAME minimax_h3_video_vae_real
      COMMAND minimax_h3_video_vae_hip_test)
    set_tests_properties(minimax_h3_video_vae_real PROPERTIES
      LABELS "gpu;hip;gfx1151;minimax-h3;video;vae;quality;external-model;very-slow"
      SKIP_RETURN_CODE 77)

    add_executable(minimax_h3_audio_vae_hip_test
      tests/models/minimax_h3_audio_vae_hip_test.hip)
    target_compile_definitions(minimax_h3_audio_vae_hip_test PRIVATE
      STRIX_H3_SOURCE_MANIFEST="${CMAKE_CURRENT_SOURCE_DIR}/src/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json"
      STRIX_H3_AUDIO_VAE_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/minimax_h3/audio-vae-oracle-v1.json")
    target_link_libraries(minimax_h3_audio_vae_hip_test PRIVATE
      strix_core hip::host)
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
      strix_deepseek_v4_flash)
    add_test(NAME deepseek_v4_flash_engine_test
      COMMAND deepseek_v4_flash_engine_test)
    set_tests_properties(deepseek_v4_flash_engine_test PROPERTIES
      LABELS "gpu;hip;gfx1151;deepseek-v4-flash;external-model;slow"
      SKIP_RETURN_CODE 77)

    add_executable(qwen_mtp_gpu_test
      tests/kernels/qwen_mtp_gpu_test.cpp)
    target_link_libraries(qwen_mtp_gpu_test PRIVATE strix_core)
    add_test(NAME qwen_mtp_gpu_test COMMAND qwen_mtp_gpu_test)
    set_tests_properties(qwen_mtp_gpu_test PROPERTIES
      LABELS "gpu;hip;gfx1151;qwen;mtp;external-model"
      SKIP_RETURN_CODE 77)

    add_executable(hipblaslt_runtime_cache_test
      tests/kernels/hipblaslt_runtime_cache_test.cpp)
    target_link_libraries(hipblaslt_runtime_cache_test PRIVATE
      strix_core hip::host)
    add_test(NAME hipblaslt_runtime_cache_test
      COMMAND hipblaslt_runtime_cache_test)
    set_tests_properties(hipblaslt_runtime_cache_test PROPERTIES
      LABELS "hip;gpu;hipblaslt;cache")

    add_executable(qwen_gpu_ops_test tests/kernels/qwen_gpu_ops_test.cpp)
    target_link_libraries(qwen_gpu_ops_test PRIVATE strix_core hip::host roc::hipblas roc::rocblas)
    add_test(NAME qwen_gpu_ops_test COMMAND qwen_gpu_ops_test)
    set_tests_properties(qwen_gpu_ops_test PROPERTIES
      LABELS "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;opt-c010-qk-rope-kv;opt-c010-residual-rmsnorm;opt-c010-ffn-swiglu;opt-c010-ssm-gate-residual;opt-c010-rmsnorm-projection;opt-c014-layer-prefetch;opt-c162-q8k-model")

    add_executable(qwen_l2_gpu_integration_test tests/kernels/qwen_l2_gpu_integration_test.cpp)
    target_link_libraries(qwen_l2_gpu_integration_test PRIVATE strix_core hip::host roc::hipblas roc::rocblas)
    add_test(NAME qwen_l2_gpu_integration_test COMMAND qwen_l2_gpu_integration_test)
    set_tests_properties(qwen_l2_gpu_integration_test PROPERTIES
      LABELS "hip;gpu;kernels;qwen;module;l2")

    add_executable(persistent_deltanet_test
      tests/kernels/persistent_deltanet_test.cpp)
    target_link_libraries(persistent_deltanet_test PRIVATE strix_core hip::host)
    add_test(NAME persistent_deltanet_test COMMAND persistent_deltanet_test)
    set_tests_properties(persistent_deltanet_test PROPERTIES
      LABELS "hip;gpu;kernels;deltanet;speculative")

    add_executable(inference_backend_gpu_test
      tests/server/inference_backend_gpu_test.cpp
      src/server/inference_backend.cpp)
    target_link_libraries(inference_backend_gpu_test PRIVATE
      strix_core strix_deepseek_v4_flash hip::host)
    add_test(NAME inference_backend_gpu_test COMMAND inference_backend_gpu_test)
    set_tests_properties(inference_backend_gpu_test PROPERTIES
      LABELS "hip;gpu;server;correctness"
      SKIP_RETURN_CODE 77)

    add_executable(deepseek_v4_flash_backend_test
      tests/server/deepseek_v4_flash_backend_test.cpp
      src/server/inference_backend.cpp)
    target_link_libraries(deepseek_v4_flash_backend_test PRIVATE
      strix_core strix_deepseek_v4_flash hip::host)
    add_test(NAME deepseek_v4_flash_backend_test
      COMMAND deepseek_v4_flash_backend_test)
    set_tests_properties(deepseek_v4_flash_backend_test PROPERTIES
      LABELS "hip;gpu;server;correctness;deepseek-v4-flash;external-model;slow"
      SKIP_RETURN_CODE 77)

    add_executable(gfx1151_acceptance_test tests/benchmarks/gfx1151_acceptance_test.cpp)
    target_link_libraries(gfx1151_acceptance_test PRIVATE strix_core hip::host roc::hipblas roc::rocblas)
    add_test(NAME gfx1151_acceptance_test COMMAND gfx1151_acceptance_test)
    set_tests_properties(gfx1151_acceptance_test PROPERTIES LABELS "hip;gpu;benchmarks")

    add_executable(speculative_verification_test tests/core/test_speculative_verification.cpp)
    target_link_libraries(speculative_verification_test PRIVATE strix_core hip::host roc::hipblas roc::rocblas)
    add_test(NAME speculative_verification_test COMMAND speculative_verification_test)
    set_tests_properties(speculative_verification_test PROPERTIES LABELS "hip;gpu;speculative")

    add_executable(draft_heads_test tests/core/draft_heads_test.cpp)
    target_link_libraries(draft_heads_test PRIVATE strix_core hip::host roc::hipblas roc::rocblas)
    add_test(NAME draft_heads_test COMMAND draft_heads_test)
    set_tests_properties(draft_heads_test PROPERTIES LABELS "hip;gpu;draft_heads;speculative")

    add_executable(npu_draft_fallback_test
      tests/core/npu_draft_fallback_test.cpp)
    target_link_libraries(npu_draft_fallback_test PRIVATE strix_core hip::host roc::hipblas roc::rocblas)
    if (ENGINE_ENABLE_XRT)
      target_link_libraries(npu_draft_fallback_test PRIVATE ${XRT_COREUTIL_LIB})
    endif()
    add_test(NAME npu_draft_fallback_test COMMAND npu_draft_fallback_test)
    set_tests_properties(npu_draft_fallback_test PROPERTIES
      LABELS "host;speculative;lookup;npu-fallback")

    add_executable(prompt_lookup_test tests/core/prompt_lookup_test.cpp)
    target_link_libraries(prompt_lookup_test PRIVATE strix_core hip::host roc::hipblas roc::rocblas)
    add_test(NAME prompt_lookup_test COMMAND prompt_lookup_test)
    set_tests_properties(prompt_lookup_test PROPERTIES LABELS "speculative;lookup;pld")

  endif()

  if (ENGINE_ENABLE_HIP AND ENGINE_ENABLE_XRT)
    add_test(NAME strix_gpu_xdna2_probe COMMAND strix)
    set_tests_properties(strix_gpu_xdna2_probe PROPERTIES
      LABELS "platform;probe;xdna2;xrt;hip;gpu")
  elseif (ENGINE_ENABLE_HIP)
    add_test(NAME strix_gpu_probe COMMAND strix)
    set_tests_properties(strix_gpu_probe PROPERTIES
      LABELS "platform;probe;hip;gpu")
  elseif (ENGINE_ENABLE_XRT)
    add_test(NAME strix_xdna2_probe COMMAND strix)
    set_tests_properties(strix_xdna2_probe PROPERTIES
      LABELS "platform;probe;xdna2;xrt")
  else()
    add_test(NAME strix_cpu_probe COMMAND strix)
    set_tests_properties(strix_cpu_probe PROPERTIES
      LABELS "platform;probe;cpu")
  endif()
endif()

# Static analysis and formatting targets
include(cmake/StaticAnalysis.cmake)
