cmake_minimum_required(VERSION 3.21)
project(strix LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(STRIX_VERSION "development" CACHE STRING "Strix build revision")
include_directories(${CMAKE_SOURCE_DIR})

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
  find_package(hip REQUIRED)
  find_package(hipblas REQUIRED)
  find_package(hipblaslt REQUIRED)
  find_package(rocblas REQUIRED)
  find_path(ROCTX_INCLUDE_DIR rocprofiler-sdk-roctx/roctx.h REQUIRED)
  find_library(ROCTX_LIBRARY NAMES rocprofiler-sdk-roctx REQUIRED)
  find_path(COMPOSABLE_KERNEL_INCLUDE_DIR ck/ck.hpp REQUIRED)

  message(STATUS "strix: ROCm/HIP enabled, architectures=${CMAKE_HIP_ARCHITECTURES}")
  message(STATUS "strix: ROCM_PATH=$ENV{ROCM_PATH}")
endif()

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
  if (DEFINED ENV{STRIX_AIE_QWEN_AIE2P_W4A8_PROGRAM_DIR}
      AND NOT "$ENV{STRIX_AIE_QWEN_AIE2P_W4A8_PROGRAM_DIR}" STREQUAL "")
    set(STRIX_AIE_QWEN_AIE2P_W4A8_PROGRAM_DIR
      "$ENV{STRIX_AIE_QWEN_AIE2P_W4A8_PROGRAM_DIR}"
      CACHE PATH "Directory containing the reviewed Qwen AIE2P W4A8 artifacts"
      FORCE)
  else()
    set(STRIX_AIE_QWEN_AIE2P_W4A8_PROGRAM_DIR
      "${STRIX_AIE_QWEN_AIE2P_W4A8_PROGRAM_DIR}"
      CACHE PATH "Directory containing the reviewed Qwen AIE2P W4A8 artifacts")
  endif()
  set(STRIX_AIE_QWEN_AIE2P_W4A8_INCLUDE_DIR
    "$ENV{STRIX_AIE_QWEN_AIE2P_W4A8_ROOT}/include")
  if (NOT STRIX_AIE_SMOKE_PROGRAM_DIR OR NOT STRIX_AIE_SMOKE_INCLUDE_DIR
      OR NOT STRIX_AIE_QWEN_MTP_RMSNORM_PROGRAM_DIR
      OR NOT STRIX_AIE_QWEN_MTP_RMSNORM_INCLUDE_DIR
      OR NOT STRIX_AIE_QWEN_MTP_EH_PROJ_PROGRAM_DIR
      OR NOT STRIX_AIE_QWEN_MTP_EH_PROJ_INCLUDE_DIR
      OR NOT STRIX_AIE_QWEN_AIE2P_W4A8_PROGRAM_DIR
      OR NOT STRIX_AIE_QWEN_AIE2P_W4A8_INCLUDE_DIR)
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
  src/tokenization/qwen_tokenizer.cpp
  src/tokenization/qwen_chat_template.cpp
  src/models/qwen_oracles.cpp
  src/models/qwen_state.cpp
  src/models/qwen_ssm.cpp
  src/models/qwen_forward.cpp
  src/models/qwen_generator.cpp
  src/core/speculative/qwen_mtp_reference.cpp
  src/testing/compare/logit_comparator.cpp
  src/eval/regrade.cpp
  src/core/hip/hipblaslt_plan_database.cpp
  src/core/quant/ggml_dequant.cpp
)
target_include_directories(strix_core PUBLIC ${CMAKE_SOURCE_DIR})

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
  target_link_libraries(strix_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if (ENGINE_ENABLE_HIP)
  target_sources(strix_core PRIVATE
    src/core/hip/qwen_gpu_attention.hip
    src/core/hip/qwen_gpu_batched_ssm.hip
    src/core/hip/qwen_gpu_blas.hip
    src/core/hip/qwen_gpu_decode_ops.hip
    src/core/hip/qwen_gpu_prefill_ops.hip
    src/core/hip/qwen_gpu_arena.cpp
    src/core/hip/qwen_gpu_decode.cpp
    src/core/hip/qwen_gpu_executor.cpp
    src/core/hip/qwen_gpu_model_loader.cpp
    src/core/hip/qwen_gpu_prefill.cpp
    src/core/hip/qwen_mtp_gpu.cpp
    src/core/speculative/draft_heads.cpp
    src/core/speculative/self_speculative.cpp
    src/core/speculative/prompt_lookup_backend.cpp
    src/core/speculative/speculative_verifier.cpp
    src/core/heterogeneous/npu_drafter.cpp
  )
  target_compile_definitions(strix_core PUBLIC ENGINE_ENABLE_HIP=1)
  target_include_directories(strix_core SYSTEM PRIVATE
    ${COMPOSABLE_KERNEL_INCLUDE_DIR})
  target_link_libraries(strix_core PUBLIC hip::host roc::hipblas
    roc::hipblaslt roc::rocblas)
endif()

if (ENGINE_ENABLE_XRT)
  target_sources(strix_core PRIVATE
    src/core/xdna2/qwen_mtp_eh_proj.cpp
    src/core/xdna2/qwen_mtp_rmsnorm.cpp
    src/core/xdna2/qwen_aie2p_w4a8.cpp)
  target_compile_definitions(strix_core PUBLIC ENGINE_ENABLE_XRT=1)
  target_compile_definitions(strix_core PRIVATE
    STRIX_AIE_QWEN_MTP_EH_PROJ_PROGRAM_DIR="${STRIX_AIE_QWEN_MTP_EH_PROJ_PROGRAM_DIR}"
    STRIX_AIE_QWEN_MTP_RMSNORM_PROGRAM_DIR="${STRIX_AIE_QWEN_MTP_RMSNORM_PROGRAM_DIR}"
    STRIX_AIE_QWEN_AIE2P_W4A8_PROGRAM_DIR="${STRIX_AIE_QWEN_AIE2P_W4A8_PROGRAM_DIR}")
  target_include_directories(strix_core SYSTEM PUBLIC
    ${XRT_INCLUDE_DIR}
    ${STRIX_AIE_QWEN_MTP_EH_PROJ_INCLUDE_DIR}
    ${STRIX_AIE_QWEN_MTP_RMSNORM_INCLUDE_DIR}
    ${STRIX_AIE_QWEN_AIE2P_W4A8_INCLUDE_DIR})
  target_link_libraries(strix_core PUBLIC
    strix_diagnostics
    ${XRT_COREUTIL_LIB}
    ${UUID_LIBRARY})
endif()

add_executable(strix-server
  src/server/main.cpp
  src/server/bench_cli.cpp
  src/server/prompt_cli.cpp
  src/server/inference_backend.cpp
  src/server/http_server.cpp
)
target_link_libraries(strix-server PRIVATE strix_diagnostics strix_core)

add_executable(strix-bench
  src/bench/main.cpp
  src/server/bench_cli.cpp
)
target_link_libraries(strix-bench PRIVATE strix_diagnostics strix_core)

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

install(TARGETS strix strix-server strix-bench DESTINATION bin)
if (ENGINE_ENABLE_HIP)
  install(TARGETS strix-kernel-bench tune_hipblaslt benchmark_ssm_replay
    DESTINATION bin)
endif()

# CTest integration
include(CTest)

if (BUILD_TESTING)
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

  add_test(NAME strix_server_serve_help COMMAND strix-server serve --help)
  set_tests_properties(strix_server_serve_help PROPERTIES LABELS "cpu;server;cli")

  add_test(NAME strix_server_prompt_smoke COMMAND strix-server prompt "Hello world")
  set_tests_properties(strix_server_prompt_smoke PROPERTIES LABELS "cpu;server;cli")

  add_executable(prompt_cli_test
    tests/server/prompt_cli_test.cpp
    src/server/prompt_cli.cpp
  )
  target_link_libraries(prompt_cli_test PRIVATE strix_core)

  add_test(NAME prompt_cli_test COMMAND prompt_cli_test)
  set_tests_properties(prompt_cli_test PROPERTIES LABELS "cpu;server;cli")

  add_executable(bench_cli_test
    tests/server/bench_cli_test.cpp
    src/server/bench_cli.cpp
  )
  target_link_libraries(bench_cli_test PRIVATE strix_core)

  add_test(NAME bench_cli_test COMMAND bench_cli_test)
  set_tests_properties(bench_cli_test PROPERTIES LABELS "cpu;server;cli")

  add_executable(bandwidth_test tests/diagnostics/bandwidth_test.cpp)
  target_compile_definitions(bandwidth_test PRIVATE TEST_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/diagnostics")
  target_link_libraries(bandwidth_test PRIVATE strix_diagnostics)

  add_test(NAME bandwidth_test COMMAND bandwidth_test)
  set_tests_properties(bandwidth_test PROPERTIES LABELS "cpu;diagnostics;bandwidth;bandwidth_schema;bandwidth_correctness")

  if (ENGINE_ENABLE_XRT)
    add_executable(xrt_smoke_test tests/device/xrt_smoke_test.cpp)
    target_compile_definitions(xrt_smoke_test PRIVATE
      STRIX_AIE_SMOKE_PROGRAM_DIR="${STRIX_AIE_SMOKE_PROGRAM_DIR}")
    target_link_libraries(xrt_smoke_test PRIVATE strix_diagnostics)
    add_test(NAME xrt_smoke_test COMMAND xrt_smoke_test)
    set_tests_properties(xrt_smoke_test PROPERTIES
      LABELS "npu;xdna2;xrt;xrt-hardware")

    add_executable(qwen_mtp_rmsnorm_test
      tests/device/qwen_mtp_rmsnorm_test.cpp)
    target_compile_definitions(qwen_mtp_rmsnorm_test PRIVATE
      STRIX_AIE_QWEN_MTP_RMSNORM_PROGRAM_DIR="${STRIX_AIE_QWEN_MTP_RMSNORM_PROGRAM_DIR}")
    target_link_libraries(qwen_mtp_rmsnorm_test PRIVATE
      strix_core
      strix_diagnostics)
    add_test(NAME qwen_mtp_rmsnorm_test COMMAND qwen_mtp_rmsnorm_test)
    set_tests_properties(qwen_mtp_rmsnorm_test PROPERTIES
      LABELS "npu;xdna2;xrt;xrt-hardware;qwen;mtp;rmsnorm")

    add_executable(qwen_mtp_eh_proj_test
      tests/device/qwen_mtp_eh_proj_test.cpp)
    target_compile_definitions(qwen_mtp_eh_proj_test PRIVATE
      STRIX_AIE_QWEN_MTP_EH_PROJ_PROGRAM_DIR="${STRIX_AIE_QWEN_MTP_EH_PROJ_PROGRAM_DIR}")
    target_link_libraries(qwen_mtp_eh_proj_test PRIVATE
      strix_core
      strix_diagnostics)
    add_test(NAME qwen_mtp_eh_proj_test COMMAND qwen_mtp_eh_proj_test)
    set_tests_properties(qwen_mtp_eh_proj_test PROPERTIES
      LABELS "npu;xdna2;xrt;xrt-hardware;qwen;mtp;gemv")

    add_executable(qwen_aie2p_w4a8_test
      tests/kernels/qwen38_27b/test_aie2p_w4a8.cpp)
    target_compile_definitions(qwen_aie2p_w4a8_test PRIVATE
      STRIX_AIE_QWEN_AIE2P_W4A8_PROGRAM_DIR="${STRIX_AIE_QWEN_AIE2P_W4A8_PROGRAM_DIR}")
    target_link_libraries(qwen_aie2p_w4a8_test PRIVATE
      strix_core
      strix_diagnostics)
    add_test(NAME qwen_aie2p_w4a8_test COMMAND qwen_aie2p_w4a8_test)
    set_tests_properties(qwen_aie2p_w4a8_test PROPERTIES
      LABELS "cpu;npu;xdna2;xrt;xrt-hardware;qwen;mtp;w4a8;gemm;gemv"
      SKIP_RETURN_CODE 77)
  endif()

  # W4A8 SHQ4-T16 shape-matrix bench (issue #35, M004-C007 P3). Compiles and
  # runs CPU-only (pack.hpp oracle); the NPU path is compiled only under
  # ENGINE_ENABLE_XRT and SKIPs (exit 77) when XDNA2 is absent.
  add_executable(qwen_aie2p_w4a8_shq4_matrix_bench
    tools/testing/qwen_aie2p_w4a8_shq4_matrix_bench.cpp)
  target_link_libraries(qwen_aie2p_w4a8_shq4_matrix_bench PRIVATE strix_core)
  if (ENGINE_ENABLE_XRT)
    target_compile_definitions(qwen_aie2p_w4a8_shq4_matrix_bench PRIVATE
      STRIX_AIE_QWEN_AIE2P_W4A8_PROGRAM_DIR="${STRIX_AIE_QWEN_AIE2P_W4A8_PROGRAM_DIR}")
  endif()
  add_test(NAME qwen_aie2p_w4a8_shq4_matrix_bench
    COMMAND qwen_aie2p_w4a8_shq4_matrix_bench)
  set_tests_properties(qwen_aie2p_w4a8_shq4_matrix_bench PROPERTIES
    LABELS "npu;xdna2;xrt;xrt-hardware;qwen;mtp;w4a8;bench;gemm;gemv"
    SKIP_RETURN_CODE 77)

  add_executable(gguf_reader_test tests/core/gguf_reader_test.cpp)
  target_link_libraries(gguf_reader_test PRIVATE strix_core)

  add_test(NAME gguf_reader_test COMMAND gguf_reader_test)
  set_tests_properties(gguf_reader_test PROPERTIES LABELS "cpu;core;gguf;model_config")

  add_executable(ggml_dequant_test tests/core/ggml_dequant_test.cpp)
  target_link_libraries(ggml_dequant_test PRIVATE strix_core)

  add_test(NAME ggml_dequant_test COMMAND ggml_dequant_test)
  set_tests_properties(ggml_dequant_test PROPERTIES LABELS "cpu;core;quant")

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
  set_tests_properties(qwen_forward_test PROPERTIES LABELS "cpu;models;forward")

  add_executable(qwen_generator_test tests/models/qwen_generator_test.cpp)
  target_link_libraries(qwen_generator_test PRIVATE strix_core)

  add_test(NAME qwen_generator_test COMMAND qwen_generator_test)
  set_tests_properties(qwen_generator_test PROPERTIES LABELS "cpu;models;generator")

  add_executable(qwen_ssm_test tests/models/qwen_ssm_test.cpp)
  target_link_libraries(qwen_ssm_test PRIVATE strix_core)

  add_test(NAME qwen_ssm_test COMMAND qwen_ssm_test)
  set_tests_properties(qwen_ssm_test PROPERTIES LABELS "cpu;models;ssm")

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
    set_tests_properties(qwen_gpu_ops_test PROPERTIES LABELS "hip;gpu;kernels")

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
      strix_core hip::host)
    add_test(NAME inference_backend_gpu_test COMMAND inference_backend_gpu_test)
    set_tests_properties(inference_backend_gpu_test PROPERTIES
      LABELS "hip;gpu;server;correctness"
      SKIP_RETURN_CODE 77)

    add_executable(gfx1151_acceptance_test tests/benchmarks/gfx1151_acceptance_test.cpp)
    target_link_libraries(gfx1151_acceptance_test PRIVATE strix_core hip::host roc::hipblas roc::rocblas)
    add_test(NAME gfx1151_acceptance_test COMMAND gfx1151_acceptance_test)
    set_tests_properties(gfx1151_acceptance_test PROPERTIES LABELS "hip;gpu;benchmarks")

    add_executable(speculative_verification_test tests/core/test_speculative_verification.cpp)
    target_link_libraries(speculative_verification_test PRIVATE strix_core hip::host roc::hipblas roc::rocblas)
    add_test(NAME speculative_verification_test COMMAND speculative_verification_test)
    set_tests_properties(speculative_verification_test PROPERTIES LABELS "hip;gpu;speculative")

    add_executable(test_draft_heads tests/core/test_draft_heads.cpp)
    target_link_libraries(test_draft_heads PRIVATE strix_core hip::host roc::hipblas roc::rocblas)
    add_test(NAME test_draft_heads COMMAND test_draft_heads)
    set_tests_properties(test_draft_heads PROPERTIES LABELS "hip;gpu;draft_heads;speculative")

    add_executable(test_npu_speculative tests/npu/test_npu_speculative.cpp)
    target_link_libraries(test_npu_speculative PRIVATE strix_core hip::host roc::hipblas roc::rocblas)
    if (ENGINE_ENABLE_XRT)
      target_link_libraries(test_npu_speculative PRIVATE ${XRT_COREUTIL_LIB})
    endif()
    add_test(NAME test_npu_speculative COMMAND test_npu_speculative)
    set_tests_properties(test_npu_speculative PROPERTIES LABELS "npu;xdna2;speculative")

    add_executable(test_prompt_lookup tests/core/test_prompt_lookup.cpp)
    target_link_libraries(test_prompt_lookup PRIVATE strix_core hip::host roc::hipblas roc::rocblas)
    add_test(NAME test_prompt_lookup COMMAND test_prompt_lookup)
    set_tests_properties(test_prompt_lookup PROPERTIES LABELS "speculative;lookup;pld")

  endif()

  if (ENGINE_ENABLE_HIP AND ENGINE_ENABLE_XRT)
    add_test(NAME strix_npu_smoke COMMAND strix)
    set_tests_properties(strix_npu_smoke PROPERTIES LABELS "npu;xdna2;xrt;hip;gpu")
  elseif (ENGINE_ENABLE_HIP)
    add_test(NAME strix_hip_smoke COMMAND strix)
    set_tests_properties(strix_hip_smoke PROPERTIES LABELS "hip;gpu")
  elseif (ENGINE_ENABLE_XRT)
    add_test(NAME strix_npu_smoke COMMAND strix)
    set_tests_properties(strix_npu_smoke PROPERTIES LABELS "npu;xdna2;xrt")
  else()
    add_test(NAME strix_cpu_smoke COMMAND strix)
    set_tests_properties(strix_cpu_smoke PROPERTIES LABELS "cpu")
  endif()
endif()

# Static analysis and formatting targets
include(cmake/StaticAnalysis.cmake)
