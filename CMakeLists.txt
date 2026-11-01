cmake_minimum_required(VERSION 3.21)
project(gufo LANGUAGES C CXX)
# Production remains Linux x86-64 on gfx1151; Windows x86-64 is a supported
# native build target sharing the same source tree.
if (NOT (CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Windows")
    OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
  message(FATAL_ERROR "Gufo requires Linux or Windows on x86-64 with an AMD gfx1151 GPU")
endif()
if (WIN32)
  # Declared before any subdirectory: windows.h must never define the
  # min/max macros that break std::min/std::max and rocPRIM device headers.
  # PSAPI_VERSION=2 resolves GetProcessMemoryInfo through kernel32 directly.
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN PSAPI_VERSION=2)
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
set(GUFO_RELEASE_VERSION "development" CACHE STRING "Gufo release version")
set(GUFO_REVISION "unknown" CACHE STRING "Gufo source revision")
include_directories(${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/include)

# Reuse the compiler cache whenever it is installed, as the CPU check
# derivations do. HIP kernels stay uncached: hipcc drive files and bitcode
# paths need their own hit-rate measurement before they join the cache.
find_program(GUFO_CCACHE_PROGRAM NAMES ccache)
if (GUFO_CCACHE_PROGRAM)
  set(CMAKE_C_COMPILER_LAUNCHER "${GUFO_CCACHE_PROGRAM}")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${GUFO_CCACHE_PROGRAM}")
  # Newer CMake gives clang-cl gcc-style depfiles (-clang:-MD), which ccache
  # rejects as unsupported; /showIncludes is cacheable.
  if (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    set(GUFO_CCACHE_CLANG_CL ON)
    foreach(lang C CXX)
      set(CMAKE_DEPFILE_FLAGS_${lang} "/showIncludes")
      set(CMAKE_${lang}_DEPFILE_FORMAT msvc)
    endforeach()
  endif()
  message(STATUS "gufo: compiling C and CXX through ${GUFO_CCACHE_PROGRAM}")
else()
  message(STATUS "gufo: ccache not found, compiling without a launcher")
endif()

# Windows host checks build only the OS leaves, the CPU cache library and
# their portable tests. The rest of the tree targets Linux only.
option(GUFO_PLATFORM_ONLY
  "Build only the platform leaves, the CPU cache and their portable tests" OFF)
if (GUFO_PLATFORM_ONLY)
  find_package(Threads REQUIRED)
  add_compile_options(-Wall -Wextra -Wpedantic -UNDEBUG)
  add_subdirectory(src/core/platform)
  add_subdirectory(src/cache)
  include(CTest)
  if (BUILD_TESTING)
    if (NOT WIN32)
      find_package(Python3 REQUIRED COMPONENTS Interpreter)
    endif()
    include(cmake/PlatformTests.cmake)
    add_subdirectory(tests/cache)
  endif()
  return()
endif()
# Linux libstdc++ must be parsed before HIP defines its __noinline__ macro.
if (NOT WIN32)
  add_compile_options(
    "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-include \"${CMAKE_SOURCE_DIR}/src/core/hip/libstdcxx_noinline.hpp\">")
endif()
find_package(ICU REQUIRED COMPONENTS uc i18n)
find_package(Threads REQUIRED)
find_package(CURL REQUIRED)
find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)
if (WIN32)
  # Windows has no system pkg-config; locate the pkgconf tool vcpkg installs.
  # .pc files resolve via CMAKE_PREFIX_PATH, which the vcpkg toolchain already
  # points at the manifest install tree.
  find_program(PKG_CONFIG_EXECUTABLE pkgconf REQUIRED
    HINTS "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/tools/pkgconf")
endif()
find_package(PkgConfig REQUIRED)
pkg_check_modules(WEBP REQUIRED IMPORTED_TARGET libwebp)
pkg_check_modules(SNDFILE REQUIRED IMPORTED_TARGET sndfile)
pkg_check_modules(SOXR REQUIRED IMPORTED_TARGET soxr)
pkg_check_modules(AV REQUIRED IMPORTED_TARGET libavformat libavcodec libavutil libswscale)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)
set(GUFO_FFMPEG_EXECUTABLE "ffmpeg" CACHE STRING
  "FFmpeg executable used for generated media")
set(GUFO_FFPROBE_EXECUTABLE "ffprobe" CACHE STRING
  "FFprobe executable used to validate generated media")

# CPU-only builds exist for host checks; production targets gfx1151 only.
option(ENGINE_ENABLE_HIP "Build the gfx1151 HIP backend" ON)
include(cmake/GufoExtensionDiscovery.cmake)
gufo_discover_extensions()
set(CMAKE_HIP_ARCHITECTURES gfx1151 CACHE STRING "HIP architecture")
if (ENGINE_ENABLE_HIP AND NOT CMAKE_HIP_ARCHITECTURES STREQUAL "gfx1151")
  message(FATAL_ERROR "Gufo kernels require CMAKE_HIP_ARCHITECTURES=gfx1151")
endif()
# Standard ROCm installations; custom prefixes use CMAKE_PREFIX_PATH. As a
# system prefix /opt/rocm is searched after CMAKE_PREFIX_PATH in the
# environment, so a development shell's ROCm wins over the host's.
if (EXISTS /opt/rocm)
  list(APPEND CMAKE_SYSTEM_PREFIX_PATH /opt/rocm)
endif()

if (ENGINE_ENABLE_HIP)
  enable_language(HIP)
  # clang-cl compiles HIP like C++ on Windows, so ccache covers it there.
  if (GUFO_CCACHE_CLANG_CL)
    set(CMAKE_HIP_COMPILER_LAUNCHER "${GUFO_CCACHE_PROGRAM}")
    set(CMAKE_DEPFILE_FLAGS_HIP "/showIncludes")
    set(CMAKE_HIP_DEPFILE_FORMAT msvc)
    message(STATUS "gufo: compiling HIP through ${GUFO_CCACHE_PROGRAM}")
  endif()
  # clang-cl maps the MSVC RelWithDebInfo /Ob1 to -finline-hint-functions for
  # the device compile too, which marks every __device__ helper without
  # inline/__forceinline__ noinline: the kernels then call them. Linux builds
  # inline them; /Ob2 restores the normal inliner for HIP sources.
  if (WIN32)
    string(REPLACE "/Ob1" "/Ob2" CMAKE_HIP_FLAGS_RELWITHDEBINFO
      "${CMAKE_HIP_FLAGS_RELWITHDEBINFO}")
    string(REPLACE "/Ob1" "/Ob2" CMAKE_HIP_FLAGS_MINSIZEREL
      "${CMAKE_HIP_FLAGS_MINSIZEREL}")
  endif()
  find_package(hip REQUIRED)
  find_package(hipblas REQUIRED)
  find_package(hipblaslt REQUIRED)
  find_package(rocblas REQUIRED)
  find_package(rocm-core CONFIG QUIET)
  find_path(HIPCUB_INCLUDE_DIR hipcub/hipcub.hpp REQUIRED)
  find_path(ROCPRIM_INCLUDE_DIR rocprim/rocprim.hpp REQUIRED)
  find_path(ROCWMMA_INCLUDE_DIR rocwmma/rocwmma.hpp REQUIRED)
  message(STATUS "gufo: HIP target ${CMAKE_HIP_ARCHITECTURES}")
  if (WIN32)
    # The AMD driver installs an older amdhip64 into System32 that shadows
    # the TheRock runtime on the DLL search path; executables must carry the
    # matching runtime beside them (the exe directory precedes System32).
    # ROCM_PATH comes from cmake/toolchains/windows-therock.cmake (HIP_PATH).
    file(GLOB GUFO_HIP_RUNTIME_DLLS "${ROCM_PATH}/bin/*.dll")
    if (NOT GUFO_HIP_RUNTIME_DLLS)
      message(FATAL_ERROR "No HIP runtime DLLs under ROCM_PATH=${ROCM_PATH}/bin")
    endif()
    message(STATUS "gufo: staging ${GUFO_HIP_RUNTIME_DLLS}")
  endif()
endif()

function(gufo_stage_hip_runtime target)
  if (ENGINE_ENABLE_HIP AND WIN32 AND GUFO_HIP_RUNTIME_DLLS)
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              ${GUFO_HIP_RUNTIME_DLLS} $<TARGET_FILE_DIR:${target}>)
    # rocBLAS/hipBLASLt resolve their Tensile kernel libraries relative to
    # their own DLL location, so the data directories ship beside the exe.
    foreach(data_dir rocblas hipblaslt)
      if (EXISTS "${ROCM_PATH}/bin/${data_dir}")
        add_custom_command(TARGET ${target} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_directory
                  "${ROCM_PATH}/bin/${data_dir}"
                  "$<TARGET_FILE_DIR:${target}>/${data_dir}")
      endif()
    endforeach()
  endif()
endfunction()

if (ENGINE_ENABLE_HIP)
  add_subdirectory(src/models/deepseek_v4_flash)
endif()

# Qwen3.8-Flash-Next — model-private CPU reference plus ROCm runtime.
add_subdirectory(src/models/qwen38_flash_next)

# Qwen3-TTS — CPU reference + HIP kernels (CustomVoice 12Hz).
add_subdirectory(src/models/qwen3_tts)

# Qwen3-ASR — model-private 1.7B speech recognition runtime.
add_subdirectory(src/models/qwen3_asr)

# EmbeddingGemma 2 is a model-private BF16 text/image embedding runtime.
add_subdirectory(src/models/embedding_gemma2)

# Compiler warnings and sanitizers
option(GUFO_ENABLE_WARNINGS "Enable compiler warnings" ON)
option(GUFO_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

if (GUFO_ENABLE_WARNINGS)
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
  endif()
endif()

if (GUFO_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

if (WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # CMake drives lld-link directly, which skips the clang-cl driver and its
  # runtime selection. Code using __builtin_cpu_supports (the disk CRC
  # dispatch) needs the builtins' __cpu_model at link time.
  # Ask the compiler in use for its runtime directory.
  execute_process(COMMAND "${CMAKE_CXX_COMPILER}" /clang:-print-runtime-dir
                  OUTPUT_VARIABLE GUFO_CLANG_RT_DIR
                  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  set(GUFO_CLANG_RT_BUILTINS
      "${GUFO_CLANG_RT_DIR}/clang_rt.builtins-x86_64.lib")
  if (EXISTS "${GUFO_CLANG_RT_BUILTINS}")
    add_link_options("${GUFO_CLANG_RT_BUILTINS}")
  else()
    message(WARNING "clang_rt.builtins not found under '${GUFO_CLANG_RT_DIR}'")
  endif()
  if (ENGINE_ENABLE_HIP)
    execute_process(COMMAND "${CMAKE_CXX_COMPILER}" /clang:-print-resource-dir
                    OUTPUT_VARIABLE GUFO_CLANG_RESOURCE_DIR
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    set(GUFO_HIP_CMATH "${GUFO_CLANG_RESOURCE_DIR}/include/__clang_hip_cmath.h")
    if (EXISTS "${GUFO_HIP_CMATH}")
      file(READ "${GUFO_HIP_CMATH}" GUFO_HIP_CMATH_CONTENT)
      if (NOT GUFO_HIP_CMATH_CONTENT MATCHES "GUFO_MSVC_CMATH_PATCH")
        message(FATAL_ERROR
          "TheRock HIP math headers are not patched for MSVC 14.40+; run: "
          "python tools/windows/patch_sdk_math.py")
      endif()
    endif()
  endif()
endif()

add_subdirectory(src/cache)

if (ENGINE_ENABLE_HIP)
  add_library(gufo_hip_host STATIC src/core/hip/host_allocations.cpp)
  target_link_libraries(gufo_hip_host PUBLIC hip::host gufo_platform)
  # HIP loads ROCr through libamdhip64's own RUNPATH, so a build tree runs
  # the unpatched runtime unless LD_LIBRARY_PATH names the patched one (the
  # Nix package wraps `gufo` for this). With GUFO_ROCR_LIBRARY_DIR (the Nix
  # shells export it) every HIP binary links that ROCr directly: the loader
  # maps it first and HIP reuses it, one soname per process.
  set(GUFO_ROCR_LIBRARY_DIR "" CACHE PATH
      "ROCr runtime directory HIP binaries load (empty: HIP's own)")
  set(gufo_rocr_dir "${GUFO_ROCR_LIBRARY_DIR}")
  if (NOT gufo_rocr_dir AND DEFINED ENV{GUFO_ROCR_LIBRARY_DIR})
    set(gufo_rocr_dir "$ENV{GUFO_ROCR_LIBRARY_DIR}")
  endif()
  if (gufo_rocr_dir AND NOT WIN32)
    find_library(GUFO_ROCR_LIBRARY hsa-runtime64 PATHS "${gufo_rocr_dir}"
                 NO_DEFAULT_PATH NO_CACHE REQUIRED)
    target_link_libraries(gufo_hip_host PUBLIC
                          "${GUFO_ROCR_LIBRARY}")
    message(STATUS "gufo: ROCr runtime ${GUFO_ROCR_LIBRARY}")
  endif()
  target_include_directories(gufo_hip_host PUBLIC ${PROJECT_SOURCE_DIR})
  add_library(gufo_cache_hip STATIC
    src/core/hip/committed_backing.cpp
    src/core/hip/component_cache_resources.cpp
    src/core/hip/staging.cpp
    src/core/hip/transfer_pool.cpp)
  target_link_libraries(gufo_cache_hip PUBLIC gufo_cache gufo_hip_host hip::host)
  if (BUILD_TESTING)
    target_compile_options(gufo_cache_hip PRIVATE -UNDEBUG)
  endif()
  if (GUFO_BUILD_TOOLS)
    add_executable(cache_idle_spill_bench tests/cache/idle_spill_bench.cpp)
    target_link_libraries(cache_idle_spill_bench PRIVATE gufo_cache_hip)
    add_executable(cache_backing_bench tests/cache/backing_bench.cpp)
    target_link_libraries(cache_backing_bench PRIVATE gufo_cache_hip)
    add_executable(cache_streaming_bench tests/cache/streaming_bench.cpp)
    target_link_libraries(cache_streaming_bench PRIVATE gufo_cache_hip)
  endif()
endif()

add_library(gufo_diagnostics STATIC
  src/core/diagnostics/gpu_queues.cpp
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
target_link_libraries(gufo_diagnostics PUBLIC gufo_platform)
target_compile_definitions(gufo_diagnostics PRIVATE
  GUFO_REVISION="${GUFO_REVISION}")

if (ENGINE_ENABLE_HIP)
  target_sources(gufo_diagnostics PRIVATE
    src/core/hip/allocation_benchmark.hip)
  target_compile_definitions(gufo_diagnostics PRIVATE ENGINE_ENABLE_HIP=1)
  if (rocm-core_FOUND)
    target_compile_definitions(gufo_diagnostics PRIVATE
      GUFO_ROCM_VERSION="${rocm-core_VERSION}")
  endif()
  target_link_libraries(gufo_diagnostics PRIVATE hip::host roc::hipblas roc::rocblas)
endif()

add_subdirectory(src/core/platform)

add_library(gufo_core STATIC
  src/core/image.cpp
  src/core/crypto/sha256.cpp
  src/core/gguf_reader.cpp
  src/core/gguf_identity.cpp
  src/core/json_constraint.cpp
  src/core/json_schema_lexeme.cpp
  src/core/json_schema_regex.cpp
  src/core/sampling.cpp
  src/core/hip/hipblaslt_plan_database.cpp
  src/core/quant/ggml_dequant.cpp
  src/core/quant/ggml_gemm.cpp
)
target_include_directories(gufo_core PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(gufo_core PUBLIC gufo_platform ICU::uc ICU::i18n
  Threads::Threads PNG::PNG JPEG::JPEG PkgConfig::WEBP CURL::libcurl
  OpenSSL::Crypto)

# Family libraries come before the model that needs them; they may depend on
# gufo_core and on each other, never on a model directory.
add_subdirectory(src/families/qwen)
add_subdirectory(src/models/qwen)
add_subdirectory(src/models/qwen_image_21)

# Qwen family consumers name the family libraries they include. Flash-Next uses
# the shared tokenizer/chat template, kernels and vision stack; Qwen3-TTS the
# shared HIP operators; Qwen-Image-2.1 the tokenizer.
target_link_libraries(gufo_qwen_image_21 PUBLIC gufo_qwen_text)
target_link_libraries(gufo_qwen3_tts PUBLIC gufo_core gufo_qwen_control_tokens)
if (ENGINE_ENABLE_HIP)
  target_link_libraries(gufo_qwen3_tts PUBLIC gufo_qwen_kernels)
  target_link_libraries(gufo_qwen38_flash_next PUBLIC
    gufo_qwen_text gufo_qwen_vision)
  target_link_libraries(gufo_deepseek_v4_flash PUBLIC gufo_core)
endif()

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
  target_link_libraries(gufo_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if (ENGINE_ENABLE_HIP)
  target_sources(gufo_core PRIVATE
    src/core/hip/snapshot_copy.hip
    src/core/hip/weight_upload.cpp
    src/core/speculative/speculative_verifier.cpp
  )
  target_compile_definitions(gufo_core PUBLIC ENGINE_ENABLE_HIP=1)
  target_include_directories(gufo_core SYSTEM PRIVATE
    ${HIPCUB_INCLUDE_DIR}
    ${ROCPRIM_INCLUDE_DIR})
  target_link_libraries(gufo_core PUBLIC gufo_hip_host hip::host roc::hipblas
    roc::hipblaslt roc::rocblas)
endif()

# Card 08 (RFC step 5): the model, comparison and eval tiers that used to be
# object files of gufo_core are separate targets now. They are added here,
# after the warning/sanitizer block, so their sources keep the compile
# environment they had inside gufo_core.
add_subdirectory(src/models/minimax_h3)
add_subdirectory(src/testing/compare)
add_subdirectory(src/eval)
# The CPU-only DeepSeek text tier is registered next to them because it must
# keep that warning/sanitizer environment; the model directory itself is added
# above, before the block, and links this library instead of compiling the two
# text sources again.
add_subdirectory(src/models/deepseek_v4_flash/text)

add_library(gufo_llm_cli STATIC
  src/cli/bench/bench.cpp
  src/cli/prompt/prompt.cpp)
# The interim edges the RFC allows between steps 5 and 6: prompt/bench still
# reach the Qwen executor and generators directly, so the CLI library links the
# Qwen model and family libraries instead of getting them through gufo_core.
target_link_libraries(gufo_llm_cli PUBLIC gufo_core gufo_compare gufo_qwen38_flash_next
  gufo_qwen27b gufo_qwen_vision gufo_qwen_text)
if(ENGINE_ENABLE_HIP)
  target_link_libraries(gufo_llm_cli PUBLIC gufo_deepseek_v4_flash)
endif()

add_subdirectory(src/ext)

# The imported DeepSeek V4 Flash runtime routes its informational messages
# through the shared logger instead of writing to stderr directly, so it needs
# the threshold that `--log-level` arms.
if (ENGINE_ENABLE_HIP)
  target_link_libraries(gufo_deepseek_v4_flash PUBLIC gufo_logging)
endif()

# RFC step 6: the text-serving tier the gufo executable used to compile into
# itself. Scheduler, runner pool, component text cache and InferenceBackend
# live here; queueing, retention/admission and API policy stay on this core
# side of the boundary, while models reach execution through gufo_ext_runtime.
add_library(gufo_serving STATIC
  src/cli/serve/continuation_cache.cpp
  src/cli/serve/continuation_disk_store.cpp
  src/cli/serve/component_text_cache.cpp
  src/cli/serve/text_model_runner.cpp
  src/cli/serve/text_generation_scheduler.cpp
  src/cli/serve/continuation_identity.cpp
  src/cli/serve/inference_backend.cpp)
target_link_libraries(gufo_serving PUBLIC gufo_ext_runtime gufo_logging
  gufo_core gufo_cache)
# The executable's own objects carried production NDEBUG, the test binaries that
# re-compiled these sources forced assertions on. One library can only pick
# one: test builds keep assertions, so a test-preset `gufo` now asserts where it
# previously did not. Production (BUILD_TESTING=OFF) is unchanged.
if (BUILD_TESTING)
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(gufo_serving PRIVATE -UNDEBUG)
  elseif (MSVC)
    target_compile_options(gufo_serving PRIVATE /UNDEBUG)
  endif()
endif()
if (ENGINE_ENABLE_HIP)
  # Interim edges the RFC allows between steps 5 and 6: the concrete text
  # runners and the model-specific identity lines still compile in these
  # sources, so the library links the models until step 10 moves them out.
  target_link_libraries(gufo_serving PUBLIC gufo_cache_hip hip::host
    gufo_deepseek_v4_flash gufo_qwen38_flash_next ${GUFO_QWEN_LIBRARIES}
    gufo_qwen_vision)
endif()

add_library(gufo_http STATIC
  src/cli/serve/image_api.cpp
  src/cli/serve/audio_asr_api.cpp
  src/cli/serve/asr_service.cpp
  src/cli/serve/embeddings_api.cpp
  src/cli/serve/embedding_service.cpp
  src/cli/serve/audio_tts_api.cpp
  src/cli/serve/tts_service.cpp
  src/cli/serve/video_api.cpp
  src/cli/serve/video_jobs.cpp
  src/cli/serve/http_server.cpp
  src/cli/serve/websocket.cpp
  src/cli/serve/asr_websocket.cpp
  src/cli/serve/tts_websocket.cpp
  src/cli/serve/openai_chat.cpp
  # The DeepSeek tool-call scanner lives with its model; its renderer stays in
  # the engine library, so no translation unit is linked twice.
  src/models/deepseek_v4_flash/tool_calls.cpp
)
target_link_libraries(gufo_http PUBLIC gufo_logging gufo_core gufo_minimax_h3 gufo_qwen3_tts gufo_qwen3_asr
  gufo_qwen_image_21 gufo_embedding_gemma2
  gufo_qwen27b gufo_qwen_vision gufo_qwen_text)

add_executable(gufo
  src/cli/main.cpp
  src/cli/extension_diagnostics.cpp
  src/cli/banner.cpp
  src/cli/eval/eval.cpp
  src/cli/transcribe/transcribe.cpp
  src/cli/video/video.cpp
  src/cli/serve/serve.cpp
  src/eval/http_client.cpp
)
target_link_libraries(gufo PRIVATE gufo_cache gufo_diagnostics gufo_core gufo_eval
  gufo_llm_cli gufo_http gufo_serving CURL::libcurl)
gufo_build_discovered_extensions()
get_property(gufo_selected_extensions GLOBAL PROPERTY GUFO_SELECTED_EXTENSIONS)
gufo_target_extensions(gufo EXTENSIONS ${gufo_selected_extensions})
gufo_stage_hip_runtime(gufo)
target_compile_definitions(gufo PRIVATE
  GUFO_RELEASE_VERSION="${GUFO_RELEASE_VERSION}"
  GUFO_REVISION="${GUFO_REVISION}"
  GUFO_EVAL_SOURCE_DATA="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/eval/antirez-ds4.json"
  GUFO_EVAL_INSTALLED_DATA="${CMAKE_INSTALL_PREFIX}/share/gufo/eval/antirez-ds4.json"
  GUFO_H3_SOURCE_MANIFEST="${CMAKE_CURRENT_SOURCE_DIR}/src/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json"
  GUFO_H3_INSTALLED_SOURCE_MANIFEST="${CMAKE_INSTALL_PREFIX}/share/gufo/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json")

if (ENGINE_ENABLE_HIP)
  target_link_libraries(gufo PRIVATE gufo_cache_hip gufo_deepseek_v4_flash)
endif()
target_link_libraries(gufo PRIVATE gufo_qwen3_tts gufo_qwen3_asr gufo_embedding_gemma2)

if (NOT WIN32)
  add_custom_target(gufo-server ALL
    COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE_NAME:gufo> ${CMAKE_CURRENT_BINARY_DIR}/gufo-server
    DEPENDS gufo
  )
else()
  add_custom_target(gufo-server ALL
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:gufo> ${CMAKE_CURRENT_BINARY_DIR}/gufo-server.exe
    DEPENDS gufo
  )
endif()
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
    GUFO_REVISION="${GUFO_REVISION}")
  target_include_directories(gufo-kernel-bench SYSTEM PRIVATE
    ${ROCTX_INCLUDE_DIR})
  target_link_libraries(gufo-kernel-bench PRIVATE
    gufo_kernel_bench_support
    gufo_diagnostics
    gufo_core
    ${GUFO_QWEN_LIBRARIES}
    gufo_qwen_kernels
    ${ROCTX_LIBRARY})

  add_executable(tune_hipblaslt tools/bench/tune_hipblaslt.cpp)
  target_link_libraries(tune_hipblaslt PRIVATE gufo_core gufo_qwen_kernels
    hip::host)

  add_executable(wmma_layout_test tools/bench/wmma_layout_test.hip)
  target_link_libraries(wmma_layout_test PRIVATE gufo_core hip::device)
endif()

install(TARGETS gufo DESTINATION bin)
install(FILES
  src/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json
  DESTINATION share/gufo/models/minimax_h3)
install(FILES
  tests/fixtures/eval/antirez-ds4.json
  DESTINATION share/gufo/eval)
install(FILES LICENSE NOTICE THIRD_PARTY_NOTICES.md DESTINATION share/licenses/gufo)
install(DIRECTORY licenses/ DESTINATION share/licenses/gufo/third-party)
install(FILES src/models/deepseek_v4_flash/LICENSE.ds4 DESTINATION share/licenses/gufo/third-party)
if (NOT WIN32)
install(CODE "execute_process(COMMAND \"${CMAKE_COMMAND}\" -E create_symlink gufo \"\$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gufo-server\")")
else()
  install(CODE "execute_process(COMMAND \"${CMAKE_COMMAND}\" -E copy \"\$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gufo.exe\" \"\$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gufo-server.exe\")")
endif()
if (ENGINE_ENABLE_HIP AND GUFO_BUILD_TOOLS)
  install(TARGETS gufo-kernel-bench tune_hipblaslt
    DESTINATION bin)
endif()

# CTest integration
include(CTest)

# Test assertions remain active when model code uses production optimization.
if (BUILD_TESTING)
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-UNDEBUG)
  elseif (MSVC)
    add_compile_options(/UNDEBUG)
  endif()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  # In-tree test support for model API tests (RFC Step 7b, Card 12).
  # Wraps HTTP and serving fixtures so tests stop reaching src/cli/ directly.
  # Excluded from production, external extensions, and installation.
  add_library(gufo_test_support STATIC tests/support/test_support.cpp)
  target_include_directories(gufo_test_support PUBLIC ${CMAKE_SOURCE_DIR})
  target_link_libraries(gufo_test_support PUBLIC gufo_http gufo_serving)

  add_subdirectory(tests/cache)
  add_subdirectory(tests/extensions)

  # Opt-in live process-failure checks, never preloaded into hosted tests or
  # production. No HIP headers are needed for these stable C ABI hooks.
  add_library(device_loss_faults SHARED EXCLUDE_FROM_ALL
    tests/functional/device_loss_faults.cpp)
  target_link_libraries(device_loss_faults PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)
  # Linux-only: the gate interposes fsync via dlsym(RTLD_NEXT) and reads
  # /proc/self/fd, neither of which exists on Windows.
  if (NOT WIN32)
  add_library(cache_disk_faults SHARED
    tests/functional/cache_disk_faults.cpp)
  target_link_libraries(cache_disk_faults PRIVATE ${CMAKE_DL_LIBS})
  endif()
  # The fault libraries are LD_PRELOADed into uninstrumented processes (the
  # test's Python, a release server); a sanitized library aborts there with
  # "ASan runtime does not come first".
  if (GUFO_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    foreach(preload device_loss_faults cache_disk_faults)
      if (TARGET ${preload})
        target_compile_options(${preload} PRIVATE -fno-sanitize=all)
        target_link_options(${preload} PRIVATE -fno-sanitize=all)
      endif()
    endforeach()
  endif()

  add_subdirectory(tests/families/qwen)
  add_subdirectory(tests/models/qwen)
  add_subdirectory(tests/models/qwen27b)
  gufo_qwen3_asr_register_tests()
  gufo_qwen_image_21_register_tests()

  add_subdirectory(tests/models/embedding_gemma2)

  add_test(NAME gufo_version COMMAND gufo --version)
  set_tests_properties(gufo_version PROPERTIES
    LABELS "cpu"
    PASS_REGULAR_EXPRESSION
      "gufo version ${GUFO_RELEASE_VERSION} \\(${GUFO_REVISION}\\)")

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
  add_test(NAME gufo_serve_embedding_help COMMAND gufo serve embedding --help)
  set_tests_properties(gufo_serve_embedding_help PROPERTIES
    LABELS "cpu;server;cli;embedding-gemma2")

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
  target_compile_definitions(json_constraint_test PRIVATE
    GUFO_LLAMA_TOOL_GRAMMAR_GOLDENS="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/llama_cpp_tool_grammar.json")
  add_test(NAME json_constraint_test COMMAND json_constraint_test)
  set_tests_properties(json_constraint_test PROPERTIES LABELS "cpu;json;sampling")

  add_executable(json_test tests/core/json_test.cpp)
  target_link_libraries(json_test PRIVATE gufo_core)
  add_test(NAME json_test COMMAND json_test)
  set_tests_properties(json_test PROPERTIES LABELS "cpu;json")

  add_executable(heap_placement_test tests/core/heap_placement_test.cpp)
  target_link_libraries(heap_placement_test PRIVATE gufo_core)
  add_test(NAME heap_placement_test COMMAND heap_placement_test)
  set_tests_properties(heap_placement_test PROPERTIES LABELS "cpu;hip;core")

  add_executable(sampling_test tests/core/sampling_test.cpp)
  target_link_libraries(sampling_test PRIVATE gufo_core)
  add_test(NAME sampling_test COMMAND sampling_test)
  set_tests_properties(sampling_test PROPERTIES LABELS "cpu;sampling")

  add_executable(logprobs_test tests/cli/logprobs_test.cpp)
  target_link_libraries(logprobs_test PRIVATE gufo_core)
  add_test(NAME logprobs_test COMMAND logprobs_test)
  set_tests_properties(logprobs_test PROPERTIES LABELS "cpu;server;api;openai")

  # openai_chat.cpp and the DeepSeek tool-call scanner come from gufo_http, and
  # the chat template from the CPU-only DeepSeek text target: this test no
  # longer compiles a second copy of any of them.
  add_executable(openai_chat_test
    tests/cli/openai_chat_test.cpp
  )
  target_link_libraries(openai_chat_test PRIVATE gufo_core gufo_logging gufo_http
    gufo_deepseek_v4_flash_text ${GUFO_QWEN_LIBRARIES})
  add_test(NAME openai_chat_test COMMAND openai_chat_test)
  set_tests_properties(openai_chat_test PROPERTIES
    LABELS "cpu;server;api;openai;streaming;tools")

  add_executable(quote_tracker_test tests/cli/quote_tracker_test.cpp)
  target_link_libraries(quote_tracker_test PRIVATE gufo_core)
  add_test(NAME quote_tracker_test COMMAND quote_tracker_test)
  set_tests_properties(quote_tracker_test PROPERTIES LABELS "cpu;server;api")

  # the trace suite simulates RLIMIT_FSIZE
  if (NOT WIN32)
  add_executable(trace_test tests/cli/trace_test.cpp)
  target_link_libraries(trace_test PRIVATE gufo_logging gufo_platform)
  add_test(NAME trace_test COMMAND trace_test)
  set_tests_properties(trace_test PROPERTIES LABELS "cpu;server;trace")
  endif()

  # These HTTP/WebSocket suites drive raw POSIX sockets or fork() directly.
  if (NOT WIN32)
  add_executable(http_server_test tests/cli/http_server_test.cpp)
  target_link_libraries(http_server_test PRIVATE gufo_http)
  add_test(NAME http_server_test COMMAND http_server_test)
  set_tests_properties(http_server_test PROPERTIES LABELS "cpu;server;http" TIMEOUT 30)

  add_executable(http_server_signals_test tests/cli/http_server_signals_test.cpp)
  target_link_libraries(http_server_signals_test PRIVATE gufo_http)
  add_test(NAME http_server_signals_test COMMAND http_server_signals_test)
  set_tests_properties(http_server_signals_test
    PROPERTIES LABELS "cpu;server;http" TIMEOUT 60)

  add_executable(audio_websocket_test tests/cli/audio_websocket_test.cpp)
  target_link_libraries(audio_websocket_test PRIVATE gufo_http)
  add_test(NAME audio_websocket_test COMMAND audio_websocket_test)
  set_tests_properties(audio_websocket_test PROPERTIES LABELS "cpu;server;audio" TIMEOUT 30)
  endif()

  add_test(NAME serve_cli_test
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli/serve_test.py
      $<TARGET_FILE:gufo>)
  set_tests_properties(serve_cli_test PROPERTIES LABELS "cpu;server;cli" TIMEOUT 30)

  add_test(NAME kernel_resources_parser_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/kernel_resources_test.py)
  set_tests_properties(kernel_resources_parser_test PROPERTIES
    LABELS "cpu;kernels" TIMEOUT 30)

  add_test(NAME check_extension_isolation_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/check_extension_isolation_test.py)
  add_test(NAME extension_isolation_report
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/ci/check-extension-isolation.py --mode report)
  add_test(NAME extension_core_model_includes_gate
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/ci/check-extension-isolation.py
      --fail-on-core-model-includes)
  add_test(NAME extension_serving_boundary_gate
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/ci/check-extension-isolation.py
      --fail-on-serving-boundary)
  add_test(NAME extension_model_core_includes_gate
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/ci/check-extension-isolation.py
      --fail-on-model-core-includes)
  add_test(NAME extension_family_graph_gate
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/ci/check-extension-isolation.py
      --fail-on-family-graph)
  add_test(NAME extension_target_ownership_gate
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/ci/check-extension-isolation.py
      --fail-on-target-ownership)
  # The checker's own tests scan the tree many times: about 25 s on Linux.
  set_tests_properties(check_extension_isolation_test PROPERTIES
    LABELS "cpu;repository" TIMEOUT 180)
  set_tests_properties(extension_isolation_report
    extension_core_model_includes_gate extension_family_graph_gate
    extension_model_core_includes_gate
    extension_serving_boundary_gate extension_target_ownership_gate
    PROPERTIES LABELS "cpu;repository" TIMEOUT 30)

  add_test(NAME check_docs_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/check_docs_test.py)
  set_tests_properties(check_docs_test PROPERTIES LABELS "cpu;docs" TIMEOUT 30)

  add_test(NAME check_pr_title_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/check_pr_title_test.py)
  set_tests_properties(check_pr_title_test PROPERTIES LABELS "cpu;docs" TIMEOUT 30)

  # It reads the gufo binary, which Windows does not build yet.
  if(ENGINE_ENABLE_HIP AND TARGET gufo)
    # A compiler update can make a kernel spill with no source change. Clang 23
    # increased the scratch of the Q8 prefill GEMM from 0 to 480 bytes. This
    # test fails if a kernel uses more scratch than the baseline allows. It
    # reads the code objects in the binary and does not use a GPU.
    get_filename_component(GUFO_HIP_BIN_DIR ${CMAKE_HIP_COMPILER} DIRECTORY)
    add_test(NAME kernel_resources_test
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/ci/check-kernel-resources.py
        $<TARGET_FILE:gufo>
        --baseline ${CMAKE_CURRENT_SOURCE_DIR}/tools/ci/kernel-resources.json
        --bundler ${GUFO_HIP_BIN_DIR}/clang-offload-bundler)
    set_tests_properties(kernel_resources_test PROPERTIES
      LABELS "cpu;kernels" TIMEOUT 120)
  endif()

  add_subdirectory(tests/models/deepseek_v4_flash)
  add_subdirectory(tests/models/qwen38_flash_next)

  if(ENGINE_ENABLE_HIP)
    add_executable(chat_template_hf_token_golden_test
      tests/models/chat_template_hf_token_golden_test.cpp)
    target_link_libraries(chat_template_hf_token_golden_test PRIVATE gufo_core
      gufo_qwen_text)
    target_compile_definitions(chat_template_hf_token_golden_test PRIVATE
      GUFO_CHAT_TEMPLATE_HF_GOLDENS="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/chat_template_hf_goldens.json")
    add_test(NAME chat_template_hf_token_golden_test
      COMMAND chat_template_hf_token_golden_test)
    set_tests_properties(chat_template_hf_token_golden_test PROPERTIES
      LABELS "gpu;hip;tokenization;chat_template;qwen;external-model"
      SKIP_RETURN_CODE 77)
  endif()

  add_executable(continuation_cache_test tests/cli/continuation_cache_test.cpp)
  target_link_libraries(continuation_cache_test PRIVATE gufo_serving)
  add_test(NAME continuation_cache_test COMMAND continuation_cache_test)
  set_tests_properties(continuation_cache_test PROPERTIES
    LABELS "cpu;server;cache;state;lifecycle")

  add_executable(component_text_cache_test
    tests/cli/component_text_cache_test.cpp)
  target_link_libraries(component_text_cache_test PRIVATE gufo_cache_fake
    gufo_serving)
  add_test(NAME component_text_cache_test COMMAND component_text_cache_test)
  set_tests_properties(component_text_cache_test PROPERTIES LABELS "cpu;server;cache;state;lifecycle")

  add_executable(text_model_runner_test tests/cli/text_model_runner_test.cpp)
  target_link_libraries(text_model_runner_test PRIVATE gufo_serving
    ${GUFO_QWEN_LIBRARIES})
  add_test(NAME text_model_runner_test COMMAND text_model_runner_test)
  set_tests_properties(text_model_runner_test PROPERTIES
    LABELS "cpu;server;runner;cache;state;lifecycle")

  # the disk-store suite exercises permission faults with chmod and lstat
  if (NOT WIN32)
  add_executable(continuation_disk_store_test
    tests/cli/continuation_disk_store_test.cpp)
  target_link_libraries(continuation_disk_store_test PRIVATE gufo_serving)
  add_test(NAME continuation_disk_store_test
    COMMAND continuation_disk_store_test)
  set_tests_properties(continuation_disk_store_test PROPERTIES
    LABELS "cpu;server;cache;disk;state;lifecycle")
  endif()

  add_executable(logit_sampler_test tests/cli/logit_sampler_test.cpp)
  target_link_libraries(logit_sampler_test PRIVATE gufo_core)
  add_test(NAME logit_sampler_test COMMAND logit_sampler_test)
  set_tests_properties(logit_sampler_test PROPERTIES
    LABELS "cpu;server;sampling")

  add_executable(text_generation_scheduler_test
    tests/cli/text_generation_scheduler_test.cpp)
  target_link_libraries(text_generation_scheduler_test PRIVATE gufo_serving)
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
  target_link_libraries(video_cli_test PRIVATE gufo_core gufo_minimax_h3)
  add_test(NAME video_cli_test COMMAND video_cli_test)
  set_tests_properties(video_cli_test PROPERTIES
    LABELS "cpu;server;cli;minimax-h3;quality")

  add_executable(video_jobs_test
    tests/cli/video_jobs_test.cpp
    src/cli/serve/video_jobs.cpp
  )
  target_link_libraries(video_jobs_test PRIVATE gufo_logging gufo_core
    gufo_minimax_h3 gufo_cache)
  add_test(NAME video_jobs_test COMMAND video_jobs_test)
  set_tests_properties(video_jobs_test PROPERTIES
    LABELS "cpu;server;video;minimax-h3;queue;cancellation;quality")

  add_executable(video_api_test
    tests/cli/video_api_test.cpp
    src/cli/serve/video_api.cpp
    src/cli/serve/video_jobs.cpp
  )
  target_link_libraries(video_api_test PRIVATE gufo_logging gufo_core
    gufo_minimax_h3 gufo_cache)
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

  add_test(NAME functional_runner_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/functional/test_runner.py)
  set_tests_properties(functional_runner_test PROPERTIES LABELS "cpu;server;functional")
  add_test(NAME cache_workloads_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/functional/test_cache_workloads.py)
  set_tests_properties(cache_workloads_test PROPERTIES LABELS "cpu;server;functional")
  # the fault runner needs the Linux-only gate library
  if (NOT WIN32)
  add_test(NAME cache_disk_faults_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/functional/cache_disk_faults_test.py
      $<TARGET_FILE:cache_disk_faults>)
  set_tests_properties(cache_disk_faults_test PROPERTIES LABELS "cpu;server;functional")
  endif()

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

  add_executable(gpu_queues_test tests/diagnostics/gpu_queues_test.cpp)
  target_link_libraries(gpu_queues_test PRIVATE gufo_diagnostics gufo_platform)
  add_test(NAME gpu_queues_test COMMAND gpu_queues_test)
  set_tests_properties(gpu_queues_test PROPERTIES LABELS "cpu;diagnostics;serving")

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

  # the prefetch fixture uses mkstemp and mmap
  if (NOT WIN32)
  add_executable(gguf_reader_test tests/core/gguf_reader_test.cpp)
  target_link_libraries(gguf_reader_test PRIVATE gufo_core)

  add_test(NAME gguf_reader_test COMMAND gguf_reader_test)
  set_tests_properties(gguf_reader_test PROPERTIES LABELS "cpu;core;gguf;model_config")
  endif()

  # the digest-cache case restores mtimes with utimensat
  if (NOT WIN32)
  add_executable(gguf_identity_test tests/core/gguf_identity_test.cpp)
  target_link_libraries(gguf_identity_test PRIVATE gufo_core)

  add_test(NAME gguf_identity_test COMMAND gguf_identity_test)
  set_tests_properties(gguf_identity_test PROPERTIES LABELS "cpu;core;gguf")
  endif()

  add_executable(minimax_h3_runtime_test
    tests/models/minimax_h3_runtime_test.cpp)
  target_link_libraries(minimax_h3_runtime_test PRIVATE gufo_core gufo_minimax_h3)
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
    tests/models/qwen3_tts/audio_api_test.cpp)
  target_link_libraries(qwen3_tts_audio_api_test PRIVATE gufo_qwen3_tts gufo_test_support)
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
  target_link_libraries(minimax_h3_tokenizer_test PRIVATE gufo_core gufo_minimax_h3)
  add_test(NAME minimax_h3_tokenizer_test COMMAND minimax_h3_tokenizer_test)
  set_tests_properties(minimax_h3_tokenizer_test PROPERTIES
    LABELS "cpu;models;minimax-h3;tokenizer;quality")

  add_executable(minimax_h3_sampling_test
    tests/models/minimax_h3_sampling_test.cpp)
  target_link_libraries(minimax_h3_sampling_test PRIVATE gufo_core gufo_minimax_h3)
  add_test(NAME minimax_h3_sampling_test COMMAND minimax_h3_sampling_test)
  set_tests_properties(minimax_h3_sampling_test PROPERTIES
    LABELS "cpu;models;minimax-h3;layout;schedule;rng;quality")

  add_executable(minimax_h3_generation_test
    tests/models/minimax_h3_generation_test.cpp)
  target_link_libraries(minimax_h3_generation_test PRIVATE gufo_core gufo_minimax_h3)
  add_test(NAME minimax_h3_generation_test
    COMMAND minimax_h3_generation_test)
  set_tests_properties(minimax_h3_generation_test PROPERTIES
    LABELS "cpu;models;minimax-h3;generation;cli;quality;cancellation")

  add_executable(minimax_h3_video_vae_test
    tests/models/minimax_h3_video_vae_test.cpp)
  target_link_libraries(minimax_h3_video_vae_test PRIVATE gufo_core gufo_minimax_h3)
  add_test(NAME minimax_h3_video_vae_plan
    COMMAND minimax_h3_video_vae_test)
  set_tests_properties(minimax_h3_video_vae_plan PROPERTIES
    LABELS "cpu;models;minimax-h3;video;vae;tiling;chunking")

  add_executable(minimax_h3_media_test
    tests/models/minimax_h3_media_test.cpp)
  target_link_libraries(minimax_h3_media_test PRIVATE gufo_core gufo_minimax_h3)
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
  target_link_libraries(exact_token_test PRIVATE gufo_core ${GUFO_QWEN_LIBRARIES})
  add_test(NAME exact_token_test COMMAND exact_token_test)
  set_tests_properties(exact_token_test PROPERTIES LABELS "cpu;quality;tokens")

  add_executable(eval_regrade_test tests/quality/eval_regrade_test.cpp)
  target_link_libraries(eval_regrade_test PRIVATE gufo_core gufo_eval)
  add_test(NAME eval_regrade_test COMMAND eval_regrade_test)
  set_tests_properties(eval_regrade_test PROPERTIES LABELS "cpu;quality;eval")

  add_test(NAME eval_http_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/test_eval_http.py
      --gufo $<TARGET_FILE:gufo>
      --data ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/eval/antirez-ds4.json)
  set_tests_properties(eval_http_test PROPERTIES
    LABELS "cpu;quality;eval;http;cli;server")

  # The eval client fixture drives raw POSIX sockets directly.
  if (NOT WIN32)
  add_executable(eval_http_client_test
    tests/quality/eval_http_client_test.cpp src/eval/http_client.cpp)
  target_include_directories(eval_http_client_test PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(eval_http_client_test PRIVATE CURL::libcurl Threads::Threads)
  add_test(NAME eval_http_client_test COMMAND eval_http_client_test)
  set_tests_properties(eval_http_client_test PROPERTIES LABELS "cpu;quality;eval;http")
  endif()

  add_executable(kernel_bench_test tests/benchmarks/kernel_bench_test.cpp)
  target_link_libraries(kernel_bench_test PRIVATE gufo_kernel_bench_support)
  add_test(NAME kernel_bench_test COMMAND kernel_bench_test)
  set_tests_properties(kernel_bench_test PROPERTIES
    LABELS "cpu;benchmarks;statistics;json")

  add_library(platform_library_fixture SHARED tests/core/platform_library_fixture.cpp)
  set_target_properties(platform_library_fixture PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tests/core/fixtures"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tests/core")
  add_executable(platform_library_test tests/core/platform_library_test.cpp)
  target_link_libraries(platform_library_test PRIVATE gufo_platform)
  add_dependencies(platform_library_test platform_library_fixture)
  add_test(NAME platform_library_test COMMAND platform_library_test
    "$<TARGET_FILE:platform_library_fixture>")
  set_tests_properties(platform_library_test PROPERTIES LABELS "cpu;platform")
  if(NOT WIN32)
    add_test(NAME platform_library_relocated_test COMMAND platform_library_test
      "/gufo-missing-build-path/$<TARGET_FILE_NAME:platform_library_fixture>")
    set_tests_properties(platform_library_relocated_test PROPERTIES
      LABELS "cpu;platform"
      ENVIRONMENT "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:platform_library_fixture>")
  endif()

  # the publish suite drives descriptors through dup2
  if (NOT WIN32)
  add_executable(platform_publish_test
    tests/core/platform_publish_test.cpp src/core/platform/publish.cpp)
  target_link_libraries(platform_publish_test PRIVATE Threads::Threads)
  if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(platform_publish_test PRIVATE "-Wl,--wrap=fsync")
  endif()
  add_test(NAME platform_publish_test COMMAND platform_publish_test)
  set_tests_properties(platform_publish_test PROPERTIES LABELS "cpu;platform")
  endif()

  # the prefault fixture maps pages with mmap
  if (NOT WIN32)
  add_executable(platform_prefault_test tests/core/platform_prefault_test.cpp)
  target_link_libraries(platform_prefault_test PRIVATE Threads::Threads)
  add_test(NAME platform_prefault_test COMMAND platform_prefault_test)
  set_tests_properties(platform_prefault_test PROPERTIES LABELS "cpu;platform")
  endif()

  include(cmake/PlatformTests.cmake)

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
      gufo_core gufo_minimax_h3 hip::host)
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
      gufo_core gufo_minimax_h3 hip::host)
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
      gufo_core gufo_minimax_h3 hip::host)
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
      gufo_core gufo_minimax_h3 hip::host)
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
      gufo_core gufo_minimax_h3 hip::host)
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
      gufo_core gufo_minimax_h3 hip::host)
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
      gufo_core gufo_minimax_h3 hip::host)
    add_test(NAME minimax_h3_audio_vae_primitives
      COMMAND minimax_h3_audio_vae_hip_test --primitives)
    set_tests_properties(minimax_h3_audio_vae_primitives PROPERTIES
      LABELS "gpu;hip;gfx1151;minimax-h3;audio;vae;analytic")
    add_test(NAME minimax_h3_audio_vae_real
      COMMAND minimax_h3_audio_vae_hip_test --real)
    set_tests_properties(minimax_h3_audio_vae_real PROPERTIES
      LABELS "gpu;hip;gfx1151;minimax-h3;audio;vae;quality;external-model;very-slow"
      SKIP_RETURN_CODE 77)

    add_executable(hip_memory_visibility_test
      tests/core/hip_memory_visibility_test.hip)
    target_link_libraries(hip_memory_visibility_test PRIVATE gufo_core hip::host)
    add_test(NAME hip_memory_visibility_test COMMAND hip_memory_visibility_test)
    add_test(NAME hip_memory_visibility_restricted_test
      COMMAND hip_memory_visibility_test --deny-long-term-pins)
    add_test(NAME hip_memory_visibility_driver_test COMMAND hip_memory_visibility_test)
    set_tests_properties(hip_memory_visibility_test hip_memory_visibility_restricted_test PROPERTIES
      LABELS "gpu;hip;gfx1151"
      ENVIRONMENT "HSA_USERPTR_FOR_PAGED_MEM=1")
    set_tests_properties(hip_memory_visibility_driver_test PROPERTIES
      LABELS "gpu;hip;gfx1151"
      ENVIRONMENT "HSA_USERPTR_FOR_PAGED_MEM=0")

    add_executable(hip_weight_upload_test tests/core/hip_weight_upload_test.cpp)
    target_link_libraries(hip_weight_upload_test PRIVATE gufo_core)
    add_test(NAME hip_weight_upload_test COMMAND hip_weight_upload_test)
    set_tests_properties(hip_weight_upload_test PROPERTIES LABELS "hip;gpu;gfx1151")

    add_executable(hipblaslt_runtime_cache_test
      tests/kernels/hipblaslt_runtime_cache_test.cpp)
    target_link_libraries(hipblaslt_runtime_cache_test PRIVATE
      gufo_core ${GUFO_QWEN_LIBRARIES} hip::host)
    add_test(NAME hipblaslt_runtime_cache_test
      COMMAND hipblaslt_runtime_cache_test)
    set_tests_properties(hipblaslt_runtime_cache_test PROPERTIES
      LABELS "hip;gpu;hipblaslt;cache")

    add_executable(persistent_deltanet_test
      tests/kernels/persistent_deltanet_test.cpp)
    target_link_libraries(persistent_deltanet_test PRIVATE gufo_core
      ${GUFO_QWEN_LIBRARIES} hip::host)
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

include(cmake/TargetGraph.cmake)
if (BUILD_TESTING)
  # Card 09 (RFC step 6): record the real link edges here, where every target
  # is defined, and let `extension_link_graph_gate` judge them.
  gufo_write_link_graph("${CMAKE_BINARY_DIR}/gufo-link-graph.json")
  add_test(NAME extension_link_graph_gate
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/ci/check-target-graph.py
      --report ${CMAKE_BINARY_DIR}/gufo-link-graph.json
      --root ${CMAKE_CURRENT_SOURCE_DIR})
  set_tests_properties(extension_link_graph_gate PROPERTIES
    LABELS "cpu;repository" TIMEOUT 60)
  add_test(NAME check_target_graph_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/check_target_graph_test.py ${CMAKE_BINARY_DIR})
  set_tests_properties(check_target_graph_test PROPERTIES
    LABELS "cpu;repository" TIMEOUT 60)
endif()

if (BUILD_TESTING)
  include(cmake/Checks.cmake)
endif()
