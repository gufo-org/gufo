cmake_minimum_required(VERSION 3.24)
project(ds4_reference LANGUAGES C CXX HIP)
set(CMAKE_C_STANDARD 11)
set(CMAKE_HIP_STANDARD 17)
find_package(hip REQUIRED)
find_package(hipblas REQUIRED)
find_package(hipblaslt REQUIRED)
find_package(rocblas REQUIRED)
find_package(Threads REQUIRED)
set(hip_sources
  ds4_rocm.cu ds4_rocm_compat.cu ds4_rocm_unavailable.cu
  cuda/mmq/ds4_ggml_stubs.cu cuda/mmq/ds4_mmq.cu
  cuda/mmq/quantize.cu cuda/mmq/mmid.cu cuda/mmq/mmvq.cu
  cuda/mmq/test/d2r_stubs.cu)
set_source_files_properties(${hip_sources} PROPERTIES LANGUAGE HIP)
add_library(ds4_reference STATIC
  ds4.c ds4_image.c ds4_distributed.c ds4_tp.c ds4_ssd.c
  ds4_layer_pack.c rax.c ds4_gpu_args.c ${hip_sources})
target_compile_definitions(ds4_reference PRIVATE
  _GNU_SOURCE DS4_ROCM_BUILD __HIP_PLATFORM_AMD__
  GGML_USE_HIP DS4_HIP_MMQ_Y=64)
target_compile_options(ds4_reference PRIVATE
  -O3 -ffast-math -fno-finite-math-only
  "$<$<COMPILE_LANGUAGE:C>:-march=znver5>")
target_include_directories(ds4_reference PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR} cuda/mmq
  ${HIPCUB_INCLUDE_DIR} ${ROCPRIM_INCLUDE_DIR} ${ROCWMMA_INCLUDE_DIR})
target_link_libraries(ds4_reference PUBLIC
  hip::host roc::hipblas roc::hipblaslt roc::rocblas Threads::Threads m stdc++)
add_executable(frontier_reference frontiers.c)
target_include_directories(frontier_reference PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(frontier_reference PRIVATE _GNU_SOURCE DS4_ROCM_BUILD)
target_compile_options(frontier_reference PRIVATE -O3 -fno-fast-math)
target_link_libraries(frontier_reference PRIVATE ds4_reference)
set_target_properties(frontier_reference PROPERTIES LINKER_LANGUAGE HIP)
add_executable(score_official gguf-tools/quality-testing/score_official.c)
target_include_directories(score_official PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(score_official PRIVATE _GNU_SOURCE DS4_ROCM_BUILD)
target_compile_options(score_official PRIVATE -O3 -fno-fast-math)
target_link_libraries(score_official PRIVATE ds4_reference)
set_target_properties(score_official PROPERTIES LINKER_LANGUAGE HIP)
install(TARGETS score_official frontier_reference RUNTIME DESTINATION bin)
