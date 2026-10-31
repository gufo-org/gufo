# Shared Qwen HIP operators. Compiled only with the HIP backend; every source
# here previously compiled inside gufo_core, so the per-source options below are
# re-anchored to this target instead of gufo_core. The list order is the order
# gufo_core used, which keeps the archive layout comparable when auditing.
add_library(gufo_qwen_kernels STATIC
  attention_batched.hip
  attention_decode_graph.hip
  attention_decode.hip
  attention_decode_verify_wave64.hip
  attention_tile.hip
  attention_wmma.hip
  dflash_kernels.hip
  embed.hip
  fused.hip
  gemv.hip
  gemv_quant.hip
  hipblas_gemm.hip
  hipblaslt_gemm.hip
  norm.hip
  prefill_embed.hip
  prefill_fp16.hip
  prefill_gemm.hip
  prefill_norm.hip
  prefill_quant_gemm.hip
  prefill_quant_wave64.hip
  prefill_residual.hip
  prefill_rope.hip
  prefill_ssm.hip
  prefill_swiglu.hip
  prefill_unpack.hip
  qkv.hip
  residual.hip
  rope.hip
  sample.hip
  small_batch_quant16_wave64.hip
  small_batch_wave64.hip
  ssm_decode_recurrence.hip
  ssm.hip
  ssm_recurrence.hip
  ssm_row_split.hip
  swiglu.hip
  unpack.hip
  verify_gemm_27b.hip)

target_include_directories(gufo_qwen_kernels PUBLIC ${PROJECT_SOURCE_DIR})
# gufo_core supplied these device headers privately, so they are repeated here
# rather than inherited.
target_include_directories(gufo_qwen_kernels SYSTEM PUBLIC
  ${HIPCUB_INCLUDE_DIR}
  ${ROCPRIM_INCLUDE_DIR})
target_link_libraries(gufo_qwen_kernels PUBLIC
  gufo_core
  gufo_qwen_hip_common)

# HIP helpers must share the kernel's wave size, so compile this unit
# separately while the other Qwen kernels retain wave32.
set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/small_batch_wave64.hip
  ${CMAKE_CURRENT_SOURCE_DIR}/prefill_quant_wave64.hip
  ${CMAKE_CURRENT_SOURCE_DIR}/attention_decode_verify_wave64.hip
  TARGET_DIRECTORY gufo_qwen_kernels
  PROPERTIES COMPILE_OPTIONS -mwavefrontsize64)
# Iterative ILP improves Q4/Q5 groups of sixteen without changing arithmetic.
# Keep it isolated: other widths regress. Line tables retain source profiling
# while avoiding an LLVM register-pressure crash with full debug information.
set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/small_batch_quant16_wave64.hip
  TARGET_DIRECTORY gufo_qwen_kernels
  PROPERTIES COMPILE_OPTIONS
  "-mwavefrontsize64;-gline-tables-only;-Xarch_device;-mllvm=-amdgpu-sched-strategy=iterative-ilp")
# SLP vectorization puts the K-quant scales and offsets of the blocked
# prefill GEMMs into wide register tuples. With clang 23, the Q3_K and Q6_K
# 128x128 tiles then spill. Disable SLP for the two units on all compilers.
set_property(
  SOURCE
    ${CMAKE_CURRENT_SOURCE_DIR}/prefill_quant_gemm.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/prefill_quant_wave64.hip
  TARGET_DIRECTORY gufo_qwen_kernels
  APPEND PROPERTY COMPILE_OPTIONS -fno-slp-vectorize)
