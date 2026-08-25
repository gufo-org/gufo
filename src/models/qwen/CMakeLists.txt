target_sources(strix_core PRIVATE
  ${PROJECT_SOURCE_DIR}/src/models/qwen/tokenizer.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/chat_template.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/oracles.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/state.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/weights.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/ssm.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/forward.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/modules/attention.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/modules/embed.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/modules/ffn.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/modules/norm.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/modules/quant_gemm.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/modules/residual.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/modules/rope.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/modules/sample.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/modules/ssm.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/modules/unembed.cpp
   ${PROJECT_SOURCE_DIR}/src/models/qwen/generator.cpp
   ${PROJECT_SOURCE_DIR}/src/models/qwen/mtp_reference.cpp
   ${PROJECT_SOURCE_DIR}/src/models/qwen/dflash_reference.cpp)

if (ENGINE_ENABLE_HIP)
  target_sources(strix_core PRIVATE
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/attention.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/batched_ssm.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/decode_ops.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/prefill_embed.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/prefill_gemm.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/prefill_quant_gemm.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/prefill_norm.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/prefill_residual.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/prefill_rope.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/prefill_ssm.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/prefill_swiglu.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/prefill_unpack.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/embed.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/residual.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/unpack.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/rope.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/sample.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/norm.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/gemv.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/gemv_quant.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/fused.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/common.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/qkv.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/ssm.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/ssm_decode_recurrence.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/ssm_recurrence.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/ssm_row_split.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/swiglu.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/swiglu_rmsnorm.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/attention_batched.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/attention_ck.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/attention_decode.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/attention_decode_graph.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/attention_split.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/attention_tile.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/attention_wmma.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/hipblas_gemm.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/hipblaslt_gemm.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/dflash_kernels.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/arena.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/decode.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/decode_step.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/executor.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/model_loader.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/prefill.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/prefill_chunk.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/ssm_replay.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/mtp/model.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/mtp/executor.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/mtp/backend.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/dflash/model.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/dflash/executor.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/dflash/backend.cpp)
endif()

if (ENGINE_ENABLE_XRT)
  target_sources(strix_core PRIVATE
    ${PROJECT_SOURCE_DIR}/src/models/qwen/xdna2/mtp_eh_proj.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/xdna2/mtp_rmsnorm.cpp)
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
