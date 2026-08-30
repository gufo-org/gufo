if (ENGINE_ENABLE_XRT)
  add_executable(qwen_mtp_xdna2_rmsnorm_test
    ${PROJECT_SOURCE_DIR}/tests/models/qwen/xdna2/mtp_rmsnorm_test.cpp)
  target_compile_definitions(qwen_mtp_xdna2_rmsnorm_test PRIVATE
    GUFO_AIE_QWEN_MTP_RMSNORM_PROGRAM_DIR="${GUFO_AIE_QWEN_MTP_RMSNORM_PROGRAM_DIR}")
  target_link_libraries(qwen_mtp_xdna2_rmsnorm_test PRIVATE
    gufo_core
    gufo_diagnostics)
  add_test(NAME qwen_mtp_xdna2_rmsnorm_test
    COMMAND qwen_mtp_xdna2_rmsnorm_test)
  set_tests_properties(qwen_mtp_xdna2_rmsnorm_test PROPERTIES
    LABELS "npu;xdna2;xrt;xrt-hardware;qwen;mtp;rmsnorm")

  add_executable(qwen_mtp_xdna2_eh_proj_test
    ${PROJECT_SOURCE_DIR}/tests/models/qwen/xdna2/mtp_eh_proj_test.cpp)
  target_compile_definitions(qwen_mtp_xdna2_eh_proj_test PRIVATE
    GUFO_AIE_QWEN_MTP_EH_PROJ_PROGRAM_DIR="${GUFO_AIE_QWEN_MTP_EH_PROJ_PROGRAM_DIR}")
  target_link_libraries(qwen_mtp_xdna2_eh_proj_test PRIVATE
    gufo_core
    gufo_diagnostics)
  add_test(NAME qwen_mtp_xdna2_eh_proj_test
    COMMAND qwen_mtp_xdna2_eh_proj_test)
  set_tests_properties(qwen_mtp_xdna2_eh_proj_test PROPERTIES
    LABELS "npu;xdna2;xrt;xrt-hardware;qwen;mtp;gemv")
endif()

add_executable(qwen_aie2p_w4a8_pack_test
  ${PROJECT_SOURCE_DIR}/tests/kernels/qwen38_27b/test_aie2p_w4a8.cpp)
target_link_libraries(qwen_aie2p_w4a8_pack_test PRIVATE gufo_core)
add_test(NAME qwen_aie2p_w4a8_pack_test
  COMMAND qwen_aie2p_w4a8_pack_test)
set_tests_properties(qwen_aie2p_w4a8_pack_test PROPERTIES
  LABELS "cpu;qwen;mtp;w4a8;quantization")

function(add_qwen_cpu_test target source labels)
  add_executable(${target} ${PROJECT_SOURCE_DIR}/${source})
  target_link_libraries(${target} PRIVATE gufo_core)
  add_test(NAME ${target} COMMAND ${target})
  set_tests_properties(${target} PROPERTIES LABELS "${labels}")
endfunction()

# Qwen tests mirror src/models/qwen so focused module, policy, and runtime
# changes have an obvious validation target while preserving CTest names.
add_qwen_cpu_test(qwen_mtp_reference_test
  tests/models/qwen/mtp/reference_test.cpp
  "cpu;core;qwen;mtp;external-model")
set_tests_properties(qwen_mtp_reference_test PROPERTIES SKIP_RETURN_CODE 77)

add_qwen_cpu_test(qwen_tokenizer_test
  tests/models/qwen/tokenization/tokenizer_test.cpp
  "cpu;tokenization;qwen")
add_qwen_cpu_test(qwen_chat_template_test
  tests/models/qwen/tokenization/chat_template_test.cpp
  "cpu;tokenization;chat_template")

add_qwen_cpu_test(qwen_cpu_oracles_test
  tests/models/qwen/cpu/oracles_test.cpp
  "cpu;models;qwen;oracles")
add_qwen_cpu_test(qwen_forward_test
  tests/models/qwen/cpu/forward_test.cpp
  "cpu;models;qwen;forward;reference")
add_qwen_cpu_test(qwen_generator_test
  tests/models/qwen/cpu/generator_test.cpp
  "cpu;models;qwen;generator")
add_qwen_cpu_test(qwen_ssm_test
  tests/models/qwen/cpu/ssm_test.cpp
  "cpu;models;qwen;ssm;reference")
add_qwen_cpu_test(qwen_oracle_test
  tests/models/qwen/cpu/model_oracle_test.cpp
  "cpu;models;qwen;oracles")

add_qwen_cpu_test(qwen_l1_norm_module_test
  tests/models/qwen/modules/norm_test.cpp
  "cpu;models;qwen;module;norm")
add_qwen_cpu_test(qwen_l1_ffn_module_test
  tests/models/qwen/modules/ffn_test.cpp
  "cpu;models;qwen;module;ffn")
add_qwen_cpu_test(qwen_l1_ssm_module_test
  tests/models/qwen/modules/ssm_test.cpp
  "cpu;models;qwen;module;ssm")

add_qwen_cpu_test(qwen_fusion_route_test
  tests/models/qwen/hip/execution_policy_test.cpp
  "cpu;models;qwen;hip;policy;fusion")
add_qwen_cpu_test(qwen_attention_policy_test
  tests/models/qwen/hip/attention_policy_test.cpp
  "cpu;models;qwen;hip;attention;dispatch")
add_qwen_cpu_test(qwen_gemm_route_test
  tests/models/qwen/gemm_route_test.cpp
  "cpu;models;qwen;gemm;dispatch")

add_qwen_cpu_test(qwen_dflash_reference_test
  tests/models/qwen/dflash/reference_test.cpp
  "cpu;core;qwen;dflash;dflash2")

if (ENGINE_ENABLE_HIP)
  add_executable(qwen_mtp_gpu_test
    ${PROJECT_SOURCE_DIR}/tests/models/qwen/hip/mtp_test.cpp)
  target_link_libraries(qwen_mtp_gpu_test PRIVATE gufo_core)
  add_test(NAME qwen_mtp_gpu_test COMMAND qwen_mtp_gpu_test)
  set_tests_properties(qwen_mtp_gpu_test PROPERTIES
    LABELS "gpu;hip;gfx1151;qwen;mtp;external-model"
    SKIP_RETURN_CODE 77)

  add_executable(qwen_dflash_gpu_test
    ${PROJECT_SOURCE_DIR}/tests/models/qwen/hip/dflash_test.cpp)
  target_link_libraries(qwen_dflash_gpu_test PRIVATE gufo_core)
  add_test(NAME qwen_dflash_gpu_test COMMAND qwen_dflash_gpu_test)
  set_tests_properties(qwen_dflash_gpu_test PROPERTIES
    LABELS "gpu;hip;gfx1151;qwen;dflash;dflash2;external-model"
    SKIP_RETURN_CODE 77)

  function(add_qwen_hip_test target source labels)
    add_executable(${target} ${PROJECT_SOURCE_DIR}/${source})
    target_link_libraries(${target} PRIVATE
      gufo_core hip::host roc::hipblas roc::rocblas)
    add_test(NAME ${target} COMMAND ${target})
    set_tests_properties(${target} PROPERTIES
      LABELS "${labels}"
      SKIP_RETURN_CODE 77)
  endfunction()

  add_qwen_hip_test(qwen_hip_device_gate_test
    tests/models/qwen/hip/device_gate_test.cpp
    "hip;qwen;test-support;device-gate")

  add_qwen_hip_test(qwen_gpu_ops_test
    tests/models/qwen/hip/basic/blas_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;basic;blas")
  add_qwen_hip_test(qwen_elementwise_ops_test
    tests/models/qwen/hip/basic/elementwise_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;basic;elementwise")

  add_qwen_hip_test(qwen_attention_decode_ops_test
    tests/models/qwen/hip/attention/backend_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;attention;decode;backend")
  add_qwen_hip_test(qwen_attention_long_context_ops_test
    tests/models/qwen/hip/attention/long_context_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;attention;decode;long-context")
  add_qwen_hip_test(qwen_attention_kv_storage_ops_test
    tests/models/qwen/hip/attention/kv_storage_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;attention;decode;long-context;snapshot;opt-c0133-fp16-kv")
  add_qwen_hip_test(qwen_attention_fusion_ops_test
    tests/models/qwen/hip/attention/qk_rope_kv_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;attention;fusion;opt-c010-qk-rope-kv")
  add_qwen_hip_test(qwen_attention_wmma_prefill_ops_test
    tests/models/qwen/hip/attention/wmma_prefill_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;attention;prefill;opt-c177-attn-wmma")
  add_qwen_hip_test(qwen_attention_split_prefill_ops_test
    tests/models/qwen/hip/attention/split_prefill_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;attention;prefill;opt-c165-attn-split")
  add_qwen_hip_test(qwen_attention_projection_ops_test
    tests/models/qwen/hip/attention/projection_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;attention;ssm;projection;opt-c010-rmsnorm-projection")
  add_qwen_hip_test(qwen_attention_component_ops_test
    tests/models/qwen/hip/attention/component_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;attention;component")

  add_qwen_hip_test(qwen_ssm_ops_test
    tests/models/qwen/hip/ssm_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;ssm;opt-c010-ssm-gate-residual;opt-c010-rmsnorm-projection;opt-c0138-bf16-recurrent")
  add_qwen_hip_test(qwen_ffn_fusion_ops_test
    tests/models/qwen/hip/ffn/swiglu_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;ffn;fusion;opt-c010-ffn-swiglu")
  add_qwen_hip_test(qwen_ffn_residual_ops_test
    tests/models/qwen/hip/ffn/residual_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;ffn;residual;opt-c010-residual-rmsnorm;opt-c010-ssm-gate-residual")

  add_qwen_hip_test(qwen_graph_prefetch_ops_test
    tests/models/qwen/hip/graph_prefetch_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;graph;prefetch;opt-c014-layer-prefetch")
  add_qwen_hip_test(qwen_quant_gemv_ops_test
    tests/models/qwen/hip/quant/q8_gemv_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;quant;gemv;q8;opt-c162-q8k-model;opt-c206-q8-small-batch")
  add_qwen_hip_test(qwen_prefill_quant_gemm_ops_test
    tests/models/qwen/hip/quant/prefill_quant_gemm_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;quant;gemm;prefill;q8;opt-c163-blocked-w8a8")
  add_qwen_hip_test(qwen_kquant_gemv_ops_test
    tests/models/qwen/hip/quant/kquant_gemv_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;quant;gemv;k-quant")
  add_qwen_hip_test(qwen_dequant_ops_test
    tests/models/qwen/hip/dequant_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;quant;dequant;opt-c162-q8k-model")
  add_qwen_hip_test(qwen_module_ops_test
    tests/models/qwen/hip/module_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;module")
  add_qwen_hip_test(qwen_l2_gpu_integration_test
    tests/models/qwen/hip/integration_test.cpp
    "hip;gpu;kernels;qwen;module;l2")
endif()
