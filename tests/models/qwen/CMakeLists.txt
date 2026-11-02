function(add_qwen_cpu_test target source labels)
  add_executable(${target} ${PROJECT_SOURCE_DIR}/${source})
  target_link_libraries(${target} PRIVATE gufo_core
    ${GUFO_QWEN_LIBRARIES})
  add_test(NAME ${target} COMMAND ${target})
  set_tests_properties(${target} PROPERTIES LABELS "${labels}")
endfunction()

# Qwen tests mirror the family libraries under src/families/qwen and the Qwen
# 27B model in src/models/qwen, so focused module, policy, and runtime changes
# have an obvious validation target while preserving CTest names. Shared code
# changes and model changes are covered from here, so the helpers link the
# whole Qwen set (src/models/qwen/CMakeLists.txt defines GUFO_QWEN_LIBRARIES).
add_qwen_cpu_test(qwen_vision_cache_test
  tests/models/qwen/vision_cache_test.cpp
  "cpu;vision;qwen;correctness")
add_qwen_cpu_test(qwen_tokenizer_test
  tests/models/qwen/tokenization/tokenizer_test.cpp
  "cpu;tokenization;qwen")
add_qwen_cpu_test(qwen_chat_template_test
  tests/models/qwen/tokenization/chat_template_test.cpp
  "cpu;tokenization;chat_template")
target_compile_definitions(qwen_chat_template_test PRIVATE
  GUFO_QWEN38_CHAT_TEMPLATE_REFERENCE="${PROJECT_SOURCE_DIR}/src/families/qwen/text/reference/chat_template.jinja"
  GUFO_CHAT_TEMPLATE_HF_GOLDENS="${PROJECT_SOURCE_DIR}/tests/fixtures/chat_template_hf_goldens.json")

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

# These four report candidate logits through the shared comparison support
# (src/testing/compare) via tests/testing/test_common.hpp.
target_link_libraries(qwen_oracle_test PRIVATE gufo_compare)
target_link_libraries(qwen_l1_norm_module_test PRIVATE gufo_compare)
target_link_libraries(qwen_l1_ffn_module_test PRIVATE gufo_compare)
target_link_libraries(qwen_l1_ssm_module_test PRIVATE gufo_compare)

add_qwen_cpu_test(qwen_fusion_route_test
  tests/models/qwen/hip/execution_policy_test.cpp
  "cpu;models;qwen;hip;policy;fusion")
add_qwen_cpu_test(qwen_attention_policy_test
  tests/models/qwen/hip/attention_policy_test.cpp
  "cpu;models;qwen;hip;attention;dispatch")
add_qwen_cpu_test(qwen_gemm_route_test
  tests/models/qwen/gemm_route_test.cpp
  "cpu;models;qwen;gemm;dispatch")

if (ENGINE_ENABLE_HIP)
  function(add_qwen_hip_test target source labels)
    add_executable(${target} ${PROJECT_SOURCE_DIR}/${source})
    target_link_libraries(${target} PRIVATE
      gufo_core ${GUFO_QWEN_LIBRARIES} gufo_qwen_kernels gufo_qwen_vision
      hip::host roc::hipblas roc::rocblas)
    add_test(NAME ${target} COMMAND ${target})
    set_tests_properties(${target} PROPERTIES
      LABELS "${labels}"
      SKIP_RETURN_CODE 77)
  endfunction()

  add_qwen_hip_test(qwen_hip_device_gate_test
    tests/models/qwen/hip/device_gate_test.cpp
    "hip;qwen;test-support;device-gate")
  add_qwen_hip_test(qwen_prompt_hidden_buffer_test
    tests/models/qwen/hip/prompt_hidden_buffer_test.cpp
    "hip;gpu;qwen")

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
  add_qwen_hip_test(qwen_attention_projection_ops_test
    tests/models/qwen/hip/attention/projection_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;attention;ssm;projection")
  add_qwen_hip_test(qwen_attention_component_ops_test
    tests/models/qwen/hip/attention/component_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;attention;component")

  add_qwen_hip_test(qwen_ssm_ops_test
    tests/models/qwen/hip/ssm_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;ssm;opt-c0138-bf16-recurrent")

  add_qwen_hip_test(qwen_graph_ops_test
    tests/models/qwen/hip/graph_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;graph")
  add_qwen_hip_test(qwen_quant_gemv_ops_test
    tests/models/qwen/hip/quant/q8_gemv_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;quant;gemv;q8;opt-c162-q8k-model;opt-c206-q8-small-batch")
  add_qwen_hip_test(qwen_prefill_quant_gemm_ops_test
    tests/models/qwen/hip/quant/prefill_quant_gemm_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;quant;gemm;prefill;q8;opt-c163-blocked-w8a8;opt-c192-swiglu-epilogue")
  add_qwen_hip_test(qwen_q4kxl_quant_ops_test
    tests/models/qwen/hip/quant/q4kxl_quant_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;quant;gemv;gemm;k-quant;iquant;opt-q4kxl")
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
