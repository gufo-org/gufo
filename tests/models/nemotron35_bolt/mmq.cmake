# GPU operator check of the vendored quantized matrix kernels against the
# core ggml dequantizers (included only when HIP is enabled).
add_executable(nemotron35_bolt_mmq_ops_test mmq_ops_test.cpp)
target_link_libraries(nemotron35_bolt_mmq_ops_test PRIVATE
  gufo_nemotron35_bolt_mmq gufo_core hip::host)
add_test(NAME nemotron35_bolt.mmq_ops COMMAND nemotron35_bolt_mmq_ops_test)
set_tests_properties(nemotron35_bolt.mmq_ops PROPERTIES
  LABELS "nemotron35_bolt;hip;gpu;gfx1151")
add_dependencies(nemotron35_bolt_tests nemotron35_bolt_mmq_ops_test)
