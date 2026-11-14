# BF16 eager-parity Qwen3 operators shared by the safetensors models. Compiled
# only with the HIP backend.
add_library(gufo_qwen_eager STATIC
  attention.hip
  ops.hip)
set_source_files_properties(attention.hip ops.hip PROPERTIES LANGUAGE HIP)
target_include_directories(gufo_qwen_eager PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_definitions(gufo_qwen_eager PUBLIC ENGINE_ENABLE_HIP=1)
target_link_libraries(gufo_qwen_eager PUBLIC hip::host)
# The fused attention keeps the options it was tuned with in Qwen-Image-2.1.
set_source_files_properties(attention.hip
  TARGET_DIRECTORY gufo_qwen_eager
  PROPERTIES COMPILE_OPTIONS
  "-O3;${GUFO_CLANG_OPTION}-ffp-contract=off;-gline-tables-only;-Xarch_device;-mllvm=-amdgpu-sched-strategy=iterative-ilp")
