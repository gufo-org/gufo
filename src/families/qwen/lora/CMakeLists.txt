# CPU-only llama.cpp GGUF LoRA reader and Q8_0 merge. Each model enumerates the
# matrices an adapter may patch and places the merged rows itself.
add_library(gufo_qwen_lora STATIC lora.cpp)
target_include_directories(gufo_qwen_lora PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_qwen_lora PUBLIC gufo_core)
# Merges must match offline merges bit for bit: no fused multiply-add.
set_source_files_properties(lora.cpp
  PROPERTIES COMPILE_OPTIONS "${GUFO_CLANG_OPTION}-ffp-contract=off")
