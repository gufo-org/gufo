# CPU-only Qwen GGUF configuration extraction. It links nothing but gufo_core's
# GGUF/config surface, so an ENGINE_ENABLE_HIP=OFF build compiles and tests it;
# that build is the gate proving config users never pull HIP.
add_library(gufo_qwen_config STATIC qwen_config.cpp)
target_include_directories(gufo_qwen_config PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_qwen_config PUBLIC gufo_core)
