# Qwen tokenizer and chat template (CPU). Flash-Next, Qwen-Image-2.1 and the
# Qwen 27B model all bind these; the DeepSeek test binds the chat template too.
add_library(gufo_qwen_text STATIC
  tokenizer.cpp
  chat_template.cpp)
target_include_directories(gufo_qwen_text PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_qwen_text PUBLIC
  gufo_core
  gufo_qwen_config
  gufo_qwen_control_tokens)
