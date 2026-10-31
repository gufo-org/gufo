# Header-only shared Qwen control-token spellings. Consumers include
# src/families/qwen/control_tokens/control_tokens.hpp; the catalogue and the
# rules for changing it live in src/families/qwen/AGENTS.md.
add_library(gufo_qwen_control_tokens INTERFACE)
target_include_directories(gufo_qwen_control_tokens INTERFACE ${PROJECT_SOURCE_DIR})
