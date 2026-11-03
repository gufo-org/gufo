# Header-only Qwen3.8 speculative draft vocabulary: the prefix-plus-table output
# head shared by the Flash-Next MTP draft and the 27B DFlash2 draft.
add_library(gufo_qwen_draft_vocab INTERFACE)
target_include_directories(gufo_qwen_draft_vocab INTERFACE ${PROJECT_SOURCE_DIR})
