# CPU-only DeepSeek V4 Flash text tier: chat-template rendering and the
# scalar sampler kernels.
#
# Card 08 (RFC step 5) gives these two sources one implementation each. Before,
# `gufo_deepseek_v4_flash` compiled them for HIP builds while
# `ds4_sampling_test`, `ds4_chat_template_test` and `openai_chat_test` each
# recompiled a copy, so a CPU build had no DeepSeek text support of its own.
# `tool_calls.cpp` deliberately stays out of this target: the tool-call scanner
# belongs to `gufo_http` until the step-10 text migration (Card 22), and
# `chat_template.cpp` needs no symbol from it (only the header-only
# `DeepSeekToolSyntax` constants).
#
# The parent model directory is registered before the root warning/sanitizer
# block, so it is added from here instead: as test-executable sources these two
# files compiled with warnings and sanitizers, and this position preserves that.
add_library(gufo_deepseek_v4_flash_text STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../chat_template.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../runtime/sampling.cpp)

target_include_directories(gufo_deepseek_v4_flash_text PUBLIC ${PROJECT_SOURCE_DIR})
# sampling.cpp includes the runtime's internal header by its plain name.
target_include_directories(gufo_deepseek_v4_flash_text SYSTEM PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../runtime)
target_compile_definitions(gufo_deepseek_v4_flash_text PRIVATE
  _GNU_SOURCE
  DS4_ROCM_BUILD=1)
# The shipped sampling codegen is preserved exactly: inside
# gufo_deepseek_v4_flash these sources built with the model's own -O3
# -ffast-math tier, and that is what the release binary must keep.
if (ENGINE_ENABLE_HIP)
  target_compile_options(gufo_deepseek_v4_flash_text PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-ffast-math;-fno-finite-math-only>)
endif()
# Assertions stay armed in test builds, where the duplicated copies had
# `-UNDEBUG` from the test scope; production builds keep NDEBUG.
if (BUILD_TESTING AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(gufo_deepseek_v4_flash_text PRIVATE -UNDEBUG)
endif()
target_link_libraries(gufo_deepseek_v4_flash_text PUBLIC gufo_core)
