# Inkling 2 Mini model extension.

# Host-only: configuration, probe, weight table and the text front-end
# (tokenizer, chat template, tool dialect). No HIP, so probes and the text
# tests run on machines without a GPU.
gufo_extension_target(
  NAME gufo_inkling2
  SOURCES
    config.cpp
    weights.cpp
    probe.cpp
    extension.cpp
    text/chat_template.cpp
    text/gguf_meta.cpp
    text/json.cpp
    text/pretokenizer.cpp
    text/tokenizer.cpp
    text/tool_dialect.cpp
    text/unicode.cpp
    text/unicode_tables.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    gufo_core
)
if(TARGET gufo_inkling2)
  target_include_directories(gufo_inkling2 PUBLIC ${PROJECT_SOURCE_DIR})
endif()

gufo_extension_target(
  NAME gufo_inkling2_hip
  SOURCES
    engine.cpp
    state_layout.cpp
    cache_adapter.cpp
    runner.cpp
    provider.cpp
    dspark.cpp
    vision/vision_image.cpp
    vision/mmproj_gguf.cpp
  LINK_LIBRARIES
    gufo_inkling2
    gufo_core
    gufo_ext_hip
    gufo_cache_hip
    hip::host
)
if(TARGET gufo_inkling2_hip)
  gufo_extension_hip_sources(gufo_inkling2_hip
    SOURCES
      hip/kernels.hip
      hip/fast.hip
      hip/dspark.hip
      vision/vision_encoder.hip
  )
  target_compile_options(gufo_inkling2_hip PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:${GUFO_CLANG_OPTION}-O3;-Wno-unused-command-line-argument>)
  target_include_directories(gufo_inkling2_hip PUBLIC ${PROJECT_SOURCE_DIR})
endif()

# Install model-owned license material only when this model is selected.
install(FILES "${GUFO_EXTENSION_BUILD_ROOT}/LICENSE.inkling2"
  DESTINATION share/licenses/gufo/third-party)
install(FILES "${GUFO_EXTENSION_BUILD_ROOT}/THIRD_PARTY.md"
  DESTINATION share/licenses/gufo/third-party
  RENAME inkling2-THIRD_PARTY.md)

gufo_extension_target(
  NAME inkling2_probe_test
  SOURCES
    tests/probe_test.cpp
  LINK_LIBRARIES
    gufo_inkling2
)
gufo_extension_target(
  NAME inkling2_text_test
  SOURCES
    tests/text/text_tests.cpp
  LINK_LIBRARIES
    gufo_inkling2
)
gufo_extension_target(
  NAME inkling2_policy_test
  SOURCES
    tests/spec/policy_test.cpp
  LINK_LIBRARIES
    gufo_inkling2
)
if(TARGET inkling2_text_test)
  target_compile_definitions(inkling2_text_test PRIVATE
    INKLING2_TEXT_GOLDENS_DIR="${GUFO_EXTENSION_BUILD_ROOT}/tests/text/goldens")
endif()

# GPU tools: llama.cpp comparison oracle, benchmark, cache bit-identity.
gufo_extension_target(
  NAME inkling2_oracle
  SOURCES
    tests/oracle/oracle.cpp
  LINK_LIBRARIES
    gufo_inkling2_hip
)
gufo_extension_target(
  NAME inkling2_bench
  SOURCES
    tests/bench/bench.cpp
  LINK_LIBRARIES
    gufo_inkling2_hip
)
gufo_extension_target(
  NAME inkling2_vision_bench
  SOURCES
    tests/bench/vision_bench.cpp
  LINK_LIBRARIES
    gufo_inkling2_hip
    hip::host
)
gufo_extension_target(
  NAME inkling2_spec_test
  SOURCES
    tests/spec/spec_test.cpp
  LINK_LIBRARIES
    gufo_inkling2_hip
)
gufo_extension_target(
  NAME inkling2_cache_test
  SOURCES
    tests/cache/cache_test.cpp
  LINK_LIBRARIES
    gufo_inkling2_hip
)

if(TARGET inkling2_probe_test)
  add_test(NAME inkling2.probe COMMAND inkling2_probe_test)
  set_tests_properties(inkling2.probe PROPERTIES LABELS inkling2)
endif()
if(TARGET inkling2_policy_test)
  add_test(NAME inkling2.policy COMMAND inkling2_policy_test)
  set_tests_properties(inkling2.policy PROPERTIES LABELS inkling2)
endif()
if(TARGET inkling2_text_test)
  add_test(NAME inkling2.text COMMAND inkling2_text_test)
  set_tests_properties(inkling2.text PROPERTIES LABELS inkling2 SKIP_RETURN_CODE 77)
endif()
