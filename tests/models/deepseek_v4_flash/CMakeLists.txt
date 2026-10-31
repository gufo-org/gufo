# Pre-migration serialization check needs neither HIP nor model weights:
# it compiles continuation_identity.cpp itself for its own fixture path and
# a HIP-free compile, instead of linking gufo_serving.
add_executable(deepseek_v4_flash_continuation_identity_test continuation_identity_test.cpp
  ${PROJECT_SOURCE_DIR}/src/cli/serve/continuation_identity.cpp)
target_include_directories(deepseek_v4_flash_continuation_identity_test PRIVATE ${PROJECT_SOURCE_DIR})
target_compile_definitions(deepseek_v4_flash_continuation_identity_test PRIVATE
  GUFO_IDENTITY_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/continuation_identity")
add_test(NAME deepseek_v4_flash_continuation_identity_test COMMAND deepseek_v4_flash_continuation_identity_test)
set_tests_properties(deepseek_v4_flash_continuation_identity_test PROPERTIES LABELS "cpu;deepseek_v4_flash;cache;correctness")

# DS4 CTest checks are registered here. Shared CLI/parser and scheduler
# tests remain with their shared components.
add_executable(ds4_sampling_test sampling_test.cpp)
target_link_libraries(ds4_sampling_test PRIVATE gufo_core gufo_deepseek_v4_flash_text)
add_test(NAME ds4.sampling COMMAND ds4_sampling_test)
set_tests_properties(ds4.sampling PROPERTIES LABELS "cpu;ds4;sampling")

add_executable(ds4_chat_template_test chat_template_test.cpp)
target_link_libraries(ds4_chat_template_test PRIVATE gufo_core
  gufo_deepseek_v4_flash_text)
target_compile_definitions(ds4_chat_template_test PRIVATE
  GUFO_DEEPSEEK_CHAT_TEMPLATE_REFERENCE="${PROJECT_SOURCE_DIR}/src/models/deepseek_v4_flash/reference/artifact_chat_template.jinja"
  GUFO_CHAT_TEMPLATE_HF_GOLDENS="${PROJECT_SOURCE_DIR}/tests/fixtures/chat_template_hf_goldens.json")
add_test(NAME ds4.template COMMAND ds4_chat_template_test)
set_tests_properties(ds4.template PROPERTIES LABELS "cpu;ds4;tokenization")

add_executable(ds4_cli_test cli_test.cpp)
target_link_libraries(ds4_cli_test PRIVATE gufo_llm_cli)
add_test(NAME ds4.cli COMMAND ds4_cli_test)
set_tests_properties(ds4.cli PROPERTIES LABELS "cpu;ds4;cli")

if(ENGINE_ENABLE_HIP)
  add_executable(ds4_markov_test markov_test.cpp)
  target_link_libraries(ds4_markov_test PRIVATE gufo_deepseek_v4_flash)
  add_test(NAME ds4.markov COMMAND ds4_markov_test)
  set_tests_properties(ds4.markov PROPERTIES
    LABELS "gpu;hip;gfx1151;ds4;quality"
    RESOURCE_LOCK ds4_model
    SKIP_RETURN_CODE 77)

  add_executable(ds4_compressor_test compressor_test.cpp)
  target_link_libraries(ds4_compressor_test PRIVATE gufo_deepseek_v4_flash)
  add_test(NAME ds4.compressor COMMAND ds4_compressor_test)
  set_tests_properties(ds4.compressor PROPERTIES
    LABELS "gpu;hip;gfx1151;ds4;quality"
    RESOURCE_LOCK ds4_model
    SKIP_RETURN_CODE 77)

  add_executable(ds4_router_test router_test.cpp)
  target_link_libraries(ds4_router_test PRIVATE gufo_deepseek_v4_flash)
  add_test(NAME ds4.router COMMAND ds4_router_test)
  set_tests_properties(ds4.router PROPERTIES
    LABELS "gpu;hip;gfx1151;ds4;quality"
    RESOURCE_LOCK ds4_model
    SKIP_RETURN_CODE 77)

  add_executable(ds4_load_test load_test.cpp)
  target_link_libraries(ds4_load_test PRIVATE gufo_deepseek_v4_flash)
  add_test(NAME ds4.load COMMAND ds4_load_test)
  set_tests_properties(ds4.load PROPERTIES LABELS "cpu;ds4;loader")

  add_test(NAME ds4.chat COMMAND ds4_cli_test --sampled-chat)
  set_tests_properties(ds4.chat PROPERTIES
    LABELS "gpu;hip;ds4;cli;external-model;slow"
    RESOURCE_LOCK ds4_model
    SKIP_RETURN_CODE 77)

  add_executable(ds4_quality_test quality_test.cpp token_goldens.cpp
    official_reference.cpp)
  target_compile_definitions(ds4_quality_test PRIVATE
    GUFO_DS4_BOUNDARY_PROMPT="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/indexer-boundary-prompt.txt"
    GUFO_DS4_OFFICIAL_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/official-0731.json"
    GUFO_CHAT_TEMPLATE_HF_GOLDENS="${PROJECT_SOURCE_DIR}/tests/fixtures/chat_template_hf_goldens.json")
  target_link_libraries(ds4_quality_test PRIVATE gufo_deepseek_v4_flash)
  add_test(NAME ds4.target COMMAND ds4_quality_test)
  add_test(NAME ds4.dspark COMMAND ds4_quality_test --dspark)
  set_tests_properties(ds4.target ds4.dspark PROPERTIES
    LABELS "gpu;hip;gfx1151;ds4;quality;external-model;slow"
    RESOURCE_LOCK ds4_model
    SKIP_RETURN_CODE 77)

  add_executable(ds4_serving_test serving_test.cpp)
  # gufo_serving still reaches the Qwen 27B model and vision stack through
  # inference_backend.cpp; step 10 removes that interim edge.
  target_link_libraries(ds4_serving_test PRIVATE gufo_serving)
  add_test(NAME ds4.serving COMMAND ds4_serving_test)
  set_tests_properties(ds4.serving PROPERTIES
    LABELS "gpu;hip;ds4;server;external-model;slow"
    RESOURCE_LOCK ds4_model
    SKIP_RETURN_CODE 77)
endif()

if(ENGINE_ENABLE_HIP)
  add_executable(ds4_quantize_test quantize_test.hip)
  target_include_directories(ds4_quantize_test PRIVATE ${PROJECT_SOURCE_DIR})
  target_compile_definitions(ds4_quantize_test PRIVATE GGML_USE_HIP GGML_HIP_NO_VMM)
  target_link_libraries(ds4_quantize_test PRIVATE gufo_deepseek_v4_flash_mmq)
  add_test(NAME ds4.quantization COMMAND ds4_quantize_test)
  set_tests_properties(ds4.quantization PROPERTIES
    LABELS "gpu;hip;gfx1151;ds4;quality"
    RESOURCE_LOCK ds4_model
    SKIP_RETURN_CODE 77)

  add_executable(ds4_projection_test projection_test.hip moe_test.hip)
  target_include_directories(ds4_projection_test SYSTEM PRIVATE ${ROCWMMA_INCLUDE_DIR})
  target_include_directories(ds4_projection_test PRIVATE ${PROJECT_SOURCE_DIR})
  target_compile_options(ds4_projection_test PRIVATE -O3 -ffast-math -fno-finite-math-only)
  target_link_libraries(ds4_projection_test PRIVATE hip::host)
  add_test(NAME ds4.projections COMMAND ds4_projection_test)
  set_tests_properties(ds4.projections PROPERTIES
    LABELS "gpu;hip;gfx1151;ds4;quality"
    RESOURCE_LOCK ds4_model
    SKIP_RETURN_CODE 77)

  add_executable(ds4_attention_test attention_test.hip indexer_test.hip)
  target_include_directories(ds4_attention_test PRIVATE ${PROJECT_SOURCE_DIR})
  target_include_directories(ds4_attention_test SYSTEM PRIVATE
    ${HIPCUB_INCLUDE_DIR} ${ROCPRIM_INCLUDE_DIR} ${ROCWMMA_INCLUDE_DIR})
  target_compile_options(ds4_attention_test PRIVATE -O3 -ffast-math -fno-finite-math-only)
  target_link_libraries(ds4_attention_test PRIVATE hip::host)
  add_test(NAME ds4.attention COMMAND ds4_attention_test)
  set_tests_properties(ds4.attention PROPERTIES
    LABELS "gpu;hip;gfx1151;ds4;quality"
    RESOURCE_LOCK ds4_model
    SKIP_RETURN_CODE 77)
endif()

add_test(NAME ds4.dataset
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/dataset_test.py)
set_tests_properties(ds4.dataset PROPERTIES
  LABELS "cpu;ds4;quality;dataset;provenance;license")

add_executable(ds4_eval_test eval_test.cpp)
target_compile_definitions(ds4_eval_test PRIVATE
  GUFO_EVAL_TEST_DATA="${PROJECT_SOURCE_DIR}/tests/fixtures/eval/antirez-ds4.json")
target_link_libraries(ds4_eval_test PRIVATE gufo_core gufo_eval)
add_test(NAME ds4.eval COMMAND ds4_eval_test)
set_tests_properties(ds4.eval PROPERTIES LABELS "cpu;ds4;quality;eval;dataset;extractor")
