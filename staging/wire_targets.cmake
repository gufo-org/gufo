# Wires the staging model engines onto the root CLI and serving targets.
# Included from the bottom of the root CMakeLists (where every consumer
# target exists) when STAGING_MODELS is ON; source paths are root-relative.

target_sources(gufo_llm_cli PRIVATE
  staging/models/gemma4/cli_runner.cpp
  staging/models/ling3_0/cli_runner.cpp
  staging/models/qwen36_a3b/cli_runner.cpp)
target_link_libraries(gufo_llm_cli PUBLIC gufo_staging_integration)

if(ENGINE_ENABLE_HIP)
  target_link_libraries(gufo_llm_cli PUBLIC gufo_qwen36_a3b_engine
    gufo_gemma4_engine gufo_ling3_0_engine gufo_nemotron_h_moe_engine
    gufo_flint_engine)
  # Extension targets link their operators PRIVATE, so transitive archives
  # must be named here for every gufo_llm_cli consumer.
  target_link_libraries(gufo_llm_cli PUBLIC gufo_qwen_hip gufo_qwen_kernels
    gufo_qwen_vision hip::host)
endif()

# Staging model engines reach execution through the dispatch chain in
# inference_backend.cpp; their serve runners compile into this tier.
if(ENGINE_ENABLE_HIP)
  target_sources(gufo_serving PRIVATE
    staging/models/flint/serve_runner.cpp
    staging/models/gemma4/serve_runner.cpp
    staging/models/glm5_next/serve_runner.cpp
    staging/models/ling3_0/serve_runner.cpp
    staging/models/nemotron_h_moe/serve_runner.cpp
    staging/models/qwen36_a3b/serve_runner.cpp)
  target_link_libraries(gufo_serving PUBLIC
    gufo_staging_integration
    gufo_gemma4_engine gufo_ling3_0_engine gufo_qwen36_a3b_engine
    gufo_nemotron_h_moe_engine gufo_flint_engine gufo_glm5_next_engine)
  # Extension targets link their operators PRIVATE, so transitive archives
  # must be named here for every gufo_serving consumer.
  target_link_libraries(gufo_serving PUBLIC gufo_qwen_hip
    gufo_qwen_kernels gufo_qwen_vision)
endif()

target_link_libraries(gufo_http PUBLIC gufo_gemma4)

target_sources(gufo PRIVATE
  staging/models/flint/serve_runner.cpp
  staging/models/gemma4/serve_runner.cpp
  staging/models/glm5_next/serve_runner.cpp
  staging/models/ling3_0/serve_runner.cpp
  staging/models/nemotron_h_moe/serve_runner.cpp
  staging/models/qwen36_a3b/serve_runner.cpp)

if(BUILD_TESTING)
  # Gemma 4 native tool-call parsing reached the test through gufo_http.
  add_subdirectory(staging/tests/models/qwen36_a3b)
  add_subdirectory(staging/tests/models/gemma4)
  add_subdirectory(staging/tests/models/ling3_0)
  add_subdirectory(staging/tests/models/glm5_next)
  add_subdirectory(staging/tests/models/nemotron_h_moe)
  add_subdirectory(staging/tests/models/flint)
endif()
