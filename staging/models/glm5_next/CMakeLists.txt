# GLM-5.3-Flash (glm5-next) model package. A hybrid trunk of KDA linear
# layers with hyper-connections and absorbed DSA layers (indexer + latent
# pooling), group-limited MoE routing with one shared expert, dense lead
# layers, and a self-speculative nextn block. Metadata loader first; the
# oracle, kernels and engine targets join in their phases with their sources.
add_library(gufo_glm5_next STATIC
  chat_template.cpp
  config.cpp
  tokenizer.cpp
  weights.cpp)

target_include_directories(gufo_glm5_next PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_glm5_next PUBLIC gufo_core Threads::Threads)

if (ENGINE_ENABLE_HIP)
  # Quantized GEMV tier: decodes the artifact's mixed map (Q4_K embedding,
  # Q8_0/Q5_K/Q6_K trunk, IQ2_XXS/IQ1_M expert gate/up, IQ3_XXS/IQ4_XS expert
  # down, Q2_K/Q3_K nextn) into the F32 dot on the fly. Linked by the
  # executor, the engine and the GPU validation targets.
  # -ffast-math is deliberately omitted so the decode math stays bit-faithful
  # to src/core/quant/ggml_dequant.cpp.
  add_library(gufo_glm5_next_kernels STATIC
    kernels/rocm/dense_wmma.hip.cpp
    kernels/rocm/gemv.hip.cpp
    kernels/rocm/ops.hip.cpp
    kernels/rocm/routed_compact.hip.cpp
    kernels/rocm/routed_wmma.hip.cpp
    kernels/rocm/attention.hip.cpp)
  set_source_files_properties(kernels/rocm/dense_wmma.hip.cpp
    PROPERTIES LANGUAGE HIP)
  set_source_files_properties(kernels/rocm/gemv.hip.cpp
    PROPERTIES LANGUAGE HIP)
  set_source_files_properties(kernels/rocm/ops.hip.cpp
    PROPERTIES LANGUAGE HIP)
  set_source_files_properties(kernels/rocm/routed_compact.hip.cpp
    PROPERTIES LANGUAGE HIP)
  set_source_files_properties(kernels/rocm/routed_wmma.hip.cpp
    PROPERTIES LANGUAGE HIP)
  set_source_files_properties(kernels/rocm/attention.hip.cpp
    PROPERTIES LANGUAGE HIP)
  set_target_properties(gufo_glm5_next_kernels PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
  target_include_directories(gufo_glm5_next_kernels PUBLIC
    ${PROJECT_SOURCE_DIR})
  # Keep -UNDEBUG on the whole kernels target: ling3 measured an NDEBUG-only
  # allocator artifact that spills hot kernels to scratch.
  target_compile_options(gufo_glm5_next_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-UNDEBUG;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  target_link_libraries(gufo_glm5_next_kernels PUBLIC gufo_core hip::host)

  # Host-side driver layer: streams the artifact (trunk + nextn in one reader)
  # to the GPU in its GGUF encoding and steps one token per call.
  add_library(gufo_glm5_next_executor STATIC
    kernels/rocm/device_model.cpp
    kernels/rocm/executor.cpp)
  set_target_properties(gufo_glm5_next_executor PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
  target_include_directories(gufo_glm5_next_executor PUBLIC
    ${PROJECT_SOURCE_DIR})
  target_link_libraries(gufo_glm5_next_executor PUBLIC
    gufo_glm5_next_kernels gufo_glm5_next gufo_core hip::host)

  # Model/Session runtime consumed by the serve seam.
  add_library(gufo_glm5_next_engine STATIC
    engine.cpp)
  set_target_properties(gufo_glm5_next_engine PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
  target_include_directories(gufo_glm5_next_engine PUBLIC
    ${PROJECT_SOURCE_DIR})
  target_link_libraries(gufo_glm5_next_engine PUBLIC gufo_glm5_next_executor)
endif()
