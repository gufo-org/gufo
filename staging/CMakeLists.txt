# Staging-only model engines. The root CMakeLists adds this directory when
# STAGING_MODELS is ON; the wiring onto root CLI and serving targets lives in
# staging/wire_targets.cmake, included once every root target exists.

if(ENGINE_ENABLE_HIP)
  # Shared dense binary16 prefill GEMM used by the staging model engines.
  add_subdirectory(core/gemm)
endif()

# Qwen3.6-35B-A3B (qwen35moe) — isolated hybrid GDN + gated-GQA + MoE engine.
add_subdirectory(models/qwen36_a3b)

# Gemma-4-26B-A4B (gemma4) — isolated hybrid SWA/full attention + MoE engine.
add_subdirectory(models/gemma4)

# Ling 3.0 Flash (bailingmoe3) — hybrid KDA + absorbed-MLA + MoE engine.
add_subdirectory(models/ling3_0)

# Nemotron 3.5 Bolt (nemotron_h_moe) — hybrid Mamba-2 + NoPE-GQA + latent
# MoE engine. Model-private kernels only; no other model tree is modified.
add_subdirectory(models/nemotron_h_moe)

# Flint Preview 40B-A2B (flint) — dense-lead MoE with gated rotary/window
# attention, QK norms and attention sinks. Metadata loader first.
add_subdirectory(models/flint)

# GLM-5.3-Flash (glm5-next) — hybrid KDA/mHC + absorbed DSA + group-limited
# MoE with a nextn draft block. Metadata loader first.
add_subdirectory(models/glm5_next)

# Serve- and CLI-seam glue consumed by gufo_llm_cli and gufo_serving.
add_subdirectory(integration)
