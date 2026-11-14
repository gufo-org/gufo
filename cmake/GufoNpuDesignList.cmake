# The in-tree NPU designs (src/core/npu/kernels/<name>), for gufo_npu_design
# (GufoNpuDesigns.cmake). Included by the main build and by the standalone
# designs-only project (cmake/npu-designs), which Nix builds as a derivation of
# its own so CI reuses it until a design changes.
gufo_npu_design(q8gemm)
gufo_npu_design(qwen3dec PROGRAMS qwen3dec qwen3dec_p
  EXTRA_OUTPUTS qwen3dec_layout.txt qwen3dec_p_layout.txt)
gufo_npu_design(qwen3pre VARIANTS 256 512 1024
  OUTPUTS qwen3pre_256.xclbin qwen3pre_256.bin qwen3pre_256_plan.txt
          qwen3pre_512.xclbin qwen3pre_512.bin qwen3pre_512_plan.txt
          qwen3pre_1024.xclbin qwen3pre_1024.bin qwen3pre_1024_plan.txt)
gufo_npu_design(bfpgemm PROGRAMS bfpgemm bfpgemm_s)

if(GUFO_NPU_DESIGNS_AVAILABLE)
  get_property(artifacts GLOBAL PROPERTY GUFO_NPU_ARTIFACTS)
  list(JOIN artifacts "\n" manifest)
  file(WRITE "${CMAKE_BINARY_DIR}/npu-artifacts.txt" "${manifest}\n")
  if(WIN32)
    set(destination "bin/npu")
  else()
    set(destination "share/gufo/npu")
  endif()
  # Per-design Nix outputs are combined before the application installs the
  # complete manifest. A selected standalone build has only a partial list.
  if(NOT GUFO_NPU_DESIGN)
    install(FILES "${CMAKE_BINARY_DIR}/npu-artifacts.txt" DESTINATION "${destination}")
  endif()
endif()
