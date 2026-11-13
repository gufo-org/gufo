# The in-tree NPU designs (src/core/npu/kernels/<name>), for gufo_npu_design
# (GufoNpuDesigns.cmake). Included by the main build and by the standalone
# designs-only project (cmake/npu-designs), which Nix builds as a derivation of
# its own so CI reuses it until a design changes.
gufo_npu_design(q8gemm)
gufo_npu_design(qwen3dec PROGRAMS qwen3dec qwen3dec_p)
gufo_npu_design(qwen3pre
  OUTPUTS qwen3pre_256.xclbin qwen3pre_256.bin qwen3pre_256_plan.txt
          qwen3pre_512.xclbin qwen3pre_512.bin qwen3pre_512_plan.txt
          qwen3pre_1024.xclbin qwen3pre_1024.bin qwen3pre_1024_plan.txt)
gufo_npu_design(bfpgemm PROGRAMS bfpgemm bfpgemm_s)
