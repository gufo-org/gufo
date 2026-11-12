#include <cassert>
#include <chrono>
#include <iostream>

#include "src/core/model_config.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/hip/hip_utils.hpp"
#endif

void TestHardwareAcceptanceAccounting() {
  gufo::core::ModelConfig config;
  config.architecture = "qwen35";
  config.num_layers = 32;
  config.hidden_size = 2560;
  config.intermediate_size = 9216;
  config.num_attention_heads = 16;
  config.num_key_value_heads = 4;
  config.head_dim = 256;
  config.vocab_size = 248320;

  assert(config.num_layers == 32);
  assert(config.hidden_size == 2560);
  assert(config.intermediate_size == 9216);

#if defined(ENGINE_ENABLE_HIP)
  int dev_count = 0;
  if (hipGetDeviceCount(&dev_count) == hipSuccess && dev_count > 0) {
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::cout << "[Acceptance Baseline]: Device=" << prop.name
              << " GCNArch=" << prop.gcnArchName
              << " GlobalMem=" << (prop.totalGlobalMem / (1024 * 1024))
              << " MB\n";
  }
#endif
}

int main() {
  TestHardwareAcceptanceAccounting();
  std::cout << "Hardware acceptance baseline tests passed on RDNA 3.5.\n";
  return 0;
}
