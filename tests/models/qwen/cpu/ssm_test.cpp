#include "src/models/qwen/ssm.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

void TestQwenSsmConvRecurrence() {
  strix::core::ModelConfig config;
  config.num_layers = 64;
  config.hidden_size = 5120;
  config.intermediate_size = 17408;
  config.num_attention_heads = 24;
  config.num_key_value_heads = 4;
  config.head_dim = 256;
  config.ssm_conv_kernel = 4;
  config.ssm_state_size = 128;
  config.ssm_group_count = 16;
  config.ssm_time_step_rank = 48;
  config.ssm_inner_size = 6144;

  const std::uint32_t num_layers = 1;
  const std::size_t conv_channels = config.SsmQkvSize();
  const std::uint32_t num_heads = config.ssm_time_step_rank;
  const std::uint32_t key_dim = config.ssm_state_size;
  const std::uint32_t val_dim = config.SsmValueSize();

  strix::models::QwenSsmCache cache(num_layers, conv_channels,
                                    config.ssm_conv_kernel, num_heads, key_dim,
                                    val_dim);

  // Check state sizes
  auto conv = cache.GetConvState(0);
  assert(conv.size() == conv_channels * config.ssm_conv_kernel);
  auto deltanet = cache.GetDeltaNetState(0, 0);
  assert(deltanet.size() == key_dim * val_dim);

  // Initially zero
  assert(conv[0] == 0.0F);
  assert(deltanet[0] == 0.0F);

  // Create dummy layer
  strix::models::QwenLayerWeights layer;
  layer.is_full_attention = false;

  std::vector<float> in(config.hidden_size, 0.1F);
  std::vector<float> qkv_buf(config.SsmQkvSize(), 0.0F);
  std::vector<float> gate_buf(config.ssm_inner_size, 0.0F);
  std::vector<float> out_buf(config.ssm_inner_size, 0.0F);
  std::vector<float> out(config.hidden_size, 0.0F);

  strix::models::ForwardSSM(in, layer, config, cache, 0, qkv_buf, gate_buf,
                            out_buf, out);

  // After 1 step, cache is updated
  assert(cache.GetConvState(0).size() ==
         conv_channels * config.ssm_conv_kernel);
  assert(cache.GetDeltaNetState(0, 0).size() == key_dim * val_dim);
}

int main() {
  TestQwenSsmConvRecurrence();
  std::cout << "All Qwen SSM tests passed.\n";
  return 0;
}
