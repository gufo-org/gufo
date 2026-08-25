#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

#include "src/models/qwen/hrx/qwen_hrx_model.hpp"

namespace {

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << '\n';
    std::exit(1);
  }
}

gufo::core::ModelConfig Qwen38Config() {
  gufo::core::ModelConfig config;
  config.num_layers = 64;
  config.hidden_size = 5120;
  config.intermediate_size = 17408;
  config.num_attention_heads = 24;
  config.num_key_value_heads = 4;
  config.head_dim = 256;
  config.vocab_size = 248320;
  config.context_length = 262144;
  return config;
}

void TestProductionQwen38Contract() {
  std::string error;
  Expect(gufo::hrx::QwenHrxModelContract::Supports(Qwen38Config(), &error),
         "Qwen3.8-27B configuration is accepted by the HRX contract");
  Expect(error.empty(), "accepted configuration has no error message");
}

void TestPrototypeAbiIsRejected() {
  auto incompatible = Qwen38Config();
  incompatible.vocab_size = 152064;
  std::string error;
  Expect(!gufo::hrx::QwenHrxModelContract::Supports(incompatible, &error),
         "prototype vocabulary ABI is rejected");
  Expect(!error.empty(), "rejection names the fixed production contract");
}

}  // namespace

int main() {
  TestProductionQwen38Contract();
  TestPrototypeAbiIsRejected();
  std::cout << "All qwen_hrx_model_contract_test assertions passed!\n";
  return 0;
}
