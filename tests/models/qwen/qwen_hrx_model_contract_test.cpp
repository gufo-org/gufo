#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

#include "src/models/qwen/hrx/qwen_hrx_contract.hpp"

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
  config.rotary_dim = 64;
  config.ssm_group_count = 16;
  config.ssm_state_size = 128;
  config.ssm_time_step_rank = 48;
  config.ssm_inner_size = 6144;
  config.vocab_size = 248320;
  config.context_length = 262144;
  return config;
}

void TestProductionQwen38Contract() {
  std::string error;
  const auto contract =
      gufo::hrx::QwenHrxArtifactContract::FromConfig(Qwen38Config(), &error);
  Expect(contract.has_value(),
         "Qwen3.8-27B configuration is accepted by the HRX contract");
  Expect(error.empty(), "accepted configuration has no error message");
  Expect(contract->HiddenSize() == 5120, "hidden width is derived");
  Expect(contract->FfnSize() == 17408, "FFN width is derived");
  Expect(contract->QHeadCount() == 24, "Q head count is derived");
  Expect(contract->KvHeadCount() == 4, "KV head count is derived");
  Expect(contract->HeadDim() == 256, "head dimension is derived");
  Expect(contract->RotaryDim() == 64, "rotary dimension is derived");
  Expect(contract->QWidth() == 6144, "Q width is derived");
  Expect(contract->KWidth() == 1024, "K width is derived");
  Expect(contract->VWidth() == 1024, "V width is derived");
  Expect(contract->QkvWidth() == 8192, "concatenated QKV width is derived");
  Expect(contract->SsmQkvWidth() == 10240, "SSM QKV width is derived");
  Expect(contract->VocabSize() == 248320, "vocabulary size is derived");
}

void TestPrototypeAbiIsRejected() {
  auto incompatible = Qwen38Config();
  incompatible.vocab_size = 152064;
  std::string error;
  Expect(!gufo::hrx::QwenHrxArtifactContract::Supports(incompatible, &error),
         "prototype vocabulary ABI is rejected");
  Expect(!error.empty(), "rejection names the fixed production contract");
}

void TestDerivedShapeMismatchIsRejected() {
  auto incompatible = Qwen38Config();
  incompatible.num_key_value_heads = 24;
  std::string error;
  Expect(!gufo::hrx::QwenHrxArtifactContract::Supports(incompatible, &error),
         "MHA dimensions cannot be used with GQA artifacts");

  incompatible = Qwen38Config();
  incompatible.ssm_inner_size = 4096;
  incompatible.ssm_time_step_rank = 32;
  Expect(!gufo::hrx::QwenHrxArtifactContract::Supports(incompatible, &error),
         "prototype SSM QKV dimensions are rejected");
}

}  // namespace

int main() {
  TestProductionQwen38Contract();
  TestPrototypeAbiIsRejected();
  TestDerivedShapeMismatchIsRejected();
  std::cout << "All qwen_hrx_model_contract_test assertions passed!\n";
  return 0;
}
