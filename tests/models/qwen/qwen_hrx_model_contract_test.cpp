#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>

#include "src/models/qwen/hrx/qwen_hrx_arena_layout.hpp"
#include "src/models/qwen/hrx/qwen_hrx_contract.hpp"
#include "src/models/qwen/hrx/qwen_hrx_tensor_binding.hpp"

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
  Expect(contract->FullAttentionLayerCount() == 16,
         "full-attention layer count follows the 3:1 pattern");
  Expect(contract->SsmLayerCount() == 48,
         "SSM layer count follows the 3:1 pattern");
  Expect(contract->FullAttentionQueryWidth() == 6144,
         "attention query width is derived");
  Expect(contract->FullAttentionQGateWidth() == 12288,
         "attention Q+gate projection width is derived");
  Expect(contract->FullAttentionKeyWidth() == 1024,
         "attention key width is derived");
  Expect(contract->FullAttentionValueWidth() == 1024,
         "attention value width is derived");
  Expect(contract->FullAttentionPackedProjectionWidth() == 14336,
         "diagnostic attention projection sum includes Q gate, K, and V");
  Expect(contract->FullAttentionPackedProjectionWidth() != 8192,
         "former gate-less packed QKV assumption is rejected");
  Expect(contract->SsmQkvWidth() == 10240, "SSM QKV width is derived");
  Expect(contract->SsmGateWidth() == 6144, "SSM gate width is derived");
  Expect(contract->SsmKeyHeadCount() == 16,
         "SSM key-head count is derived");
  Expect(contract->SsmValueHeadCount() == 48,
         "SSM recurrent/value-head count is derived");
  Expect(contract->SsmValueHeadCount() != 16,
         "former key-head/recurrent-head conflation is rejected");
  Expect(contract->SsmKeyDim() == 128, "SSM key dimension is derived");
  Expect(contract->SsmValueDim() == 128, "SSM value dimension is derived");
  Expect(contract->SsmAlphaBetaWidth() == 48,
         "SSM alpha/beta width follows recurrent heads");
  Expect(contract->SsmConvKernel() == 4, "SSM convolution width is derived");
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

  incompatible = Qwen38Config();
  incompatible.ssm_time_step_rank = 16;
  incompatible.ssm_inner_size = 2048;
  Expect(!strix::hrx::QwenHrxArtifactContract::Supports(incompatible, &error),
         "former 16 recurrent-head assumption is rejected");

  incompatible = Qwen38Config();
  incompatible.full_attention_interval = 8;
  Expect(!strix::hrx::QwenHrxArtifactContract::Supports(incompatible, &error),
         "non-3:1 full-attention/SSM layer pattern is rejected");

  incompatible = Qwen38Config();
  incompatible.ssm_conv_kernel = 3;
  Expect(!strix::hrx::QwenHrxArtifactContract::Supports(incompatible, &error),
         "non-production convolution width is rejected");
}

void TestTypedTensorPayloadValidation() {
  std::array<std::uint16_t, 6> payload{};
  const strix::models::QwenTensorRef bf16{
      .data = payload.data(),
      .type = strix::core::GgmlType::kBF16,
      .num_elements = payload.size(),
      .available_bytes = payload.size() * sizeof(std::uint16_t),
  };
  const auto matrix_elements = strix::hrx::HrxMatrixElementCount(2, 3);
  Expect(matrix_elements.has_value() && *matrix_elements == payload.size(),
         "2x3 BF16 matrix shape has the exact payload element count");
  std::string error;
  Expect(strix::hrx::ValidateHrxTensorPayload(
             bf16, strix::core::GgmlType::kBF16, *matrix_elements, &error),
         "exact BF16 matrix type, shape, and storage are accepted");
  Expect(error.empty(), "accepted tensor payload clears the error");

  auto incompatible = bf16;
  incompatible.type = strix::core::GgmlType::kF32;
  Expect(!strix::hrx::ValidateHrxTensorPayload(
             incompatible, strix::core::GgmlType::kBF16, payload.size(),
             &error),
         "F32 payload is rejected for a BF16 artifact operand");

  incompatible = bf16;
  incompatible.type = strix::core::GgmlType::kQ8_0;
  Expect(!strix::hrx::ValidateHrxTensorPayload(
             incompatible, strix::core::GgmlType::kBF16, payload.size(),
             &error),
         "quantized payload is rejected for a BF16 artifact operand");

  std::array<std::uint8_t, 34> q8_payload{};
  const strix::models::QwenTensorRef q8{
      .data = q8_payload.data(),
      .type = strix::core::GgmlType::kQ8_0,
      .num_elements = 32,
      .available_bytes = q8_payload.size(),
  };
  Expect(strix::hrx::ValidateHrxTensorPayload(
             q8, strix::core::GgmlType::kQ8_0, 32, &error),
         "exact Q8_0 block payload is accepted by the typed contract");

  incompatible = bf16;
  incompatible.num_elements = payload.size() - 1;
  Expect(!strix::hrx::ValidateHrxTensorPayload(
             incompatible, strix::core::GgmlType::kBF16, payload.size(),
             &error),
         "wrong logical element count is rejected");

  incompatible = bf16;
  incompatible.available_bytes =
      payload.size() * sizeof(std::uint16_t) - 1;
  Expect(!strix::hrx::ValidateHrxTensorPayload(
             incompatible, strix::core::GgmlType::kBF16, payload.size(),
             &error),
         "truncated encoded storage is rejected");
}

void TestCheckedMatrixElementCount() {
  const auto elements = strix::hrx::HrxMatrixElementCount(2, 3);
  Expect(elements.has_value() && *elements == 6,
         "valid matrix dimensions produce the exact element count");
  Expect(!strix::hrx::HrxMatrixElementCount(0, 3).has_value(),
         "zero matrix dimensions are rejected");
  Expect(!strix::hrx::HrxMatrixElementCount(
              std::numeric_limits<std::size_t>::max(), 2)
              .has_value(),
         "matrix element-count overflow is rejected");
}

void TestArenaLayout() {
  const auto contract =
      strix::hrx::QwenHrxArtifactContract::FromConfig(Qwen38Config());
  Expect(contract.has_value(), "arena test has a valid production contract");
  std::string error;
  const auto layout =
      strix::hrx::QwenHrxArenaLayout::Create(*contract, 8, &error);
  Expect(layout.has_value(), "eight-token native HRX arena layout is valid");
  Expect(error.empty(), "valid arena layout clears the error");
  Expect(layout->hidden_bytes == 5120U * sizeof(float),
         "hidden scratch is one F32 token");
  Expect(layout->token_bytes == sizeof(std::uint32_t) &&
             layout->position_bytes == sizeof(std::uint32_t),
         "token and position parameters are stable uint32 buffers");
  Expect(layout->attention_q_gate_bytes == 12288U * sizeof(float),
         "attention Q+gate scratch uses production width");
  Expect(layout->ssm_qkv_bytes == 10240U * sizeof(float),
         "SSM QKV scratch uses production width");
  Expect(layout->ssm_recurrent_output_bytes == 48U * 128U * sizeof(float),
         "SSM recurrent output uses 48 value heads");
  Expect(layout->kv_cache_bytes ==
             16U * 2U * 4U * 8U * 256U * sizeof(float),
         "KV cache allocates only full-attention layers and requested context");
  Expect(layout->ssm_conv_state_bytes ==
             64U * 10240U * 4U * sizeof(float),
         "convolution state covers all layer slots");
  Expect(layout->ssm_recurrent_state_bytes ==
             64U * 48U * 128U * 128U * sizeof(float),
         "DeltaNet state uses 48 recurrent heads");
  Expect(layout->saved_ssm_conv_state_bytes == layout->ssm_conv_state_bytes,
         "saved convolution state mirrors live state");
  Expect(layout->saved_ssm_recurrent_state_bytes ==
             layout->ssm_recurrent_state_bytes,
         "saved DeltaNet state mirrors live state");

  Expect(!strix::hrx::QwenHrxArenaLayout::Create(*contract, 0, &error)
              .has_value(),
         "zero-context arena is rejected");
  Expect(!strix::hrx::HrxCheckedProduct(
              {std::numeric_limits<std::size_t>::max(), 2})
              .has_value(),
         "arena byte-size overflow is rejected");
}

}  // namespace

int main() {
  TestProductionQwen38Contract();
  TestPrototypeAbiIsRejected();
  TestDerivedShapeMismatchIsRejected();
  TestTypedTensorPayloadValidation();
  TestCheckedMatrixElementCount();
  TestArenaLayout();
  std::cout << "All qwen_hrx_model_contract_test assertions passed!\n";
  return 0;
}
