#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>

#include "src/models/qwen/hrx/qwen_hrx_arena_layout.hpp"
#include "src/models/qwen/hrx/qwen_hrx_capabilities.hpp"
#include "src/models/qwen/hrx/qwen_hrx_contract.hpp"
#include "src/models/qwen/hrx/qwen_hrx_manifest.hpp"
#include "src/models/qwen/hrx/qwen_hrx_policy.hpp"
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
  Expect(contract->SsmKeyHeadCount() == 16, "SSM key-head count is derived");
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
  Expect(!gufo::hrx::QwenHrxArtifactContract::Supports(incompatible, &error),
         "former 16 recurrent-head assumption is rejected");

  incompatible = Qwen38Config();
  incompatible.full_attention_interval = 8;
  Expect(!gufo::hrx::QwenHrxArtifactContract::Supports(incompatible, &error),
         "non-3:1 full-attention/SSM layer pattern is rejected");

  incompatible = Qwen38Config();
  incompatible.ssm_conv_kernel = 3;
  Expect(!gufo::hrx::QwenHrxArtifactContract::Supports(incompatible, &error),
         "non-production convolution width is rejected");
}

void TestTypedTensorPayloadValidation() {
  std::array<std::uint16_t, 6> payload{};
  const gufo::models::QwenTensorRef bf16{
      .data = payload.data(),
      .type = gufo::core::GgmlType::kBF16,
      .num_elements = payload.size(),
      .available_bytes = payload.size() * sizeof(std::uint16_t),
  };
  const auto matrix_elements = gufo::hrx::HrxMatrixElementCount(2, 3);
  Expect(matrix_elements.has_value() && *matrix_elements == payload.size(),
         "2x3 BF16 matrix shape has the exact payload element count");
  std::string error;
  Expect(gufo::hrx::ValidateHrxTensorPayload(bf16, gufo::core::GgmlType::kBF16,
                                             *matrix_elements, &error),
         "exact BF16 matrix type, shape, and storage are accepted");
  Expect(error.empty(), "accepted tensor payload clears the error");

  auto incompatible = bf16;
  incompatible.type = gufo::core::GgmlType::kF32;
  Expect(!gufo::hrx::ValidateHrxTensorPayload(
             incompatible, gufo::core::GgmlType::kBF16, payload.size(), &error),
         "F32 payload is rejected for a BF16 artifact operand");

  incompatible = bf16;
  incompatible.type = gufo::core::GgmlType::kQ8_0;
  Expect(!gufo::hrx::ValidateHrxTensorPayload(
             incompatible, gufo::core::GgmlType::kBF16, payload.size(), &error),
         "quantized payload is rejected for a BF16 artifact operand");

  std::array<std::uint8_t, 34> q8_payload{};
  const gufo::models::QwenTensorRef q8{
      .data = q8_payload.data(),
      .type = gufo::core::GgmlType::kQ8_0,
      .num_elements = 32,
      .available_bytes = q8_payload.size(),
  };
  Expect(gufo::hrx::ValidateHrxTensorPayload(q8, gufo::core::GgmlType::kQ8_0,
                                             32, &error),
         "exact Q8_0 block payload is accepted by the typed contract");

  incompatible = bf16;
  incompatible.num_elements = payload.size() - 1;
  Expect(!gufo::hrx::ValidateHrxTensorPayload(
             incompatible, gufo::core::GgmlType::kBF16, payload.size(), &error),
         "wrong logical element count is rejected");

  incompatible = bf16;
  incompatible.available_bytes = payload.size() * sizeof(std::uint16_t) - 1;
  Expect(!gufo::hrx::ValidateHrxTensorPayload(
             incompatible, gufo::core::GgmlType::kBF16, payload.size(), &error),
         "truncated encoded storage is rejected");
}

void TestCheckedMatrixElementCount() {
  const auto elements = gufo::hrx::HrxMatrixElementCount(2, 3);
  Expect(elements.has_value() && *elements == 6,
         "valid matrix dimensions produce the exact element count");
  Expect(!gufo::hrx::HrxMatrixElementCount(0, 3).has_value(),
         "zero matrix dimensions are rejected");
  Expect(!gufo::hrx::HrxMatrixElementCount(
              std::numeric_limits<std::size_t>::max(), 2)
              .has_value(),
         "matrix element-count overflow is rejected");
}

void TestArenaLayout() {
  const auto contract =
      gufo::hrx::QwenHrxArtifactContract::FromConfig(Qwen38Config());
  Expect(contract.has_value(), "arena test has a valid production contract");
  std::string error;
  const auto layout =
      gufo::hrx::QwenHrxArenaLayout::Create(*contract, 8, &error);
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
  Expect(layout->kv_cache_bytes == 16U * 2U * 4U * 8U * 256U * sizeof(float),
         "KV cache allocates only full-attention layers and requested context");
  Expect(layout->ssm_conv_state_bytes == 64U * 10240U * 4U * sizeof(float),
         "convolution state covers all layer slots");
  Expect(layout->ssm_recurrent_state_bytes ==
             64U * 48U * 128U * 128U * sizeof(float),
         "DeltaNet state uses 48 recurrent heads");
  Expect(layout->saved_ssm_conv_state_bytes == layout->ssm_conv_state_bytes,
         "saved convolution state mirrors live state");
  Expect(layout->saved_ssm_recurrent_state_bytes ==
             layout->ssm_recurrent_state_bytes,
         "saved DeltaNet state mirrors live state");

  Expect(
      !gufo::hrx::QwenHrxArenaLayout::Create(*contract, 0, &error).has_value(),
      "zero-context arena is rejected");
  Expect(!gufo::hrx::HrxCheckedProduct(
              {std::numeric_limits<std::size_t>::max(), 2})
              .has_value(),
         "arena byte-size overflow is rejected");
}

void TestManifestValidation() {
  const auto contract =
      gufo::hrx::QwenHrxArtifactContract::FromConfig(Qwen38Config());
  Expect(contract.has_value(), "valid production contract for manifest test");

  // 1. Builtin manifest creation
  auto manifest = gufo::hrx::HrxArtifactManifest::CreateBuiltin();
  Expect(manifest != nullptr, "builtin manifest creates");
  Expect(manifest->SchemaVersion() == "1.0.0", "manifest schema is 1.0.0");
  Expect(manifest->ModelKind() == "qwen3.8-27b",
         "manifest model kind is qwen3.8-27b");
  Expect(manifest->Target() == "gfx1151", "manifest target is gfx1151");
  Expect(manifest->WaveSize() == 32, "manifest wave size is 32");
  Expect(manifest->HrxAbiRevision() == "hrx-loom-v1",
         "manifest ABI is hrx-loom-v1");
  Expect(!manifest->Entries().empty(), "manifest contains entries");

  std::string error;

  // 2. Wrong schema version rejection
  manifest->SetSchemaVersion("2.0.0");
  Expect(!manifest->ValidateDirectory("/nonexistent", *contract, &error),
         "wrong schema version is rejected");
  Expect(!error.empty(), "rejection error is populated");
  manifest->SetSchemaVersion("1.0.0");

  // 3. Wrong model kind rejection
  manifest->SetModelKind("llama-3-8b");
  Expect(!manifest->ValidateDirectory("/nonexistent", *contract, &error),
         "wrong model kind is rejected");
  manifest->SetModelKind("qwen3.8-27b");

  // 4. Wrong target rejection
  manifest->SetTarget("gfx1100");
  Expect(!manifest->ValidateDirectory("/nonexistent", *contract, &error),
         "wrong target is rejected");
  manifest->SetTarget("gfx1151");

  // 5. Wrong wave size rejection
  manifest->SetWaveSize(64);
  Expect(!manifest->ValidateDirectory("/nonexistent", *contract, &error),
         "wrong wave size is rejected");
  manifest->SetWaveSize(32);

  // 6. Wrong ABI revision rejection
  manifest->SetHrxAbiRevision("legacy-abi-v0");
  Expect(!manifest->ValidateDirectory("/nonexistent", *contract, &error),
         "wrong ABI revision is rejected");
  manifest->SetHrxAbiRevision("hrx-loom-v1");

  // 7. Duplicate entry name rejection
  auto dup_manifest = gufo::hrx::HrxArtifactManifest::CreateBuiltin();
  gufo::hrx::HrxArtifactManifestEntry dup_entry;
  dup_entry.name = "qwen_argmax";
  dup_entry.filename = "qwen_argmax_dup.fb";
  dup_entry.export_name = "qwen_argmax_dup";
  dup_entry.binding_count = 2;
  dup_manifest->AddEntry(dup_entry);
  Expect(!dup_manifest->ValidateDirectory("/nonexistent", *contract, &error),
         "duplicate entry name is rejected");

  // 8. Missing required file rejection in directory
  Expect(!manifest->ValidateDirectory("/nonexistent_empty_dir_xyz", *contract,
                                      &error),
         "missing required artifact file in directory is rejected");
  Expect(error.find("missing:") != std::string::npos,
         "rejection error specifies missing artifact");
}

void TestHrxCapabilityNegotiation() {
  const auto config = Qwen38Config();
  std::string error;

  // 1. Builtin manifest with all entries present -> capable
  auto manifest = gufo::hrx::HrxArtifactManifest::CreateBuiltin();
  for (auto& entry : manifest->MutableEntries()) {
    entry.sha256 =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
  }
  auto report = gufo::hrx::ProbeHrxCapabilities(*manifest, config, &error);
  Expect(report.is_capable, "full manifest reports capable");
  Expect(report.missing_required_primitives.empty(),
         "no missing required primitives");
  Expect(report.available_required_primitives.size() == 5,
         "5 required primitives available");
  Expect(!report.ToString().empty(), "report string is non-empty");

  // 2. Missing Embedding
  auto no_emb = gufo::hrx::HrxArtifactManifest::CreateBuiltin();
  for (auto& entry : no_emb->MutableEntries()) {
    if (entry.name != "qwen_q8_embedding") {
      entry.sha256 = "aaa";
    }
  }
  auto rep_no_emb = gufo::hrx::ProbeHrxCapabilities(*no_emb, config, &error);
  Expect(!rep_no_emb.is_capable, "missing embedding is not capable");
  Expect(std::find(rep_no_emb.missing_required_primitives.begin(),
                   rep_no_emb.missing_required_primitives.end(),
                   "Embedding") != rep_no_emb.missing_required_primitives.end(),
         "missing primitives list includes Embedding");

  // 3. Missing SSM
  auto no_ssm = gufo::hrx::HrxArtifactManifest::CreateBuiltin();
  for (auto& entry : no_ssm->MutableEntries()) {
    if (entry.name != "qwen_deltanet_recurrence") {
      entry.sha256 = "aaa";
    }
  }
  auto rep_no_ssm = gufo::hrx::ProbeHrxCapabilities(*no_ssm, config, &error);
  Expect(!rep_no_ssm.is_capable, "missing SSM is not capable");
  Expect(std::find(rep_no_ssm.missing_required_primitives.begin(),
                   rep_no_ssm.missing_required_primitives.end(),
                   "SSM") != rep_no_ssm.missing_required_primitives.end(),
         "missing primitives list includes SSM");

  // 4. Missing Attention
  auto no_attn = gufo::hrx::HrxArtifactManifest::CreateBuiltin();
  for (auto& entry : no_attn->MutableEntries()) {
    if (entry.name != "qwen_attention_decode") {
      entry.sha256 = "aaa";
    }
  }
  auto rep_no_attn = gufo::hrx::ProbeHrxCapabilities(*no_attn, config, &error);
  Expect(!rep_no_attn.is_capable, "missing Attention is not capable");
  Expect(
      std::find(rep_no_attn.missing_required_primitives.begin(),
                rep_no_attn.missing_required_primitives.end(),
                "Attention") != rep_no_attn.missing_required_primitives.end(),
      "missing primitives list includes Attention");

  // 5. Missing FFN
  auto no_ffn = gufo::hrx::HrxArtifactManifest::CreateBuiltin();
  for (auto& entry : no_ffn->MutableEntries()) {
    if (entry.name != "qwen_q8_gemv_k17408") {
      entry.sha256 = "aaa";
    }
  }
  auto rep_no_ffn = gufo::hrx::ProbeHrxCapabilities(*no_ffn, config, &error);
  Expect(!rep_no_ffn.is_capable, "missing FFN is not capable");
  Expect(std::find(rep_no_ffn.missing_required_primitives.begin(),
                   rep_no_ffn.missing_required_primitives.end(),
                   "FFN") != rep_no_ffn.missing_required_primitives.end(),
         "missing primitives list includes FFN");

  // 6. Missing Argmax
  auto no_argmax = gufo::hrx::HrxArtifactManifest::CreateBuiltin();
  for (auto& entry : no_argmax->MutableEntries()) {
    if (entry.name != "qwen_argmax") {
      entry.sha256 = "aaa";
    }
  }
  auto rep_no_argmax =
      gufo::hrx::ProbeHrxCapabilities(*no_argmax, config, &error);
  Expect(!rep_no_argmax.is_capable, "missing Argmax is not capable");
  Expect(std::find(rep_no_argmax.missing_required_primitives.begin(),
                   rep_no_argmax.missing_required_primitives.end(),
                   "Argmax") != rep_no_argmax.missing_required_primitives.end(),
         "missing primitives list includes Argmax");
}

void TestHrxExecutionPolicyParsing() {
  std::string error;

  // Default / empty / none
  auto p_none = gufo::hrx::QwenHrxExecutionPolicy::Parse("none", &error);
  Expect(!p_none.fused_swiglu_bf16 && !p_none.fused_down_residual_bf16,
         "none clears all fusions");
  Expect(p_none.ToString() == "none", "none serializes as 'none'");

  // All
  auto p_all = gufo::hrx::QwenHrxExecutionPolicy::Parse("all", &error);
  Expect(p_all.fused_swiglu_bf16 && p_all.fused_down_residual_bf16 &&
             p_all.fused_rmsnorm_qkv_bf16 && p_all.fused_rope_kv_bf16,
         "all enables all fusions");

  // Specific fusions
  auto p_swiglu = gufo::hrx::QwenHrxExecutionPolicy::Parse("swiglu", &error);
  Expect(p_swiglu.fused_swiglu_bf16 && !p_swiglu.fused_down_residual_bf16,
         "swiglu enables swiglu fusion only");

  auto p_down =
      gufo::hrx::QwenHrxExecutionPolicy::Parse("down-residual", &error);
  Expect(p_down.fused_down_residual_bf16 && !p_down.fused_swiglu_bf16,
         "down-residual enables down-residual fusion only");

  auto p_combo = gufo::hrx::QwenHrxExecutionPolicy::Parse(
      "swiglu,down-residual,rope-kv", &error);
  Expect(p_combo.fused_swiglu_bf16 && p_combo.fused_down_residual_bf16 &&
             p_combo.fused_rope_kv_bf16 && !p_combo.fused_rmsnorm_qkv_bf16,
         "combo enables requested fusions");
  Expect(p_combo.ToString() == "swiglu,down-residual,rope-kv",
         "combo serializes correctly");

  auto p_wmma =
      gufo::hrx::QwenHrxExecutionPolicy::Parse("wmma-prefill", &error);
  Expect(p_wmma.wmma_prefill && p_wmma.int8_prefill &&
             p_wmma.chunked_prefill,
         "wmma-prefill enables its int8 and chunked prerequisites");
  Expect(p_wmma.ToString() ==
             "chunked-prefill,int8-prefill,wmma-prefill",
         "wmma-prefill serializes with explicit prerequisites");

  // Invalid flag
  auto p_bad =
      gufo::hrx::QwenHrxExecutionPolicy::Parse("invalid-flag-123", &error);
  Expect(!error.empty() &&
             error.find("Unknown HRX fusion flag") != std::string::npos,
         "invalid fusion flag is reported");
}

}  // namespace

int main() {
  TestProductionQwen38Contract();
  TestPrototypeAbiIsRejected();
  TestDerivedShapeMismatchIsRejected();
  TestTypedTensorPayloadValidation();
  TestCheckedMatrixElementCount();
  TestArenaLayout();
  TestManifestValidation();
  TestHrxCapabilityNegotiation();
  TestHrxExecutionPolicyParsing();
  std::cout << "All qwen_hrx_model_contract_test assertions passed!\n";
  return 0;
}
