// DSpine GGUF loading (QwenDFlashWeights::LoadFromGguf) on tiny synthetic
// files in the layout of llama.cpp's DSpine converter: the checks llama.cpp's
// port guards (src/models/dflash.cpp:403-427 at d493f841b) plus the older
// layout. No model weights and no GPU.
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/models/qwen/dflash_weights.hpp"
#include "tests/fixtures/gguf_builder.hpp"

namespace {

using gufo::core::GgmlType;
using gufo::speculative::QwenDFlashWeights;

int failures = 0;

void Expect(bool ok, std::string_view what) {
  std::printf("%-58s %s\n", std::string(what).c_str(), ok ? "ok" : "FAIL");
  failures += ok ? 0 : 1;
}

constexpr std::uint64_t kHidden = 64, kFfn = 128, kHeadDim = 32;
constexpr std::uint64_t kVocab = 16, kCode = 8, kRank = 4;

struct Variant {
  std::uint32_t block_size = 16;
  std::uint32_t top_k = 16;
  bool causal = true;
  bool selector_tensor = false;  // a DFlash-2 selector beside the codebook
  bool selector_metadata = false;
  bool per_layer_bridge = true;  // false = the older layout
};

std::vector<std::uint8_t> BuildDSpine(const Variant& variant) {
  gufo::test_fixture::GgufBuilder builder;
  builder.AddMetadataString("general.architecture", "dflash");
  builder.AddMetadataUint32("dflash.embedding_length", kHidden);
  builder.AddMetadataUint32("dflash.feed_forward_length", kFfn);
  builder.AddMetadataUint32("dflash.attention.head_count", 2);
  builder.AddMetadataUint32("dflash.attention.head_count_kv", 1);
  builder.AddMetadataUint32("dflash.attention.key_length", kHeadDim);
  builder.AddMetadataUint32("dflash.rope.dimension_count", kHeadDim);
  builder.AddMetadataUint32("dflash.context_length", 4096);
  builder.AddMetadataUint32("dflash.block_count", 1);
  builder.AddMetadataFloat32("dflash.rope.freq_base", 1e7F);
  builder.AddMetadataUint32("dflash.block_size", variant.block_size);
  builder.AddMetadataUint32("dflash.selector_top_k", variant.top_k);
  builder.AddMetadataUint32("dflash.attention.sliding_window", 8192);
  builder.AddMetadataUint32("tokenizer.ggml.mask_token_id", 5);
  builder.AddMetadataBool("dflash.attention.causal", variant.causal);
  builder.AddMetadataInt32Array("dflash.target_layers", {2});
  if (variant.selector_metadata)
    builder.AddMetadataUint32("dflash.selector_rank", 4);

  std::uint64_t offset = 0;
  const auto tensor = [&](std::string_view name,
                          std::vector<std::uint64_t> dims) {
    builder.AddTensor(name, dims, GgmlType::kF32, offset);
    std::uint64_t elements = 1;
    for (const auto dim : dims)
      elements *= dim;
    offset += (elements * sizeof(float) + 31U) / 32U * 32U;
  };
  tensor("dspine_codebook.weight", {kCode, kVocab});
  tensor("dspine_gate.weight", {kCode, kRank});
  tensor("dspine_pred_gate.weight", {kCode, kRank});
  tensor("dspine_prev.weight", {kCode, kRank});
  tensor("dspine_gate.bias", {kRank});
  if (variant.per_layer_bridge) {
    tensor("blk.0.dspine_read.weight", {kHidden, kCode});
    tensor("blk.0.dspine_write.weight", {kRank, kHidden});
  }
  if (variant.selector_tensor)
    tensor("selector_predecessor.weight", {4, kVocab});
  tensor("fc.weight", {kHidden, kHidden});
  tensor("enc.output_norm.weight", {kHidden});
  tensor("output_norm.weight", {kHidden});
  tensor("blk.0.attn_norm.weight", {kHidden});
  tensor("blk.0.attn_q.weight", {kHidden, 2 * kHeadDim});
  tensor("blk.0.attn_k.weight", {kHidden, kHeadDim});
  tensor("blk.0.attn_v.weight", {kHidden, kHeadDim});
  tensor("blk.0.attn_output.weight", {2 * kHeadDim, kHidden});
  tensor("blk.0.attn_q_norm.weight", {kHeadDim});
  tensor("blk.0.attn_k_norm.weight", {kHeadDim});
  tensor("blk.0.ffn_norm.weight", {kHidden});
  tensor("blk.0.ffn_gate.weight", {kHidden, kFfn});
  tensor("blk.0.ffn_up.weight", {kHidden, kFfn});
  tensor("blk.0.ffn_down.weight", {kFfn, kHidden});
  return builder.Build(offset);
}

/// Loads `variant`; returns the loader's error ("" on success).
std::string Load(const Variant& variant,
                 std::optional<QwenDFlashWeights>* out = nullptr) {
  const auto binary = BuildDSpine(variant);
  std::string error;
  const auto reader =
      gufo::core::GgufReader::OpenMemory(binary.data(), binary.size(), &error);
  if (reader == nullptr)
    return "reader: " + error;
  auto weights = QwenDFlashWeights::LoadFromGguf(*reader, &error);
  if (weights.has_value())
    error.clear();
  else if (error.empty())
    error = "(no message)";
  if (out != nullptr)
    *out = std::move(weights);
  return error;
}

bool Contains(const std::string& text, std::string_view part) {
  return text.find(part) != std::string::npos;
}

}  // namespace

int main() {
  std::optional<QwenDFlashWeights> weights;
  const auto ok = Load({}, &weights);
  Expect(ok.empty(), "current layout loads: " + ok);
  if (weights.has_value()) {
    const auto& draft = weights->dflash_config;
    Expect(weights->dspine.has_value() && draft.causal &&
               draft.selector_top_k == 16 && draft.conv_kernel_size == 0 &&
               draft.selector_rank == 0,
           "DSpine detected, causal, K = 16, no conv or selector");
    Expect(weights->config.vocab_size == kVocab &&
               weights->dspine->code_size == kCode &&
               weights->dspine->rank == kRank &&
               weights->dspine->read.size() == 1 &&
               weights->dspine->write.size() == 1 &&
               weights->selector_predecessor.empty() &&
               weights->layers[0].attention_conv_projection.empty(),
           "vocab = codebook rows, C, R, one read/write per layer");
  }

  // K1 (llama.cpp dflash.cpp:410): one file is one method; K >= 2; block >= 2.
  auto error = Load({.selector_tensor = true});
  Expect(Contains(error, "must not carry"),
         "K1 refuses a selector tensor beside the codebook: " + error);
  error = Load({.selector_metadata = true});
  Expect(Contains(error, "must not carry"),
         "K1 refuses selector metadata beside the codebook: " + error);
  error = Load({.top_k = 1});
  Expect(Contains(error, "invalid diffusion topology"),
         "K1 refuses selector_top_k < 2: " + error);
  error = Load({.block_size = 1});
  Expect(Contains(error, "invalid diffusion topology"),
         "K1 refuses block_size < 2: " + error);
  error = Load({.top_k = 17});
  Expect(Contains(error, "invalid diffusion topology"),
         "refuses selector_top_k > 16 (the GPU kernels' bound): " + error);
  error = Load({.causal = false});
  Expect(Contains(error, "invalid diffusion topology"),
         "refuses a non-causal DSpine file: " + error);

  // K3: the older converter output has no per-layer read/write tensors.
  error = Load({.per_layer_bridge = false});
  Expect(Contains(error, "older layout"),
         "K3 refuses the older layout with a clear error: " + error);

  std::printf("qwen_dspine_loader_test: %s (%d failures)\n",
              failures == 0 ? "ALL TESTS PASSED" : "FAILED", failures);
  return failures == 0 ? 0 : 1;
}
