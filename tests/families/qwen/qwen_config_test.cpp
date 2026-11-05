#include "src/families/qwen/config/qwen_config.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "tests/fixtures/gguf_builder.hpp"

// CPU-only coverage for the family configuration extractor. The expectations
// are the ones the core GGUF reader test carried before ExtractQwenConfig moved
// out of GgufReader; they name shapes and rejection text, never a helper that
// derives them from the production definitions.

namespace {

using gufo::test_fixture::GgufBuilder;

void Expect(bool condition, std::string_view msg) {
  if (!condition) {
    std::cerr << "Assertion failed: " << msg << "\n";
    std::exit(1);
  }
}

void TestQwen35ConfigExtraction() {
  GgufBuilder builder;
  builder.AddMetadataString("general.architecture", "qwen35");
  builder.AddMetadataString("general.name", "qwen3.5-4b-text");
  builder.AddMetadataUint32("qwen35.block_count", 36);
  builder.AddMetadataUint32("qwen35.embedding_length", 2560);
  builder.AddMetadataUint32("qwen35.feed_forward_length", 9728);
  builder.AddMetadataUint32("qwen35.attention.head_count", 20);
  builder.AddMetadataUint32("qwen35.attention.head_count_kv", 4);
  builder.AddMetadataUint32("qwen35.attention.key_length", 128);
  builder.AddMetadataUint32("qwen35.context_length", 32768);
  builder.AddMetadataUint32("qwen35.full_attention_interval", 4);
  builder.AddMetadataFloat32("qwen35.rope.freq_base", 1000000.0F);

  builder.AddTensor("token_embd.weight", {2560, 8}, gufo::core::GgmlType::kBF16,
                    0);
  builder.AddTensor("blk.0.attn_q.weight", {256, 8},
                    gufo::core::GgmlType::kQ4_K, 64);
  builder.AddTensor("mtp.0.proj.weight", {256, 8}, gufo::core::GgmlType::kQ8_0,
                    128);

  auto binary = builder.Build(512);
  std::string err;
  auto reader =
      gufo::core::GgufReader::OpenMemory(binary.data(), binary.size(), &err);
  Expect(reader != nullptr, "Reader open succeeds: " + err);

  auto config = gufo::models::qwen::ExtractQwenConfig(*reader, &err);
  Expect(config.has_value(), "ExtractQwenConfig succeeds: " + err);
  Expect(config->num_layers == 36, "36 layers");
  Expect(config->hidden_size == 2560, "2560 hidden size");
  Expect(config->head_dim == 128, "128 head dim");
  Expect(config->full_attention_interval == 4, "full_attention_interval 4");
  Expect(config->mtp_num_layers == 1, "MTP layer count 1");
  Expect(config->is_text_only, "is_text_only true");
  Expect(config->IsValidQwen(), "IsValidQwen true");
}

void TestQwen38_27BConfigExtraction() {
  GgufBuilder builder;
  builder.AddMetadataString("general.architecture", "qwen35");
  builder.AddMetadataString("general.name", "Qwen3.8-27B");
  builder.AddMetadataUint32("qwen35.block_count", 65);
  builder.AddMetadataUint32("qwen35.embedding_length", 5120);
  builder.AddMetadataUint32("qwen35.feed_forward_length", 17408);
  builder.AddMetadataUint32("qwen35.attention.head_count", 24);
  builder.AddMetadataUint32("qwen35.attention.head_count_kv", 4);
  builder.AddMetadataUint32("qwen35.attention.key_length", 256);
  builder.AddMetadataUint32("qwen35.context_length", 262144);
  builder.AddMetadataUint32("qwen35.full_attention_interval", 4);
  builder.AddMetadataUint32("qwen35.nextn_predict_layers", 1);
  builder.AddMetadataUint32("qwen35.ssm.conv_kernel", 4);
  builder.AddMetadataUint32("qwen35.ssm.state_size", 128);
  builder.AddMetadataUint32("qwen35.ssm.group_count", 16);
  builder.AddMetadataUint32("qwen35.ssm.time_step_rank", 48);
  builder.AddMetadataUint32("qwen35.ssm.inner_size", 6144);
  builder.AddMetadataUint32("qwen35.rope.dimension_count", 64);
  builder.AddMetadataFloat32("qwen35.rope.freq_base", 10000000.0F);

  builder.AddTensor("token_embd.weight", {5120, 8}, gufo::core::GgmlType::kBF16,
                    0);
  builder.AddTensor("blk.3.attn_q.weight", {256, 8},
                    gufo::core::GgmlType::kQ4_K, 64);
  builder.AddTensor("blk.63.ffn_gate.weight", {256, 8},
                    gufo::core::GgmlType::kQ4_K, 128);
  builder.AddTensor("blk.64.nextn.enorm.weight", {5120},
                    gufo::core::GgmlType::kQ8_0, 192);

  auto binary = builder.Build(512);

  std::string err;
  auto reader =
      gufo::core::GgufReader::OpenMemory(binary.data(), binary.size(), &err);
  Expect(reader != nullptr, "Reader open succeeds: " + err);
  Expect(reader->GetTensorCount() == 4, "4 tensors parsed for 27B");

  // Config extraction and verification for 27B
  auto config = gufo::models::qwen::ExtractQwenConfig(*reader, &err);
  Expect(config.has_value(), "ExtractModelConfig succeeds for 27B: " + err);
  Expect(config->num_layers == 64, "64 layers for 27B");
  Expect(config->hidden_size == 5120, "5120 hidden size for 27B");
  Expect(config->intermediate_size == 17408, "17408 intermediate size for 27B");
  Expect(config->num_attention_heads == 24, "24 attention heads for 27B");
  Expect(config->num_key_value_heads == 4, "4 KV heads for 27B");
  Expect(config->head_dim == 256, "256 head dim for 27B");
  Expect(config->context_length == 262144, "262144 context length for 27B");
  Expect(config->full_attention_interval == 4,
         "full_attention_interval 4 for 27B");
  Expect(config->mtp_num_layers == 1, "MTP layer count 1 for 27B");
  Expect(config->AttentionSize() == 6144, "6144 full-attention width for 27B");
  Expect(config->FullAttentionLayerCount() == 16,
         "16 full-attention layers for 27B");
  Expect(config->SsmQkvSize() == 10240, "10240 SSM QKV width for 27B");
  Expect(config->SsmValueSize() == 128, "128 SSM value width for 27B");
  Expect(config->is_text_only, "is_text_only true for 27B");
  Expect(config->IsValidQwen(), "IsValidQwen true for 27B");
}
void TestVisionExclusionRejection() {
  GgufBuilder builder;
  builder.AddMetadataString("general.architecture", "qwen35");
  builder.AddMetadataUint32("qwen35.block_count", 36);
  builder.AddMetadataUint32("qwen35.embedding_length", 2560);
  builder.AddMetadataUint32("qwen35.feed_forward_length", 9728);
  builder.AddMetadataUint32("qwen35.attention.head_count", 20);
  builder.AddMetadataUint32("qwen35.attention.head_count_kv", 4);
  builder.AddMetadataUint32("qwen35.attention.key_length", 128);

  // Add vision encoder tensor
  builder.AddTensor("model.visual.patch_embed.weight", {768, 3, 14, 14},
                    gufo::core::GgmlType::kF16, 0);

  auto binary = builder.Build();

  std::string err;
  auto reader =
      gufo::core::GgufReader::OpenMemory(binary.data(), binary.size(), &err);
  Expect(reader != nullptr, "Reader open succeeds");
  Expect(reader->HasVisionTensors(), "Vision tensors detected");

  // Must fail closed on text-only contract
  auto config = gufo::models::qwen::ExtractQwenConfig(*reader, &err);
  Expect(!config.has_value(),
         "Config extraction fails closed when vision weights present");
  Expect(err.find("vision") != std::string::npos,
         "Error mentions vision rejection");
}
}  // namespace

int main() {
  std::cout << "Running Qwen family config tests...\n";
  TestQwen35ConfigExtraction();
  TestQwen38_27BConfigExtraction();
  TestVisionExclusionRejection();
  std::cout << "All Qwen family config tests passed successfully!\n";
  return 0;
}
