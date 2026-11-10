#include "src/models/qwen/probe.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/families/qwen/config/qwen_config.hpp"
#include "tests/fixtures/gguf_builder.hpp"

// Provider selection for the qwen35 GGUF family. The Qwen3.5-4B artifact
// (32 layers, 2560 hidden, no embedded MTP layer) belongs to the qwen35-4b
// provider; the 5120-hidden Qwen3.8 shapes belong to the Qwen3.8 providers.
// Expectations are written against the published Unsloth Qwen3.5-4B-Q4_K_M
// metadata, not derived from the probe implementation.

namespace {

using gufo::test_fixture::GgufBuilder;

void Expect(bool condition, std::string_view msg) {
  if (!condition) {
    std::cerr << "Assertion failed: " << msg << "\n";
    std::exit(1);
  }
}

// One openable reader per probe input; the probes read metadata only.
std::unique_ptr<gufo::core::GgufReader> Open(
    const std::vector<std::uint8_t>& binary, std::string& err) {
  auto reader =
      gufo::core::GgufReader::OpenMemory(binary.data(), binary.size(), &err);
  Expect(reader != nullptr, "Reader open succeeds: " + err);
  return reader;
}

// The reader rejects duplicate metadata keys, so every variant writes its
// own complete artifact instead of overriding a shared builder.
GgufBuilder Qwen35_4BArtifact(std::string_view arch = "qwen35",
                              std::uint32_t embedding_length = 2560,
                              bool embedded_mtp_layer = false) {
  GgufBuilder builder;
  builder.AddMetadataString("general.architecture", arch);
  builder.AddMetadataString("general.name", "Qwen3.5-4B");
  builder.AddMetadataUint32("qwen35.block_count", 32);
  builder.AddMetadataUint32("qwen35.embedding_length", embedding_length);
  builder.AddMetadataUint32("qwen35.feed_forward_length", 9216);
  builder.AddMetadataUint32("qwen35.attention.head_count", 16);
  builder.AddMetadataUint32("qwen35.attention.head_count_kv", 4);
  builder.AddMetadataUint32("qwen35.attention.key_length", 256);
  builder.AddMetadataUint32("qwen35.context_length", 262144);
  builder.AddMetadataUint32("qwen35.full_attention_interval", 4);
  builder.AddMetadataUint32("qwen35.ssm.conv_kernel", 4);
  builder.AddMetadataUint32("qwen35.ssm.state_size", 128);
  builder.AddMetadataUint32("qwen35.ssm.group_count", 16);
  builder.AddMetadataUint32("qwen35.ssm.time_step_rank", 32);
  builder.AddMetadataUint32("qwen35.ssm.inner_size", 4096);
  builder.AddMetadataUint32("qwen35.rope.dimension_count", 64);
  builder.AddMetadataFloat32("qwen35.rope.freq_base", 10000000.0F);
  if (embedded_mtp_layer) {
    // The 27B artifact counts its MTP layer inside block_count.
    builder.AddMetadataUint32("qwen35.nextn_predict_layers", 1);
  }
  builder.AddTensor("token_embd.weight", {2560, 8}, gufo::core::GgmlType::kBF16,
                    0);
  return builder;
}

void TestQwen35_4BProbe() {
  auto binary = Qwen35_4BArtifact().Build(512);
  std::string err;
  auto reader = Open(binary, err);

  Expect(gufo::models::qwen::ProbeQwen35_4BGguf(*reader),
         "Qwen3.5-4B artifact probes as qwen35-4b");
  Expect(!gufo::models::qwen::ProbeQwenGguf(*reader),
         "Qwen3.5-4B artifact is not claimed by the 27B probe");

  auto config = gufo::models::qwen::ExtractQwenConfig(*reader, &err);
  Expect(config.has_value(), "ExtractQwenConfig succeeds: " + err);
  Expect(config->num_layers == 32, "32 layers");
  Expect(config->hidden_size == 2560, "2560 hidden size");
  Expect(config->intermediate_size == 9216, "9216 intermediate size");
  Expect(config->head_dim == 256, "256 head dim");
  Expect(config->rotary_dim == 64, "64 rotary dim");
  Expect(config->full_attention_interval == 4, "full_attention_interval 4");
  Expect(config->ssm_time_step_rank == 32, "SSM time_step_rank 32");
  Expect(config->ssm_inner_size == 4096, "SSM inner_size 4096");
  Expect(config->SsmValueSize() == 128, "SSM value size 128");
  Expect(config->mtp_num_layers == 0, "no embedded MTP layer");
  Expect(config->is_text_only, "is_text_only true");
  Expect(config->IsValidQwen(), "IsValidQwen true");
}

void TestLargerQwen35ShapesRejected() {
  {
    auto binary = Qwen35_4BArtifact("qwen35", 5120).Build(512);
    std::string err;
    auto reader = Open(binary, err);
    Expect(!gufo::models::qwen::ProbeQwen35_4BGguf(*reader),
           "2560-hidden probe rejects the 5120 shape");
  }
  {
    auto binary = Qwen35_4BArtifact("qwen35", 2560, true).Build(512);
    std::string err;
    auto reader = Open(binary, err);
    Expect(!gufo::models::qwen::ProbeQwen35_4BGguf(*reader),
           "Probe rejects qwen35 artifacts with an embedded MTP layer");
  }
  {
    auto binary = Qwen35_4BArtifact("qwen38").Build(512);
    std::string err;
    auto reader = Open(binary, err);
    Expect(!gufo::models::qwen::ProbeQwen35_4BGguf(*reader),
           "Probe requires the qwen35 architecture prefix");
  }
}

}  // namespace

int main() {
  TestQwen35_4BProbe();
  TestLargerQwen35ShapesRejected();
  std::cout << "qwen probe tests passed\n";
  return 0;
}
