#include "src/core/gguf_reader.hpp"

#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/model_config.hpp"

namespace {

void Expect(bool condition, std::string_view msg) {
  if (!condition) {
    std::cerr << "Assertion failed: " << msg << "\n";
    std::exit(1);
  }
}

// Helper to construct a synthetic valid GGUF binary in memory
class GgufBuilder {
public:
  GgufBuilder(std::uint32_t version = 3) {
    // Magic "GGUF"
    const char magic[4] = {'G', 'G', 'U', 'F'};
    AppendBytes(magic, 4);
    AppendPod(version);
    // Reserve space for tensor_count and metadata_count (filled on Build)
    tensor_count_pos_ = buffer_.size();
    AppendPod(static_cast<std::uint64_t>(0));
    metadata_count_pos_ = buffer_.size();
    AppendPod(static_cast<std::uint64_t>(0));
  }

  void AddMetadataString(std::string_view key, std::string_view val) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(strix::core::GgufValueType::kString));
    AppendString(val);
    metadata_count_++;
  }

  void AddMetadataUint32(std::string_view key, std::uint32_t val) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(strix::core::GgufValueType::kUint32));
    AppendPod(val);
    metadata_count_++;
  }

  void AddMetadataUint64(std::string_view key, std::uint64_t val) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(strix::core::GgufValueType::kUint64));
    AppendPod(val);
    metadata_count_++;
  }

  void AddMetadataInt32(std::string_view key, std::int32_t val) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(strix::core::GgufValueType::kInt32));
    AppendPod(val);
    metadata_count_++;
  }

  void AddMetadataFloat32(std::string_view key, float val) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(strix::core::GgufValueType::kFloat32));
    AppendPod(val);
    metadata_count_++;
  }

  void AddMetadataBool(std::string_view key, bool val) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(strix::core::GgufValueType::kBool));
    std::uint8_t b = val ? 1 : 0;
    AppendPod(b);
    metadata_count_++;
  }

  void AddMetadataInt32Array(std::string_view key,
                             const std::vector<std::int32_t>& values) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(strix::core::GgufValueType::kArray));
    AppendPod(static_cast<std::uint32_t>(strix::core::GgufValueType::kInt32));
    AppendPod(static_cast<std::uint64_t>(values.size()));
    for (const auto value : values) {
      AppendPod(value);
    }
    metadata_count_++;
  }

  void AddTensor(std::string_view name, const std::vector<std::uint64_t>& dims,
                 strix::core::GgmlType type, std::uint64_t offset) {
    tensors_to_write_.push_back({std::string(name), dims, type, offset});
  }

  std::vector<std::uint8_t> Build(std::size_t payload_bytes = 1024) {
    // Patch metadata count
    std::memcpy(buffer_.data() + metadata_count_pos_, &metadata_count_,
                sizeof(metadata_count_));

    // Write tensors
    std::uint64_t tensor_count = tensors_to_write_.size();
    std::memcpy(buffer_.data() + tensor_count_pos_, &tensor_count,
                sizeof(tensor_count));

    for (const auto& t : tensors_to_write_) {
      AppendString(t.name);
      AppendPod(static_cast<std::uint32_t>(t.dims.size()));
      for (auto d : t.dims) {
        AppendPod(d);
      }
      AppendPod(static_cast<std::uint32_t>(t.type));
      AppendPod(t.offset);
    }

    // Align to 32 bytes
    std::size_t rem = buffer_.size() % 32;
    if (rem != 0) {
      buffer_.resize(buffer_.size() + (32 - rem), 0);
    }

    // Append payload dummy data
    buffer_.resize(buffer_.size() + payload_bytes, 0xAB);
    return buffer_;
  }

private:
  struct TensorRecord {
    std::string name;
    std::vector<std::uint64_t> dims;
    strix::core::GgmlType type;
    std::uint64_t offset;
  };

  void AppendBytes(const void* data, std::size_t len) {
    const auto* p = static_cast<const std::uint8_t*>(data);
    buffer_.insert(buffer_.end(), p, p + len);
  }

  template<typename T>
  void AppendPod(T val) {
    AppendBytes(&val, sizeof(T));
  }

  void AppendString(std::string_view s) {
    std::uint64_t len = s.size();
    AppendPod(len);
    AppendBytes(s.data(), len);
  }

  std::vector<std::uint8_t> buffer_;
  std::size_t tensor_count_pos_{0};
  std::size_t metadata_count_pos_{0};
  std::uint64_t metadata_count_{0};
  std::vector<TensorRecord> tensors_to_write_;
};

void TestBasicGgufParsing() {
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
  builder.AddMetadataInt32Array("qwen35.rope.dimension_sections",
                                {64, 0, 0, 0});

  builder.AddTensor("token_embd.weight", {248320, 2560},
                    strix::core::GgmlType::kBF16, 0);
  builder.AddTensor("blk.0.attn_q.weight", {2560, 2560},
                    strix::core::GgmlType::kStrixSHQ4_T16, 64);
  builder.AddTensor("mtp.0.proj.weight", {2560, 2560},
                    strix::core::GgmlType::kStrixSHQ8_T16, 128);

  auto binary = builder.Build(512);

  std::string err;
  auto reader =
      strix::core::GgufReader::OpenMemory(binary.data(), binary.size(), &err);
  Expect(reader != nullptr, "Reader open succeeds: " + err);
  Expect(reader->GetVersion() == 3, "GGUF version 3");
  Expect(reader->GetTensorCount() == 3, "3 tensors parsed");
  Expect(reader->GetMetadataCount() == 12, "12 metadata entries");

  // Metadata retrieval
  Expect(reader->GetMetadataString("general.architecture") == "qwen35",
         "Architecture is qwen35");
  Expect(reader->GetMetadataUint32("qwen35.block_count") == 36, "36 layers");
  Expect(reader->GetMetadataUint32("qwen35.embedding_length") == 2560,
         "Hidden size 2560");
  const auto* rope_sections =
      reader->FindMetadata("qwen35.rope.dimension_sections");
  Expect(rope_sections != nullptr, "signed integer array found");
  Expect(
      std::holds_alternative<std::vector<std::int64_t>>(rope_sections->value),
      "signed integer array retains its type");
  Expect(std::get<std::vector<std::int64_t>>(rope_sections->value) ==
             std::vector<std::int64_t>({64, 0, 0, 0}),
         "signed integer array values match");

  // Tensor inspection
  const auto* t0 = reader->FindTensor("token_embd.weight");
  Expect(t0 != nullptr, "token_embd.weight found");
  Expect(t0->dimensions.size() == 2, "2 dimensions");
  Expect(t0->dimensions[0] == 248320 && t0->dimensions[1] == 2560,
         "Dimensions match");
  Expect(t0->type == strix::core::GgmlType::kBF16, "BF16 type");
  Expect(t0->data != nullptr, "Valid memory mapped data pointer");

  const auto* t1 = reader->FindTensor("blk.0.attn_q.weight");
  Expect(t1 != nullptr, "blk.0.attn_q.weight found");
  Expect(t1->type == strix::core::GgmlType::kStrixSHQ4_T16,
         "Strix SHQ4_T16 type");

  // MTP presence
  Expect(reader->HasMtpTensors(), "MTP tensors detected");
  Expect(!reader->HasVisionTensors(), "No vision tensors");

  // Config extraction
  auto config = reader->ExtractModelConfig(&err);
  Expect(config.has_value(), "ExtractModelConfig succeeds: " + err);
  Expect(config->num_layers == 36, "36 layers");
  Expect(config->hidden_size == 2560, "2560 hidden size");
  Expect(config->head_dim == 128, "128 head dim");
  Expect(config->full_attention_interval == 4, "full_attention_interval 4");
  Expect(config->mtp_num_layers == 1, "MTP layer count 1");
  Expect(config->is_text_only, "is_text_only true");
  Expect(config->IsValidQwen(), "IsValidQwen true");
}

void TestQwen38_27BParsing() {
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

  builder.AddTensor("token_embd.weight", {248320, 5120},
                    strix::core::GgmlType::kBF16, 0);
  builder.AddTensor("blk.3.attn_q.weight", {12288, 5120},
                    strix::core::GgmlType::kStrixSHQ4_T16, 64);
  builder.AddTensor("blk.63.ffn_gate.weight", {17408, 5120},
                    strix::core::GgmlType::kStrixSHQ4_T16, 128);
  builder.AddTensor("blk.64.nextn.enorm.weight", {5120},
                    strix::core::GgmlType::kStrixSHQ8_T16, 192);

  auto binary = builder.Build(512);

  std::string err;
  auto reader =
      strix::core::GgufReader::OpenMemory(binary.data(), binary.size(), &err);
  Expect(reader != nullptr, "Reader open succeeds: " + err);
  Expect(reader->GetTensorCount() == 4, "4 tensors parsed for 27B");

  // Config extraction and verification for 27B
  auto config = reader->ExtractModelConfig(&err);
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

void WriteBinaryFile(const std::filesystem::path& path,
                     const std::vector<std::uint8_t>& data) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size()));
  Expect(out.good(), "Write GGUF test shard");
}

void TestSplitGgufDiscovery() {
  const auto temp_dir =
      std::filesystem::temp_directory_path() /
      ("strix-gguf-reader-" + std::to_string(static_cast<long>(getpid())));
  std::filesystem::create_directories(temp_dir);
  const auto first_path = temp_dir / "model-00001-of-00002.gguf";
  const auto second_path = temp_dir / "model-00002-of-00002.gguf";

  GgufBuilder first;
  first.AddMetadataString("general.architecture", "qwen35");
  first.AddMetadataUint32("split.no", 0);
  first.AddMetadataUint32("split.count", 2);
  first.AddMetadataInt32("split.tensors.count", 2);
  first.AddTensor("token_embd.weight", {8, 4}, strix::core::GgmlType::kBF16, 0);

  GgufBuilder second;
  second.AddMetadataUint32("split.no", 1);
  second.AddMetadataUint32("split.count", 2);
  second.AddMetadataInt32("split.tensors.count", 2);
  second.AddTensor("output_norm.weight", {4}, strix::core::GgmlType::kF32, 0);

  WriteBinaryFile(first_path, first.Build(128));
  WriteBinaryFile(second_path, second.Build(128));

  std::string err;
  const auto reader = strix::core::GgufReader::OpenFile(second_path, &err);
  Expect(reader != nullptr, "Split reader opens from any shard: " + err);
  Expect(reader->GetTensorCount() == 2, "Split reader merges tensor indexes");
  Expect(reader->FindTensor("token_embd.weight") != nullptr,
         "Tensor from first shard found");
  Expect(reader->FindTensor("output_norm.weight") != nullptr,
         "Tensor from second shard found");
  Expect(reader->GetMappedRegions().size() == 2,
         "Split reader exposes two mapped regions");
  Expect(reader->GetData() == nullptr,
         "Split reader has no single contiguous backing pointer");

  std::filesystem::remove_all(temp_dir);
}

void TestVisionExclusionValidation() {
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
                    strix::core::GgmlType::kF16, 0);

  auto binary = builder.Build();

  std::string err;
  auto reader =
      strix::core::GgufReader::OpenMemory(binary.data(), binary.size(), &err);
  Expect(reader != nullptr, "Reader open succeeds");
  Expect(reader->HasVisionTensors(), "Vision tensors detected");

  // Must fail closed on text-only contract
  auto config = reader->ExtractModelConfig(&err);
  Expect(!config.has_value(),
         "Config extraction fails closed when vision weights present");
  Expect(err.find("vision") != std::string::npos,
         "Error mentions vision rejection");
}

void TestMalformedGgufRejection() {
  // Bad magic
  const std::uint8_t bad_magic[24] = {'N', 'O', 'P', 'E', 3, 0, 0, 0};
  std::string err;
  auto r1 =
      strix::core::GgufReader::OpenMemory(bad_magic, sizeof(bad_magic), &err);
  Expect(r1 == nullptr, "Bad magic rejected");

  // Truncated buffer
  const std::uint8_t truncated[10] = {'G', 'G', 'U', 'F'};
  auto r2 =
      strix::core::GgufReader::OpenMemory(truncated, sizeof(truncated), &err);
  Expect(r2 == nullptr, "Truncated buffer rejected");
}

}  // namespace

int main() {
  std::cout << "Running GgufReader unit tests...\n";
  TestBasicGgufParsing();
  TestQwen38_27BParsing();
  TestSplitGgufDiscovery();
  TestVisionExclusionValidation();
  TestMalformedGgufRejection();
  std::cout << "All GgufReader tests passed successfully!\n";
  return 0;
}
