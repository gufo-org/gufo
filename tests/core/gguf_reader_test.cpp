#include "src/core/gguf_reader.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/mapped_prefetch.hpp"
#include "src/core/model_config.hpp"
#include "src/core/quant/ggml_dequant.hpp"

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
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kString));
    AppendString(val);
    metadata_count_++;
  }

  void AddMetadataUint32(std::string_view key, std::uint32_t val) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kUint32));
    AppendPod(val);
    metadata_count_++;
  }

  void AddMetadataUint64(std::string_view key, std::uint64_t val) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kUint64));
    AppendPod(val);
    metadata_count_++;
  }

  void AddMetadataInt32(std::string_view key, std::int32_t val) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kInt32));
    AppendPod(val);
    metadata_count_++;
  }

  void AddMetadataFloat32(std::string_view key, float val) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kFloat32));
    AppendPod(val);
    metadata_count_++;
  }

  void AddMetadataBool(std::string_view key, bool val) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kBool));
    std::uint8_t b = val ? 1 : 0;
    AppendPod(b);
    metadata_count_++;
  }

  void AddMetadataInt32Array(std::string_view key,
                             const std::vector<std::int32_t>& values) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kArray));
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kInt32));
    AppendPod(static_cast<std::uint64_t>(values.size()));
    for (const auto value : values) {
      AppendPod(value);
    }
    metadata_count_++;
  }

  template<class T>
  void AddMetadataArray(std::string_view key, gufo::core::GgufValueType type,
                        const std::vector<T>& values) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kArray));
    AppendPod(static_cast<std::uint32_t>(type));
    AppendPod(static_cast<std::uint64_t>(values.size()));
    for (const auto value : values)
      AppendPod(value);
    ++metadata_count_;
  }

  void AddTensor(std::string_view name, const std::vector<std::uint64_t>& dims,
                 gufo::core::GgmlType type, std::uint64_t offset) {
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

    // Fixtures carry the complete declared payload, just like real files.
    for (const auto& tensor : tensors_to_write_) {
      std::size_t elements = 1;
      for (auto dimension : tensor.dims)
        elements *= dimension;
      payload_bytes =
          std::max(payload_bytes,
                   static_cast<std::size_t>(tensor.offset) +
                       gufo::quant::EncodedSizeBytes(tensor.type, elements));
    }
    // Append payload dummy data
    buffer_.resize(buffer_.size() + payload_bytes, 0xAB);
    return buffer_;
  }

private:
  struct TensorRecord {
    std::string name;
    std::vector<std::uint64_t> dims;
    gufo::core::GgmlType type;
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
  Expect(t0->dimensions[0] == 2560 && t0->dimensions[1] == 8,
         "Dimensions match");
  Expect(t0->type == gufo::core::GgmlType::kBF16, "BF16 type");
  Expect(t0->data != nullptr, "Valid memory mapped data pointer");

  const auto* t1 = reader->FindTensor("blk.0.attn_q.weight");
  Expect(t1 != nullptr, "blk.0.attn_q.weight found");
  Expect(t1->type == gufo::core::GgmlType::kQ4_K, "Q4_K type");

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
      ("gufo-gguf-reader-" + std::to_string(static_cast<long>(getpid())));
  std::filesystem::create_directories(temp_dir);
  const auto first_path = temp_dir / "model-00001-of-00002.gguf";
  const auto second_path = temp_dir / "model-00002-of-00002.gguf";

  GgufBuilder first;
  first.AddMetadataString("general.architecture", "qwen35");
  first.AddMetadataUint32("split.no", 0);
  first.AddMetadataUint32("split.count", 2);
  first.AddMetadataInt32("split.tensors.count", 2);
  first.AddTensor("token_embd.weight", {8, 4}, gufo::core::GgmlType::kBF16, 0);

  GgufBuilder second;
  second.AddMetadataUint32("split.no", 1);
  second.AddMetadataUint32("split.count", 2);
  second.AddMetadataInt32("split.tensors.count", 2);
  second.AddTensor("output_norm.weight", {4}, gufo::core::GgmlType::kF32, 0);

  WriteBinaryFile(first_path, first.Build(128));
  WriteBinaryFile(second_path, second.Build(128));

  std::string err;
  const auto reader = gufo::core::GgufReader::OpenFile(second_path, &err);
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
                    gufo::core::GgmlType::kF16, 0);

  auto binary = builder.Build();

  std::string err;
  auto reader =
      gufo::core::GgufReader::OpenMemory(binary.data(), binary.size(), &err);
  Expect(reader != nullptr, "Reader open succeeds");
  Expect(reader->HasVisionTensors(), "Vision tensors detected");

  // Must fail closed on text-only contract
  auto config = reader->ExtractModelConfig(&err);
  Expect(!config.has_value(),
         "Config extraction fails closed when vision weights present");
  Expect(err.find("vision") != std::string::npos,
         "Error mentions vision rejection");
}

void TestQuantizationLabel() {
  // The label the benchmark table prints, derived from the tensor table rather
  // than from `general.file_type`: the Unsloth UD-Q8_K_XL artifact records
  // ftype 15 (Q4_K_M) over Q8_0 weights, so a recorded ftype cannot be trusted
  // and must not override what the tensors say.
  {
    GgufBuilder builder;
    builder.AddMetadataString("general.architecture", "qwen35");
    builder.AddMetadataUint32("general.file_type", 15);
    builder.AddTensor("token_embd.weight", {256, 256},
                      gufo::core::GgmlType::kQ8_0, 0);
    builder.AddTensor("blk.0.attn_norm.weight", {64},
                      gufo::core::GgmlType::kF32, 64);
    builder.AddTensor("blk.1.attn_norm.weight", {64},
                      gufo::core::GgmlType::kF32, 128);
    builder.AddTensor("blk.2.attn_norm.weight", {64},
                      gufo::core::GgmlType::kF32, 192);
    auto binary = builder.Build(512);
    std::string err;
    auto reader =
        gufo::core::GgufReader::OpenMemory(binary.data(), binary.size(), &err);
    Expect(reader != nullptr, "Q8_0 label reader opens: " + err);
    Expect(reader->GetQuantizationLabel() == "Q8_0",
           "dominant type beats a stale ftype and F32 norms outnumbering it");
  }
  {
    // A mixed shard: no single format holds four fifths of the weights, so the
    // label has to say so instead of claiming the file is its largest format.
    GgufBuilder builder;
    builder.AddMetadataString("general.architecture", "qwen35");
    builder.AddTensor("blk.0.ffn_up.weight", {256, 16},
                      gufo::core::GgmlType::kQ5_K, 0);
    builder.AddTensor("blk.0.ffn_gate.weight", {256, 8},
                      gufo::core::GgmlType::kIQ4_XS, 64);
    builder.AddTensor("blk.0.attn_qkv.weight", {256, 4},
                      gufo::core::GgmlType::kQ4_K, 128);
    auto binary = builder.Build(512);
    std::string err;
    auto reader =
        gufo::core::GgufReader::OpenMemory(binary.data(), binary.size(), &err);
    Expect(reader != nullptr, "mixed label reader opens: " + err);
    Expect(reader->GetQuantizationLabel() == "Q5_K mixed",
           "a shard with no dominant format is reported as mixed");
  }
  {
    GgufBuilder builder;
    builder.AddMetadataString("general.architecture", "qwen35");
    builder.AddTensor("blk.0.ffn_up.weight", {256, 256},
                      gufo::core::GgmlType::kBF16, 0);
    auto binary = builder.Build(512);
    std::string err;
    auto reader =
        gufo::core::GgufReader::OpenMemory(binary.data(), binary.size(), &err);
    Expect(reader != nullptr, "BF16 label reader opens: " + err);
    Expect(reader->GetQuantizationLabel() == "BF16",
           "a single-format artifact is named without a suffix");
  }
}

void TestMalformedGgufRejection() {
  // Bad magic
  const std::uint8_t bad_magic[24] = {'N', 'O', 'P', 'E', 3, 0, 0, 0};
  std::string err;
  auto r1 =
      gufo::core::GgufReader::OpenMemory(bad_magic, sizeof(bad_magic), &err);
  Expect(r1 == nullptr, "Bad magic rejected");

  // Truncated buffer
  const std::uint8_t truncated[10] = {'G', 'G', 'U', 'F'};
  auto r2 =
      gufo::core::GgufReader::OpenMemory(truncated, sizeof(truncated), &err);
  Expect(r2 == nullptr, "Truncated buffer rejected");
}

void TestHostileHeaders() {
  const auto rejects = [](const std::vector<std::uint8_t>& bytes) {
    std::string error;
    Expect(
        !gufo::core::GgufReader::OpenMemory(bytes.data(), bytes.size(), &error),
        "malformed GGUF must fail closed");
    Expect(!error.empty(), "failed parse explains rejection");
  };
  const auto patch = [](auto& bytes, std::size_t offset, auto value) {
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
  };
  for (std::size_t offset : {8U, 16U}) {
    auto bytes = GgufBuilder{}.Build(0);
    patch(bytes, offset, UINT64_MAX);
    rejects(bytes);
  }
  for (std::uint32_t alignment : {0U, 3U, UINT32_MAX}) {
    GgufBuilder builder;
    builder.AddMetadataUint32("general.alignment", alignment);
    rejects(builder.Build());
  }
  {
    GgufBuilder builder;
    builder.AddMetadataString("x", "y");
    auto original = builder.Build();
    for (std::size_t offset : {24U, 37U}) {
      auto bytes = original;
      patch(bytes, offset, UINT64_MAX);
      rejects(bytes);
    }
    auto bytes = original;
    patch(bytes, 33, std::uint32_t{0x108});  // Must not truncate to STRING.
    rejects(bytes);
  }
  for (const auto type :
       {gufo::core::GgufValueType::kString, gufo::core::GgufValueType::kUint64,
        gufo::core::GgufValueType::kInt32,
        gufo::core::GgufValueType::kFloat32}) {
    GgufBuilder builder;
    builder.AddMetadataArray<std::uint64_t>("x", type, {});
    auto bytes = builder.Build();
    patch(bytes, 41, UINT64_MAX);
    rejects(bytes);
  }
  {
    GgufBuilder builder;
    builder.AddMetadataUint32("x", 0);
    builder.AddMetadataUint32("x", 1);
    rejects(builder.Build());
  }
  {
    GgufBuilder builder;
    builder.AddTensor("x", {32}, gufo::core::GgmlType::kQ8_0, 0);
    auto original = builder.Build(34);
    auto bytes = original;
    bytes.pop_back();
    rejects(bytes);  // Tensor start fits, final byte does not.
    bytes = original;
    patch(bytes, 37, UINT64_MAX);  // Dimension/product overflow.
    rejects(bytes);
    bytes = original;
    patch(bytes, 45, std::uint32_t{0x10008});  // No narrowing of storage type.
    rejects(bytes);
    bytes = original;
    patch(bytes, 49, UINT64_MAX);  // Tensor offset overflow.
    rejects(bytes);
    bytes = original;
    patch(bytes, 49, std::uint64_t{1});  // Unaligned offset.
    rejects(bytes);
  }
}

void TestVisionMetadataArrays() {
  GgufBuilder builder;
  builder.AddMetadataArray<float>("clip.vision.image_mean",
                                  gufo::core::GgufValueType::kFloat32,
                                  {0.5F, 0.5F, 0.5F});
  builder.AddMetadataArray<double>(
      "float64-array", gufo::core::GgufValueType::kFloat64, {1.25, -0.25});
  builder.AddMetadataArray<std::uint8_t>("clip.vision.is_deepstack_layers",
                                         gufo::core::GgufValueType::kBool,
                                         {0, 1, 0});
  auto bytes = builder.Build();
  auto reader = gufo::core::GgufReader::OpenMemory(bytes.data(), bytes.size());
  Expect(reader != nullptr, "vision metadata parses");
  Expect(std::get<std::vector<double>>(
             reader->FindMetadata("clip.vision.image_mean")->value) ==
             std::vector<double>({0.5, 0.5, 0.5}),
         "vision normalization metadata is retained");
  Expect(std::get<std::vector<double>>(
             reader->FindMetadata("float64-array")->value) ==
             std::vector<double>({1.25, -0.25}),
         "float64 array precision is retained");
  Expect(std::get<std::vector<std::uint64_t>>(
             reader->FindMetadata("clip.vision.is_deepstack_layers")->value) ==
             std::vector<std::uint64_t>({0, 1, 0}),
         "deepstack flags are retained");
}

void TestIntegerRoutingTensor() {
  GgufBuilder builder;
  builder.AddMetadataString("general.architecture", "deepseek4");
  // DS4 uses dense int32 token-to-expert tables, not quantized rows.
  builder.AddTensor("blk.0.ffn_gate_tid2eid.weight", {6, 64},
                    gufo::core::GgmlType::kI32, 0);
  auto bytes = builder.Build(6 * 64 * sizeof(std::int32_t));
  std::string error;
  auto reader =
      gufo::core::GgufReader::OpenMemory(bytes.data(), bytes.size(), &error);
  Expect(reader != nullptr, "integer expert routing table loads: " + error);
  const auto* table = reader->FindTensor("blk.0.ffn_gate_tid2eid.weight");
  Expect(table && table->type == gufo::core::GgmlType::kI32 &&
             table->size_bytes == 6 * 64 * sizeof(std::int32_t),
         "integer routing payload retains its storage type and exact extent");
  bytes.pop_back();
  Expect(
      !gufo::core::GgufReader::OpenMemory(bytes.data(), bytes.size(), &error),
      "truncated integer routing table is rejected");
  Expect(gufo::quant::EncodedSizeBytes(
             gufo::core::GgmlType::kI32,
             std::numeric_limits<std::size_t>::max() / 4 + 1) == 0,
         "integer routing extent overflow is rejected");
}

void TestMappedPrefetch() {
  char path[] = "/tmp/gufo-prefetch-XXXXXX";
  const int fd = mkstemp(path);
  Expect(fd >= 0, "create prefetch fixture");
  unlink(path);
  constexpr std::size_t bytes = (17U << 20) + 7;
  Expect(ftruncate(fd, bytes) == 0, "size two-chunk prefetch fixture");
  const char marker = 'Q';
  Expect(pwrite(fd, &marker, 1, bytes - 1) == 1, "write tail marker");
  auto* data = static_cast<const char*>(
      mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0));
  Expect(data != MAP_FAILED, "map prefetch fixture");
  gufo::core::PrefaultMappedRange(data + 13, bytes - 13);
  Expect(data[13] == 0 && data[bytes - 1] == marker,
         "parallel unaligned prefetch preserves the complete readable range");
  gufo::core::PrefaultMappedRange(nullptr, 0);
  Expect(ftruncate(fd, 4096) == 0, "truncate mapped fixture");
  bool rejected = false;
  try {
    gufo::core::PrefaultMappedRange(data, bytes);
  } catch (const std::system_error&) {
    rejected = true;
  }
  Expect(rejected, "failed parallel reads are joined and propagated");
  munmap(const_cast<char*>(data), bytes);
  close(fd);
}

}  // namespace

int main() {
  TestMappedPrefetch();
  TestIntegerRoutingTensor();
  TestVisionMetadataArrays();
  std::cout << "Running GgufReader unit tests...\n";
  TestBasicGgufParsing();
  TestQwen38_27BParsing();
  TestSplitGgufDiscovery();
  TestVisionExclusionValidation();
  TestQuantizationLabel();
  TestMalformedGgufRejection();
  TestHostileHeaders();
  std::cout << "All GgufReader tests passed successfully!\n";
  return 0;
}
