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
#include "tests/fixtures/gguf_builder.hpp"

namespace {

using gufo::test_fixture::GgufBuilder;

void Expect(bool condition, std::string_view msg) {
  if (!condition) {
    std::cerr << "Assertion failed: " << msg << "\n";
    std::exit(1);
  }
}

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

  // ExtractQwenConfig assertions moved with the function to
  // tests/families/qwen/qwen_config_test.cpp.
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

void TestNarrowIntegerArrays() {
  // Writers store small integer lists in the narrowest type. Each width must
  // decode, and an array must never read back as a scalar.
  using gufo::core::GgufValueType;
  GgufBuilder builder;
  builder.AddMetadataArray<std::uint8_t>("u8", GgufValueType::kUint8,
                                         {0, 255, 7});
  builder.AddMetadataArray<std::uint16_t>("u16", GgufValueType::kUint16,
                                          {65535, 2});
  builder.AddMetadataArray<std::int8_t>("i8", GgufValueType::kInt8, {-128, 5});
  builder.AddMetadataArray<std::int16_t>("i16", GgufValueType::kInt16,
                                         {-32768, 32767});
  auto bytes = builder.Build();
  auto reader = gufo::core::GgufReader::OpenMemory(bytes.data(), bytes.size());
  Expect(reader != nullptr, "narrow integer arrays parse");
  Expect(
      std::get<std::vector<std::uint64_t>>(reader->FindMetadata("u8")->value) ==
          std::vector<std::uint64_t>({0, 255, 7}),
      "uint8 array is decoded");
  Expect(std::get<std::vector<std::uint64_t>>(
             reader->FindMetadata("u16")->value) ==
             std::vector<std::uint64_t>({65535, 2}),
         "uint16 array is decoded");
  Expect(
      std::get<std::vector<std::int64_t>>(reader->FindMetadata("i8")->value) ==
          std::vector<std::int64_t>({-128, 5}),
      "int8 array is decoded with its sign");
  Expect(
      std::get<std::vector<std::int64_t>>(reader->FindMetadata("i16")->value) ==
          std::vector<std::int64_t>({-32768, 32767}),
      "int16 array is decoded with its sign");
  Expect(!reader->GetMetadataUint32("u8") && !reader->GetMetadataUint64("i16"),
         "an integer array is not a scalar");
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
  TestNarrowIntegerArrays();
  std::cout << "Running GgufReader unit tests...\n";
  TestBasicGgufParsing();
  TestSplitGgufDiscovery();
  TestQuantizationLabel();
  TestMalformedGgufRejection();
  TestHostileHeaders();
  std::cout << "All GgufReader tests passed successfully!\n";
  return 0;
}
