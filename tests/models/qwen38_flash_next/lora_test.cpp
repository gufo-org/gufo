// Flash-Next LoRA merge: the Q8_0 encoder against gguf-py goldens, merges
// against integer arithmetic, adapter validation and cache identities.
#include "src/models/qwen38_flash_next/lora.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "gufo/ext/sha256.hpp"
#include "src/models/qwen38_flash_next/continuation_identity.hpp"
#include "tests/fixtures/gguf_builder.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
using gufo::core::GgmlType;

namespace {

void Check(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

// gguf-py (llama.cpp's quantize_row_q8_0_ref port) on the same inputs; see
// the generator in the change description. Block one has rounding ties.
constexpr std::array<std::uint32_t, 32> kTiesIn = {
    0x40200000, 0xc0200000, 0x3f000000, 0xbf000000, 0x40600000, 0xc0600000,
    0x42fd0000, 0xc2fd0000, 0x3fa00000, 0xbfe00000, 0x42fe0000, 0x00000000,
    0x40f00000, 0xc0f00000, 0x42c8fae1, 0xc2c9051f, 0xc1000000, 0xc0e00000,
    0xc0c00000, 0xc0a00000, 0xc0800000, 0xc0400000, 0xc0000000, 0xbf800000,
    0x00000000, 0x3f800000, 0x40000000, 0x40400000, 0x40800000, 0x40a00000,
    0x40c00000, 0x40e00000};
constexpr std::array<std::uint8_t, 34> kTiesOut = {
    0x00, 0x3c, 0x03, 0xfd, 0x01, 0xff, 0x04, 0xfc, 0x7f, 0x81, 0x01, 0xfe,
    0x7f, 0x00, 0x08, 0xf8, 0x64, 0x9b, 0xf8, 0xf9, 0xfa, 0xfb, 0xfc, 0xfd,
    0xfe, 0xff, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07};
constexpr std::array<std::uint32_t, 32> kMixedIn = {
    0xbd0c4bb5, 0xbea45612, 0x3c59ce47, 0x3e4bd770, 0xbe96bbc8, 0x3f32d3f9,
    0x3da2ceed, 0xbe3d38db, 0x3ee6de29, 0x3da0e7d5, 0xbec88671, 0xbf0661df,
    0xbe45081a, 0x3e884f7d, 0xbed1baf3, 0x3e5da54f, 0xbe5b4b73, 0xbf1da486,
    0xbed7724a, 0x3e7b4d74, 0x3e0139e1, 0xbe1a68c0, 0xbca74096, 0xbe971fe5,
    0xbe0ed161, 0x3decfad5, 0xbe324661, 0x3e62b7aa, 0xbe321a86, 0xbe52b70f,
    0x3f2d024f, 0x3e3c783e};
constexpr std::array<std::uint8_t, 34> kMixedOut = {
    0xa2, 0x1d, 0xfa, 0xc6, 0x02, 0x24, 0xca, 0x7f, 0x0e, 0xde, 0x52, 0x0e,
    0xb9, 0xa1, 0xdd, 0x30, 0xb6, 0x27, 0xd9, 0x90, 0xb3, 0x2d, 0x17, 0xe5,
    0xfc, 0xca, 0xe7, 0x15, 0xe0, 0x28, 0xe0, 0xdb, 0x7b, 0x21};
// Block scale below the smallest normal half.
constexpr std::array<std::uint32_t, 32> kTinyIn = {
    0xb9b23e8b, 0xb99f7d15, 0x38264987, 0x394a80de, 0xb963d633, 0xb8b5aebe,
    0xb8f51cbf, 0x3989447a, 0xb9fca1d9, 0xb92590fb, 0xb94a40c3, 0x38f29596,
    0x3789f21b, 0x39444e12, 0x39b30eb4, 0x3a00dd6e, 0x398aacca, 0xb8c0a041,
    0x37a44cca, 0x3a2529d4, 0x39e6c825, 0xb89e3066, 0xb9104679, 0x39fc72b4,
    0xba297f3f, 0xb981446d, 0x3932ac4f, 0x39fecf16, 0x39b003c0, 0xb754728d,
    0x38bc4595, 0xb8adb861};
constexpr std::array<std::uint8_t, 34> kTinyOut = {
    0x55, 0x00, 0xbd, 0xc4, 0x08, 0x26, 0xd5, 0xef, 0xe9, 0x33, 0xa1, 0xe1,
    0xda, 0x17, 0x03, 0x25, 0x43, 0x61, 0x34, 0xee, 0x04, 0x7c, 0x56, 0xf1,
    0xe5, 0x5f, 0x81, 0xd0, 0x21, 0x5f, 0x42, 0xfe, 0x12, 0xf0};

void CheckEncoder() {
  const auto encode = [](const std::array<std::uint32_t, 32>& bits) {
    std::array<float, 32> values{};
    for (std::size_t i = 0; i < 32; ++i)
      values[i] = std::bit_cast<float>(bits[i]);
    std::array<std::uint8_t, 34> out{};
    Check(qfn::QuantizeRowQ8_0(values.data(), out.data(), 32),
          "finite block rejected");
    return out;
  };
  Check(encode(kTiesIn) == kTiesOut, "Q8_0 ties differ from gguf-py");
  Check(encode(kMixedIn) == kMixedOut, "Q8_0 block differs from gguf-py");
  Check(encode(kTinyIn) == kTinyOut, "Q8_0 subnormal scale differs");
  Check(encode({}) == std::array<std::uint8_t, 34>{}, "zero block not zero");
  // 1e-39: the block scale is subnormal and its reciprocal overflows.
  std::array<std::uint32_t, 32> subnormal{};
  subnormal.fill(0x000ae398);
  Check(encode(subnormal) == std::array<std::uint8_t, 34>{},
        "block with an infinite reciprocal scale not zero");

  // NumPy float16 casts of the same float32 bits.
  constexpr std::array<std::pair<std::uint32_t, std::uint16_t>, 15> halves = {{
      {0x3f800000, 0x3c00},
      {0x3f801000, 0x3c00},
      {0x3f803000, 0x3c02},
      {0x477fe000, 0x7bff},
      {0x477fef00, 0x7bff},
      {0x477ff000, 0x7c00},
      {0x38800000, 0x0400},
      {0x33000000, 0x0000},
      {0x33c00000, 0x0002},
      {0x33800000, 0x0001},
      {0x387fe000, 0x0400},
      {0x80000000, 0x8000},
      {0xb7fba882, 0x81f7},
      {0x3dcccccd, 0x2e66},
      {0x322bcc77, 0x0000},
  }};
  for (const auto& [in, out] : halves)
    Check(qfn::FloatToHalf(std::bit_cast<float>(in)) == out,
          "half rounding of " + std::to_string(in));
  const float huge = 1.0e7F * 127.0F;
  std::array<float, 32> overflow{};
  overflow[3] = huge;
  std::array<std::uint8_t, 34> out{};
  Check(!qfn::QuantizeRowQ8_0(overflow.data(), out.data(), 32),
        "block scale overflow accepted");
}

// A Q8_0 matrix with every block scale 1, so codes are the weights.
struct Matrix {
  std::uint64_t cols;
  std::uint64_t rows;
  std::vector<std::uint8_t> bytes;
  qfn::TensorRef ref;

  Matrix(const char* name, std::uint64_t c, std::uint64_t r)
      : cols(c), rows(r) {
    bytes.resize(cols / 32 * 34 * rows);
    for (std::uint64_t row = 0; row < rows; ++row)
      for (std::uint64_t col = 0; col < cols; ++col)
        Set(row, col, Code(row, col));
    ref.data = bytes.data();
    ref.type = GgmlType::kQ8_0;
    ref.cols = cols;
    ref.rows = rows;
    ref.name = name;
  }
  // Asymmetric codes; the last column of each block holds the block maximum
  // and is never patched, so every merged block keeps scale 1.
  static int Code(std::uint64_t row, std::uint64_t col) {
    return col % 32 == 31 ? 127
                          : static_cast<int>((row * 7 + col * 3) % 41) - 20;
  }
  void Set(std::uint64_t row, std::uint64_t col, int code) {
    auto* block = bytes.data() + (row * (cols / 32) + col / 32) * 34;
    const std::uint16_t one = 0x3c00;
    std::memcpy(block, &one, 2);
    block[2 + col % 32] =
        static_cast<std::uint8_t>(static_cast<std::int8_t>(code));
  }
};

struct Tensor {
  std::string name;
  std::vector<std::uint64_t> dims;
  std::vector<float> values;
  GgmlType type{GgmlType::kF32};
};

struct AdapterSpec {
  std::string architecture{"qwen4exp"};
  std::string general_type{"adapter"};
  std::string adapter_type{"lora"};
  bool has_alpha{true};
  float alpha{1.0F};
  std::vector<Tensor> tensors;
};

std::vector<std::uint8_t> Build(const AdapterSpec& spec) {
  gufo::test_fixture::GgufBuilder builder;
  builder.AddMetadataString("general.architecture", spec.architecture);
  builder.AddMetadataString("general.type", spec.general_type);
  builder.AddMetadataString("adapter.type", spec.adapter_type);
  if (spec.has_alpha)
    builder.AddMetadataFloat32("adapter.lora.alpha", spec.alpha);
  std::uint64_t offset = 0;
  std::vector<std::uint64_t> offsets;
  for (const auto& t : spec.tensors) {
    builder.AddTensor(t.name, t.dims, t.type, offset);
    offsets.push_back(offset);
    offset += (t.values.size() * 4 + 31) / 32 * 32;
  }
  // The payload follows the header and is exactly `offset` bytes long.
  auto bytes = builder.Build(offset);
  const std::size_t start = bytes.size() - offset;
  for (std::size_t i = 0; i < spec.tensors.size(); ++i) {
    const auto& values = spec.tensors[i].values;
    std::memcpy(bytes.data() + start + offsets[i], values.data(),
                values.size() * 4);
  }
  return bytes;
}

std::unique_ptr<qfn::LoraAdapter> Open(const std::vector<std::uint8_t>& bytes,
                                       std::span<const qfn::TensorRef> targets,
                                       float scale, std::string* error,
                                       std::string label = "test.gguf") {
  auto reader =
      gufo::core::GgufReader::OpenMemory(bytes.data(), bytes.size(), error);
  Check(reader != nullptr, "synthetic adapter does not parse: " + *error);
  return qfn::OpenLoraAdapter(std::move(reader), std::move(label), scale,
                              "qwen4exp", targets, error);
}

// lora_a: rank rows of `cols`; lora_b: `rows` rows of rank. Values are
// multiples of 1/4 (A) and small integers (B), so every update is exact.
float A(std::uint32_t k, std::uint64_t col) {
  return col % 32 == 31 ? 0.0F
                        : 0.25F * (static_cast<float>(col % 29) - 14.0F) *
                              (k == 0 ? 1.0F : -0.5F);
}
float B(std::uint32_t k, std::uint64_t row) {
  constexpr float kB[2][3] = {{2.0F, -1.0F, -3.0F}, {1.0F, 4.0F, -2.0F}};
  return kB[k][row];
}

AdapterSpec Pair(const std::string& base, std::uint64_t cols,
                 std::uint64_t rows, std::uint32_t rank) {
  Tensor a{base + ".lora_a", {cols, rank}, {}};
  Tensor b{base + ".lora_b", {rank, rows}, {}};
  for (std::uint32_t k = 0; k < rank; ++k)
    for (std::uint64_t c = 0; c < cols; ++c)
      a.values.push_back(A(k, c));
  for (std::uint64_t r = 0; r < rows; ++r)
    for (std::uint32_t k = 0; k < rank; ++k)
      b.values.push_back(B(k, r));
  AdapterSpec spec;
  spec.tensors = {a, b};
  return spec;
}

// Expected code in exact integer arithmetic: round-half-away of the base
// code plus the update, all multiples of 1/16 here.
int Expected(std::uint64_t row, std::uint64_t col, std::uint32_t rank,
             double multiplier) {
  double update = 0;
  for (std::uint32_t k = 0; k < rank; ++k)
    update += static_cast<double>(B(k, row)) * A(k, col);
  const long sixteenths =
      Matrix::Code(row, col) * 16L + std::lround(update * multiplier * 16.0);
  const long magnitude = (std::labs(sixteenths) + 8) / 16;
  return static_cast<int>(sixteenths < 0 ? -magnitude : magnitude);
}

void CheckMerge(std::uint32_t rank, float alpha, float scale) {
  Matrix m("blk.0.ssm_out.weight", 64, 3);
  const std::vector<qfn::TensorRef> targets{m.ref};
  auto spec = Pair("blk.0.ssm_out.weight", 64, 3, rank);
  spec.alpha = alpha;
  std::string error;
  const auto bytes = Build(spec);
  const auto adapter = Open(bytes, targets, scale, &error);
  Check(adapter != nullptr, "valid adapter rejected: " + error);
  const auto* patch = adapter->Find("blk.0.ssm_out.weight");
  Check(patch != nullptr && patch->rank == rank, "patch not found");
  // Rows one and two only, as a chunked upload asks for them.
  std::vector<std::uint8_t> out(2 * m.ref.RowBytes());
  Check(adapter->MergeRows(*patch, 1, 2, out.data(), &error), error);
  const double multiplier = scale * alpha / rank;
  for (std::uint64_t row = 1; row < 3; ++row) {
    for (std::uint64_t col = 0; col < 64; ++col) {
      const auto* block = out.data() + ((row - 1) * 2 + col / 32) * 34;
      Check(block[0] == 0x00 && block[1] == 0x3c, "block scale changed");
      const auto code = static_cast<std::int8_t>(block[2 + col % 32]);
      Check(code == Expected(row, col, rank, multiplier),
            "rank " + std::to_string(rank) + " scale " + std::to_string(scale) +
                " row " + std::to_string(row) + " col " + std::to_string(col) +
                ": " + std::to_string(code) + " vs " +
                std::to_string(Expected(row, col, rank, multiplier)));
    }
  }
}

void CheckHandValues() {
  // Row 2, column 5: code (14 + 15) % 41 - 20 = 9; A = 0.25 * (5 - 14) =
  // -2.25, B = -3: 9 + 6.75 = 15.75 -> 16. Column 6: 12 + 6 = 18 -> 18.
  // Row 1, column 1: (7 + 3) % 41 - 20 = -10; -1 * -3.25 = 3.25 -> -6.75 -> -7.
  Matrix m("blk.0.ssm_out.weight", 64, 3);
  const std::vector<qfn::TensorRef> targets{m.ref};
  std::string error;
  const auto bytes = Build(Pair("blk.0.ssm_out.weight", 64, 3, 1));
  const auto adapter = Open(bytes, targets, 1.0F, &error);
  std::vector<std::uint8_t> out(3 * m.ref.RowBytes());
  Check(adapter->MergeRows(*adapter->Find("blk.0.ssm_out.weight"), 0, 3,
                           out.data(), &error),
        error);
  const auto code = [&](int row, int col) {
    return static_cast<std::int8_t>(
        out[(row * 2 + col / 32) * 34 + 2 + col % 32]);
  };
  Check(code(2, 5) == 16 && code(2, 6) == 18 && code(1, 1) == -7,
        "hand-computed merge values differ: " + std::to_string(code(2, 5)) +
            ", " + std::to_string(code(2, 6)) + ", " +
            std::to_string(code(1, 1)));
}

void CheckSubset() {
  Matrix first("blk.0.ssm_out.weight", 64, 3);
  Matrix second("blk.3.attn_output.weight", 64, 3);
  const std::vector<qfn::TensorRef> targets{first.ref, second.ref};
  std::string error;
  const auto bytes = Build(Pair("blk.3.attn_output.weight", 64, 3, 1));
  const auto adapter = Open(bytes, targets, 1.0F, &error);
  Check(adapter != nullptr && adapter->size() == 1,
        "subset rejected: " + error);
  Check(adapter->Find("blk.0.ssm_out.weight") == nullptr &&
            adapter->Find("blk.3.attn_output.weight") != nullptr,
        "subset patches the wrong tensor");
}

void Reject(const AdapterSpec& spec, std::span<const qfn::TensorRef> targets,
            const std::string& expected, float scale = 1.0F) {
  std::string error;
  const auto bytes = Build(spec);
  const auto adapter = Open(bytes, targets, scale, &error);
  Check(adapter == nullptr, "accepted: " + expected);
  Check(error.find(expected) != std::string::npos &&
            error.find("test.gguf") != std::string::npos,
        "error '" + error + "' does not name '" + expected + "'");
}

void CheckRejections() {
  Matrix m("blk.0.ssm_out.weight", 64, 3);
  Matrix bf16("blk.1.ffn_down_shexp.weight", 64, 3);
  bf16.ref.type = GgmlType::kBF16;
  const std::vector<qfn::TensorRef> targets{m.ref, bf16.ref};
  const auto valid = Pair("blk.0.ssm_out.weight", 64, 3, 1);

  auto spec = valid;
  spec.architecture = "qwen3";
  Reject(spec, targets, "general.architecture is 'qwen3', expected 'qwen4exp'");
  spec = valid;
  spec.general_type = "model";
  Reject(spec, targets, "general.type is 'model', expected 'adapter'");
  spec = valid;
  spec.adapter_type = "control_vector";
  Reject(spec, targets, "adapter.type is 'control_vector', expected 'lora'");
  spec = valid;
  spec.has_alpha = false;
  Reject(spec, targets, "adapter.lora.alpha is missing");
  for (float alpha : {0.0F, -1.0F, std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::quiet_NaN()}) {
    spec = valid;
    spec.alpha = alpha;
    Reject(spec, targets, "expected a positive finite value");
  }
  spec = valid;
  spec.tensors[0].name = "blk.0.ssm_out.weight.lora_c";
  Reject(spec, targets, "blk.0.ssm_out.weight.lora_c has an unknown suffix");
  spec = valid;
  spec.tensors.pop_back();
  Reject(spec, targets,
         "blk.0.ssm_out.weight.lora_b is missing, found only "
         "blk.0.ssm_out.weight.lora_a");
  {
    // The GGUF reader already refuses a repeated tensor name.
    spec = valid;
    spec.tensors.push_back(spec.tensors[1]);
    const auto bytes = Build(spec);
    std::string error;
    Check(gufo::core::GgufReader::OpenMemory(bytes.data(), bytes.size(),
                                             &error) == nullptr &&
              error.find("Duplicate") != std::string::npos,
          "duplicate LoRA half accepted");
  }
  // F16 and BF16 factors are accepted (tests/families/qwen/lora_test.cpp).
  spec = valid;
  spec.tensors[0].type = GgmlType::kQ8_0;
  spec.tensors[0].values.resize(32);
  Reject(spec, targets,
         "blk.0.ssm_out.weight.lora_a is Q8_0, expected F32, F16 or BF16");
  spec = valid;
  spec.tensors[0].dims = {32, 1};
  spec.tensors[0].values.resize(32);
  Reject(spec, targets,
         "blk.0.ssm_out.weight.lora_a has shape [32, 1], expected [64, rank]");
  spec = valid;
  spec.tensors[1].dims = {1, 2};
  spec.tensors[1].values.resize(2);
  Reject(spec, targets,
         "blk.0.ssm_out.weight.lora_b has shape [1, 2], expected [rank, 3]");
  spec = Pair("blk.0.ssm_out.weight", 64, 3, 2);
  spec.tensors[1] = valid.tensors[1];
  Reject(spec, targets, "has rank 2 in .lora_a but 1 in .lora_b");
  spec = Pair("blk.0.attn_q.weight", 64, 3, 1);
  Reject(spec, targets, "target blk.0.attn_q.weight is not a patchable");
  spec = Pair("blk.48.attn_output.weight", 64, 3, 1);
  Reject(spec, targets, "target blk.48.attn_output.weight is not a patchable");
  spec = Pair("blk.1.ffn_down_shexp.weight", 64, 3, 1);
  Reject(spec, targets,
         "target blk.1.ffn_down_shexp.weight is BF16 in the model, expected "
         "Q8_0");
  spec = valid;
  spec.tensors[1].values[2] = std::numeric_limits<float>::infinity();
  Reject(spec, targets,
         "blk.0.ssm_out.weight.lora_b has a non-finite value at "
         "element 2");
  spec = valid;
  spec.tensors.clear();
  Reject(spec, targets, "contains no LoRA tensors");
  Reject(valid, targets, "is not finite",
         std::numeric_limits<float>::infinity());
}

void CheckOptions() {
  qfn::LoraSelection selection;
  std::string error;
  Check(qfn::ResolveLoraOptions({}, &selection, &error) &&
            selection.path.empty() && selection.scale == 1.0F,
        "no adapter by default");
  Check(qfn::ResolveLoraOptions({{"--lora", std::string("x.gguf")}}, &selection,
                                &error) &&
            selection.path == "x.gguf" && selection.scale == 1.0F,
        "--lora alone keeps scale 1");
  Check(qfn::ResolveLoraOptions(
            {{"--lora", std::string("x.gguf")}, {"--lora-scale", -0.5}},
            &selection, &error) &&
            selection.scale == -0.5F,
        "--lora-scale not applied");
  Check(!qfn::ResolveLoraOptions({{"--lora-scale", 0.5}}, &selection, &error) &&
            error == "--lora-scale requires --lora",
        "--lora-scale without --lora accepted");
  for (double bad : {std::numeric_limits<double>::infinity(),
                     std::numeric_limits<double>::quiet_NaN(), 1e300}) {
    Check(!qfn::ResolveLoraOptions(
              {{"--lora", std::string("x.gguf")}, {"--lora-scale", bad}},
              &selection, &error),
          "non-finite --lora-scale accepted");
  }
}

void CheckIdentity() {
  Matrix m("blk.0.ssm_out.weight", 64, 3);
  const std::vector<qfn::TensorRef> targets{m.ref};
  const auto bytes = Build(Pair("blk.0.ssm_out.weight", 64, 3, 1));
  std::string error;
  const auto first = Open(bytes, targets, 1.0F, &error, "/a/adapter.gguf");
  const auto moved = Open(bytes, targets, 1.0F, &error, "/b/copy.gguf");
  const auto scaled = Open(bytes, targets, 0.5F, &error);
  auto changed_spec = Pair("blk.0.ssm_out.weight", 64, 3, 1);
  changed_spec.tensors[1].values[0] = 2.5F;
  const auto changed_bytes = Build(changed_spec);
  const auto changed = Open(changed_bytes, targets, 1.0F, &error);
  Check(first->CacheIdentity() == moved->CacheIdentity(),
        "identity depends on the adapter path");
  Check(first->CacheIdentity() == "lora;merge=q8-requant-v1;sha256=" +
                                      first->sha256() + ";scale_f32=0x3f800000",
        "identity format: " + first->CacheIdentity());
  Check(first->sha256() == gufo::crypto::Sha256Hex(bytes),
        "digest is not of the adapter bytes");
  Check(first->CacheIdentity() != scaled->CacheIdentity(),
        "scales alias in the identity");
  Check(first->CacheIdentity() != changed->CacheIdentity(),
        "adapter contents alias in the identity");

  const std::string artifact(64, 'a');
  const auto stock = qfn::QwenFlashNextCompatibilityIdentity(
      artifact, "", false, 4096, 4, 1, "absolute-v1");
  const auto adapted = qfn::QwenFlashNextCompatibilityIdentity(
      artifact, "", false, 4096, 4, 1, "absolute-v1",
      gufo::ext::kGgufIdentityScheme, gufo::ext::kGgufIdentityScheme,
      first->CacheIdentity());
  const std::string stock_text(stock.begin(), stock.end());
  const std::string adapted_text(adapted.begin(), adapted.end());
  Check(stock_text.find("\nadapters=none\n") != std::string::npos,
        "stock identity lost adapters=none");
  Check(adapted_text.find("\nadapters=" + first->CacheIdentity() + "\n") !=
                std::string::npos &&
            adapted_text.find("adapters=none") == std::string::npos,
        "adapter identity line missing");
}

}  // namespace

int main() {
  CheckEncoder();
  CheckHandValues();
  CheckMerge(1, 1.0F, 1.0F);
  CheckMerge(2, 1.0F, 1.0F);
  CheckMerge(1, 1.0F, -1.0F);
  CheckMerge(1, 1.0F, 0.0F);
  CheckMerge(2, 4.0F, 0.5F);
  CheckSubset();
  CheckRejections();
  CheckOptions();
  CheckIdentity();
  std::cout << "Flash-Next LoRA checks passed.\n";
}
