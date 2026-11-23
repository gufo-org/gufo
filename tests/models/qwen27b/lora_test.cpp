// Qwen3.8 27B LoRA targets: which bound tensors an adapter may patch, their
// shapes from the GGUF dimensions, the choice between merging and the side
// path, the side path's factor layout and limits, the cache identity and the
// CPU path's refusal. The merge arithmetic is tested with the shared library
// (tests/families/qwen).
#include "src/models/qwen/lora.hpp"

#include <bit>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "gufo/ext/experimental/ggml_dequant.hpp"
#include "tests/families/qwen/lora_fixture.hpp"

namespace qwen = gufo::models::qwen;
using gufo::core::GgmlType;
using lora_fixture::Tensor;

namespace {

void Check(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

constexpr std::uint64_t kHidden = 32;
constexpr std::uint64_t kInner = 64;          // ssm_out columns
constexpr std::uint64_t kAttention = 64;      // attn_output columns
constexpr std::uint64_t kIntermediate = 256;  // ffn_down columns

std::vector<std::uint8_t> Zeros(GgmlType type, std::uint64_t elements) {
  return std::vector<std::uint8_t>(
      gufo::quant::EncodedSizeBytes(type, elements));
}

/// A two-layer trunk: blk.0 is a GDN layer, blk.1 full attention bound
/// through the `attn_out.weight` alias; blk.1's ffn_down is Q5_K, as in
/// UD-Q4_K_XL. The MTP-style blk.2 and the embeddings are never targets.
struct Model {
  std::vector<std::uint8_t> file;
  std::unique_ptr<gufo::core::GgufReader> reader;
  gufo::models::QwenModelWeights weights;

  /// `attention_cols` and `attention_type` shape blk.1's attn_out.weight.
  /// `shared_ffn_down` makes blk.1's ffn_down a Q8_0 tensor that shares
  /// blk.0's bytes.
  explicit Model(bool transposed_ffn_down = false, bool shared_ssm_out = false,
                 std::uint64_t attention_cols = kAttention,
                 GgmlType attention_type = GgmlType::kQ8_0,
                 bool shared_ffn_down = false) {
    const auto q8 = [](std::string name, std::uint64_t cols) {
      return Tensor{std::move(name),
                    {cols, kHidden},
                    GgmlType::kQ8_0,
                    Zeros(GgmlType::kQ8_0, cols * kHidden)};
    };
    std::vector<Tensor> tensors{
        q8("blk.0.ssm_out.weight", kInner),
        q8("blk.0.ffn_down.weight", kIntermediate),
        {"blk.1.attn_out.weight",
         {attention_cols, kHidden},
         attention_type,
         Zeros(attention_type, attention_cols * kHidden)},
        {"blk.1.ffn_down.weight",
         {kIntermediate, kHidden},
         GgmlType::kQ5_K,
         Zeros(GgmlType::kQ5_K, kIntermediate * kHidden)},
        q8("blk.2.attn_output.weight", kAttention),
        q8("output.weight", kHidden),
        q8("token_embd.weight", kHidden),
    };
    if (transposed_ffn_down) {
      tensors[1].dims = {kHidden, kIntermediate};
    }
    if (shared_ffn_down) {
      Tensor twin = tensors[1];
      twin.name = "blk.1.ffn_down.weight";
      twin.shares_previous = true;
      tensors.erase(tensors.begin() + 3);
      tensors.insert(tensors.begin() + 2, std::move(twin));
    }
    if (shared_ssm_out) {
      // A second name for blk.0.ssm_out.weight's bytes.
      Tensor twin = tensors[0];
      twin.name = "blk.0.ssm_twin.weight";
      twin.shares_previous = true;
      tensors.insert(tensors.begin() + 1, std::move(twin));
    }
    gufo::test_fixture::GgufBuilder builder;
    builder.AddMetadataString("general.architecture", "qwen35");
    file = lora_fixture::Build(std::move(builder), tensors);
    reader = lora_fixture::Read(file);

    auto& c = weights.config;
    c.num_layers = 2;
    c.hidden_size = kHidden;
    c.ssm_inner_size = kInner;
    c.num_attention_heads = 2;
    c.head_dim = static_cast<std::uint32_t>(attention_cols / 2);
    c.intermediate_size = kIntermediate;
    const auto bind = [&](const char* name) {
      const auto* info = reader->FindTensor(name);
      Check(info != nullptr, name);
      return gufo::models::QwenTensorRef{.data = info->data,
                                         .type = info->type,
                                         .num_elements = info->ElementCount()};
    };
    weights.layers.resize(2);
    weights.layers[0].ssm_out = bind("blk.0.ssm_out.weight");
    weights.layers[0].ffn_down = bind("blk.0.ffn_down.weight");
    weights.layers[1].is_full_attention = true;
    weights.layers[1].attn_output = bind("blk.1.attn_out.weight");
    weights.layers[1].ffn_down = bind("blk.1.ffn_down.weight");
    weights.output = bind("output.weight");
    weights.token_embd = bind("token_embd.weight");
  }
};

std::vector<Tensor> Pair(const std::string& base, std::uint64_t cols,
                         std::uint64_t rows, std::uint32_t rank = 3) {
  return {{base + ".lora_a",
           {cols, rank},
           GgmlType::kF16,
           std::vector<std::uint8_t>(cols * rank * 2)},
          {base + ".lora_b",
           {rank, rows},
           GgmlType::kF16,
           std::vector<std::uint8_t>(rows * rank * 2)}};
}

/// Factor `index` of a pair: exact in F16 and BF16, and different for A
/// and B.
float FactorValue(std::size_t index, bool b) {
  return (static_cast<float>(index % 7) - 3.0F) * (b ? 0.5F : 0.25F);
}

/// `Pair` with values `FactorValue` stored as `type`.
std::vector<Tensor> ValuedPair(const std::string& base, std::uint64_t cols,
                               std::uint64_t rows, GgmlType type) {
  constexpr std::uint32_t kRank = 3;
  const auto encode = [type](std::uint64_t count, bool b) {
    std::vector<std::uint8_t> bytes;
    for (std::uint64_t i = 0; i < count; ++i) {
      const float value = FactorValue(i, b);
      std::uint32_t word = std::bit_cast<std::uint32_t>(value);
      std::size_t size = 4;
      if (type == GgmlType::kF16) {
        word = qwen::lora::FloatToHalf(value);
        size = 2;
      } else if (type == GgmlType::kBF16) {
        word >>= 16;  // Exact: the values have short mantissas.
        size = 2;
      }
      for (std::size_t k = 0; k < size; ++k)
        bytes.push_back(static_cast<std::uint8_t>(word >> (8 * k)));
    }
    return bytes;
  };
  return {{base + ".lora_a", {cols, kRank}, type, encode(cols * kRank, false)},
          {base + ".lora_b", {kRank, rows}, type, encode(rows * kRank, true)}};
}

std::vector<Tensor> Join(std::vector<std::vector<Tensor>> pairs) {
  std::vector<Tensor> all;
  for (auto& pair : pairs)
    for (auto& t : pair)
      all.push_back(std::move(t));
  return all;
}

std::unique_ptr<qwen::lora::LoraAdapter> Open(
    const Model& model, const std::vector<Tensor>& pairs, std::string* error,
    qwen::lora::LoraTargetFormats formats =
        qwen::lora::LoraTargetFormats::kQ8_0,
    float scale = 1.0F) {
  const auto targets = qwen::LoraTargets(*model.reader, model.weights, error);
  Check(targets.has_value(), "targets rejected: " + *error);
  return qwen::lora::LoraAdapter::Open(
      lora_fixture::Read(lora_fixture::Adapter("qwen35", 3.0F, pairs)),
      "test.gguf", scale, "qwen35", *targets, qwen::kLoraExpectedTargets, error,
      formats);
}

void Reject(const Model& model, const std::vector<Tensor>& pairs,
            const std::string& expected) {
  std::string error;
  Check(Open(model, pairs, &error) == nullptr &&
            error.find(expected) != std::string::npos,
        "error '" + error + "' does not name '" + expected + "'");
}

void CheckTargets() {
  const Model model;
  std::string error;
  const auto targets = qwen::LoraTargets(*model.reader, model.weights, &error);
  Check(targets.has_value(), "targets rejected: " + error);
  struct Expected {
    std::string name;
    std::uint64_t cols;
    GgmlType type;
  };
  const std::vector<Expected> expected{
      {"blk.0.ssm_out.weight", kInner, GgmlType::kQ8_0},
      {"blk.0.ffn_down.weight", kIntermediate, GgmlType::kQ8_0},
      {"blk.1.attn_out.weight", kAttention, GgmlType::kQ8_0},
      {"blk.1.ffn_down.weight", kIntermediate, GgmlType::kQ5_K},
  };
  Check(targets->size() == expected.size(), "target count");
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const auto& t = (*targets)[i];
    Check(t.name == expected[i].name && t.cols == expected[i].cols &&
              t.rows == kHidden && t.type == expected[i].type &&
              t.data == model.reader->FindTensor(t.name)->data,
          "target " + expected[i].name);
  }

  const Model transposed(true);
  Check(!qwen::LoraTargets(*transposed.reader, transposed.weights, &error) &&
            error ==
                "LoRA target blk.0.ffn_down.weight has shape [32, 256], "
                "expected [256, 32]",
        "transposed model tensor accepted: " + error);
}

void CheckAdapters() {
  const Model model;
  std::string error;
  auto all = Pair("blk.0.ssm_out.weight", kInner, kHidden);
  for (auto& t : Pair("blk.0.ffn_down.weight", kIntermediate, kHidden))
    all.push_back(t);
  for (auto& t : Pair("blk.1.attn_out.weight", kAttention, kHidden))
    all.push_back(t);
  const auto adapter = Open(model, all, &error);
  Check(adapter != nullptr && adapter->size() == 3,
        "Q8_0 targets rejected: " + error);

  // A subset of the targets is a valid adapter.
  const auto subset =
      Open(model, Pair("blk.1.attn_out.weight", kAttention, kHidden), &error);
  Check(subset != nullptr && subset->size() == 1 &&
            subset->Find("blk.1.attn_out.weight") != nullptr,
        "subset rejected: " + error);

  // Swapped axes with the same element count.
  Reject(model, Pair("blk.0.ffn_down.weight", kHidden, kIntermediate),
         "blk.0.ffn_down.weight.lora_a has shape [32, 3], expected [256, "
         "rank]");
  auto half = Pair("blk.0.ssm_out.weight", kInner, kHidden);
  half.pop_back();
  Reject(model, half,
         "blk.0.ssm_out.weight.lora_b is missing, found only "
         "blk.0.ssm_out.weight.lora_a");
  for (const auto& name :
       {"output.weight", "token_embd.weight", "blk.2.attn_output.weight",
        "blk.64.attn_output.weight", "blk.64.nextn.eh_proj.weight",
        "blk.0.attn_qkv.weight"}) {
    Reject(model, Pair(name, kHidden, kHidden),
           "target " + std::string(name) +
               " is not a patchable trunk tensor, expected "
               "blk.N.ssm_out.weight, blk.N.attn_output.weight or "
               "blk.N.ffn_down.weight of a trunk layer");
  }
  Reject(model, Pair("blk.1.ffn_down.weight", kIntermediate, kHidden),
         "target blk.1.ffn_down.weight is Q5_K in the model, expected Q8_0");
}

// The loader overwrites a target in place, so a target whose bytes another
// tensor shares is refused; other targets of the same model still merge.
void CheckSharedBytes() {
  const Model model(false, true);
  std::string error;
  const auto shared =
      Open(model, Pair("blk.0.ssm_out.weight", kInner, kHidden), &error);
  Check(shared != nullptr, "adapter rejected: " + error);
  Check(!qwen::LoraPatchesOwnTheirBytes(*model.reader, *shared, &error) &&
            error ==
                "LoRA target blk.0.ssm_out.weight shares its bytes with "
                "blk.0.ssm_twin.weight in the model",
        "shared target accepted: " + error);
  const auto own = Open(
      model, Pair("blk.0.ffn_down.weight", kIntermediate, kHidden), &error);
  Check(own != nullptr &&
            qwen::LoraPatchesOwnTheirBytes(*model.reader, *own, &error),
        "unshared target rejected: " + error);
}

constexpr auto kAny = qwen::lora::LoraTargetFormats::kAny;

// An adapter whose targets are all Q8_0 merges, even on a mixed trunk; one
// with any other target takes the side path for every target, its Q8_0
// ones included.
void CheckModes() {
  const Model model;
  std::string error;
  const auto q8 =
      Open(model,
           Join({Pair("blk.0.ssm_out.weight", kInner, kHidden),
                 Pair("blk.1.attn_out.weight", kAttention, kHidden)}),
           &error, kAny);
  Check(q8 != nullptr && qwen::LoraMergesInPlace(*q8),
        "Q8_0 targets do not merge: " + error);
  const auto mixed =
      Open(model,
           Join({Pair("blk.0.ssm_out.weight", kInner, kHidden),
                 Pair("blk.1.ffn_down.weight", kIntermediate, kHidden)}),
           &error, kAny);
  Check(mixed != nullptr && !qwen::LoraMergesInPlace(*mixed),
        "a Q5_K target merges: " + error);
  const auto path =
      qwen::PlanLoraSidePath(*model.reader, *mixed, model.weights, &error);
  Check(path.has_value() && path->slots.size() == 2,
        "mixed adapter has no side path: " + error);
  // Patches come in name order: the Q8_0 blk.0.ssm_out first.
  Check(path->slots[0].layer == 0 && !path->slots[0].ffn_down &&
            path->slots[1].layer == 1 && path->slots[1].ffn_down,
        "side path skips the Q8_0 target");
}

// Factors of each storage type are widened into one buffer, A and B
// 64-byte aligned, beside the projection they patch.
void CheckSidePathLayout() {
  const Model model;
  std::string error;
  const auto adapter = Open(
      model,
      Join({ValuedPair("blk.0.ssm_out.weight", kInner, kHidden, GgmlType::kF32),
            ValuedPair("blk.1.attn_out.weight", kAttention, kHidden,
                       GgmlType::kF16),
            ValuedPair("blk.1.ffn_down.weight", kIntermediate, kHidden,
                       GgmlType::kBF16)}),
      &error, kAny, 0.5F);
  Check(adapter != nullptr, "adapter rejected: " + error);
  const auto path =
      qwen::PlanLoraSidePath(*model.reader, *adapter, model.weights, &error);
  Check(path.has_value() && path->slots.size() == 3, "no side path: " + error);
  struct Expected {
    std::uint32_t layer;
    bool ffn_down;
    std::uint64_t in;
  };
  const Expected expected[] = {
      {0, false, kInner}, {1, false, kAttention}, {1, true, kIntermediate}};
  for (std::size_t i = 0; i < 3; ++i) {
    const auto& slot = path->slots[i];
    const auto& e = expected[i];
    const std::string label = "slot " + std::to_string(i);
    Check(slot.layer == e.layer && slot.ffn_down == e.ffn_down &&
              slot.in == e.in && slot.out == kHidden && slot.rank == 3 &&
              slot.multiplier == 0.5F,
          label + " descriptor");
    Check(slot.a % 16 == 0 && slot.b % 16 == 0 &&
              slot.b >= slot.a + 3 * slot.in &&
              slot.b + 3 * kHidden <= path->factors.size(),
          label + " offsets");
    for (std::size_t k = 0; k < 3 * slot.in; ++k)
      Check(path->factors[slot.a + k] == FactorValue(k, false),
            label + " A value " + std::to_string(k));
    for (std::size_t k = 0; k < 3 * kHidden; ++k)
      Check(path->factors[slot.b + k] == FactorValue(k, true),
            label + " B value " + std::to_string(k));
  }
}

// Projections of two layers may share bytes; the side path places a patch
// by its tensor name, so patching the later one updates only that layer.
void CheckSidePathSharedBytes() {
  const Model model(false, false, kAttention, GgmlType::kQ8_0, true);
  Check(model.weights.layers[0].ffn_down.data ==
            model.weights.layers[1].ffn_down.data,
        "fixture: ffn_down bytes are not shared");
  std::string error;
  const auto later =
      Open(model, Pair("blk.1.ffn_down.weight", kIntermediate, kHidden), &error,
           qwen::lora::LoraTargetFormats::kAny);
  Check(later != nullptr, "adapter rejected: " + error);
  const auto path =
      qwen::PlanLoraSidePath(*model.reader, *later, model.weights, &error);
  Check(path.has_value() && path->slots.size() == 1 &&
            path->slots[0].layer == 1 && path->slots[0].ffn_down,
        "patch of blk.1.ffn_down placed elsewhere: " + error);
}

void CheckSidePathLimits() {
  const Model model;
  std::string error;
  const auto wide =
      Open(model, Pair("blk.1.ffn_down.weight", kIntermediate, kHidden, 65),
           &error, kAny);
  Check(wide != nullptr, "rank 65 adapter rejected: " + error);
  Check(!qwen::PlanLoraSidePath(*model.reader, *wide, model.weights, &error) &&
            error ==
                "LoRA target blk.1.ffn_down.weight has rank 65, the side "
                "path accepts 64 at most",
        "rank 65 accepted: " + error);
  const auto widest =
      Open(model, Pair("blk.1.ffn_down.weight", kIntermediate, kHidden, 64),
           &error, kAny);
  Check(widest != nullptr && qwen::PlanLoraSidePath(*model.reader, *widest,
                                                    model.weights, &error)
                                 .has_value(),
        "rank 64 rejected: " + error);

  // Scale and alpha are finite, their product is not.
  const auto overflow =
      Open(model, Pair("blk.1.ffn_down.weight", kIntermediate, kHidden, 1),
           &error, kAny, 3.0e38F);
  Check(overflow != nullptr, "large-scale adapter rejected: " + error);
  Check(!qwen::PlanLoraSidePath(*model.reader, *overflow, model.weights,
                                &error) &&
            error ==
                "LoRA target blk.1.ffn_down.weight has a non-finite "
                "multiplier (scale * alpha / rank)",
        "non-finite multiplier accepted: " + error);

  // An F32 attention output 6 columns wide.
  const Model narrow(false, false, 6, GgmlType::kF32);
  const auto odd =
      Open(narrow, Pair("blk.1.attn_out.weight", 6, kHidden), &error, kAny);
  Check(odd != nullptr, "6-column adapter rejected: " + error);
  Check(!qwen::PlanLoraSidePath(*narrow.reader, *odd, narrow.weights, &error) &&
            error ==
                "LoRA target blk.1.attn_out.weight has 6 columns, the side "
                "path needs a multiple of 4",
        "6 columns accepted: " + error);
}

// The identity names the algorithm that applies the update and the scale's
// float32 bits; scale 0 and negative scales are ordinary scales.
void CheckIdentity() {
  const Model model;
  std::string error;
  const auto q8_pairs = Pair("blk.0.ssm_out.weight", kInner, kHidden);
  const auto side_pairs = Pair("blk.1.ffn_down.weight", kIntermediate, kHidden);
  const auto expect = [&](const std::vector<Tensor>& pairs, float scale,
                          const std::string& algorithm, const std::string& bits,
                          float multiplier) {
    const auto adapter = Open(model, pairs, &error, kAny, scale);
    Check(adapter != nullptr, "adapter rejected: " + error);
    const std::string identity = qwen::LoraAdapterIdentity(*adapter);
    Check(identity == "lora;merge=" + algorithm + ";sha256=" +
                          adapter->sha256() + ";scale_f32=0x" + bits,
          "identity " + identity);
    Check(adapter->Multiplier(adapter->patches().begin()->second) == multiplier,
          "multiplier for " + identity);
    return identity;
  };
  expect(q8_pairs, 1.0F, "q8-requant-v1", "3f800000", 1.0F);
  expect(side_pairs, 1.0F, "low-rank-fp32-v1", "3f800000", 1.0F);
  expect(side_pairs, 0.5F, "low-rank-fp32-v1", "3f000000", 0.5F);
  expect(side_pairs, 0.0F, "low-rank-fp32-v1", "00000000", 0.0F);
  expect(side_pairs, -0.5F, "low-rank-fp32-v1", "bf000000", -0.5F);
  const auto zero = Open(model, side_pairs, &error, kAny, 0.0F);
  const auto path =
      qwen::PlanLoraSidePath(*model.reader, *zero, model.weights, &error);
  Check(path.has_value() && path->slots[0].multiplier == 0.0F,
        "scale 0 has no side path: " + error);
}

void CheckCpuRefusal() {
  std::string error;
  Check(qwen::ValidateCpuLoraOptions({}, &error), "no adapter refused");
  Check(!qwen::ValidateCpuLoraOptions({{"--lora", std::string("a.gguf")}},
                                      &error) &&
            error.starts_with("--lora requires the ROCm backend"),
        "CPU path accepted an adapter: " + error);
  Check(!qwen::ValidateCpuLoraOptions({{"--lora-scale", 0.5}}, &error) &&
            error == "--lora-scale requires --lora",
        "CPU path accepted --lora-scale alone: " + error);
}

}  // namespace

int main() {
  CheckTargets();
  CheckAdapters();
  CheckSharedBytes();
  CheckModes();
  CheckSidePathLayout();
  CheckSidePathSharedBytes();
  CheckSidePathLimits();
  CheckIdentity();
  CheckCpuRefusal();
  std::cout << "Qwen 27B LoRA checks passed.\n";
}
