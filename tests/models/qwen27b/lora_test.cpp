// Qwen3.8 27B LoRA targets: which bound tensors an adapter may patch, their
// shapes from the GGUF dimensions, and the CPU path's refusal. The merge
// arithmetic is tested with the shared library (tests/families/qwen).
#include "src/models/qwen/lora.hpp"

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

  explicit Model(bool transposed_ffn_down = false,
                 bool shared_ssm_out = false) {
    const auto q8 = [](std::string name, std::uint64_t cols) {
      return Tensor{std::move(name),
                    {cols, kHidden},
                    GgmlType::kQ8_0,
                    Zeros(GgmlType::kQ8_0, cols * kHidden)};
    };
    std::vector<Tensor> tensors{
        q8("blk.0.ssm_out.weight", kInner),
        q8("blk.0.ffn_down.weight", kIntermediate),
        q8("blk.1.attn_out.weight", kAttention),
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
    c.head_dim = kAttention / 2;
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

std::unique_ptr<qwen::lora::LoraAdapter> Open(const Model& model,
                                              const std::vector<Tensor>& pairs,
                                              std::string* error) {
  const auto targets = qwen::LoraTargets(*model.reader, model.weights, error);
  Check(targets.has_value(), "targets rejected: " + *error);
  return qwen::lora::LoraAdapter::Open(
      lora_fixture::Read(lora_fixture::Adapter("qwen35", 3.0F, pairs)),
      "test.gguf", 1.0F, "qwen35", *targets, qwen::kLoraExpectedTargets, error);
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
  CheckCpuRefusal();
  std::cout << "Qwen 27B LoRA checks passed.\n";
}
