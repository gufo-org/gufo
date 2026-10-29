#include "src/models/nemotron35_bolt/config.hpp"

#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace nb = gufo::models::nemotron35_bolt;
using gufo::core::GgufValueType;

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

/// Metadata-only GGUF image; Config reads no tensors.
class MetadataGguf {
public:
  void String(std::string_view key, std::string_view value) {
    Key(key, GgufValueType::kString);
    Text(value);
  }
  void U32(std::string_view key, std::uint32_t value) {
    Key(key, GgufValueType::kUint32);
    Pod(value);
  }
  void F32(std::string_view key, float value) {
    Key(key, GgufValueType::kFloat32);
    Pod(value);
  }
  void Bool(std::string_view key, bool value) {
    Key(key, GgufValueType::kBool);
    Pod(static_cast<std::uint8_t>(value));
  }
  void I32Array(std::string_view key, const std::vector<std::int32_t>& v) {
    Key(key, GgufValueType::kArray);
    Pod(static_cast<std::uint32_t>(GgufValueType::kInt32));
    Pod(static_cast<std::uint64_t>(v.size()));
    for (auto x : v)
      Pod(x);
  }
  void BoolArray(std::string_view key, const std::vector<bool>& v) {
    Key(key, GgufValueType::kArray);
    Pod(static_cast<std::uint32_t>(GgufValueType::kBool));
    Pod(static_cast<std::uint64_t>(v.size()));
    for (bool x : v)
      Pod(static_cast<std::uint8_t>(x));
  }
  std::vector<std::uint8_t> Build() const {
    std::vector<std::uint8_t> out = {'G', 'G', 'U', 'F'};
    auto put = [&](auto v) {
      const auto* p = reinterpret_cast<const std::uint8_t*>(&v);
      out.insert(out.end(), p, p + sizeof(v));
    };
    put(static_cast<std::uint32_t>(3));
    put(static_cast<std::uint64_t>(0));
    put(count_);
    out.insert(out.end(), body_.begin(), body_.end());
    out.resize((out.size() + 31) / 32 * 32, 0);
    return out;
  }

private:
  template<class T>
  void Pod(T v) {
    const auto* p = reinterpret_cast<const std::uint8_t*>(&v);
    body_.insert(body_.end(), p, p + sizeof(v));
  }
  void Text(std::string_view s) {
    Pod(static_cast<std::uint64_t>(s.size()));
    body_.insert(body_.end(), s.begin(), s.end());
  }
  void Key(std::string_view key, GgufValueType type) {
    Text(key);
    Pod(static_cast<std::uint32_t>(type));
    ++count_;
  }
  std::vector<std::uint8_t> body_;
  std::uint64_t count_{0};
};

// The pinned artifact's block order: M = Mamba-2, E = MoE, * = attention.
constexpr std::string_view kLayout =
    "MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*"
    "EMEMEMEM*EMEMEMEME";

struct Profile {
  std::uint32_t hidden = 4096;
  std::uint32_t gating = 2;
  std::size_t kv_entries = kLayout.size();
  bool kv_on_first_moe = false;
};

std::optional<nb::Config> Parse(const Profile& p, std::string* error) {
  MetadataGguf g;
  const std::string a = "nemotron_h_moe.";
  g.String("general.architecture", "nemotron_h_moe");
  g.U32(a + "block_count", static_cast<std::uint32_t>(kLayout.size()));
  g.U32(a + "context_length", 262144);
  g.U32(a + "embedding_length", p.hidden);
  g.U32(a + "vocab_size", 131072);
  g.U32(a + "attention.head_count", 32);
  g.F32(a + "attention.layer_norm_rms_epsilon", 1e-5F);
  g.F32(a + "attention.layer_norm_epsilon", 1e-5F);
  g.U32(a + "attention.key_length", 128);
  g.U32(a + "attention.value_length", 128);
  g.U32(a + "expert_count", 496);
  g.U32(a + "expert_group_count", 1);
  g.U32(a + "expert_group_used_count", 1);
  g.U32(a + "expert_gating_func", p.gating);
  g.U32(a + "expert_shared_feed_forward_length", 5376);
  g.U32(a + "expert_shared_count", 1);
  g.Bool(a + "expert_weights_norm", true);
  g.F32(a + "expert_weights_scale", 5.0F);
  g.U32(a + "moe_latent_size", 1024);
  g.U32(a + "rope.dimension_count", 128);
  g.U32(a + "ssm.conv_kernel", 4);
  g.U32(a + "ssm.state_size", 128);
  g.U32(a + "ssm.group_count", 8);
  g.U32(a + "ssm.inner_size", 8192);
  g.U32(a + "ssm.time_step_rank", 128);

  std::vector<std::int32_t> kv;
  std::vector<std::int32_t> ff;
  std::vector<std::int32_t> used;
  bool first_moe = true;
  for (std::size_t i = 0; i < kLayout.size(); ++i) {
    const bool moe = kLayout[i] == 'E';
    // One narrower block, as in the pinned file, keeps widths per block.
    const std::int32_t width = i == 3 ? 1536 : 2048;
    kv.push_back(
        kLayout[i] == '*' || (moe && first_moe && p.kv_on_first_moe) ? 2 : 0);
    ff.push_back(moe ? width : 0);
    used.push_back(moe ? 9 : 0);
    first_moe = first_moe && !moe;
  }
  kv.resize(p.kv_entries);
  g.I32Array(a + "attention.head_count_kv", kv);
  g.I32Array(a + "feed_forward_length", ff);
  g.I32Array(a + "expert_feed_forward_length", ff);
  g.I32Array(a + "expert_used_count", used);

  const auto bytes = g.Build();
  auto reader =
      gufo::core::GgufReader::OpenMemory(bytes.data(), bytes.size(), error);
  Require(reader != nullptr, "synthetic metadata GGUF must load");
  return nb::Config::FromGguf(*reader, error);
}

void CheckPinnedProfile() {
  std::string error;
  const auto c = Parse({}, &error);
  Require(c.has_value(), "pinned profile rejected: " + error);
  std::size_t counts[3] = {};
  for (auto b : c->blocks)
    ++counts[static_cast<int>(b)];
  Require(counts[0] == 40 && counts[1] == 8 && counts[2] == 40,
          "blocks must split into 40 Mamba, 8 attention and 40 MoE");
  Require(c->blocks[0] == nb::BlockType::kMamba &&
              c->blocks[1] == nb::BlockType::kMoe &&
              c->blocks[7] == nb::BlockType::kAttention,
          "block classification follows head_count_kv and ff widths");
  Require(c->expert_ff[3] == 1536 && c->expert_ff[1] == 2048 &&
              c->expert_ff[0] == 0 && c->experts_used[1] == 9 &&
              c->experts_used[7] == 0,
          "expert widths and top-k are kept per block");
  Require(c->num_heads == 32 && c->num_kv_heads == 2,
          "attention geometry comes from attention blocks");
  Require(c->SsmInProjection() == 18560 && c->SsmConvChannels() == 10240 &&
              c->SsmHeadDim() == 64,
          "Mamba-2 projection layout is z|x|B|C|dt");
  Require(c->expert_weights_norm && c->expert_weights_scale == 5.0F,
          "routing weight normalization and scale");
}

void CheckRejections() {
  std::string error;
  Profile p;
  p.gating = 1;
  Require(!Parse(p, &error), "softmax gating accepted");
  p = {};
  p.kv_entries = kLayout.size() - 1;
  Require(!Parse(p, &error), "short per-block array accepted");
  p = {};
  p.hidden = 4224;
  Require(!Parse(p, &error), "unsupported hidden width accepted");
  p = {};
  p.kv_on_first_moe = true;
  Require(!Parse(p, &error), "MoE block with attention heads accepted");
}

struct DrafterProfile {
  bool causal = false;
  std::uint32_t selector_rank = 0;
  std::int32_t last_target_layer = 86;
  bool mask_token = true;
  std::uint32_t block_size = 8;
};

std::optional<nb::DrafterConfig> ParseDrafter(const DrafterProfile& p,
                                              std::string* error) {
  std::string target_error;
  const auto target = Parse({}, &target_error);
  Require(target.has_value(), "pinned profile rejected: " + target_error);
  MetadataGguf g;
  g.String("general.architecture", "dflash");
  g.U32("dflash.block_count", 5);
  g.U32("dflash.context_length", 262144);
  g.U32("dflash.embedding_length", 4096);
  g.U32("dflash.feed_forward_length", 12288);
  g.U32("dflash.attention.head_count", 32);
  g.U32("dflash.attention.head_count_kv", 2);
  g.F32("dflash.rope.freq_base", 10000.0F);
  g.F32("dflash.attention.layer_norm_rms_epsilon", 1e-6F);
  g.U32("dflash.attention.key_length", 128);
  g.U32("dflash.attention.value_length", 128);
  g.U32("dflash.block_size", p.block_size);
  g.U32("dflash.conv_kernel_size", 2);
  g.U32("dflash.conv_group_size", 16);
  g.U32("dflash.selector_rank", p.selector_rank);
  g.U32("dflash.selector_top_k", 0);
  g.I32Array("dflash.target_layers", {2, 19, 36, 52, 69, p.last_target_layer});
  g.U32("dflash.attention.sliding_window", 1024);
  g.BoolArray("dflash.attention.sliding_window_pattern",
              {true, true, true, true, true});
  g.Bool("dflash.attention.causal", p.causal);
  g.U32("lilicorr.hidden_size", 1024);
  g.U32("lilicorr.candidate_topk", 8);
  g.U32("lilicorr.num_heads", 8);
  g.U32("lilicorr.num_layers", 2);
  g.U32("lilicorr.factor_dim", 1024);
  g.U32("lilicorr.mlp_hidden_size", 2048);
  g.F32("lilicorr.vector_eps", 1e-4F);
  g.F32("lilicorr.logit_scale", 8.0F);
  if (p.mask_token)
    g.U32("tokenizer.ggml.mask_token_id", 990);
  const auto bytes = g.Build();
  auto reader =
      gufo::core::GgufReader::OpenMemory(bytes.data(), bytes.size(), error);
  Require(reader != nullptr, "synthetic drafter GGUF must load");
  return nb::DrafterConfig::FromGguf(*reader, *target, error);
}

void CheckDrafter() {
  std::string error;
  const auto c = ParseDrafter({}, &error);
  Require(c.has_value(), "pinned drafter rejected: " + error);
  Require(c->target_layers == std::vector<std::uint32_t>{2, 19, 36, 52, 69, 86},
          "drafter target layers");
  Require(c->FeatureSize() == 6 * 4096 && c->ConvProjection() == 1024 &&
              c->Slots() == 7 && c->mask_token == 990 &&
              c->sliding_window == 1024,
          "drafter geometry");
  DrafterProfile p;
  p.causal = true;
  Require(!ParseDrafter(p, &error), "causal drafter accepted");
  p = {};
  p.selector_rank = 64;
  Require(!ParseDrafter(p, &error), "DFlash2 selector accepted");
  p = {};
  p.last_target_layer = 88;
  Require(!ParseDrafter(p, &error), "target layer past the trunk accepted");
  p = {};
  p.mask_token = false;
  Require(!ParseDrafter(p, &error), "drafter without a mask token accepted");
  p = {};
  p.block_size = 16;
  Require(!ParseDrafter(p, &error), "other block size accepted");
}

}  // namespace

int main() {
  try {
    CheckPinnedProfile();
    CheckRejections();
    CheckDrafter();
  } catch (const std::exception& e) {
    std::cerr << "nemotron35_bolt config_test: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
