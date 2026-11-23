#include "src/models/inkling2/config.hpp"

#include <cmath>
#include <optional>
#include <string_view>
#include <variant>
#include <vector>

namespace gufo::models::inkling2 {

std::uint32_t Inkling2Config::PhysOfVisit(std::uint32_t v) const {
  const auto core_visits = n_passes * CoreLen();
  if (v < core_start) {
    return v;
  }
  if (v < core_start + core_visits) {
    return core_start + (v - core_start) % CoreLen();
  }
  return v - (n_passes - 1) * CoreLen();
}

namespace {

/// Collects the first mismatch while a file is checked field by field.
class Checker {
public:
  Checker(const core::GgufReader& reader, std::string prefix)
      : reader_(reader), prefix_(std::move(prefix)) {}

  [[nodiscard]] bool ok() const { return why_.empty(); }
  [[nodiscard]] const std::string& why() const { return why_; }

  void Fail(std::string message) {
    if (why_.empty()) {
      why_ = std::move(message);
    }
  }

  std::string Key(std::string_view key) const {
    return prefix_ + "." + std::string(key);
  }

  /// `<arch>.<key>` must be the unsigned integer `expected`.
  void Expect(std::string_view key, std::uint32_t expected) {
    const auto name = Key(key);
    const auto value = reader_.GetMetadataUint32(name);
    if (!value) {
      Fail(name + " is missing");
    } else if (*value != expected) {
      Fail(name + " is " + std::to_string(*value) + ", expected " +
           std::to_string(expected));
    }
  }

  /// `<arch>.<key>` must be the float `expected` (float32 storage tolerance).
  void ExpectFloat(std::string_view key, float expected) {
    const auto name = Key(key);
    const auto value = reader_.GetMetadataFloat32(name);
    if (!value) {
      Fail(name + " is missing");
    } else if (std::fabs(*value - expected) > 1e-5F * std::fabs(expected)) {
      Fail(name + " is " + std::to_string(*value) + ", expected " +
           std::to_string(expected));
    }
  }

  /// An integer array as unsigned numbers.
  std::vector<std::uint64_t> Array(std::string_view key) {
    const auto name = Key(key);
    const auto* meta = reader_.FindMetadata(name);
    if (meta == nullptr) {
      Fail(name + " is missing");
      return {};
    }
    if (const auto* values =
            std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
      return *values;
    }
    // Writers may store the counts as signed integers.
    if (const auto* values =
            std::get_if<std::vector<std::int64_t>>(&meta->value)) {
      std::vector<std::uint64_t> out;
      for (const auto value : *values) {
        out.push_back(
            static_cast<std::uint64_t>(value));  // negatives never match
      }
      return out;
    }
    Fail(name + " is not an integer array");
    return {};
  }

  /// `<arch>.<key>` is an array of `count` copies of `expected`.
  void ExpectConstantArray(std::string_view key, std::size_t count,
                           std::uint32_t expected) {
    const auto values = Array(key);
    if (!ok()) {
      return;
    }
    if (values.size() != count) {
      Fail(Key(key) + " has " + std::to_string(values.size()) +
           " entries, expected " + std::to_string(count));
      return;
    }
    for (const auto value : values) {
      if (value != expected) {
        Fail(Key(key) + " holds " + std::to_string(value) + ", expected " +
             std::to_string(expected) + " everywhere");
        return;
      }
    }
  }

private:
  const core::GgufReader& reader_;
  std::string prefix_;
  std::string why_;
};

}  // namespace

bool ParseConfig(const core::GgufReader& reader, Inkling2Config* out,
                 std::string* why) {
  const auto fail = [&](std::string message) {
    if (why != nullptr) {
      *why = std::move(message);
    }
    return false;
  };
  const auto arch = reader.GetMetadataString("general.architecture");
  if (!arch) {
    return fail("general.architecture is missing");
  }
  // The layout mainline llama.cpp loads: 24 physical blocks and the loop
  // keys `loop.start/end/count/gate_rank/varying`.
  if (*arch != "inkling2_loop") {
    return fail("architecture '" + std::string(*arch) +
                "' is not inkling2_loop");
  }
  Inkling2Config config;
  Checker c(reader, std::string(*arch));

  c.Expect("block_count", config.n_phys);
  c.Expect("loop.count", config.n_passes);
  c.Expect("loop.start", config.core_start);
  c.Expect("loop.end", config.core_end);
  // `loop.varying` is absent from older conversions; only a true value
  // (variable pass count) is refused.
  if (const auto varying = reader.GetMetadataBool(c.Key("loop.varying"))) {
    if (*varying) {
      c.Fail(c.Key("loop.varying") +
             " is true; variable passes are not supported");
    }
  }
  c.Expect("loop.gate_rank", config.gate_rank);
  c.Expect("context_length", config.n_ctx_train);
  c.Expect("embedding_length", config.n_embd);
  c.Expect("feed_forward_length", config.n_ff_expert);
  c.Expect("expert_feed_forward_length", config.n_ff_expert);
  c.Expect("expert_count", config.n_expert);
  c.Expect("expert_used_count", config.n_expert_used);
  c.Expect("vocab_size", config.n_vocab);
  c.Expect("vocab_size_unpadded", config.n_vocab_text);
  c.Expect("attention.key_length", config.head_dim);
  c.Expect("attention.value_length", config.head_dim);
  c.Expect("attention.key_length_swa", config.head_dim);
  c.Expect("attention.value_length_swa", config.head_dim);
  c.Expect("attention.sliding_window", config.n_swa);
  c.Expect("attention.relative_length", config.rel_dim);
  c.Expect("attention.relative_extent", config.rel_extent);
  c.Expect("attention.log_scale_floor", config.log_scale_floor);
  c.Expect("rope.dimension_count", 0);  // no RoPE
  c.Expect("shortconv.l_cache", config.conv_kernel);
  c.Expect("attn_res.block_size", config.attn_res_block);
  c.ExpectFloat("attention.layer_norm_rms_epsilon", config.rms_eps);
  c.ExpectFloat("attention.qk_clip", config.qk_clip);
  c.ExpectFloat("attention.relative_clip", config.rel_clip);
  c.ExpectFloat("attention.log_scale_alpha", config.log_scale_alpha);
  c.ExpectFloat("expert_weights_scale", config.expert_weights_scale);
  c.ExpectFloat("activation.situ_beta", config.situ_beta);
  c.ExpectFloat("activation.situ_linear_beta", config.situ_linear_beta);
  c.ExpectFloat("final_logit_softcapping", config.logit_softcap);

  // Per-layer arrays have one entry per physical block.
  const std::size_t entries = config.n_phys;
  c.ExpectConstantArray("attention.head_count", entries, config.n_head);
  c.ExpectConstantArray("attention.head_count_kv", entries, config.n_head_kv);
  const auto pattern = c.Array("attention.sliding_window_pattern");
  if (c.ok() && pattern.size() != entries) {
    c.Fail(c.Key("attention.sliding_window_pattern") + " has " +
           std::to_string(pattern.size()) + " entries, expected " +
           std::to_string(entries));
  }
  if (!c.ok()) {
    return fail(c.why());
  }
  // The pattern is true for sliding-window layers, so a global layer is
  // `false`.
  std::array<bool, Inkling2Config::kPhysLayers> global{};
  for (std::uint32_t il = 0; il < entries; ++il) {
    global[il] = pattern[il] == 0;
  }
  for (std::uint32_t il = 0; il < config.n_phys; ++il) {
    // Global attention at layers 5, 11, 17, 23: the last of every six.
    const bool expected = il % 6 == 5;
    if (global[il] != expected) {
      return fail(c.Key("attention.sliding_window_pattern") + " marks layer " +
                  std::to_string(il) +
                  (expected ? " as sliding-window" : " as global") +
                  "; expected global layers 5, 11, 17, 23");
    }
    config.global_layer[il] = global[il];
  }
  if (out != nullptr) {
    *out = config;
  }
  return true;
}

}  // namespace gufo::models::inkling2
