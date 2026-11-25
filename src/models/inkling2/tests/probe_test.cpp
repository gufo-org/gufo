// Refusal and normalisation tests on synthetic GGUF headers. No GPU, no model
// files: weights_test covers the real artifacts.

#include "src/models/inkling2/probe.hpp"

#include <cstdio>
#include <gufo/ext/gguf_reader.hpp>
#include <string>
#include <vector>

#include "src/models/inkling2/config.hpp"
#include "src/models/inkling2/tensor_ref.hpp"
#include "src/models/inkling2/tests/gguf_builder.hpp"
#include "src/models/inkling2/weights.hpp"

namespace {

using namespace gufo::models::inkling2;
using namespace inkling2_test;

int g_failures = 0;

#define CHECK(cond)                                                         \
  do {                                                                      \
    if (!(cond)) {                                                          \
      std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, \
                   #cond);                                                  \
      ++g_failures;                                                         \
    }                                                                       \
  } while (false)

std::vector<Kv> ValidMeta() {
  const std::string p = "inkling2_loop";
  const std::uint32_t n = 24;
  std::vector<bool> swa(n, true);
  for (std::uint32_t il = 0; il < n; ++il) {
    swa[il] = il % 6 != 5;  // layers 5, 11, 17, 23 are global
  }
  std::vector<Kv> kv = {
      {"general.architecture", p},
      {p + ".block_count", n},
      {p + ".context_length", 1048576U},
      {p + ".embedding_length", 2304U},
      {p + ".feed_forward_length", 1152U},
      {p + ".expert_feed_forward_length", 1152U},
      {p + ".attention.head_count",
       U32Array{std::vector<std::uint32_t>(n, 24)}},
      {p + ".attention.head_count_kv",
       U32Array{std::vector<std::uint32_t>(n, 8)}},
      {p + ".attention.layer_norm_rms_epsilon", 1e-6F},
      {p + ".expert_count", 128U},
      {p + ".expert_used_count", 8U},
      {p + ".attention.key_length", 128U},
      {p + ".attention.value_length", 128U},
      {p + ".attention.key_length_swa", 128U},
      {p + ".attention.value_length_swa", 128U},
      {p + ".loop.gate_rank", 64U},
      {p + ".vocab_size", 201024U},
      {p + ".vocab_size_unpadded", 200058U},
      {p + ".rope.dimension_count", 0U},
      {p + ".attention.sliding_window", 1024U},
      {p + ".attention.sliding_window_pattern", BoolArray{swa}},
      {p + ".shortconv.l_cache", 4U},
      {p + ".expert_weights_scale", 8.0F},
      {p + ".activation.situ_beta", 4.0F},
      {p + ".activation.situ_linear_beta", 25.0F},
      {p + ".attn_res.block_size", 6U},
      {p + ".final_logit_softcapping", 30.0F},
      {p + ".attention.relative_length", 16U},
      {p + ".attention.relative_extent", 1024U},
      {p + ".attention.qk_clip", 22.627417F},
      {p + ".attention.relative_clip", 8.0F},
      {p + ".attention.log_scale_floor", 8192U},
      {p + ".attention.log_scale_alpha", 0.1F},
  };
  kv.push_back({p + ".loop.count", 3U});
  kv.push_back({p + ".loop.start", 6U});
  kv.push_back({p + ".loop.end", 18U});
  kv.push_back({p + ".loop.varying", false});
  return kv;
}

std::vector<Kv> With(std::vector<Kv> kv, const std::string& key, Value value) {
  for (auto& entry : kv) {
    if (entry.key == key) {
      entry.value = std::move(value);
      return kv;
    }
  }
  kv.push_back({key, std::move(value)});
  return kv;
}

std::vector<Kv> Without(std::vector<Kv> kv, const std::string& key) {
  std::erase_if(kv, [&](const Kv& e) { return e.key == key; });
  return kv;
}

/// Probes a synthetic file; returns the refusal reason ("" when accepted).
std::string Probe(const std::vector<Kv>& kv,
                  const std::vector<TensorSpec>& tensors = {}) {
  const auto image = BuildGguf(kv, tensors);
  std::string error;
  const auto reader = gufo::core::GgufReader::OpenMemory(
      image.data(), image.size(), &error, {.allow_unsized_tensors = true});
  if (!reader) {
    return "reader: " + error;
  }
  std::string reason;
  return ProbeInkling2Gguf(*reader, &reason) ? "" : reason;
}

bool Has(const std::string& text, const std::string& part) {
  const bool found = text.find(part) != std::string::npos;
  if (!found) {
    std::fprintf(stderr, "  reason '%s' lacks '%s'\n", text.c_str(),
                 part.c_str());
  }
  return found;
}

void TestNvfp4Layout() {
  static_assert(sizeof(float) == 4);
  CHECK(kNvfp4BlockElements == 64 && kNvfp4BlockBytes == 36);
  // One expert tensor of layers 0..19: 2304 x 2304 x 128 values.
  CHECK(Nvfp4Bytes(2304ULL * 2304 * 128) == 2304ULL * 2304 * 128 / 64 * 36);
  CHECK(Nvfp4Bytes(2304ULL * 2304 * 128) == 382205952ULL);
  CHECK(Nvfp4Bytes(100) == 0);
  CHECK(IsSupportedType(kNvfp4) && IsSupportedType(GgmlType::kIQ4_XS));
  CHECK(!IsSupportedType(GgmlType::kQ4_0) && !IsSupportedType(GgmlType::kQ2_K));
}

void TestConfig() {
  {
    const auto image = BuildGguf(ValidMeta(), {});
    std::string error;
    const auto reader = gufo::core::GgufReader::OpenMemory(
        image.data(), image.size(), &error, {.allow_unsized_tensors = true});
    CHECK(reader != nullptr);
    Inkling2Config config;
    std::string why;
    CHECK(ParseConfig(*reader, &config, &why));
    CHECK(config.n_phys == 24 && config.n_visits == 48);
    for (std::uint32_t il = 0; il < 24; ++il) {
      CHECK(config.global_layer[il] ==
            (il == 5 || il == 11 || il == 17 || il == 23));
    }
    // Visits: prelude 0-5, core 6-17 three times, postlude 18-23.
    CHECK(config.PhysOfVisit(5) == 5 && config.PhysOfVisit(6) == 6);
    CHECK(config.PhysOfVisit(17) == 17 && config.PhysOfVisit(18) == 6);
    CHECK(config.PhysOfVisit(41) == 17 && config.PhysOfVisit(42) == 18);
    CHECK(config.PhysOfVisit(47) == 23);
    // With no tensors the probe stops at the first missing one.
    CHECK(Has(Probe(ValidMeta()), "missing tensor token_embd.weight"));
  }
}

void TestRefusals() {
  const auto loop = ValidMeta();
  CHECK(Has(Probe(With(loop, "general.architecture", std::string("qwen4exp"))),
            "is not inkling2_loop"));
  // Unsloth's earlier root layout of the same tensors is no longer read.
  CHECK(Has(Probe(With(loop, "general.architecture", std::string("inkling2"))),
            "is not inkling2_loop"));
  CHECK(Has(Probe(Without(loop, "general.architecture")),
            "general.architecture is missing"));
  CHECK(Has(Probe(With(loop, "inkling2_loop.block_count", 48U)),
            "block_count is 48, expected 24"));
  CHECK(Has(Probe(With(loop, "inkling2_loop.loop.count", 2U)),
            "loop.count is 2"));
  CHECK(Has(Probe(With(loop, "inkling2_loop.loop.end", 17U)), "end is 17"));
  CHECK(Has(Probe(With(loop, "inkling2_loop.loop.varying", true)),
            "variable passes"));
  CHECK(Has(Probe(Without(loop, "inkling2_loop.loop.start")),
            "start is missing"));
  CHECK(Has(Probe(With(loop, "inkling2_loop.expert_count", 256U)),
            "expert_count is 256"));
  CHECK(Has(Probe(With(loop, "inkling2_loop.attention.qk_clip", 11.0F)),
            "qk_clip"));
  CHECK(Has(Probe(With(loop, "inkling2_loop.rope.dimension_count", 128U)),
            "rope.dimension_count is 128"));
  CHECK(Has(Probe(With(loop, "inkling2_loop.attention.head_count",
                       U32Array{std::vector<std::uint32_t>(23, 24)})),
            "has 23 entries, expected 24"));
  auto heads = std::vector<std::uint32_t>(24, 24);
  heads[7] = 16;
  CHECK(Has(
      Probe(With(loop, "inkling2_loop.attention.head_count", U32Array{heads})),
      "holds 16"));
  // Layer 4 global instead of sliding-window.
  std::vector<bool> swa(24, true);
  for (const auto il : {4, 11, 17, 23}) {
    swa[il] = false;
  }
  CHECK(Has(Probe(With(loop, "inkling2_loop.attention.sliding_window_pattern",
                       BoolArray{swa})),
            "marks layer 4 as global"));
}

void TestTensorRefusals() {
  const auto meta = ValidMeta();
  // token_embd: type is checked before the payload size matters.
  CHECK(Has(
      Probe(meta,
            {{"token_embd.weight", {2304, 8}, 2 /*Q4_0*/, 2304 / 32 * 18 * 8}}),
      "is Q4_0, not an allowed weight type"));
  CHECK(Has(Probe(meta, {{"token_embd.weight", {2304, 8}, 0, 2304 * 8 * 4}}),
            "has shape [2304, 8], expected [2304, 201024]"));
  // NVFP4 is only for experts.
  CHECK(Has(
      Probe(meta, {{"token_embd.weight", {2304, 8}, 40, 2304 / 64 * 36 * 8}}),
      "is NVFP4, not an allowed weight type"));
  CHECK(Has(
      Probe(meta,
            {{"token_embd.weight", {2304, 8}, 3 /*Q4_1*/, 2304 / 32 * 20 * 8}}),
      "Q4_1"));
}

}  // namespace

int main() {
  TestNvfp4Layout();
  TestConfig();
  TestRefusals();
  TestTensorRefusals();
  if (g_failures == 0) {
    std::puts("inkling2 probe test: OK");
  }
  return g_failures == 0 ? 0 : 1;
}
