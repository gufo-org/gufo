#include "src/models/qwen38_flash_next/config.hpp"

#include <cassert>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <variant>

namespace qfn = gufo::models::qwen38_flash_next;

namespace {

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

void CheckSidecarCompatibility() {
  qfn::Config trunk;
  trunk.num_layers = 48;
  trunk.context_length = 262144;
  trunk.hidden_size = 2560;
  trunk.hc_count = 4;
  trunk.hc_low_rank = 320;
  trunk.num_experts = 512;
  trunk.num_experts_used = 10;
  trunk.expert_ff = trunk.shared_expert_ff = 640;
  trunk.num_heads = 24;
  trunk.num_kv_heads = 2;
  trunk.head_dim = 256;
  trunk.rotary_dim = 64;
  trunk.rope_theta = 1e7F;
  trunk.rope_sections = {11, 11, 10, 0};
  auto sidecar = trunk;
  sidecar.nextn_layers = 1;
  Require(sidecar.MtpMatches(trunk), "matching MTP constants rejected");
  for (auto member : {&qfn::Config::hc_low_rank, &qfn::Config::context_length,
                      &qfn::Config::rotary_dim, &qfn::Config::num_experts_used,
                      &qfn::Config::shared_expert_ff}) {
    auto wrong = sidecar;
    ++(wrong.*member);
    Require(!wrong.MtpMatches(trunk),
            "dimension-compatible MTP with different semantics accepted");
  }
  for (auto member : {&qfn::Config::rms_eps, &qfn::Config::rope_theta}) {
    auto wrong = sidecar;
    wrong.*member *= 2.0F;
    Require(!wrong.MtpMatches(trunk),
            "MTP numerical constant mismatch accepted");
  }
  auto wrong = sidecar;
  std::swap(wrong.rope_sections[0], wrong.rope_sections[2]);
  Require(!wrong.MtpMatches(trunk), "MTP RoPE partition mismatch accepted");
  sidecar.ple_layer = -1;
  trunk.ple_layer = 1;
  Require(sidecar.MtpMatches(trunk),
          "trunk-only PLE incorrectly required in MTP");

  // The MTP sidecar never carries rope scaling; the trunk's governs both.
  trunk.rope_scaling = {.factor = 2.5F, .original_context = 262144};
  Require(sidecar.MtpMatches(trunk), "YaRN trunk rejected an unscaled sidecar");
}

using Value =
    std::variant<std::uint32_t, float, std::string, std::vector<std::uint64_t>,
                 std::vector<std::int64_t>, std::vector<double>>;
using Metadata = std::map<std::string, Value>;

// Minimal metadata-only GGUF: no model or GPU is needed to test the loader.
std::optional<qfn::Config> Parse(const Metadata& fields, bool trunk = true) {
  std::vector<std::uint8_t> bytes;
  const auto pod = [&]<class T>(T value) {
    const auto begin = bytes.size();
    bytes.resize(begin + sizeof(value));
    std::memcpy(bytes.data() + begin, &value, sizeof(value));
  };
  const auto text = [&](const std::string& value) {
    pod(std::uint64_t{value.size()});
    bytes.insert(bytes.end(), value.begin(), value.end());
  };
  pod(std::uint32_t{0x46554747});
  pod(std::uint32_t{3});
  pod(std::uint64_t{0});
  pod(std::uint64_t{fields.size()});
  for (const auto& [key, value] : fields) {
    text(key);
    std::visit(
        [&](const auto& item) {
          using T = std::decay_t<decltype(item)>;
          if constexpr (std::is_same_v<T, std::uint32_t>) {
            pod(std::uint32_t{4});
            pod(item);
          } else if constexpr (std::is_same_v<T, float>) {
            pod(std::uint32_t{6});
            pod(item);
          } else if constexpr (std::is_same_v<T, std::string>) {
            pod(std::uint32_t{8});
            text(item);
          } else {
            pod(std::uint32_t{9});
            using E = typename T::value_type;
            pod(std::uint32_t{std::is_same_v<E, std::uint64_t>  ? 10U
                              : std::is_same_v<E, std::int64_t> ? 11U
                                                                : 12U});
            pod(std::uint64_t{item.size()});
            for (const auto entry : item)
              pod(entry);
          }
        },
        value);
  }
  bytes.resize((bytes.size() + 31) / 32 * 32);
  std::string error;
  const auto reader =
      gufo::core::GgufReader::OpenMemory(bytes.data(), bytes.size(), &error);
  assert(reader && error.empty());
  const auto result = qfn::Config::FromGguf(*reader, trunk, &error);
  assert(result.has_value() == error.empty());
  return result;
}

Metadata ValidMetadata() {
  Metadata fields{{"general.architecture", std::string{"qwen4exp"}}};
  for (const auto& [key, value] :
       std::initializer_list<std::pair<const char*, std::uint32_t>>{
           {"block_count", 48},
           {"embedding_length", 2560},
           {"context_length", 262144},
           {"hyper_connection.count", 4},
           {"hyper_connection.low_rank", 320},
           {"full_attention_interval", 4},
           {"attention.head_count", 24},
           {"attention.head_count_kv", 2},
           {"attention.key_length", 256},
           {"rope.dimension_count", 64},
           {"attention.indexer.head_count", 4},
           {"attention.indexer.key_length", 128},
           {"attention.indexer.top_k", 2048},
           {"ssm.conv_kernel", 4},
           {"ssm.state_size", 128},
           {"ssm.group_count", 16},
           {"ssm.time_step_rank", 48},
           {"ssm.inner_size", 6144},
           {"expert_count", 512},
           {"expert_used_count", 10},
           {"expert_feed_forward_length", 640},
           {"expert_shared_feed_forward_length", 640}}) {
    fields[std::string{"qwen4exp."} + key] = value;
  }
  fields["qwen4exp.attention.layer_norm_rms_epsilon"] = 1e-6F;
  fields["qwen4exp.rope.freq_base"] = 1e7F;
  fields["qwen4exp.rope.dimension_sections"] =
      std::vector<std::uint64_t>{11, 11, 10, 0};
  std::vector<std::uint64_t> ratios(48);
  for (std::size_t i = 3; i < ratios.size(); i += 4)
    ratios[i] = 4;
  fields["qwen4exp.attention.compress_ratios"] = ratios;
  return fields;
}

void CheckMalformedMetadata() {
  const auto valid = ValidMetadata();
  assert(Parse(valid));
  for (const auto key : {"qwen4exp.rope.dimension_sections",
                         "qwen4exp.attention.compress_ratios"}) {
    auto wrong = valid;
    wrong.erase(key);
    assert(!Parse(wrong));
    for (const Value value :
         {Value{std::uint32_t{4}}, Value{std::string{"4"}},
          Value{std::vector<std::uint64_t>{}},
          Value{std::vector<std::int64_t>{-1}}, Value{std::vector<double>{4.0}},
          Value{
              std::vector<double>{std::numeric_limits<double>::quiet_NaN()}}}) {
      wrong = valid;
      wrong[key] = value;
      assert(!Parse(wrong));
    }
  }
  for (const auto sections :
       {std::vector<std::uint64_t>{11, 11, 10},
        std::vector<std::uint64_t>{11, 11, 10, 0, 0},
        std::vector<std::uint64_t>{11, 11, 10, 1},
        std::vector<std::uint64_t>{11, 11, 11, 0},
        std::vector<std::uint64_t>{10, 11, 11, 0},
        std::vector<std::uint64_t>{0, 16, 16, 0},
        std::vector<std::uint64_t>{UINT64_MAX, 11, 10, 0}}) {
    auto wrong = valid;
    wrong["qwen4exp.rope.dimension_sections"] = sections;
    assert(!Parse(wrong));
  }
  for (const auto& [index, value] :
       std::initializer_list<std::pair<std::size_t, std::uint64_t>>{
           {0, 4}, {3, 0}, {7, 8}, {3, 4097}, {3, UINT64_MAX}}) {
    auto wrong = valid;
    auto& ratios = std::get<std::vector<std::uint64_t>>(
        wrong["qwen4exp.attention.compress_ratios"]);
    ratios[index] = value;
    assert(!Parse(wrong));
  }
  for (const auto& [key, value] :
       {std::pair{"attention.indexer.key_length", 32U},
        std::pair{"attention.indexer.head_count", 0U},
        std::pair{"attention.indexer.head_count", 8U},
        std::pair{"attention.indexer.key_length", 256U},
        std::pair{"attention.key_length", 128U},
        std::pair{"attention.value_length", 128U},
        std::pair{"embedding_length", 5120U}, std::pair{"ssm.state_size", 64U},
        std::pair{"expert_count", 2048U}, std::pair{"expert_used_count", 11U},
        std::pair{"attention.indexer.top_k", 2047U},
        std::pair{"nextn_predict_layers", 48U}}) {
    auto wrong = valid;
    wrong[std::string{"qwen4exp."} + key] = value;
    assert(!Parse(wrong));
  }
  for (const Value value :
       {Value{std::string{"4"}}, Value{std::vector<std::int64_t>{-1}},
        Value{std::vector<std::uint64_t>{48}}}) {
    auto wrong = valid;
    wrong["qwen4exp.ple.layers"] = value;
    assert(!Parse(wrong));
  }
  auto sidecar = valid;
  sidecar["qwen4exp.block_count"] = std::uint32_t{49};
  sidecar["qwen4exp.nextn_predict_layers"] = std::uint32_t{1};
  std::get<std::vector<std::uint64_t>>(
      sidecar["qwen4exp.attention.compress_ratios"])
      .push_back(0);
  auto sparse_sidecar = sidecar;
  std::get<std::vector<std::uint64_t>>(
      sparse_sidecar["qwen4exp.attention.compress_ratios"])
      .back() = 4;
  assert(Parse(sparse_sidecar, false));
  std::get<std::vector<std::uint64_t>>(
      sparse_sidecar["qwen4exp.attention.compress_ratios"])
      .back() = 8;
  assert(!Parse(sparse_sidecar, false));
  const auto draft = Parse(sidecar, false);
  const auto trunk = Parse(valid);
  assert(draft && trunk && draft->MtpMatches(*trunk));
}

void CheckScaledContext() {
  qfn::Config c;
  c.context_length = 262144;
  c.compress_ratio = 4;
  Require(c.MaxContextLength() == 262144, "scaling off must keep native");
  Require(c.SparseMaskWords(c.context_length) == 2048,
          "native mask row width changed");
  c.rope_scaling = {.factor = 1.5625F, .original_context = 262144};
  Require(c.MaxContextLength() == 409600, "YaRN 1.5625 must reach 409600");
  Require(c.SparseMaskWords(409600) == 3200, "mask rows must cover 409600");
  c.rope_scaling.factor = 2.5F;
  Require(c.MaxContextLength() == 655360, "YaRN 2.5 must reach 655360");
  Require(c.SparseMaskWords(655360) == 5120, "mask rows must cover 655360");
  const auto parsed = Parse(ValidMetadata());
  Require(parsed && parsed->rope_scaling == qfn::RopeScaling{},
          "the GGUF loader must default rope scaling off (it sets no such "
          "keys today)");
}

// The checks FromGguf ends with, as a non-GGUF weight source calls them.
void CheckValidated() {
  const auto parsed = Parse(ValidMetadata());
  Require(parsed.has_value(), "valid metadata must parse");
  std::string error;
  const auto again = qfn::Config::Validated(*parsed, true, &error);
  Require(again && again->num_layers == parsed->num_layers &&
              again->ple_layer == parsed->ple_layer && error.empty(),
          "Validated must accept and keep a parsed Config");
  auto wrong = *parsed;
  wrong.num_experts_used = wrong.num_experts + 1;
  Require(!qfn::Config::Validated(wrong, true, &error) && !error.empty(),
          "Validated must reject an unsupported shape");
  wrong = *parsed;
  wrong.hidden_size = 2048;
  error.clear();
  Require(!qfn::Config::Validated(wrong, true, &error) &&
              error == "unsupported Qwen3.8-Flash-Next kernel geometry",
          "Validated must reject another kernel geometry");
  // The PLE tables the n-gram reader divides and indexes by, on the
  // fixture with the production PLE geometry added.
  auto ple = *parsed;
  ple.ple_layer = 0;
  ple.ple_ngram_size = 3;
  ple.ple_heads_per_ngram = 8;
  ple.ple_heads = 16;
  ple.ple_head_dim = 160;
  ple.ple_conv_kernel = 4;
  for (std::uint32_t h = 0; h < ple.ple_heads; ++h) {
    ple.ple_head_offsets[h] = h * 100;
    ple.ple_head_vocab[h] = 100;
  }
  ple.ple_rows = 1600;
  Require(ple.IsLinearLayer(0), "the fixture's layer 0 is a linear layer");
  error.clear();
  Require(
      qfn::Config::Validated(ple, true, &error).has_value(),
      ("Validated must accept the production PLE geometry: " + error).c_str());
  wrong = ple;
  wrong.ple_layer = static_cast<std::int32_t>(wrong.num_layers);
  Require(!qfn::Config::Validated(wrong, true, &error),
          "Validated must reject a PLE layer outside the trunk");
  wrong = ple;
  wrong.ple_heads_per_ngram = 0;
  Require(!qfn::Config::Validated(wrong, true, &error),
          "Validated must reject an empty PLE n-gram");
  wrong = ple;
  wrong.ple_head_vocab[0] = 0;
  Require(!qfn::Config::Validated(wrong, true, &error),
          "Validated must reject an empty PLE head");
  wrong = ple;
  wrong.ple_rows += 1;
  error.clear();
  Require(!qfn::Config::Validated(wrong, true, &error) &&
              error == "PLE row count does not match its head ranges",
          "Validated must reject PLE rows that do not match the heads");
}

}  // namespace

int main() {
  CheckSidecarCompatibility();
  CheckMalformedMetadata();
  CheckScaledContext();
  CheckValidated();
  std::cout << "Flash-Next metadata checks passed.\n";
}
