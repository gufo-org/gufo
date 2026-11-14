// Binds an EXL3 checkpoint of Qwen3.8-Flash-Next to the runtime's weight
// structures. The checkpoint keeps Hugging Face tensor layouts; the runtime
// was written for the official GGUF conversion, so this binder applies the
// same conversions that artifact carries:
// - RMSNorm gammas stored as w are uploaded as (1 + w), except the gated
//   DeltaNet norm;
// - gated DeltaNet value heads are grouped by key head in the checkpoint
//   (v = g * 3 + r) and tiled in the runtime (v = r * 16 + g);
// - A_log becomes A = -exp(A_log);
// - the indexer's fused q|k projection is split into its two outputs.
// Hyper-connection and PLE projections are quantized to Q8_0, the format
// of the official artifact, so they take its fused decode kernels.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>

#include "src/models/qwen38_flash_next/config.hpp"
#include "src/models/qwen38_flash_next/exl3/format.hpp"
#include "src/models/qwen38_flash_next/exl3/pack.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace gufo::models::qwen38_flash_next {
namespace {

using exl3::DType;
using exl3::Pack;
using exl3::PackTensor;

constexpr std::string_view kTrunk = "model.language_model.";

float Bf16ToFloat(std::uint16_t v) {
  const std::uint32_t bits = static_cast<std::uint32_t>(v) << 16;
  float f = 0.0F;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

/// ggml's reference Q8_0 quantizer (quantize_row_q8_0_ref).
void QuantizeQ8_0(const float* x, std::uint8_t* out, std::size_t n) {
  for (std::size_t b = 0; b < n / 32; ++b) {
    const float* v = x + b * 32;
    float amax = 0.0F;
    for (int i = 0; i < 32; ++i)
      amax = std::max(amax, std::fabs(v[i]));
    const float d = amax / 127.0F;
    const float id = d != 0.0F ? 1.0F / d : 0.0F;
    std::uint8_t* block = out + b * 34;
    const std::uint16_t dh = exl3::FloatToHalf(d);
    std::memcpy(block, &dh, sizeof(dh));
    for (int i = 0; i < 32; ++i) {
      const auto q = static_cast<std::int8_t>(std::lround(v[i] * id));
      std::memcpy(block + 2 + i, &q, 1);
    }
  }
}

struct Binder {
  const Pack& pack;
  HostTensors& host;
  std::string* error;
  bool ok{true};

  void Fail(std::string message) {
    if (ok && error != nullptr)
      *error = std::move(message);
    ok = false;
  }

  const PackTensor* Need(const std::string& name) {
    const PackTensor* t = pack.Find(name);
    if (t == nullptr)
      Fail("EXL3 checkpoint misses tensor " + name);
    return t;
  }

  std::string_view Name(std::string name) {
    host.names.push_back(std::move(name));
    return host.names.back();
  }

  /// Float values of a 16- or 32-bit float tensor.
  std::vector<float> Floats(const PackTensor& t) {
    std::vector<float> v(t.Elements());
    const auto* p16 = static_cast<const std::uint16_t*>(t.data);
    for (std::size_t i = 0; i < v.size(); ++i) {
      switch (t.dtype) {
        case DType::kF16:
          v[i] = exl3::HalfToFloat(p16[i]);
          break;
        case DType::kBF16:
          v[i] = Bf16ToFloat(p16[i]);
          break;
        case DType::kF32:
          std::memcpy(&v[i], static_cast<const float*>(t.data) + i, 4);
          break;
        default:
          Fail("tensor " + t.name + " is not a float tensor");
          return {};
      }
    }
    return v;
  }

  TensorRef HostRef(std::string_view name, core::GgmlType type,
                    std::vector<std::uint8_t> bytes, std::uint64_t cols,
                    std::uint64_t rows) {
    host.buffers.push_back(std::move(bytes));
    TensorRef r;
    r.data = host.buffers.back().data();
    r.type = type;
    r.cols = cols;
    r.rows = rows;
    r.shard = kHostShard;
    r.name = name;
    return r;
  }

  TensorRef F32(std::string_view name, const std::vector<float>& v,
                std::uint64_t cols, std::uint64_t rows) {
    std::vector<std::uint8_t> bytes(v.size() * sizeof(float));
    std::memcpy(bytes.data(), v.data(), bytes.size());
    return HostRef(name, core::GgmlType::kF32, std::move(bytes), cols, rows);
  }

  /// A float tensor as F32 rows of `cols`, plus `bias` (the norm fold).
  TensorRef Float(const std::string& name, float bias = 0.0F) {
    const PackTensor* t = Need(name);
    if (t == nullptr)
      return {};
    auto v = Floats(*t);
    for (auto& x : v)
      x += bias;
    const std::uint64_t cols = t->shape.back();
    return F32(t->name, v, cols, v.size() / cols);
  }

  /// An [out, in] float matrix quantized to Q8_0 rows.
  TensorRef Q8(const std::string& name) {
    const PackTensor* t = Need(name);
    if (t == nullptr)
      return {};
    if (t->shape.size() != 2 || t->shape[1] % 32 != 0) {
      Fail("Q8_0 conversion needs a 2-D matrix with 32-wide rows: " + name);
      return {};
    }
    const auto v = Floats(*t);
    std::vector<std::uint8_t> bytes(v.size() / 32 * 34);
    QuantizeQ8_0(v.data(), bytes.data(), v.size());
    return HostRef(t->name, core::GgmlType::kQ8_0, std::move(bytes),
                   t->shape[1], t->shape[0]);
  }

  /// A mapped tensor used as is (the BF16 token embedding).
  TensorRef Mapped(const std::string& name, core::GgmlType type) {
    const PackTensor* t = Need(name);
    if (t == nullptr)
      return {};
    TensorRef r;
    r.data = t->data;
    r.type = type;
    r.cols = t->shape.back();
    r.rows = t->Elements() / r.cols;
    r.file_offset = t->file_offset;
    r.shard = t->file;
    r.name = t->name;
    return r;
  }

  bool Part(const std::string& prefix, Exl3Ref& ref) {
    const PackTensor* trellis = Need(prefix + ".trellis");
    const PackTensor* suh = Need(prefix + ".suh");
    const PackTensor* svh = Need(prefix + ".svh");
    const PackTensor* mul1 = Need(prefix + ".mul1");
    if (!ok)
      return false;
    // Reading a marker faults in its page: check the value once per stack
    // (an expert stack has hundreds of parts) and only presence after that.
    std::uint32_t marker = exl3::kMul1;
    if (ref.parts.empty()) {
      marker = 0;
      if (mul1->size_bytes == 4)
        std::memcpy(&marker, mul1->data, 4);
    }
    if (marker != exl3::kMul1 || mul1->size_bytes != 4 ||
        pack.Find(prefix + ".mcg") != nullptr ||
        pack.Find(prefix + ".bias") != nullptr) {
      Fail("unsupported EXL3 codebook or bias in " + prefix);
      return false;
    }
    if (trellis->dtype != DType::kI16 || trellis->shape.size() != 3 ||
        suh->dtype != DType::kF16 || svh->dtype != DType::kF16 ||
        suh->shape.size() != 1 || svh->shape.size() != 1 ||
        trellis->shape[2] % 16 != 0) {
      Fail("malformed EXL3 linear " + prefix);
      return false;
    }
    const auto in = static_cast<std::uint32_t>(suh->shape[0]);
    const auto out = static_cast<std::uint32_t>(svh->shape[0]);
    const int k = static_cast<int>(trellis->shape[2] / 16);
    if (trellis->shape[0] * 16 != in || trellis->shape[1] * 16 != out ||
        in % 128 != 0 || out % 128 != 0 || k < exl3::kMinTileK ||
        k > exl3::kMaxTileK) {
      Fail("unsupported EXL3 linear shape or width in " + prefix +
           " (K=" + std::to_string(k) + ")");
      return false;
    }
    if (ref.parts.empty()) {
      ref.k = k;
      ref.src_in = in;
      ref.src_out = out;
    } else if (ref.k != k || ref.src_in != in || ref.src_out != out) {
      Fail("EXL3 expert stack mixes shapes or widths: " + prefix);
      return false;
    }
    if (suh->file != trellis->file || svh->file != trellis->file) {
      Fail("EXL3 linear spans files: " + prefix);
      return false;
    }
    ref.parts.push_back({trellis->file, trellis->file_offset, suh->file_offset,
                         svh->file_offset,
                         static_cast<const std::uint16_t*>(suh->data),
                         static_cast<const std::uint16_t*>(svh->data)});
    return true;
  }

  TensorRef Linear(const std::string& prefix,
                   std::vector<std::uint32_t> in_blocks = {},
                   std::vector<std::uint32_t> out_blocks = {}) {
    auto ref = std::make_shared<Exl3Ref>();
    if (!Part(prefix, *ref))
      return {};
    ref->in_blocks = std::move(in_blocks);
    ref->out_blocks = std::move(out_blocks);
    return FromRef(Name(prefix), std::move(ref));
  }

  TensorRef Experts(const std::string& prefix, std::uint32_t count,
                    const std::string& matrix) {
    auto ref = std::make_shared<Exl3Ref>();
    ref->parts.reserve(count);
    for (std::uint32_t e = 0; e < count; ++e) {
      if (!Part(prefix + ".experts." + std::to_string(e) + "." + matrix, *ref))
        return {};
    }
    return FromRef(Name(prefix + ".experts." + matrix), std::move(ref));
  }

  static TensorRef FromRef(std::string_view name,
                           std::shared_ptr<const Exl3Ref> ref) {
    TensorRef r;
    r.cols = ref->In();
    r.rows = ref->Out();
    r.experts = ref->parts.size();
    r.name = name;
    r.exl3 = std::move(ref);
    return r;
  }

  HcMixer Mixer(const std::string& prefix, bool inject) {
    HcMixer m;
    m.norm = Float(prefix + ".hc_norm.weight", 1.0F);
    m.down = Q8(prefix + ".input_mix_weight_down.weight");
    m.up = Q8(prefix + ".input_mix_weight_up.weight");
    if (inject)
      m.inject = Float(prefix + ".block_inject_weight.weight");
    return m;
  }

  void Attention(const std::string& p, const Config& c, LayerWeights& l) {
    l.attn_q = Linear(p + "q_proj");
    l.attn_k = Linear(p + "k_proj");
    l.attn_v = Linear(p + "v_proj");
    l.attn_out = Linear(p + "o_proj");
    l.attn_q_norm = Float(p + "q_norm.weight", 1.0F);
    l.attn_k_norm = Float(p + "k_norm.weight", 1.0F);
    // The fused indexer projection is [q (heads * dim) | k (dim)].
    const std::uint32_t q_blocks = c.indexer_heads * c.indexer_head_dim / 128;
    const std::uint32_t k_blocks = c.indexer_head_dim / 128;
    std::vector<std::uint32_t> q(q_blocks);
    std::iota(q.begin(), q.end(), 0U);
    std::vector<std::uint32_t> k(k_blocks);
    std::iota(k.begin(), k.end(), q_blocks);
    l.indexer_q = Linear(p + "indexer.index_qk_proj", {}, q);
    l.indexer_k = Linear(p + "indexer.index_qk_proj", {}, k);
    l.indexer_q_norm = Float(p + "indexer.q_layernorm.weight", 1.0F);
    l.indexer_k_norm = Float(p + "indexer.k_layernorm.weight", 1.0F);
    if (ok && (l.attn_q.rows != 2 * c.AttentionQDim() ||
               l.attn_k.rows != c.AttentionKvDim() ||
               l.attn_v.rows != c.AttentionKvDim() ||
               l.attn_out.cols != c.AttentionQDim() ||
               l.indexer_q.exl3->src_out !=
                   (c.indexer_heads + 1) * c.indexer_head_dim)) {
      Fail("attention projections disagree with the configuration: " + p);
    }
  }

  void Moe(const std::string& p, const Config& c, LayerWeights& l) {
    l.router = Float(p + "gate.weight");
    l.shexp_gate_inp = Float(p + "shared_expert_gate.weight");
    l.ffn_gate_exps =
        Experts(p.substr(0, p.size() - 1), c.num_experts, "gate_proj");
    l.ffn_up_exps =
        Experts(p.substr(0, p.size() - 1), c.num_experts, "up_proj");
    l.ffn_down_exps =
        Experts(p.substr(0, p.size() - 1), c.num_experts, "down_proj");
    l.shexp_gate = Linear(p + "shared_expert.gate_proj");
    l.shexp_up = Linear(p + "shared_expert.up_proj");
    l.shexp_down = Linear(p + "shared_expert.down_proj");
    if (ok && (l.router.rows != c.num_experts ||
               l.ffn_gate_exps.rows != c.expert_ff ||
               l.ffn_down_exps.rows != c.hidden_size ||
               l.shexp_gate.rows != c.shared_expert_ff)) {
      Fail("expert projections disagree with the configuration: " + p);
    }
  }

  /// Runtime value head j is checkpoint head perm[j].
  static std::vector<std::uint32_t> ValueHeadOrder(const Config& c) {
    const std::uint32_t groups = c.ssm_num_k_heads;
    const std::uint32_t per = c.ssm_num_v_heads / c.ssm_num_k_heads;
    std::vector<std::uint32_t> perm(c.ssm_num_v_heads);
    for (std::uint32_t r = 0; r < per; ++r)
      for (std::uint32_t g = 0; g < groups; ++g)
        perm[r * groups + g] = g * per + r;
    return perm;
  }

  void DeltaNet(const std::string& p, const Config& c, LayerWeights& l) {
    const auto heads = ValueHeadOrder(c);
    // Value heads are 128 wide, one Hadamard block each.
    if (c.ssm_head_dim != 128 || c.SsmKeyDim() % 128 != 0) {
      Fail("gated DeltaNet head reorder needs 128-wide heads");
      return;
    }
    const std::uint32_t qk_blocks = 2 * c.SsmKeyDim() / 128;
    std::vector<std::uint32_t> qkv(qk_blocks + heads.size());
    std::iota(qkv.begin(), qkv.begin() + qk_blocks, 0U);
    for (std::size_t j = 0; j < heads.size(); ++j)
      qkv[qk_blocks + j] = qk_blocks + heads[j];
    l.ssm_qkv = Linear(p + "in_proj_qkv", {}, qkv);
    l.ssm_gate = Linear(p + "in_proj_z", {}, heads);
    l.ssm_out = Linear(p + "out_proj", heads, {});

    // Per-head vectors and rows in runtime head order.
    const auto per_head = [&](const std::string& name, bool exp_neg) {
      const PackTensor* t = Need(name);
      if (t == nullptr)
        return TensorRef{};
      const auto v = Floats(*t);
      const std::size_t row = v.size() / heads.size();
      std::vector<float> out(v.size());
      for (std::size_t j = 0; j < heads.size(); ++j) {
        for (std::size_t i = 0; i < row; ++i) {
          const float x = v[heads[j] * row + i];
          out[j * row + i] = exp_neg ? -std::exp(x) : x;
        }
      }
      return F32(t->name, out, row == 1 ? heads.size() : row,
                 row == 1 ? 1 : heads.size());
    };
    l.ssm_a = per_head(p + "A_log", true);
    l.ssm_dt = per_head(p + "dt_bias", false);
    l.ssm_alpha = per_head(p + "in_proj_a.weight", false);
    l.ssm_beta = per_head(p + "in_proj_b.weight", false);
    l.ssm_norm = Float(p + "norm.weight");
    // conv1d [channels, 1, kernel]: q|k channels keep their order, value
    // channels move with their heads.
    if (const PackTensor* t = Need(p + "conv1d.weight")) {
      const auto v = Floats(*t);
      const std::uint32_t kernel = c.ssm_conv_kernel;
      const std::uint32_t channels = c.SsmConvChannels();
      if (v.size() != static_cast<std::size_t>(channels) * kernel) {
        Fail("gated DeltaNet conv1d shape mismatch: " + p);
        return;
      }
      std::vector<float> out(v.size());
      const std::uint32_t qk = 2 * c.SsmKeyDim();
      for (std::uint32_t ch = 0; ch < channels; ++ch) {
        std::uint32_t src = ch;
        if (ch >= qk) {
          const std::uint32_t head = (ch - qk) / c.ssm_head_dim;
          src = qk + heads[head] * c.ssm_head_dim + (ch - qk) % c.ssm_head_dim;
        }
        std::copy_n(v.begin() + static_cast<std::size_t>(src) * kernel, kernel,
                    out.begin() + static_cast<std::size_t>(ch) * kernel);
      }
      l.ssm_conv1d = F32(t->name, out, kernel, channels);
    }
    if (ok && (l.ssm_qkv.rows != c.SsmConvChannels() ||
               l.ssm_gate.rows != c.SsmValueDim() ||
               l.ssm_out.cols != c.SsmValueDim())) {
      Fail("gated DeltaNet projections disagree with the configuration: " + p);
    }
  }

  void Ple(const std::string& p, LayerWeights& l) {
    l.ple_key = Q8(p + "key_proj.weight");
    l.ple_value = Q8(p + "value_proj.weight");
    l.ple_norm_key = Float(p + "norm_key.weight", 1.0F);
    l.ple_norm_query = Float(p + "norm_query.weight", 1.0F);
    l.ple_norm_conv = Float(p + "norm_conv.weight", 1.0F);
    l.ple_conv1d = Float(p + "conv1d.weight");
    if (ok) {
      // [channels, 1, kernel] -> rows of `kernel` taps, as the GGUF stores.
      l.ple_conv1d.cols = 4;
      l.ple_conv1d.rows = l.ple_conv1d.ElementCount() / 4;
    }
  }
};

std::optional<std::vector<std::uint64_t>> I64Values(const PackTensor* t) {
  if (t == nullptr || t->dtype != DType::kI64)
    return std::nullopt;
  std::vector<std::uint64_t> v(t->Elements());
  for (std::size_t i = 0; i < v.size(); ++i) {
    std::int64_t x = 0;
    std::memcpy(&x, static_cast<const std::uint8_t*>(t->data) + 8 * i, 8);
    if (x < 0)
      return std::nullopt;
    v[i] = static_cast<std::uint64_t>(x);
  }
  return v;
}

/// The layer index of the n-gram injection (the layer with ple tensors).
std::int32_t FindPleLayer(const Pack& pack, std::uint32_t layers) {
  for (std::uint32_t i = 0; i < layers; ++i) {
    if (pack.Find(std::string(kTrunk) + "layers." + std::to_string(i) +
                  ".ple.key_proj.weight") != nullptr)
      return static_cast<std::int32_t>(i);
  }
  return -1;
}

std::string NgramPrefix(std::int32_t layer) {
  return std::string(kTrunk) + "layers." + std::to_string(layer) +
         ".ple.ple_embedding.ngram_embedding.";
}

}  // namespace

std::optional<Config> Config::FromExl3(const Pack& pack, bool mtp,
                                       std::string* error_msg) {
  const json::Value& t = pack.TextConfig();
  const auto u32 = [&](const char* key) {
    return static_cast<std::uint32_t>(t.member_size(key));
  };
  Config c;
  c.num_layers = u32("num_hidden_layers");
  c.nextn_layers = mtp ? u32("mtp_num_hidden_layers") : 0;
  c.num_layers_all = c.num_layers + c.nextn_layers;
  c.hidden_size = u32("hidden_size");
  c.vocab_size = u32("vocab_size");
  c.context_length = u32("max_position_embeddings");
  c.rms_eps = static_cast<float>(t.member_double("rms_norm_eps", 1e-6));
  c.hc_count = u32("hc_count");
  c.hc_low_rank = u32("hc_lowrank");
  c.full_attention_interval = u32("full_attention_interval");
  c.num_heads = u32("num_attention_heads");
  c.num_kv_heads = u32("num_key_value_heads");
  c.head_dim = u32("head_dim");
  const json::Value* rope = t.find("rope_parameters");
  const double partial =
      rope != nullptr
          ? rope->member_double("partial_rotary_factor",
                                t.member_double("partial_rotary_factor", 0.0))
          : t.member_double("partial_rotary_factor", 0.0);
  c.rotary_dim = static_cast<std::uint32_t>(std::lround(c.head_dim * partial));
  c.rope_theta = static_cast<float>(
      rope != nullptr ? rope->member_double("rope_theta") : 0.0);
  if (rope != nullptr) {
    if (const json::Value* sections = rope->find("mrope_section");
        sections != nullptr && sections->size() == 3) {
      for (std::size_t i = 0; i < 3; ++i)
        c.rope_sections[i] =
            static_cast<std::uint32_t>(sections->items()[i].as_size());
    }
  }
  c.indexer_heads = u32("indexer_n_heads");
  c.indexer_head_dim = u32("indexer_head_dim");
  c.indexer_top_k = u32("indexer_budget");
  c.compress_ratio = u32("indexer_compress_ratio");
  c.ssm_conv_kernel = u32("linear_conv_kernel_dim");
  c.ssm_head_dim = u32("linear_key_head_dim");
  c.ssm_num_k_heads = u32("linear_num_key_heads");
  c.ssm_num_v_heads = u32("linear_num_value_heads");
  c.ssm_inner_size = c.ssm_num_v_heads * u32("linear_value_head_dim");
  c.num_experts = u32("num_experts");
  c.num_experts_used = u32("num_experts_per_tok");
  c.expert_ff = u32("moe_intermediate_size");
  c.shared_expert_ff = u32("shared_expert_intermediate_size");

  const auto fail = [&](std::string message) -> std::optional<Config> {
    if (error_msg != nullptr)
      *error_msg = std::move(message);
    return std::nullopt;
  };
  if (u32("linear_value_head_dim") != c.ssm_head_dim)
    return fail("gated DeltaNet key and value head widths differ");
  if (const json::Value* types = t.find("layer_types");
      types == nullptr || types->size() != c.num_layers) {
    return fail("config.json layer_types must describe every layer");
  } else {
    for (std::uint32_t i = 0; i < c.num_layers; ++i) {
      const bool full = types->items()[i].str() == "full_attention";
      if (full == c.IsLinearLayer(i))
        return fail("layer_types disagree with full_attention_interval");
    }
  }

  c.ple_layer = FindPleLayer(pack, c.num_layers);
  if (c.ple_layer >= 0) {
    const std::string prefix = NgramPrefix(c.ple_layer);
    c.ple_ngram_size = u32("ngram_size");
    c.ple_heads_per_ngram = u32("heads_per_ngram");
    c.ple_conv_kernel = u32("ple_conv_kernel_size");
    c.ple_eos_token = u32("eos_token_id");
    c.ple_head_dim = exl3::kNgramRowDim;
    if (c.ple_ngram_size < 2 || c.ple_ngram_size > Config::kMaxPleNgram ||
        c.ple_heads_per_ngram == 0 ||
        c.ple_heads_per_ngram > Config::kMaxPleHeads / (c.ple_ngram_size - 1))
      return fail("unsupported PLE n-gram geometry");
    c.ple_heads = (c.ple_ngram_size - 1) * c.ple_heads_per_ngram;
    const auto multipliers = I64Values(pack.Find(prefix + "layer_multipliers"));
    const auto offsets = I64Values(pack.Find(prefix + "head_offsets"));
    const auto vocab = I64Values(pack.Find(prefix + "head_vocab_sizes"));
    if (!multipliers || !offsets || !vocab ||
        multipliers->size() != c.ple_ngram_size ||
        offsets->size() != c.ple_heads || vocab->size() != c.ple_heads)
      return fail("PLE hash tables must match the head count");
    std::copy_n(multipliers->begin(), c.ple_ngram_size,
                c.ple_multipliers.begin());
    for (std::uint32_t h = 0; h < c.ple_heads; ++h) {
      if ((*vocab)[h] == 0 || (*vocab)[h] > UINT32_MAX ||
          (*offsets)[h] > UINT32_MAX - (*vocab)[h])
        return fail("PLE head range does not fit a 32-bit row index");
      c.ple_head_offsets[h] = static_cast<std::uint32_t>((*offsets)[h]);
      c.ple_head_vocab[h] = static_cast<std::uint32_t>((*vocab)[h]);
      c.ple_rows = std::max(c.ple_rows, (*offsets)[h] + (*vocab)[h]);
    }
  }
  return Config::Validated(c, !mtp, error_msg);
}

std::optional<ModelWeights> ModelWeights::BindExl3(const Pack& pack,
                                                   HostTensors& host,
                                                   std::string* error_msg) {
  auto config = Config::FromExl3(pack, false, error_msg);
  if (!config)
    return std::nullopt;
  ModelWeights w;
  w.config = *config;
  const Config& c = w.config;
  Binder b{pack, host, error_msg};
  const std::string trunk(kTrunk);
  w.token_embd = b.Mapped(trunk + "embed_tokens.weight", core::GgmlType::kBF16);
  if (const PackTensor* t = pack.Find(trunk + "embed_tokens.weight");
      t != nullptr && t->dtype != DType::kBF16) {
    b.Fail("the token embedding must be BF16");
  }
  w.output = b.Linear("lm_head");
  w.hc_head = b.Mixer(trunk + "hyper_connection_mixer", false);
  if (b.ok &&
      (w.token_embd.cols != c.hidden_size ||
       w.token_embd.rows != c.vocab_size || w.output.rows != c.vocab_size ||
       w.output.cols != c.hidden_size)) {
    b.Fail("embedding or head shape disagrees with the configuration");
  }
  w.layers.resize(c.num_layers);
  for (std::uint32_t i = 0; i < c.num_layers && b.ok; ++i) {
    LayerWeights& l = w.layers[i];
    const std::string p = trunk + "layers." + std::to_string(i) + ".";
    l.linear = c.IsLinearLayer(i);
    l.hc_attn = b.Mixer(p + "attn_hyper_connection", true);
    l.hc_ffn = b.Mixer(p + "mlp_hyper_connection", true);
    if (l.linear) {
      b.DeltaNet(p + "linear_attn.", c, l);
    } else {
      b.Attention(p + "self_attn.", c, l);
    }
    if (c.IsPleLayer(i))
      b.Ple(p + "ple.", l);
    b.Moe(p + "mlp.", c, l);
  }
  if (c.ple_layer >= 0 && b.ok) {
    const std::string prefix = NgramPrefix(c.ple_layer);
    auto ngram = std::make_shared<Exl3NgramRef>();
    // The table is one tensor or contiguous shards of one width.
    const PackTensor* first = pack.Find(prefix + "trellis");
    std::uint64_t rows = 0;
    std::uint64_t words = 0;
    std::uint64_t next_offset = 0;
    const auto add = [&](const PackTensor* t) {
      if (t->dtype != DType::kI16 || t->shape.size() != 2 ||
          (words != 0 && t->shape[1] != words) ||
          (rows != 0 &&
           (t->file != ngram->file || t->file_offset != next_offset))) {
        b.Fail("n-gram table shards are not one contiguous table");
        return;
      }
      if (rows == 0) {
        ngram->file = t->file;
        ngram->file_offset = t->file_offset;
        words = t->shape[1];
      }
      rows += t->shape[0];
      next_offset = t->file_offset + t->size_bytes;
    };
    if (first != nullptr) {
      add(first);
    } else {
      for (std::uint32_t s = 0;; ++s) {
        const PackTensor* t =
            pack.Find(prefix + "shard_" + std::to_string(s) + ".trellis");
        if (t == nullptr)
          break;
        add(t);
      }
    }
    ngram->rows = rows;
    ngram->k = static_cast<int>((words - 1) * 16 / exl3::kNgramRowDim);
    if (b.ok &&
        (rows < c.ple_rows || words == 0 ||
         exl3::NgramRowWords(ngram->k) != words ||
         (ngram->k != 4 && ngram->k != 5 && ngram->k != 6 && ngram->k != 8))) {
      b.Fail("unsupported n-gram table geometry");
    }
    if (const PackTensor* bias = b.Need(prefix + "head_bias");
        bias != nullptr) {
      if (bias->dtype != DType::kF16 ||
          bias->Elements() !=
              static_cast<std::uint64_t>(c.ple_heads) * c.ple_head_dim) {
        b.Fail("n-gram head bias shape mismatch");
      } else {
        const auto* p = static_cast<const std::uint16_t*>(bias->data);
        ngram->head_bias.assign(p, p + bias->Elements());
      }
    }
    w.ple_exl3 = std::move(ngram);
  }
  if (!b.ok)
    return std::nullopt;
  return w;
}

std::optional<MtpWeights> MtpWeights::BindExl3(const Pack& pack,
                                               const Config& trunk,
                                               HostTensors& host,
                                               std::string* error_msg) {
  if (pack.Find("mtp.fc_hidden.trellis") == nullptr)
    return std::nullopt;
  auto config = Config::FromExl3(pack, true, error_msg);
  if (!config)
    return std::nullopt;
  if (!config->MtpMatches(trunk)) {
    if (error_msg != nullptr)
      *error_msg = "MTP block does not match the trunk";
    return std::nullopt;
  }
  MtpWeights m;
  m.config = *config;
  m.config.ple_layer = -1;
  Binder b{pack, host, error_msg};
  LayerWeights& l = m.block;
  l.linear = false;
  const std::string p = "mtp.layers.0.";
  l.hc_attn = b.Mixer(p + "attn_hyper_connection", true);
  l.hc_ffn = b.Mixer(p + "mlp_hyper_connection", true);
  b.Attention(p + "self_attn.", m.config, l);
  b.Moe(p + "mlp.", m.config, l);
  l.nextn_enorm = b.Float("mtp.pre_fc_norm_embedding.weight", 1.0F);
  l.nextn_hnorm = b.Float("mtp.pre_fc_norm_hidden.weight", 1.0F);
  l.nextn_fc_embedding = b.Linear("mtp.fc_embedding");
  l.nextn_fc_hidden = b.Linear("mtp.fc_hidden");
  l.nextn_head = b.Mixer("mtp.hyper_connection_mixer", false);
  if (!b.ok)
    return std::nullopt;
  return m;
}

}  // namespace gufo::models::qwen38_flash_next
