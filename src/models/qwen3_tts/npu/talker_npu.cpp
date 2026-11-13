#include "src/models/qwen3_tts/npu/talker_npu.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "gufo/ext/experimental/npu.hpp"
#include "src/models/qwen3_tts/loader.hpp"
#include "src/models/qwen3_tts/npu/npu_common.hpp"
#include "src/models/qwen3_tts/prompt.hpp"
#include "src/models/qwen3_tts/sampling.hpp"
#include "src/models/qwen3_tts/tensor.hpp"

namespace gufo::models::qwen3_tts::npu {
namespace {

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

std::string AsciiLower(std::string_view value) {
  std::string result(value);
  std::ranges::transform(result, result.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return result;
}

std::uint32_t SelectMainCode(
    std::span<const float> logits,
    std::span<const std::uint8_t> generated_first_codes,
    std::size_t generation_step, std::uint32_t eos_token,
    const SamplingOptions& sampling, std::mt19937* random,
    std::vector<TokenScore>* scratch) {
  std::vector<TokenScore>& candidates = *scratch;
  candidates.clear();
  candidates.reserve(2049);
  for (std::uint32_t token = 0; token < logits.size(); ++token) {
    if ((token >= 2048 && token != eos_token) ||
        (generation_step < 2 && token == eos_token))
      continue;
    float score = logits[token];
    if (token < generated_first_codes.size() &&
        generated_first_codes[token] != 0)
      score = score < 0.0F ? score * sampling.repetition_penalty
                           : score / sampling.repetition_penalty;
    candidates.push_back({token, score});
  }
  return SampleCodec(candidates, sampling.top_k, sampling.top_p,
                     sampling.temperature, random);
}

// The NPU program for many-row products (prompt prefill, text projection).
struct Engine {
  gufo::npu::Device device;
  gufo::npu::BfpGemm bfp;
};

// A matrix for decode rows (on the CPU) and, when `bfp` >= 0, for many-row
// products on BfpGemm.
struct Linear {
  CpuQ8 decode;
  int bfp{-1};
};

// One stack of Qwen3 decoder layers (the talker or the code predictor) with
// its KV cache.
struct Stack {
  struct Layer {
    Linear qkv, o, gate_up, down;
    std::vector<float> input_norm, post_norm, q_norm, k_norm;
  };
  std::size_t hidden{}, inner{}, heads{}, kv_heads{}, head_dim{}, capacity{};
  float eps{}, theta{};
  std::vector<Layer> layers;
  std::vector<float> final_norm;
  std::vector<float> key_cache, value_cache;  // [layer][kv][capacity][dim]
  std::vector<float> cos_table, sin_table;    // [capacity][dim / 2], BF16

  // Scratch.
  std::vector<float> norm, qkv, attn, proj, gate_up, act;
  double gemm_ms{};
  std::size_t gemm_runs{};

  void Gemm(Engine& engine, const Linear& l, const float* x, std::size_t rows,
            float* y) {
    const auto t0 = std::chrono::steady_clock::now();
    if (rows <= 2 || l.bfp < 0)
      Gemv(l.decode, x, rows, y);
    else
      engine.bfp.Run(l.bfp, x, int(rows), y);
    gemm_ms += std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0)
                   .count();
    ++gemm_runs;
  }

  void Prepare() {
    key_cache.assign(layers.size() * kv_heads * capacity * head_dim, 0.0f);
    value_cache.assign(key_cache.size(), 0.0f);
    const std::size_t half = head_dim / 2;
    cos_table.resize(capacity * half);
    sin_table.resize(capacity * half);
    for (std::size_t p = 0; p < capacity; ++p)
      for (std::size_t d = 0; d < half; ++d) {
        const float frequency =
            1.0F / std::pow(theta, float(2 * d) / float(head_dim));
        const float angle = float(p) * frequency;
        cos_table[p * half + d] = Bf16(std::cos(angle));
        sin_table[p * half + d] = Bf16(std::sin(angle));
      }
  }

  // Normalizes and rotates one head in place, as NormalizeAndRotateHead.
  void NormRotate(float* x, const float* weight, std::size_t position) const {
    float sum = 0;
    for (std::size_t i = 0; i < head_dim; ++i) {
      x[i] = Bf16(x[i]);
      sum += x[i] * x[i];
    }
    const float inv = 1.0F / std::sqrt(sum / float(head_dim) + eps);
    for (std::size_t i = 0; i < head_dim; ++i)
      x[i] = Bf16(Bf16(x[i] * inv) * weight[i]);
    const std::size_t half = head_dim / 2;
    const float* c = cos_table.data() + position * half;
    const float* s = sin_table.data() + position * half;
    for (std::size_t d = 0; d < half; ++d) {
      const float a = x[d], b = x[d + half];
      const float a_cos = Bf16(a * c[d]), b_sin = Bf16(b * s[d]);
      const float b_cos = Bf16(b * c[d]), a_sin = Bf16(a * s[d]);
      x[d] = Bf16(a_cos - b_sin);
      x[d + half] = Bf16(b_cos + a_sin);
    }
  }

  // Runs `rows` rows at positions start.. through every layer. `x` holds the
  // BF16 residual stream [rows][hidden] and ends as the post-final-norm
  // state is left in `norm` (BF16).
  void Forward(Engine& gemm, float* x, std::size_t rows, std::size_t start) {
    if (start + rows > capacity)
      throw std::length_error("Qwen3-TTS NPU context capacity exceeded");
    const std::size_t qw = heads * head_dim, kw = kv_heads * head_dim;
    const std::size_t row_w = qw + 2 * kw;
    norm.resize(rows * hidden);
    qkv.resize(rows * row_w);
    attn.resize(rows * qw);
    proj.resize(rows * hidden);
    gate_up.resize(rows * 2 * inner);
    act.resize(rows * inner);
    ResidualNorm(x, nullptr, norm.data(), rows, hidden,
                 layers.front().input_norm.data(), eps);
    for (std::size_t l = 0; l < layers.size(); ++l) {
      const Layer& w = layers[l];
      Gemm(gemm, w.qkv, norm.data(), rows, qkv.data());
      float* kc = key_cache.data() + l * kv_heads * capacity * head_dim;
      float* vc = value_cache.data() + l * kv_heads * capacity * head_dim;
      // A single decode row is too little work for the thread pool.
      gufo::npu::host::ParallelFor(
          rows * (heads + kv_heads), rows == 1 ? rows * (heads + kv_heads) : 4,
          [&](std::size_t b, std::size_t e) {
            for (std::size_t i = b; i < e; ++i) {
              const std::size_t r = i / (heads + kv_heads),
                                h = i % (heads + kv_heads);
              float* row = qkv.data() + r * row_w;
              if (h < heads) {
                NormRotate(row + h * head_dim, w.q_norm.data(), start + r);
                continue;
              }
              const std::size_t kh = h - heads;
              float* k = row + qw + kh * head_dim;
              float* v = row + qw + kw + kh * head_dim;
              NormRotate(k, w.k_norm.data(), start + r);
              const std::size_t at = (kh * capacity + start + r) * head_dim;
              for (std::size_t d = 0; d < head_dim; ++d) {
                kc[at + d] = k[d];
                vc[at + d] = Bf16(v[d]);
              }
            }
          });
      Attend(kc, vc, rows, start);
      Gemm(gemm, w.o, attn.data(), rows, proj.data());
      ResidualNorm(x, proj.data(), norm.data(), rows, hidden,
                   w.post_norm.data(), eps);
      Gemm(gemm, w.gate_up, norm.data(), rows, gate_up.data());
      for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t i = 0; i < inner; ++i) {
          const float g = Bf16(gate_up[r * 2 * inner + i]);
          const float u = Bf16(gate_up[r * 2 * inner + inner + i]);
          const float a = Bf16(g / (1.0F + std::exp(-g)));
          act[r * inner + i] = Bf16(a * u);
        }
      Gemm(gemm, w.down, act.data(), rows, proj.data());
      ResidualNorm(x, proj.data(), norm.data(), rows, hidden,
                   l + 1 < layers.size() ? layers[l + 1].input_norm.data()
                                         : final_norm.data(),
                   eps);
    }
  }

  // Causal attention of the rows over the cache, as Bfloat16AttentionKernel.
  void Attend(const float* kc, const float* vc, std::size_t rows,
              std::size_t start) {
    const std::size_t group = heads / kv_heads;
    const float scale = 1.0F / std::sqrt(float(head_dim));
    const std::size_t qw = heads * head_dim,
                      row_w = qw + 2 * kv_heads * head_dim;
    const std::size_t work = rows * heads * (start + rows) * head_dim;
    gufo::npu::host::ParallelFor(
        rows * heads, work < (std::size_t(1) << 18) ? rows * heads : 1,
        [&](std::size_t b, std::size_t e) {
          std::vector<float> scores;
          std::vector<float> acc(head_dim);
          for (std::size_t i = b; i < e; ++i) {
            const std::size_t r = i / heads, h = i % heads, kh = h / group;
            const float* q = qkv.data() + r * row_w + h * head_dim;
            const std::size_t visible = start + r + 1;
            const float* keys = kc + kh * capacity * head_dim;
            const float* values = vc + kh * capacity * head_dim;
            scores.resize(visible);
            float top = -INFINITY;
            for (std::size_t p = 0; p < visible; ++p) {
              const float dot =
                  gufo::npu::host::Dot(q, keys + p * head_dim, head_dim);
              scores[p] = Bf16(Bf16(dot) * scale);
              top = std::max(top, scores[p]);
            }
            float total = 0;
            for (float& s : scores) {
              s = std::exp(s - top);
              total += s;
            }
            const float inverse = 1.0F / total;
            std::fill(acc.begin(), acc.end(), 0.0F);
            for (std::size_t p = 0; p < visible; ++p)
              gufo::npu::host::Axpy(Bf16(scores[p] * inverse),
                                    values + p * head_dim, acc.data(),
                                    head_dim);
            float* out = attn.data() + r * qw + h * head_dim;
            for (std::size_t d = 0; d < head_dim; ++d)
              out[d] = Bf16(acc[d]);
          }
        });
  }
};

}  // namespace

struct TalkerNpuRuntime::Impl {
  LoadResult model;
  Engine engine;
  std::size_t maximum_tokens{};
  std::size_t talker_tokens{};
  Stack talker, predictor;
  // [codec_head; small_to_mtp_projection] on the talker output.
  CpuQ8 head;
  int text_fc1{}, text_fc2{};
  std::vector<float> text_fc1_bias, text_fc2_bias, projection_bias;
  std::vector<CpuQ8> predictor_heads;
  const std::uint16_t* text_embedding{};
  const std::uint16_t* codec_embedding{};
  std::vector<const std::uint16_t*> predictor_embeddings;
  // Projected predictor inputs: row c of table g is small_to_mtp_projection
  // of codec embedding c (g = 0: talker codec; g >= 1: predictor table g -
  // 1), BF16 as the GPU path rounds them. [table][code][predictor hidden].
  std::vector<float> projected;
  std::vector<std::uint8_t> generated_first_code_flags;
  std::vector<TokenScore> candidate_scratch;
  std::mt19937* sampling_rng{nullptr};
  std::size_t predictor_top_k{50};
  float predictor_temperature{0.9F};
  float predictor_top_p{1.0F};
  bool profile{std::getenv("GUFO_NPU_PROFILE") != nullptr};
  double talker_ms{}, predictor_ms{};
  // The talker's decode steps on the fused NPU program (one dispatch per
  // token) when its shape fits; CPU products otherwise.
  gufo::npu::Qwen3Decoder fused;
  bool skip_talker_decode{};

  const Tensor& Find(const std::string& name) const {
    const Tensor* t = model.store->Find(name);
    if (t == nullptr || t->dtype != DType::kBF16)
      throw std::runtime_error("Qwen3-TTS NPU: missing BF16 tensor " + name);
    return *t;
  }
  std::vector<float> Vector(const std::string& name) const {
    return Bf16Vector(Find(name));
  }
  std::vector<const Tensor*> Parts(std::initializer_list<std::string> names) {
    std::vector<const Tensor*> parts;
    for (const std::string& name : names)
      parts.push_back(&Find(name));
    return parts;
  }
  // A layer matrix: decode rows on the CPU; with `prefill`, also on BfpGemm.
  Linear Load(std::initializer_list<std::string> names, bool prefill) {
    const auto parts = Parts(names);
    Linear l;
    if (!(prefill && skip_talker_decode))
      l.decode = MakeCpuQ8(parts);
    if (prefill)
      l.bfp = UploadBfp(engine.bfp, parts);
    return l;
  }

  // The talker prefills prompts on BfpGemm; the predictor's small matrices
  // stay on the CPU.
  void LoadStack(Stack* s, const std::string& prefix, std::size_t count,
                 bool talker_stack) {
    const bool prefill = talker_stack;
    for (std::size_t l = 0; l < count; ++l) {
      const std::string p = prefix + std::to_string(l) + ".";
      Stack::Layer layer;
      layer.qkv =
          Load({p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight",
                p + "self_attn.v_proj.weight"},
               prefill);
      layer.o = Load({p + "self_attn.o_proj.weight"}, prefill);
      layer.gate_up =
          Load({p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"}, prefill);
      layer.down = Load({p + "mlp.down_proj.weight"}, prefill);
      layer.input_norm = Vector(p + "input_layernorm.weight");
      layer.post_norm = Vector(p + "post_attention_layernorm.weight");
      layer.q_norm = Vector(p + "self_attn.q_norm.weight");
      layer.k_norm = Vector(p + "self_attn.k_norm.weight");
      s->layers.push_back(std::move(layer));
    }
    s->Prepare();
  }

  template<typename Config>
  void OpenFused(const Config& tc) {
    using Fused = gufo::npu::Qwen3Decoder;
    const char* env = std::getenv("GUFO_NPU_FUSED");
    if (!(env && std::string(env) == "1"))
      return;
    if (tc.num_hidden_layers != std::uint32_t(Fused::kLayers) ||
        tc.hidden_size != std::uint32_t(Fused::kHidden) ||
        tc.num_attention_heads != std::uint32_t(Fused::kHeads) ||
        tc.num_key_value_heads != std::uint32_t(Fused::kKvHeads) ||
        tc.head_dim != std::uint32_t(Fused::kHeadDim) ||
        tc.intermediate_size != std::uint32_t(Fused::kFfn))
      return;
    try {
      fused = Fused::Open(engine.device, gufo::npu::FindDesign("qwen3dec"));
      if (std::size_t(fused.capacity()) < maximum_tokens)
        fused = Fused();
    } catch (const gufo::npu::Error&) {
      fused = Fused();
    }
  }

  template<typename Config>
  void LoadFused(const Config& tc) {
    auto bf = [&](const std::string& n) {
      return reinterpret_cast<const std::uint16_t*>(Find(n).data);
    };
    for (std::uint32_t l = 0; l < tc.num_hidden_layers; ++l) {
      const std::string p = "talker.model.layers." + std::to_string(l) + ".";
      const Stack::Layer& w = talker.layers[l];
      fused.SetLayer(
          int(l),
          {bf(p + "self_attn.q_proj.weight"), bf(p + "self_attn.k_proj.weight"),
           bf(p + "self_attn.v_proj.weight"), bf(p + "self_attn.o_proj.weight"),
           bf(p + "mlp.gate_proj.weight"), bf(p + "mlp.up_proj.weight"),
           bf(p + "mlp.down_proj.weight"), w.input_norm.data(),
           w.post_norm.data(), w.q_norm.data(), w.k_norm.data()});
    }
    // One head for the codec logits and the predictor's input projection.
    const Tensor& codec = Find("talker.codec_head.weight");
    const Tensor& proj =
        Find("talker.code_predictor.small_to_mtp_projection.weight");
    std::vector<std::uint16_t> rows(codec.byte_count / 2 + proj.byte_count / 2);
    std::memcpy(rows.data(), codec.data, codec.byte_count);
    std::memcpy(rows.data() + codec.byte_count / 2, proj.data, proj.byte_count);
    fused.SetHead(rows.data(), int(rows.size() / tc.hidden_size),
                  talker.final_norm.data());
    fused.SetRopeTheta(tc.rope_theta);
  }

  // The fused program's cache gets the CPU prefill's entries of positions
  // [start, start + rows).
  void HandOff(std::size_t start, std::size_t rows) {
    const std::size_t kvh = talker.kv_heads, hd = talker.head_dim,
                      cap = talker.capacity, kw = kvh * hd;
    std::vector<float> k(rows * kw), v(rows * kw);
    for (std::size_t l = 0; l < talker.layers.size(); ++l) {
      const float* kc = talker.key_cache.data() + l * kvh * cap * hd;
      const float* vc = talker.value_cache.data() + l * kvh * cap * hd;
      for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t h = 0; h < kvh; ++h) {
          std::memcpy(k.data() + r * kw + h * hd,
                      kc + (h * cap + start + r) * hd, hd * 4);
          std::memcpy(v.data() + r * kw + h * hd,
                      vc + (h * cap + start + r) * hd, hd * 4);
        }
      fused.WriteKv(int(l), int(start), int(rows), k.data(), kw, v.data(), kw);
    }
  }

  void Load(const std::string& model_root) {
    model = LoadModelDirectory(model_root);
    if (!model.ok)
      throw std::runtime_error(model.error);
    const auto& tc = model.config.talker;
    const auto& pc = model.config.code_predictor;
    engine.device = gufo::npu::Device::Open();
    engine.bfp = gufo::npu::BfpGemm::Open(engine.device,
                                          gufo::npu::FindDesign("bfpgemm"),
                                          std::size_t(1900) << 20);
    talker.hidden = tc.hidden_size;
    talker.inner = tc.intermediate_size;
    talker.heads = tc.num_attention_heads;
    talker.kv_heads = tc.num_key_value_heads;
    talker.head_dim = tc.head_dim;
    talker.capacity = maximum_tokens;
    talker.eps = tc.rms_norm_eps;
    talker.theta = tc.rope_theta;
    talker.final_norm = Vector("talker.model.norm.weight");
    OpenFused(tc);
    skip_talker_decode = bool(fused);
    LoadStack(&talker, "talker.model.layers.", tc.num_hidden_layers, true);
    skip_talker_decode = false;
    if (fused)
      LoadFused(tc);
    predictor.hidden = pc.hidden_size;
    predictor.inner = pc.intermediate_size;
    predictor.heads = pc.num_attention_heads;
    predictor.kv_heads = pc.num_key_value_heads;
    predictor.head_dim = pc.head_dim;
    predictor.capacity = tc.num_code_groups;
    predictor.eps = pc.rms_norm_eps;
    predictor.theta = pc.rope_theta;
    predictor.final_norm = Vector("talker.code_predictor.model.norm.weight");
    LoadStack(&predictor, "talker.code_predictor.model.layers.",
              pc.num_hidden_layers, false);
    head = MakeCpuQ8(
        Parts({"talker.codec_head.weight",
               "talker.code_predictor.small_to_mtp_projection.weight"}));
    projection_bias =
        Vector("talker.code_predictor.small_to_mtp_projection.bias");
    text_fc1 = UploadBfp(engine.bfp,
                         Parts({"talker.text_projection.linear_fc1.weight"}));
    text_fc2 = UploadBfp(engine.bfp,
                         Parts({"talker.text_projection.linear_fc2.weight"}));
    text_fc1_bias = Vector("talker.text_projection.linear_fc1.bias");
    text_fc2_bias = Vector("talker.text_projection.linear_fc2.bias");
    for (std::uint32_t g = 0; g + 1 < tc.num_code_groups; ++g)
      predictor_heads.push_back(MakeCpuQ8(Parts(
          {"talker.code_predictor.lm_head." + std::to_string(g) + ".weight"})));
    text_embedding = reinterpret_cast<const std::uint16_t*>(
        Find("talker.model.text_embedding.weight").data);
    codec_embedding = reinterpret_cast<const std::uint16_t*>(
        Find("talker.model.codec_embedding.weight").data);
    for (std::uint32_t g = 0; g + 1 < tc.num_code_groups; ++g)
      predictor_embeddings.push_back(reinterpret_cast<const std::uint16_t*>(
          Find("talker.code_predictor.model.codec_embedding." +
               std::to_string(g) + ".weight")
              .data));
    ProjectTables();
  }

  // Precomputes the projected predictor input of every code.
  void ProjectTables() {
    const std::size_t th = talker.hidden, ph = predictor.hidden,
                      codes = model.config.code_predictor.vocab_size;
    std::size_t pn = 0, pk = 0;
    const std::vector<float> projection_rows = ConcatRows(
        Parts({"talker.code_predictor.small_to_mtp_projection.weight"}), &pn,
        &pk);
    const CpuQ8 projection = QuantizeQ8(projection_rows, pn, pk);
    const std::size_t tables = predictor_embeddings.size();  // 15: 0 + 14 used
    projected.resize(tables * codes * ph);
    std::vector<float> rows(codes * th);
    for (std::size_t t = 0; t < tables; ++t) {
      const std::uint16_t* table =
          t == 0 ? codec_embedding : predictor_embeddings[t - 1];
      gufo::npu::host::Bf16ToFloat(table, rows.data(), codes * th);
      float* out = projected.data() + t * codes * ph;
      gufo::npu::host::ParallelFor(codes, 8, [&](std::size_t b, std::size_t e) {
        for (std::size_t c = b; c < e; ++c)
          projection.Rows(rows.data() + c * th, out + c * ph, 0, ph);
      });
      for (std::size_t c = 0; c < codes; ++c)
        for (std::size_t i = 0; i < ph; ++i)
          out[c * ph + i] = Bf16(out[c * ph + i] + projection_bias[i]);
    }
  }

  std::vector<float> ProjectText(std::span<const std::uint32_t> ids) {
    const auto& tc = model.config.talker;
    if (ids.empty())
      return {};
    for (std::uint32_t id : ids)
      if (id >= tc.text_vocab_size)
        throw std::out_of_range("Qwen3-TTS text token is out of range");
    const std::size_t rows = ids.size(), th = tc.text_hidden_size,
                      h = tc.hidden_size;
    std::vector<float> x(rows * th), y(rows * th), out(rows * h);
    for (std::size_t r = 0; r < rows; ++r)
      gufo::npu::host::Bf16ToFloat(text_embedding + std::size_t(ids[r]) * th,
                                   x.data() + r * th, th);
    engine.bfp.Run(text_fc1, x.data(), int(rows), y.data());
    for (std::size_t r = 0; r < rows; ++r)
      for (std::size_t i = 0; i < th; ++i) {
        const float biased = Bf16(y[r * th + i] + text_fc1_bias[i]);
        x[r * th + i] = Bf16(biased / (1.0F + std::exp(-biased)));
      }
    engine.bfp.Run(text_fc2, x.data(), int(rows), out.data());
    for (std::size_t r = 0; r < rows; ++r)
      for (std::size_t i = 0; i < h; ++i)
        out[r * h + i] = Bf16(out[r * h + i] + text_fc2_bias[i]);
    return out;
  }

  std::vector<float> LookupCodec(std::span<const std::uint32_t> ids) const {
    const auto& tc = model.config.talker;
    std::vector<float> out(ids.size() * tc.hidden_size);
    for (std::size_t r = 0; r < ids.size(); ++r) {
      if (ids[r] >= tc.vocab_size)
        throw std::out_of_range("Qwen3-TTS codec token is out of range");
      gufo::npu::host::Bf16ToFloat(
          codec_embedding + std::size_t(ids[r]) * tc.hidden_size,
          out.data() + r * tc.hidden_size, tc.hidden_size);
    }
    return out;
  }

  // Sum of a frame's 16 codec embeddings, plus text when given, as
  // BatchedCodecEmbeddingSumKernel.
  void FrameEmbedding(std::span<const std::uint32_t> codes, const float* text,
                      float* out) const {
    const std::size_t h = model.config.talker.hidden_size;
    for (std::size_t d = 0; d < h; ++d) {
      float sum = 0;
      for (std::size_t g = 0; g < codes.size(); ++g) {
        const std::uint16_t* table =
            g == 0 ? codec_embedding : predictor_embeddings[g - 1];
        sum +=
            gufo::npu::host::Bf16ToFloat(table[std::size_t(codes[g]) * h + d]);
      }
      if (text != nullptr) {
        sum = Bf16(sum);
        sum += text[d];
      }
      out[d] = Bf16(sum);
    }
  }

  std::vector<float> ReferenceCodecEmbeddings(
      std::span<const std::uint32_t> codes, std::size_t frames) const {
    const auto& tc = model.config.talker;
    std::vector<float> out(frames * tc.hidden_size);
    for (std::size_t f = 0; f < frames; ++f)
      FrameEmbedding(codes.subspan(f * tc.num_code_groups, tc.num_code_groups),
                     nullptr, out.data() + f * tc.hidden_size);
    return out;
  }

  // Runs the talker over `rows` input embeddings; returns the codec logits
  // and the final hidden state of the last row.
  void TalkerStep(std::vector<float> x, std::size_t rows,
                  std::vector<float>* logits, std::vector<float>* last_hidden,
                  std::vector<float>* projected_hidden) {
    const auto t0 = std::chrono::steady_clock::now();
    const std::size_t h = talker.hidden;
    const std::size_t vocab = model.config.talker.vocab_size,
                      ph = predictor.hidden;
    std::vector<float> out(vocab + ph);
    if (fused) {
      // Prefill all but the last row on BfpGemm (when there are enough rows
      // for it), then each remaining row on the fused program.
      std::size_t done = 0;
      if (rows > 3) {
        talker.Forward(engine, x.data(), rows - 1, talker_tokens);
        HandOff(talker_tokens, rows - 1);
        talker_tokens += rows - 1;
        done = rows - 1;
      }
      for (std::size_t r = done; r < rows; ++r) {
        fused.Start(x.data() + r * h, int(talker_tokens++));
        fused.Wait();
      }
      std::vector<float> hidden(h);
      fused.Hidden(hidden.data());
      last_hidden->resize(h);
      gufo::npu::host::RmsNormRows(hidden.data(), last_hidden->data(), 1, h,
                                   talker.final_norm.data(), talker.eps);
      for (float& v : *last_hidden)
        v = Bf16(v);
      const auto lg = fused.logits();
      std::copy(lg.begin(), lg.begin() + std::ptrdiff_t(vocab + ph),
                out.begin());
    } else {
      talker.Forward(engine, x.data(), rows, talker_tokens);
      talker_tokens += rows;
      last_hidden->assign(talker.norm.end() - std::ptrdiff_t(h),
                          talker.norm.end());
      Gemv(head, last_hidden->data(), 1, out.data());
    }
    logits->resize(vocab);
    for (std::size_t i = 0; i < vocab; ++i)
      (*logits)[i] = Bf16(out[i]);
    projected_hidden->resize(ph);
    for (std::size_t i = 0; i < ph; ++i)
      (*projected_hidden)[i] = Bf16(out[vocab + i] + projection_bias[i]);
    talker_ms += std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - t0)
                     .count();
  }

  // Expands one first code into the frame's 16 codes.
  std::vector<std::uint32_t> PredictFrame(const std::vector<float>& projected0,
                                          std::uint32_t first_code) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto& pc = model.config.code_predictor;
    const std::size_t ph = predictor.hidden, codes = pc.vocab_size;
    std::vector<std::uint32_t> frame{first_code};
    std::vector<float> x(2 * ph);
    std::copy(projected0.begin(), projected0.end(), x.begin());
    std::copy_n(projected.data() + std::size_t(first_code) * ph, ph,
                x.begin() + std::ptrdiff_t(ph));
    std::vector<float> logits(codes);
    std::size_t rows = 2, position = 0;
    for (std::size_t g = 0; g < predictor_heads.size(); ++g) {
      predictor.Forward(engine, x.data(), rows, position);
      position += rows;
      Gemv(predictor_heads[g], predictor.norm.data() + (rows - 1) * ph, 1,
           logits.data());
      candidate_scratch.clear();
      for (std::uint32_t t = 0; t < codes; ++t)
        candidate_scratch.push_back({t, Bf16(logits[t])});
      const std::uint32_t code =
          sampling_rng != nullptr
              ? SampleCodec(candidate_scratch, predictor_top_k, predictor_top_p,
                            predictor_temperature, sampling_rng)
              : SampleCodec(candidate_scratch, 0, 1.0F, 1.0F, nullptr);
      frame.push_back(code);
      if (g + 1 < predictor_heads.size()) {
        rows = 1;
        x.assign(projected.data() + ((g + 1) * codes + code) * ph,
                 projected.data() + ((g + 1) * codes + code + 1) * ph);
      }
    }
    predictor_ms += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    return frame;
  }
};

TalkerNpuRuntime::TalkerNpuRuntime(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
TalkerNpuRuntime::~TalkerNpuRuntime() = default;

std::unique_ptr<TalkerNpuRuntime> TalkerNpuRuntime::Create(
    const std::string& model_root, std::size_t maximum_tokens,
    std::string* error) {
  if (maximum_tokens == 0) {
    SetError(error, "Qwen3-TTS NPU talker capacity must be positive");
    return nullptr;
  }
  auto impl = std::make_unique<Impl>();
  impl->maximum_tokens = maximum_tokens;
  try {
    impl->Load(model_root);
  } catch (const std::exception& e) {
    SetError(error, std::string("Qwen3-TTS NPU talker: ") + e.what());
    return nullptr;
  }
  return std::unique_ptr<TalkerNpuRuntime>(
      new TalkerNpuRuntime(std::move(impl)));
}

bool TalkerNpuRuntime::BuildCustomVoicePrompt(
    std::span<const std::uint32_t> input_ids,
    std::span<const std::uint32_t> instruction_ids, std::string_view speaker,
    std::string_view language, hip::CustomVoicePromptOutput* output,
    std::string* error) {
  return BuildConditionedPrompt(input_ids, instruction_ids, speaker, language,
                                true, output, error);
}

bool TalkerNpuRuntime::BuildVoiceDesignPrompt(
    std::span<const std::uint32_t> input_ids,
    std::span<const std::uint32_t> instruction_ids, std::string_view language,
    hip::TalkerPromptOutput* output, std::string* error) {
  return BuildConditionedPrompt(input_ids, instruction_ids, {}, language, false,
                                output, error);
}

// As TalkerHipRuntime::BuildConditionedPrompt.
bool TalkerNpuRuntime::BuildConditionedPrompt(
    std::span<const std::uint32_t> input_ids,
    std::span<const std::uint32_t> instruction_ids, std::string_view speaker,
    std::string_view language, bool include_speaker,
    hip::TalkerPromptOutput* output, std::string* error) {
  if (output == nullptr) {
    SetError(error, "Qwen3-TTS prompt output must not be null");
    return false;
  }
  *output = {};
  const auto& config = impl_->model.config;
  const auto& talker = config.talker;
  if (input_ids.size() < 8) {
    SetError(error, "Qwen3-TTS assistant prompt is too short");
    return false;
  }
  try {
    std::string normalized_language = AsciiLower(language);
    if (normalized_language.empty())
      normalized_language = "auto";
    std::optional<std::uint32_t> speaker_token;
    if (include_speaker) {
      const std::string normalized_speaker = AsciiLower(speaker);
      const auto speaker_id = talker.spk_id.find(normalized_speaker);
      if (speaker_id == talker.spk_id.end())
        throw std::invalid_argument("unsupported Qwen3-TTS speaker: " +
                                    std::string(speaker));
      speaker_token = speaker_id->second;
      if (normalized_language == "chinese" || normalized_language == "auto") {
        const auto dialect = talker.spk_dialect.find(normalized_speaker);
        if (dialect != talker.spk_dialect.end() && dialect->second.has_value())
          normalized_language = *dialect->second;
      }
    }
    std::vector<std::uint32_t> codec_ids;
    if (normalized_language == "auto") {
      codec_ids = {talker.codec_nothink_id, talker.codec_think_bos_id,
                   talker.codec_think_eos_id};
    } else {
      const auto language_id =
          talker.codec_language_id.find(normalized_language);
      if (language_id == talker.codec_language_id.end())
        throw std::invalid_argument("unsupported Qwen3-TTS language: " +
                                    std::string(language));
      codec_ids = {talker.codec_think_id, talker.codec_think_bos_id,
                   language_id->second, talker.codec_think_eos_id};
    }
    if (speaker_token.has_value())
      codec_ids.push_back(*speaker_token);
    codec_ids.push_back(talker.codec_pad_id);
    codec_ids.push_back(talker.codec_bos_id);
    const std::array<std::uint32_t, 3> special_ids{config.tts_bos_token_id,
                                                   config.tts_eos_token_id,
                                                   config.tts_pad_token_id};
    const std::vector<float> special = impl_->ProjectText(special_ids);
    const std::size_t hidden = talker.hidden_size;
    const std::span<const float> tts_bos(special.data(), hidden);
    const std::span<const float> tts_eos(special.data() + hidden, hidden);
    const std::span<const float> tts_pad(special.data() + 2 * hidden, hidden);
    std::vector<float> instruction;
    if (!instruction_ids.empty())
      instruction = impl_->ProjectText(instruction_ids);
    const std::vector<float> role = impl_->ProjectText(input_ids.first(3));
    const std::vector<float> codec = impl_->LookupCodec(codec_ids);
    const auto text_ids = input_ids.subspan(3, input_ids.size() - 8);
    const std::vector<float> text = impl_->ProjectText(text_ids);
    const std::array<std::uint32_t, 1> codec_pad{talker.codec_pad_id};
    const std::vector<float> codec_pad_embedding =
        impl_->LookupCodec(codec_pad);
    const NonIclPromptInput prompt_input{
        .hidden_size = hidden,
        .instruction = instruction,
        .role = role,
        .codec = codec,
        .text = text,
        .tts_bos = tts_bos,
        .tts_eos = tts_eos,
        .tts_pad = tts_pad,
        .codec_pad = codec_pad_embedding,
    };
    if (!BuildNonIclPrompt(prompt_input, output, error))
      return false;
    if (output->tokens == 0 || output->tokens > impl_->maximum_tokens)
      throw std::length_error("Qwen3-TTS prompt exceeds runtime capacity");
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    *output = {};
    return false;
  }
}

// As TalkerHipRuntime::BuildVoiceClonePrompt.
bool TalkerNpuRuntime::BuildVoiceClonePrompt(
    const hip::VoiceClonePromptInput& input, hip::TalkerPromptOutput* output,
    std::string* error) {
  if (output == nullptr) {
    SetError(error, "Qwen3-TTS prompt output must not be null");
    return false;
  }
  *output = {};
  const auto& config = impl_->model.config;
  const auto& talker = config.talker;
  if (input.input_ids.size() < 8) {
    SetError(error, "Qwen3-TTS assistant prompt is too short");
    return false;
  }
  if (input.speaker_embedding.size() != talker.hidden_size) {
    SetError(error, "Qwen3-TTS speaker embedding shape is invalid");
    return false;
  }
  if (input.icl_mode &&
      (input.reference_ids.size() < 5 || input.reference_frames == 0 ||
       input.reference_codes.size() !=
           input.reference_frames * talker.num_code_groups)) {
    SetError(error, "Qwen3-TTS reference prompt is invalid");
    return false;
  }
  try {
    std::string normalized_language = AsciiLower(input.language);
    if (normalized_language.empty())
      normalized_language = "auto";
    std::vector<std::uint32_t> codec_ids;
    if (normalized_language == "auto") {
      codec_ids = {talker.codec_nothink_id, talker.codec_think_bos_id,
                   talker.codec_think_eos_id};
    } else {
      const auto language_id =
          talker.codec_language_id.find(normalized_language);
      if (language_id == talker.codec_language_id.end())
        throw std::invalid_argument("unsupported Qwen3-TTS language: " +
                                    std::string(input.language));
      codec_ids = {talker.codec_think_id, talker.codec_think_bos_id,
                   language_id->second, talker.codec_think_eos_id};
    }
    const std::array<std::uint32_t, 3> special_ids{config.tts_bos_token_id,
                                                   config.tts_eos_token_id,
                                                   config.tts_pad_token_id};
    const std::vector<float> special = impl_->ProjectText(special_ids);
    const std::size_t hidden = talker.hidden_size;
    const std::span<const float> tts_bos(special.data(), hidden);
    const std::span<const float> tts_eos(special.data() + hidden, hidden);
    const std::span<const float> tts_pad(special.data() + 2 * hidden, hidden);
    const std::vector<float> role =
        impl_->ProjectText(input.input_ids.first(3));
    std::vector<float> codec = impl_->LookupCodec(codec_ids);
    codec.insert(codec.end(), input.speaker_embedding.begin(),
                 input.speaker_embedding.end());
    const std::array<std::uint32_t, 2> codec_suffix{talker.codec_pad_id,
                                                    talker.codec_bos_id};
    const std::vector<float> suffix = impl_->LookupCodec(codec_suffix);
    codec.insert(codec.end(), suffix.begin(), suffix.end());
    const auto target_ids =
        input.input_ids.subspan(3, input.input_ids.size() - 8);
    if (!input.icl_mode) {
      const std::vector<float> text = impl_->ProjectText(target_ids);
      const std::array<std::uint32_t, 1> codec_pad{talker.codec_pad_id};
      const std::vector<float> codec_pad_embedding =
          impl_->LookupCodec(codec_pad);
      const NonIclPromptInput prompt_input{
          .hidden_size = hidden,
          .instruction = {},
          .role = role,
          .codec = codec,
          .text = text,
          .tts_bos = tts_bos,
          .tts_eos = tts_eos,
          .tts_pad = tts_pad,
          .codec_pad = codec_pad_embedding,
      };
      if (!BuildNonIclPrompt(prompt_input, output, error))
        return false;
    } else {
      const auto reference_text_ids =
          input.reference_ids.subspan(3, input.reference_ids.size() - 5);
      std::vector<std::uint32_t> combined_ids(reference_text_ids.begin(),
                                              reference_text_ids.end());
      combined_ids.insert(combined_ids.end(), target_ids.begin(),
                          target_ids.end());
      const std::vector<float> combined_text = impl_->ProjectText(combined_ids);
      const std::array<std::uint32_t, 1> reference_bos{talker.codec_bos_id};
      std::vector<float> reference_codec = impl_->LookupCodec(reference_bos);
      for (std::size_t f = 0; f < input.reference_frames; ++f) {
        const auto codes = input.reference_codes.subspan(
            f * talker.num_code_groups, talker.num_code_groups);
        for (std::size_t g = 0; g < codes.size(); ++g) {
          const std::uint32_t vocabulary =
              g == 0 ? talker.vocab_size : config.code_predictor.vocab_size;
          if (codes[g] >= vocabulary)
            throw std::out_of_range(
                "Qwen3-TTS reference codec token is out of range");
        }
      }
      const std::vector<float> frames = impl_->ReferenceCodecEmbeddings(
          input.reference_codes, input.reference_frames);
      reference_codec.insert(reference_codec.end(), frames.begin(),
                             frames.end());
      const IclPromptInput prompt_input{
          .hidden_size = hidden,
          .role = role,
          .codec = codec,
          .combined_text = combined_text,
          .reference_codec = reference_codec,
          .tts_bos = tts_bos,
          .tts_eos = tts_eos,
          .tts_pad = tts_pad,
      };
      if (!BuildIclPrompt(prompt_input, output, error))
        return false;
    }
    if (output->tokens == 0 || output->tokens > impl_->maximum_tokens)
      throw std::length_error("Qwen3-TTS prompt exceeds runtime capacity");
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    *output = {};
    return false;
  }
}

// As TalkerHipRuntime::Generate.
bool TalkerNpuRuntime::Generate(
    const hip::TalkerPromptOutput& prompt, std::size_t maximum_new_tokens,
    const SamplingOptions& sampling, const CancellationCheck& is_cancelled,
    hip::TalkerGenerationOutput* output, std::string* error,
    const std::function<bool(const hip::TalkerGenerationOutput&)>& on_codes) {
  if (output == nullptr) {
    SetError(error, "Qwen3-TTS generation output must not be null");
    return false;
  }
  *output = {};
  Impl& m = *impl_;
  const auto& config = m.model.config.talker;
  if (prompt.tokens == 0 ||
      prompt.embeddings.size() != prompt.tokens * config.hidden_size ||
      prompt.tts_pad.size() != config.hidden_size ||
      prompt.trailing_text.empty() ||
      prompt.trailing_text.size() % config.hidden_size != 0 ||
      maximum_new_tokens == 0) {
    SetError(error, "Qwen3-TTS generation prompt is invalid");
    return false;
  }
  if (!ValidSamplingOptions(sampling)) {
    SetError(error, "Qwen3-TTS sampling options are invalid");
    return false;
  }
  if (prompt.tokens + maximum_new_tokens > m.maximum_tokens + 1) {
    SetError(error, "Qwen3-TTS prompt and frames exceed the context");
    return false;
  }
  std::mt19937 random(sampling.seed);
  m.sampling_rng = sampling.predictor_sample ? &random : nullptr;
  m.predictor_top_k = sampling.predictor_top_k;
  m.predictor_temperature = sampling.predictor_temperature;
  m.predictor_top_p = sampling.predictor_top_p;
  struct Reset {
    Impl* impl;
    ~Reset() { impl->sampling_rng = nullptr; }
  } const reset{&m};
  m.talker_ms = m.predictor_ms = 0;
  m.talker.gemm_ms = m.predictor.gemm_ms = 0;
  m.talker.gemm_runs = m.predictor.gemm_runs = 0;
  const auto t0 = std::chrono::steady_clock::now();
  try {
    const auto cancelled = [&] {
      if (!is_cancelled || !is_cancelled())
        return false;
      SetError(error, "Qwen3-TTS native generation cancelled");
      *output = {};
      return true;
    };
    if (cancelled())
      return false;
    m.talker_tokens = 0;
    std::vector<float> logits, last_hidden, projected0;
    m.TalkerStep(prompt.embeddings, prompt.tokens, &logits, &last_hidden,
                 &projected0);
    const double prefill_ms = m.talker_ms;
    output->code_groups = config.num_code_groups;
    output->codes.reserve(maximum_new_tokens * config.num_code_groups);
    m.generated_first_code_flags.assign(config.vocab_size, 0);
    const std::size_t trailing_rows =
        prompt.trailing_text.size() / config.hidden_size;
    std::vector<float> embedding(config.hidden_size);
    for (std::size_t step = 0; step < maximum_new_tokens; ++step) {
      if (cancelled())
        return false;
      const std::uint32_t first_code = SelectMainCode(
          logits, m.generated_first_code_flags, step, config.codec_eos_token_id,
          sampling, sampling.sample ? &random : nullptr, &m.candidate_scratch);
      if (first_code == config.codec_eos_token_id)
        break;
      m.generated_first_code_flags[first_code] = 1;
      const std::vector<std::uint32_t> frame =
          m.PredictFrame(projected0, first_code);
      output->codes.insert(output->codes.end(), frame.begin(), frame.end());
      ++output->frames;
      if (on_codes && !on_codes(*output)) {
        SetError(error, "Qwen3-TTS audio stream cancelled or failed");
        *output = {};
        return false;
      }
      if (step + 1 == maximum_new_tokens)
        break;
      const float* text = step < trailing_rows ? prompt.trailing_text.data() +
                                                     step * config.hidden_size
                                               : prompt.tts_pad.data();
      m.FrameEmbedding(frame, text, embedding.data());
      m.TalkerStep(embedding, 1, &logits, &last_hidden, &projected0);
    }
    if (m.profile)
      std::fprintf(stderr,
                   "qwen3_tts npu: %zu prompt rows, %zu frames, prefill %.1f "
                   "ms, talker %.1f ms, predictor %.1f ms, total %.1f ms; "
                   "gemm: talker %zu runs %.1f ms, predictor %zu runs %.1f "
                   "ms\n",
                   prompt.tokens, output->frames, prefill_ms,
                   m.talker_ms - prefill_ms, m.predictor_ms,
                   std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - t0)
                       .count(),
                   m.talker.gemm_runs, m.talker.gemm_ms, m.predictor.gemm_runs,
                   m.predictor.gemm_ms);
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    *output = {};
    return false;
  }
}

}  // namespace gufo::models::qwen3_tts::npu
