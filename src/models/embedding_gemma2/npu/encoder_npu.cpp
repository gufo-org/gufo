#include "src/models/embedding_gemma2/npu/encoder_npu.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <span>
#include <utility>

#include "gufo/ext/experimental/npu.hpp"
#include "src/models/embedding_gemma2/config.hpp"
#include "src/models/embedding_gemma2/loader.hpp"
#include "src/models/embedding_gemma2/tensor.hpp"

namespace gufo::models::embedding_gemma2::npu {
namespace host = gufo::npu::host;

namespace {

// Sequences at least this long run attention products on the NPU.
constexpr std::size_t kNpuAttentionRows = 128;

struct Layer {
  int qkv, o, gate_up, down, ple_gate, ple_proj;
  std::vector<float> input_norm, post_attn_norm, pre_ffn_norm, post_ffn_norm,
      q_norm, k_norm, ple_norm;
  float scalar;
};

std::vector<float> ToFloat(const Tensor& t) {
  const std::size_t n = std::size_t(*t.NumElements());
  std::vector<float> out(n);
  host::Bf16ToFloat(reinterpret_cast<const std::uint16_t*>(t.data), out.data(),
                    n);
  return out;
}

}  // namespace

struct Encoder::Impl {
  gufo::npu::Device device;
  gufo::npu::BfpGemm gemm;
  LoadResult loaded;
  const TextConfig* config{};
  const std::uint16_t* embed{};
  int ple_model, projection;
  std::vector<float> ple_norm, final_norm;
  std::vector<Layer> layers;
  std::mutex mutex;
  bool profile{std::getenv("GUFO_NPU_PROFILE") != nullptr};
  ModelConfig text_only;

  const Tensor& Find(const std::string& name) const {
    const Tensor* t = loaded.store->Find(std::string(kTextTensorPrefix) + name);
    if (!t || t->dtype != DType::kBF16)
      throw std::runtime_error("missing BF16 tensor " + name);
    return *t;
  }
  std::vector<float> Vector(const std::string& name) const {
    return ToFloat(Find(name));
  }
  // Uploads the row-wise concatenation of BF16 matrices [n_i][k].
  int Upload(std::initializer_list<std::string> names) {
    std::vector<std::uint16_t> rows;
    int n = 0, k = 0;
    for (const std::string& name : names) {
      const Tensor& t = Find(name);
      n += int(t.shape[0]);
      k = int(t.shape[1]);
      const auto* p = reinterpret_cast<const std::uint16_t*>(t.data);
      rows.insert(rows.end(), p, p + t.shape[0] * t.shape[1]);
    }
    return gemm.UploadBf16(n, k, rows);
  }

  void Load(const std::string& model_dir) {
    loaded = LoadModelDirectory(model_dir, false, false);
    if (!loaded.ok)
      throw std::runtime_error(loaded.error);
    config = &loaded.config.text;
    text_only = loaded.config;
    text_only.supported_vision = text_only.supported_video =
        text_only.supported_audio = false;
    device = gufo::npu::Device::Open();
    gemm = gufo::npu::BfpGemm::Open(device, gufo::npu::FindDesign("bfpgemm"),
                                    std::size_t(512) << 20);
    embed = reinterpret_cast<const std::uint16_t*>(
        Find("embed_tokens.weight").data);
    ple_model = Upload({"ple.per_layer_model_projection.weight"});
    projection = Upload({"embedding_projection.weight"});
    ple_norm = Vector("ple.per_layer_projection_norm.weight");
    final_norm = Vector("norm.weight");
    for (std::uint32_t l = 0; l < config->num_hidden_layers; ++l) {
      const std::string p = "layers." + std::to_string(l) + ".";
      Layer layer;
      layer.qkv =
          Upload({p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight",
                  p + "self_attn.v_proj.weight"});
      layer.o = Upload({p + "self_attn.o_proj.weight"});
      layer.gate_up =
          Upload({p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"});
      layer.down = Upload({p + "mlp.down_proj.weight"});
      layer.ple_gate = Upload({p + "ple_block.per_layer_input_gate.weight"});
      layer.ple_proj = Upload({p + "ple_block.per_layer_projection.weight"});
      layer.input_norm = Vector(p + "input_layernorm.weight");
      layer.post_attn_norm = Vector(p + "post_attention_layernorm.weight");
      layer.pre_ffn_norm = Vector(p + "pre_feedforward_layernorm.weight");
      layer.post_ffn_norm = Vector(p + "post_feedforward_layernorm.weight");
      layer.q_norm = Vector(p + "self_attn.q_norm.weight");
      layer.k_norm = Vector(p + "self_attn.k_norm.weight");
      layer.ple_norm = Vector(p + "ple_block.post_per_layer_input_norm.weight");
      layer.scalar = Vector(p + "layer_scalar")[0];
      layers.push_back(std::move(layer));
    }
  }

  // Attention of the long sequences: per sequence and head, scores = Q K^T
  // and out = P V on the NPU, with keys and values uploaded per request.
  template<typename Visible>
  void NpuAttention(const float* qkv, std::size_t row_w, std::size_t heads,
                    std::size_t kv, std::size_t hd,
                    const std::vector<std::size_t>& starts,
                    const std::vector<char>& long_seq, const Visible& visible,
                    float* context, std::size_t qw) {
    struct Item {
      std::size_t seq, head, len;
      std::vector<float> q, scores, out;
    };
    std::vector<Item> items;
    for (std::size_t s = 0; s + 1 < starts.size(); ++s)
      if (long_seq[s])
        for (std::size_t h = 0; h < heads; ++h)
          items.push_back({s, h, starts[s + 1] - starts[s], {}, {}, {}});
    if (items.empty())
      return;
    using Clock = std::chrono::steady_clock;
    auto ms = [](Clock::time_point a) {
      return std::chrono::duration<double, std::milli>(Clock::now() - a)
          .count();
    };
    auto t = Clock::now();
    const std::size_t mark = gemm.Mark();
    std::vector<int> keys, values;  // per (sequence, kv head)
    for (std::size_t s = 0; s + 1 < starts.size(); ++s) {
      if (!long_seq[s])
        continue;
      const std::size_t len = starts[s + 1] - starts[s];
      const float* base = qkv + starts[s] * row_w + qw;
      for (std::size_t h = 0; h < kv; ++h) {
        keys.push_back(gemm.Upload(int(len), int(hd), base + h * hd, row_w, 1));
        values.push_back(
            gemm.Upload(int(hd), int(len), base + kv * hd + h * hd, 1, row_w));
      }
    }
    host::ParallelFor(items.size(), 1, [&](std::size_t b, std::size_t e) {
      for (std::size_t i = b; i < e; ++i) {
        Item& it = items[i];
        it.q.resize(it.len * hd);
        for (std::size_t r = 0; r < it.len; ++r)
          std::memcpy(it.q.data() + r * hd,
                      qkv + (starts[it.seq] + r) * row_w + it.head * hd,
                      hd * sizeof(float));
        it.scores.resize(it.len * it.len);
        it.out.resize(it.len * hd);
      }
    });
    auto pair_index = [&](const Item& it) {
      std::size_t k = 0;
      for (std::size_t s = 0; s < it.seq; ++s)
        k += long_seq[s] ? kv : 0;
      return k + it.head / (heads / kv);
    };
    const double t_upload = ms(t);
    t = Clock::now();
    std::vector<gufo::npu::BfpGemm::Call> calls;
    for (Item& it : items)
      calls.push_back(
          {keys[pair_index(it)], it.q.data(), int(it.len), it.scores.data()});
    gemm.Run(calls);
    const double t_qk = ms(t);
    t = Clock::now();
    host::ParallelFor(items.size() * 64, 1, [&](std::size_t b, std::size_t e) {
      for (std::size_t w = b; w < e; ++w) {
        Item& it = items[w / 64];
        const std::size_t part = w % 64;
        const std::size_t r0 = it.len * part / 64,
                          r1 = it.len * (part + 1) / 64;
        for (std::size_t r = r0; r < r1; ++r) {
          std::size_t lo, hi;
          visible(starts[it.seq] + r, starts[it.seq], starts[it.seq + 1], &lo,
                  &hi);
          lo -= starts[it.seq];
          hi -= starts[it.seq];
          float* row = it.scores.data() + r * it.len;
          float top = -INFINITY;
          for (std::size_t t = lo; t < hi; ++t)
            top = std::max(top, row[t]);
          float sum = 0;
          for (std::size_t t = 0; t < it.len; ++t) {
            const float p = (t >= lo && t < hi) ? std::exp(row[t] - top) : 0.0f;
            row[t] = p;
            sum += p;
          }
          const float inv = 1.0f / sum;
          for (std::size_t t = lo; t < hi; ++t)
            row[t] *= inv;
        }
      }
    });
    const double t_soft = ms(t);
    t = Clock::now();
    calls.clear();
    for (Item& it : items)
      calls.push_back({values[pair_index(it)], it.scores.data(), int(it.len),
                       it.out.data()});
    gemm.Run(calls);
    gemm.Release(mark);
    if (profile)
      std::fprintf(stderr,
                   "attention: upload %.2f qk %.2f softmax %.2f pv %.2f ms\n",
                   t_upload, t_qk, t_soft, ms(t));
    for (const Item& it : items)
      for (std::size_t r = 0; r < it.len; ++r)
        std::memcpy(context + (starts[it.seq] + r) * qw + it.head * hd,
                    it.out.data() + r * hd, hd * sizeof(float));
  }

  // Encodes the packed sequences [starts[i], starts[i + 1]) of `tokens`;
  // returns the pooled, normalized embeddings.
  std::vector<std::vector<float>> Run(const std::vector<std::uint32_t>& tokens,
                                      const std::vector<std::size_t>& starts,
                                      const std::function<bool()>& cancelled) {
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    double gemm_ms = 0;
    auto timed = [&](auto&& fn) {
      const auto a = Clock::now();
      fn();
      gemm_ms +=
          std::chrono::duration<double, std::milli>(Clock::now() - a).count();
    };
    const TextConfig& c = *config;
    const std::size_t rows = tokens.size(), hidden = c.hidden_size,
                      per_layer = c.hidden_size_per_layer_input,
                      layers_n = c.num_hidden_layers,
                      heads = c.num_attention_heads,
                      inner = c.intermediate_size;
    const float eps = c.rms_norm_eps;
    std::vector<std::int32_t> positions(rows);
    std::vector<std::size_t> seq_of(rows);
    for (std::size_t s = 0; s + 1 < starts.size(); ++s)
      for (std::size_t r = starts[s]; r < starts[s + 1]; ++r) {
        positions[r] = std::int32_t(r - starts[s]);
        seq_of[r] = s;
      }

    std::vector<char> long_seq(starts.size() - 1, 0);
    for (std::size_t q = 0; q + 1 < starts.size(); ++q)
      long_seq[q] = starts[q + 1] - starts[q] >= kNpuAttentionRows;

    std::vector<float> state(rows * hidden);
    for (std::size_t r = 0; r < rows; ++r)
      for (std::size_t i = 0; i < hidden; ++i)
        state[r * hidden + i] = host::Bf16ToFloat(host::FloatToBf16(
            host::Bf16ToFloat(embed[std::size_t(tokens[r]) * hidden + i]) *
            kEmbedScale));

    std::vector<float> ple(rows * layers_n * per_layer);
    timed([&] { gemm.Run(ple_model, state.data(), int(rows), ple.data()); });
    const float ple_scale = 1.0f / std::sqrt(float(hidden));
    for (float& v : ple)
      v *= ple_scale;
    host::RmsNormRows(ple.data(), ple.data(), rows * layers_n, per_layer,
                      ple_norm.data(), eps);

    std::size_t max_q = 0, max_kv = 0, max_hd = 0;
    for (const auto& g : c.layers) {
      max_q = std::max<std::size_t>(max_q, heads * g.head_dim);
      max_kv =
          std::max<std::size_t>(max_kv, g.num_key_value_heads * g.head_dim);
      max_hd = std::max<std::size_t>(max_hd, g.head_dim);
    }
    std::vector<float> normed(rows * hidden), qkv(rows * (max_q + 2 * max_kv)),
        context(rows * max_q), tmp(rows * hidden), gate_up(rows * 2 * inner),
        act(rows * std::max(inner, per_layer));

    for (std::size_t l = 0; l < layers_n; ++l) {
      if (cancelled && cancelled())
        return {};
      const Layer& w = layers[l];
      const LayerGeometry& g = c.layers[l];
      const std::size_t hd = g.head_dim, kv = g.num_key_value_heads;
      const std::size_t qw = heads * hd, kw = kv * hd, row_w = qw + 2 * kw;
      host::RmsNormRows(state.data(), normed.data(), rows, hidden,
                        w.input_norm.data(), eps);
      timed([&] { gemm.Run(w.qkv, normed.data(), int(rows), qkv.data()); });
      host::ParallelFor(rows, 16, [&](std::size_t b, std::size_t e) {
        for (std::size_t r = b; r < e; ++r) {
          float* q = qkv.data() + r * row_w;
          host::RmsNormRows(q, q, heads, hd, w.q_norm.data(), eps);
          host::RmsNormRows(q + qw, q + qw, kv, hd, w.k_norm.data(), eps);
          host::RmsNormRows(q + qw + kw, q + qw + kw, kv, hd, nullptr, eps);
        }
      });
      // Rotary embeddings on q and k: k heads follow q heads in each row.
      host::RopeNeox(qkv.data(), rows, row_w, heads + kv, hd, hd,
                     positions.data(), g.rope_theta);
      const bool windowed = g.type == LayerType::kSlidingAttention;
      const std::size_t window = c.sliding_window;
      auto visible = [&](std::size_t r, std::size_t lo, std::size_t hi,
                         std::size_t* b, std::size_t* e) {
        if (windowed) {
          lo = std::max(lo, r > window ? r - window : 0);
          hi = std::min(hi, r + window + 1);
        }
        *b = lo;
        *e = hi;
      };
      // Long sequences: QK^T and PV on the NPU, the softmax on the CPU.
      timed([&] {
        NpuAttention(qkv.data(), row_w, heads, kv, hd, starts, long_seq,
                     visible, context.data(), qw);
      });
      host::Attention(
          {.q = qkv.data(),
           .q_stride = row_w,
           .k = qkv.data() + qw,
           .k_stride = row_w,
           .v = qkv.data() + qw + kw,
           .v_stride = row_w,
           .out = context.data(),
           .out_stride = qw,
           .rows = rows,
           .heads = heads,
           .kv_heads = kv,
           .dim = hd,
           .dim_v = hd,
           .scale = 1.0f},
          [&](std::size_t r, std::size_t* b, std::size_t* e) {
            const std::size_t s = seq_of[r];
            if (long_seq[s]) {
              *b = *e = 0;  // done on the NPU; leave its rows
              return;
            }
            visible(r, starts[s], starts[s + 1], b, e);
          },
          /*skip_empty=*/true);
      timed([&] { gemm.Run(w.o, context.data(), int(rows), tmp.data()); });
      host::RmsNormRows(tmp.data(), tmp.data(), rows, hidden,
                        w.post_attn_norm.data(), eps);
      for (std::size_t i = 0; i < state.size(); ++i)
        state[i] += tmp[i];

      host::RmsNormRows(state.data(), normed.data(), rows, hidden,
                        w.pre_ffn_norm.data(), eps);
      timed([&] {
        gemm.Run(w.gate_up, normed.data(), int(rows), gate_up.data());
      });
      host::GatedRows(gate_up.data(), 2 * inner, gate_up.data() + inner,
                      2 * inner, act.data(), inner, rows, inner,
                      host::Activation::kGeluTanh);
      timed([&] { gemm.Run(w.down, act.data(), int(rows), tmp.data()); });
      host::RmsNormRows(tmp.data(), tmp.data(), rows, hidden,
                        w.post_ffn_norm.data(), eps);
      for (std::size_t i = 0; i < state.size(); ++i)
        state[i] += tmp[i];

      timed([&] { gemm.Run(w.ple_gate, state.data(), int(rows), act.data()); });
      host::ParallelFor(rows, 64, [&](std::size_t b, std::size_t e) {
        for (std::size_t r = b; r < e; ++r) {
          float* a = act.data() + r * per_layer;
          const float* p = ple.data() + (r * layers_n + l) * per_layer;
          for (std::size_t i = 0; i < per_layer; ++i)
            a[i] = host::Activate(host::Activation::kGeluTanh, a[i]) * p[i];
        }
      });
      timed([&] { gemm.Run(w.ple_proj, act.data(), int(rows), tmp.data()); });
      host::RmsNormRows(tmp.data(), tmp.data(), rows, hidden, w.ple_norm.data(),
                        eps);
      for (std::size_t i = 0; i < state.size(); ++i)
        state[i] = (state[i] + tmp[i]) * w.scalar;
    }
    host::RmsNormRows(state.data(), state.data(), rows, hidden,
                      final_norm.data(), eps);
    const std::size_t dim = c.embedding_dim;
    std::vector<float> projected(rows * dim);
    timed([&] {
      gemm.Run(projection, state.data(), int(rows), projected.data());
    });
    std::vector<std::vector<float>> out;
    for (std::size_t s = 0; s + 1 < starts.size(); ++s) {
      std::vector<double> sum(dim, 0.0);
      for (std::size_t r = starts[s]; r < starts[s + 1]; ++r)
        for (std::size_t i = 0; i < dim; ++i)
          sum[i] += projected[r * dim + i];
      double norm = 0;
      for (double v : sum)
        norm += v * v;
      const double inv = norm > 0 ? 1.0 / std::sqrt(norm) : 0.0;
      std::vector<float> e(dim);
      for (std::size_t i = 0; i < dim; ++i)
        e[i] = float(sum[i] * inv);
      out.push_back(std::move(e));
    }
    if (profile)
      std::fprintf(
          stderr, "embedding_gemma2 npu: %zu tokens, %.2f ms (gemm %.2f ms)\n",
          rows,
          std::chrono::duration<double, std::milli>(Clock::now() - t0).count(),
          gemm_ms);
    return out;
  }
};

Encoder::Encoder() : impl_(std::make_unique<Impl>()) {}

const ModelConfig& Encoder::config() const {
  return impl_->text_only;
}
Encoder::~Encoder() = default;

std::unique_ptr<Encoder> Encoder::Create(const std::string& model_dir,
                                         std::string* error) {
  std::unique_ptr<Encoder> encoder(new Encoder());
  try {
    encoder->impl_->Load(model_dir);
  } catch (const std::exception& e) {
    if (error)
      *error = std::string("NPU encoder: ") + e.what();
    return nullptr;
  }
  return encoder;
}

bool Encoder::EmbedMultimodal(const std::vector<EncoderSequence>& sequences,
                              std::vector<std::vector<float>>* embeddings,
                              std::string* error,
                              const std::function<bool()>& is_cancelled) {
  std::vector<std::uint32_t> tokens;
  std::vector<std::size_t> starts{0};
  for (const EncoderSequence& s : sequences) {
    if (!s.images.empty() || !s.audio.empty()) {
      if (error)
        *error = "embedding_gemma2: --npu embeds text only";
      return false;
    }
    if (s.tokens.empty()) {
      if (error)
        *error = "embedding_gemma2: empty sequence";
      return false;
    }
    for (std::uint32_t t : s.tokens)
      if (t >= impl_->config->vocab_size) {
        if (error)
          *error = "embedding_gemma2: token id out of range";
        return false;
      }
    tokens.insert(tokens.end(), s.tokens.begin(), s.tokens.end());
    starts.push_back(tokens.size());
  }
  std::lock_guard lock(impl_->mutex);
  try {
    auto out = impl_->Run(tokens, starts, is_cancelled);
    if (out.size() != sequences.size()) {
      if (error)
        *error = "embedding_gemma2: request cancelled";
      return false;
    }
    *embeddings = std::move(out);
  } catch (const std::exception& e) {
    if (error)
      *error = std::string("embedding_gemma2 NPU: ") + e.what();
    return false;
  }
  return true;
}

}  // namespace gufo::models::embedding_gemma2::npu
