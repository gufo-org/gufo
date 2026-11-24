#include "src/models/embedding_gemma2/npu/audio_npu.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <utility>

#include "src/models/embedding_gemma2/loader.hpp"
#include "src/models/embedding_gemma2/tensor.hpp"

namespace gufo::models::embedding_gemma2::npu {
namespace host = gufo::npu::host;

namespace {

constexpr std::size_t kWidth = 1024, kHeads = 8, kHeadDim = 128, kFfn = 4096;
constexpr std::size_t kLayers = 12, kOut = 1536, kText = 512, kBins = 128;
constexpr std::size_t kChunk = 12, kContext = 24, kRelative = 13;
constexpr std::size_t kMaxFrames = 2999;
constexpr float kEps = 1e-6f;

std::vector<float> ToFloat(const Tensor& t) {
  const std::size_t n = std::size_t(*t.NumElements());
  std::vector<float> out(n);
  host::Bf16ToFloat(reinterpret_cast<const std::uint16_t*>(t.data), out.data(),
                    n);
  return out;
}

// One clipped linear: y = clip(clip(x) W^T).
struct Linear {
  int id;
  float in_min, in_max, out_min, out_max;
};

struct FeedForward {
  Linear up, down;
  std::vector<float> pre_norm, post_norm;
};

struct Layer {
  FeedForward ffn1, ffn2;
  Linear q, k, v, post, start, end;
  std::vector<float> relative;   // [13][1024]: positions x relative_k_proj^T
  std::vector<float> dim_scale;  // softplus(per_dim_scale) x query scale
  std::vector<float> depthwise;  // [1024][5]
  std::vector<float> lconv_pre, conv_norm, norm_pre_attn, norm_post_attn,
      norm_out;
};

void Clip(const float* x, float* y, std::size_t n, float lo, float hi) {
  host::ParallelFor(n, std::size_t(1) << 15, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i)
      y[i] = std::min(std::max(x[i], lo), hi);
  });
}

}  // namespace

struct AudioTower::Impl {
  gufo::npu::BfpGemm& gemm;
  std::vector<float> conv0_w, conv0_norm, conv1_w, conv1_norm, out_bias;
  int input_proj, output_proj, projection;
  std::vector<Layer> layers;
  // Request buffers, kept across requests.
  std::vector<float> conv0, conv1, state, update, clipped[3], temp, q, k, v,
      context, projected;

  explicit Impl(gufo::npu::BfpGemm& g) : gemm(g) {}

  void Load(const std::string& model_dir) {
    auto loaded = LoadModelDirectory(model_dir, false, true);
    if (!loaded.ok)
      throw std::runtime_error(loaded.error);
    const TensorStore& store = *loaded.store;
    auto find = [&](const std::string& name) -> const Tensor& {
      const Tensor* t = store.Find(name);
      if (!t || t->dtype != DType::kBF16)
        throw std::runtime_error("missing BF16 tensor " + name);
      return *t;
    };
    auto vec = [&](const std::string& name) { return ToFloat(find(name)); };
    auto scalar = [&](const std::string& name) { return vec(name).at(0); };
    auto upload = [&](const std::string& name) {
      const Tensor& t = find(name);
      const auto* p = reinterpret_cast<const std::uint16_t*>(t.data);
      return gemm.UploadBf16(
          int(t.shape[0]), int(t.shape[1]),
          std::span<const std::uint16_t>(p, t.shape[0] * t.shape[1]));
    };
    auto linear = [&](const std::string& p) {
      return Linear{upload(p + ".linear.weight"), scalar(p + ".input_min"),
                    scalar(p + ".input_max"), scalar(p + ".output_min"),
                    scalar(p + ".output_max")};
    };
    const std::string s = "audio_tower.subsample_conv_projection.";
    conv0_w = vec(s + "layer0.conv.weight");
    conv0_norm = vec(s + "layer0.norm.weight");
    conv1_w = vec(s + "layer1.conv.weight");
    conv1_norm = vec(s + "layer1.norm.weight");
    input_proj = upload(s + "input_proj_linear.weight");
    output_proj = upload("audio_tower.output_proj.weight");
    out_bias = vec("audio_tower.output_proj.bias");
    projection = upload("embed_audio.embedding_projection.weight");

    // Sinusoidal relative positions [13][1024], as the official float32
    // inv_timescales (sin half, then cos half; relative distance 12 - t).
    std::vector<float> positions(kRelative * kWidth);
    for (std::size_t t = 0; t < kRelative; ++t)
      for (std::size_t d = 0; d < 512; ++d) {
        const float inv = std::exp(-float(d) * float(std::log(10000.0) / 511));
        const float angle = float(12 - int(t)) * inv;
        positions[t * kWidth + d] = std::sin(angle);
        positions[t * kWidth + d + 512] = std::cos(angle);
      }
    const float qscale = (1.0f / std::sqrt(128.0f)) / std::log(2.0f);
    for (std::size_t i = 0; i < kLayers; ++i) {
      const std::string p = "audio_tower.layers." + std::to_string(i) + ".";
      Layer l;
      for (auto [ff, name] : {std::pair{&l.ffn1, "feed_forward1."},
                              std::pair{&l.ffn2, "feed_forward2."}}) {
        ff->up = linear(p + name + "ffw_layer_1");
        ff->down = linear(p + name + "ffw_layer_2");
        ff->pre_norm = vec(p + name + "pre_layer_norm.weight");
        ff->post_norm = vec(p + name + "post_layer_norm.weight");
      }
      l.q = linear(p + "self_attn.q_proj");
      l.k = linear(p + "self_attn.k_proj");
      l.v = linear(p + "self_attn.v_proj");
      l.post = linear(p + "self_attn.post");
      l.start = linear(p + "lconv1d.linear_start");
      l.end = linear(p + "lconv1d.linear_end");
      // relative = positions x relative_k_proj^T, exactly (f32, once).
      const std::vector<float> rk = vec(p + "self_attn.relative_k_proj.weight");
      l.relative.assign(kRelative * kWidth, 0.0f);
      host::ParallelFor(
          kRelative * kWidth, 64, [&](std::size_t b, std::size_t e) {
            for (std::size_t i2 = b; i2 < e; ++i2) {
              const std::size_t t = i2 / kWidth, o = i2 % kWidth;
              l.relative[i2] =
                  host::Dot(&positions[t * kWidth], &rk[o * kWidth], kWidth);
            }
          });
      const std::vector<float> pds = vec(p + "self_attn.per_dim_scale");
      l.dim_scale.resize(kHeadDim);
      for (std::size_t d = 0; d < kHeadDim; ++d)
        l.dim_scale[d] = qscale * std::log1p(std::exp(pds[d]));
      l.depthwise = vec(p + "lconv1d.depthwise_conv1d.weight");
      l.lconv_pre = vec(p + "lconv1d.pre_layer_norm.weight");
      l.conv_norm = vec(p + "lconv1d.conv_norm.weight");
      l.norm_pre_attn = vec(p + "norm_pre_attn.weight");
      l.norm_post_attn = vec(p + "norm_post_attn.weight");
      l.norm_out = vec(p + "norm_out.weight");
      layers.push_back(std::move(l));
    }
  }

  static void Grow(std::vector<float>& v, std::size_t n) {
    if (v.size() < n)
      v.resize(n);
  }

  void Linear1(const Linear& lin, const float* x, float* y, std::size_t n,
               std::size_t width, std::size_t outputs) {
    Grow(clipped[0], n * width);
    Clip(x, clipped[0].data(), n * width, lin.in_min, lin.in_max);
    gemm.Run(lin.id, clipped[0].data(), int(n), y);
    Clip(y, y, n * outputs, lin.out_min, lin.out_max);
  }

  void Feed(const FeedForward& ff, std::size_t n) {
    host::RmsNormRows(state.data(), update.data(), n, kWidth,
                      ff.pre_norm.data(), kEps);
    Linear1(ff.up, update.data(), temp.data(), n, kWidth, kFfn);
    host::ActivateRows(temp.data(), n, kFfn, nullptr, host::Activation::kSilu);
    Linear1(ff.down, temp.data(), update.data(), n, kFfn, kWidth);
    host::RmsNormRows(update.data(), update.data(), n, kWidth,
                      ff.post_norm.data(), kEps);
    // Both conformer feed-forwards add half their update.
    host::ParallelFor(n * kWidth, std::size_t(1) << 15,
                      [&](std::size_t b, std::size_t e) {
                        for (std::size_t i = b; i < e; ++i)
                          state[i] += 0.5f * update[i];
                      });
  }

  // Subsampling convolution, 3x3 stride 2 pad 1, channels last: rows of
  // `frames` x `bins` x `in` to (frames + 1) / 2 x (bins + 1) / 2 x `out`;
  // input times at or past `valid` count as zero. Then LayerNorm (no bias)
  // over the output channels and ReLU.
  static void Conv(const float* x, const float* w, const float* norm, float* y,
                   int frames, int bins, int in, int out, int valid) {
    const int ob = (bins + 1) / 2, of = (frames + 1) / 2;
    // Weights as [dt][df][in][out] so each tap updates `out` outputs.
    std::vector<float> wt(std::size_t(9) * in * out);
    for (int oc = 0; oc < out; ++oc)
      for (int ic = 0; ic < in; ++ic)
        for (int t = 0; t < 9; ++t)
          wt[(std::size_t(t) * in + ic) * out + oc] =
              w[(std::size_t(oc) * in + ic) * 9 + t];
    host::ParallelFor(
        std::size_t(of) * ob, 8, [&](std::size_t b, std::size_t e) {
          for (std::size_t i = b; i < e; ++i) {
            const int t = int(i) / ob, f = int(i) % ob;
            float* o = y + i * out;
            std::fill_n(o, out, 0.0f);
            for (int dt = 0; dt < 3; ++dt) {
              const int time = t * 2 + dt - 1;
              if (time < 0 || time >= frames || time >= valid)
                continue;
              for (int df = 0; df < 3; ++df) {
                const int bin = f * 2 + df - 1;
                if (bin < 0 || bin >= bins)
                  continue;
                const float* xs = x + (std::size_t(time) * bins + bin) * in;
                const float* ws =
                    wt.data() + std::size_t(dt * 3 + df) * in * out;
                for (int ic = 0; ic < in; ++ic)
                  host::Axpy(xs[ic], ws + std::size_t(ic) * out, o,
                             std::size_t(out));
              }
            }
            // LayerNorm (mean-centred, no bias) and ReLU.
            double mean = 0;
            for (int c = 0; c < out; ++c)
              mean += o[c];
            const float m = float(mean / out);
            double var = 0;
            for (int c = 0; c < out; ++c)
              var += double(o[c] - m) * (o[c] - m);
            const float s = 1.0f / std::sqrt(float(var / out) + kEps);
            for (int c = 0; c < out; ++c)
              o[c] = std::max(0.0f, (o[c] - m) * s * norm[c]);
          }
        });
  }

  // Chunked local attention (chunks of 12, 12 keys of left context), with
  // the relative position term and a 50 tanh logit cap, as audio_runtime.
  void Attention(const Layer& l, std::size_t n, std::size_t valid) {
    const float kscale = std::log(1.0f + std::exp(1.0f)) / std::log(2.0f);
    host::ParallelFor(n * kHeads, 8, [&](std::size_t b, std::size_t e) {
      float query[kHeadDim], scores[kContext], acc[kHeadDim];
      for (std::size_t item = b; item < e; ++item) {
        const std::size_t t = item / kHeads, h = item % kHeads;
        const std::size_t chunk = t / kChunk * kChunk;
        for (std::size_t d = 0; d < kHeadDim; ++d)
          query[d] = q[t * kWidth + h * kHeadDim + d] * l.dim_scale[d];
        float top = -1e9f;
        for (std::size_t c = 0; c < kContext; ++c) {
          const long key = long(chunk) - long(kChunk) + long(c);
          float dot = 0;
          if (key >= 0 && key < long(n))
            dot +=
                host::Dot(query, &k[std::size_t(key) * kWidth + h * kHeadDim],
                          kHeadDim) *
                kscale;
          const long rel = long(c) - long(t % kChunk);
          if (rel >= 0 && rel <= 12)
            dot += host::Dot(
                query, &l.relative[std::size_t(rel) * kWidth + h * kHeadDim],
                kHeadDim);
          const bool allowed = key >= 0 && key < long(valid) &&
                               key <= long(t) && key > long(t) - 12;
          scores[c] = allowed ? 50.0f * std::tanh(dot / 50.0f) : -1e9f;
          top = std::max(top, scores[c]);
        }
        float denom = 0;
        for (float& s : scores) {
          s = std::exp(s - top);
          denom += s;
        }
        std::fill_n(acc, kHeadDim, 0.0f);
        for (std::size_t c = 0; c < kContext; ++c) {
          const long key = long(chunk) - long(kChunk) + long(c);
          if (key >= 0 && key < long(n))
            host::Axpy(scores[c] / denom,
                       &v[std::size_t(key) * kWidth + h * kHeadDim], acc,
                       kHeadDim);
        }
        std::memcpy(&context[t * kWidth + h * kHeadDim], acc, sizeof(acc));
      }
    });
  }

  std::vector<float> Run(const AudioFeatures& features,
                         const std::function<bool()>& cancelled) {
    const int frames = int(features.mask.size());
    const int valid_frames =
        int(std::count(features.mask.begin(), features.mask.end(), 1));
    const int n0 = (frames + 1) / 2, n = (n0 + 1) / 2;
    const std::size_t valid = features.soft_tokens();
    if (frames <= 0 || frames > int(kMaxFrames) || valid == 0 ||
        features.values.size() != std::size_t(frames) * kBins)
      throw std::invalid_argument(
          "embedding_gemma2: invalid audio feature shape");
    const std::size_t rows = std::size_t(n);
    Grow(conv0, std::size_t(n0) * 64 * 128);
    Grow(conv1, rows * 32 * 32);
    for (auto* b : {&state, &update, &q, &k, &v, &context})
      Grow(*b, rows * kWidth);
    Grow(temp, rows * kFfn);
    Grow(projected, rows * kOut);

    Conv(features.values.data(), conv0_w.data(), conv0_norm.data(),
         conv0.data(), frames, int(kBins), 1, 128, valid_frames);
    Conv(conv0.data(), conv1_w.data(), conv1_norm.data(), conv1.data(), n0, 64,
         128, 32, (valid_frames + 1) / 2);
    gemm.Run(input_proj, conv1.data(), n, state.data());

    for (const Layer& l : layers) {
      if (cancelled && cancelled())
        throw std::runtime_error("audio encoding cancelled");
      Feed(l.ffn1, rows);
      host::RmsNormRows(state.data(), update.data(), rows, kWidth,
                        l.norm_pre_attn.data(), kEps);
      {
        std::vector<gufo::npu::BfpGemm::Call> calls;
        const Linear* lin[3] = {&l.q, &l.k, &l.v};
        float* out[3] = {q.data(), k.data(), v.data()};
        for (int i = 0; i < 3; ++i) {
          Grow(clipped[i], rows * kWidth);
          Clip(update.data(), clipped[i].data(), rows * kWidth, lin[i]->in_min,
               lin[i]->in_max);
          calls.push_back({lin[i]->id, clipped[i].data(), n, out[i]});
        }
        gemm.Run(calls);
        for (int i = 0; i < 3; ++i)
          Clip(out[i], out[i], rows * kWidth, lin[i]->out_min, lin[i]->out_max);
      }
      Attention(l, rows, valid);
      Linear1(l.post, context.data(), update.data(), rows, kWidth, kWidth);
      host::AddNormRows(state.data(), update.data(), rows, kWidth,
                        l.norm_post_attn.data(), kEps, 1.0f);
      // Light convolution: GLU, causal depthwise k = 5, norm, SiLU.
      host::RmsNormRows(state.data(), update.data(), rows, kWidth,
                        l.lconv_pre.data(), kEps);
      Linear1(l.start, update.data(), temp.data(), rows, kWidth, 2 * kWidth);
      host::ParallelFor(rows, 16, [&](std::size_t b, std::size_t e) {
        for (std::size_t t = b; t < e; ++t)
          for (std::size_t c = 0; c < kWidth; ++c) {
            const float a = temp[t * 2 * kWidth + c],
                        g = temp[t * 2 * kWidth + kWidth + c];
            context[t * kWidth + c] = a / (1.0f + std::exp(-g));
          }
      });
      host::ParallelFor(rows, 16, [&](std::size_t b, std::size_t e) {
        for (std::size_t t = b; t < e; ++t)
          for (std::size_t c = 0; c < kWidth; ++c) {
            float sum = 0;
            for (std::size_t tap = 0; tap < 5; ++tap)
              if (t + tap >= 4)
                sum += context[(t + tap - 4) * kWidth + c] *
                       l.depthwise[c * 5 + tap];
            update[t * kWidth + c] = sum;
          }
      });
      host::RmsNormRows(update.data(), update.data(), rows, kWidth,
                        l.conv_norm.data(), kEps);
      host::ActivateRows(update.data(), rows, kWidth, nullptr,
                         host::Activation::kSilu);
      Linear1(l.end, update.data(), temp.data(), rows, kWidth, kWidth);
      host::ParallelFor(rows * kWidth, std::size_t(1) << 15,
                        [&](std::size_t b, std::size_t e) {
                          for (std::size_t i = b; i < e; ++i)
                            state[i] += temp[i];
                        });
      Feed(l.ffn2, rows);
      host::RmsNormRows(state.data(), state.data(), rows, kWidth,
                        l.norm_out.data(), kEps);
    }
    gemm.Run(output_proj, state.data(), n, projected.data());
    host::ParallelFor(rows, 16, [&](std::size_t b, std::size_t e) {
      for (std::size_t t = b; t < e; ++t)
        for (std::size_t c = 0; c < kOut; ++c)
          projected[t * kOut + c] += out_bias[c];
    });
    host::RmsNormRows(projected.data(), projected.data(), valid, kOut, nullptr,
                      kEps);
    std::vector<float> out(valid * kText);
    gemm.Run(projection, projected.data(), int(valid), out.data());
    host::RoundBf16(out.data(), out.size());
    return out;
  }
};

AudioTower::AudioTower(gufo::npu::BfpGemm& gemm, const std::string& model_dir)
    : impl_(std::make_unique<Impl>(gemm)) {
  impl_->Load(model_dir);
}

AudioTower::~AudioTower() = default;

std::vector<float> AudioTower::Run(const AudioFeatures& features,
                                   const std::function<bool()>& cancelled) {
  return impl_->Run(features, cancelled);
}

}  // namespace gufo::models::embedding_gemma2::npu
