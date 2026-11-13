#include "src/models/qwen3_asr/npu/runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

#include "gufo/ext/experimental/npu.hpp"
#include "src/models/qwen3_asr/config.hpp"
#include "src/models/qwen3_asr/loader.hpp"
#include "src/models/qwen3_asr/npu/split_q8.hpp"
#include "src/models/qwen3_asr/tensor.hpp"

namespace gufo::models::qwen3_asr::npu {
namespace host = gufo::npu::host;

namespace {

constexpr std::size_t kMelBins = 128, kChunkFrames = 100, kChannels = 480;
constexpr std::size_t kConvFrequency = 16, kConvTime = 13;
constexpr std::size_t kFlattened = kChannels * kConvFrequency;
constexpr std::size_t kHidden = 1024, kFfn = 4096, kHeads = 16, kHeadDim = 64;
constexpr std::size_t kOutput = 2048;
constexpr float kLayerNormEps = 1.0e-5f;
constexpr std::uint32_t kEndOfText = 151643, kImEnd = 151645;

using Clock = std::chrono::steady_clock;
double Ms(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

float Round(float v) {
  return host::Bf16ToFloat(host::FloatToBf16(v));
}

struct EncoderLayer {
  int qkv, out, fc1, fc2;
  std::vector<float> qkv_bias, out_bias, fc1_bias, fc2_bias, ln1_w, ln1_b,
      ln2_w, ln2_b;
};

struct DecoderLayer {
  int qkv_b, o_b, gate_up_b, down_b;                       // BfpGemm (prefill)
  std::unique_ptr<SplitQ8> qkv_q, o_q, gate_up_q, down_q;  // decode
  std::vector<float> input_norm, post_norm, q_norm, k_norm;
};

}  // namespace

struct Runtime::Impl {
  LoadResult model;
  const TextConfig* text{};
  std::size_t capacity{};
  gufo::npu::Device device;
  gufo::npu::BfpGemm bfp;
  gufo::npu::Q8Gemm q8;
  bool profile{std::getenv("GUFO_NPU_PROFILE") != nullptr};
  double gemm_ms{}, head_ms{};

  // Audio encoder.
  std::vector<float> conv1_w, conv1_b, conv2_b, conv3_b;
  int conv2, conv3, conv_out, proj1, proj2;
  std::vector<float> proj1_b, proj2_b, ln_post_w, ln_post_b;
  std::vector<EncoderLayer> enc;

  // Text decoder.
  const std::uint16_t* embed{};
  std::vector<DecoderLayer> dec;
  std::vector<float> final_norm;
  std::unique_ptr<SplitQ8> lm_head;
  std::unique_ptr<Lane> lane;
  // Decode on the fused NPU program (one dispatch per token) when the
  // decoder's shape fits it; SplitQ8 products otherwise.
  gufo::npu::Qwen3Decoder fused;
  std::vector<float> k_cache, v_cache;  // [layer][position][kv dim]

  const Tensor& Find(const std::string& name) const {
    const Tensor* t = model.store->Find(name);
    if (!t || t->dtype != DType::kBF16)
      throw std::runtime_error("Qwen3-ASR NPU: missing BF16 tensor " + name);
    return *t;
  }
  std::vector<float> Vector(const std::string& name) const {
    const Tensor& t = Find(name);
    std::vector<float> out(std::size_t(*t.NumElements()));
    host::Bf16ToFloat(reinterpret_cast<const std::uint16_t*>(t.data),
                      out.data(), out.size());
    return out;
  }
  // Row-wise concatenation [sum n][k] of BF16 matrices (2D, or flattened
  // past the first dimension).
  std::vector<std::uint16_t> Rows(std::initializer_list<std::string> names,
                                  int* n, int* k) const {
    std::vector<std::uint16_t> rows;
    *n = 0;
    for (const std::string& name : names) {
      const Tensor& t = Find(name);
      const std::size_t elems = std::size_t(*t.NumElements());
      *n += int(t.shape[0]);
      *k = int(elems / t.shape[0]);
      const auto* p = reinterpret_cast<const std::uint16_t*>(t.data);
      rows.insert(rows.end(), p, p + elems);
    }
    return rows;
  }
  int UploadBfp(std::initializer_list<std::string> names) {
    int n = 0, k = 0;
    const auto rows = Rows(names, &n, &k);
    return bfp.UploadBf16(n, k, rows);
  }
  std::unique_ptr<SplitQ8> UploadSplit(
      std::initializer_list<std::string> names) {
    int n = 0, k = 0;
    const auto rows = Rows(names, &n, &k);
    return std::make_unique<SplitQ8>(q8, rows.data(), n, k, NpuRows(n, k));
  }
  void Load(const std::string& root, std::size_t capacity_tokens) {
    model = LoadModelDirectory(root);
    if (!model.ok)
      throw std::runtime_error(model.error);
    text = &model.config.text;
    capacity = capacity_tokens;
    device = gufo::npu::Device::Open();
    bfp = gufo::npu::BfpGemm::Open(device, gufo::npu::FindDesign("bfpgemm"),
                                   std::size_t(2040) << 20);
    const std::string a = "thinker.audio_tower.";
    conv1_w = Vector(a + "conv2d1.weight");
    conv1_b = Vector(a + "conv2d1.bias");
    conv2 = UploadBfp({a + "conv2d2.weight"});
    conv2_b = Vector(a + "conv2d2.bias");
    conv3 = UploadBfp({a + "conv2d3.weight"});
    conv3_b = Vector(a + "conv2d3.bias");
    conv_out = UploadBfp({a + "conv_out.weight"});
    for (std::size_t l = 0; l < 24; ++l) {
      const std::string p = a + "layers." + std::to_string(l) + ".";
      EncoderLayer e;
      e.qkv = UploadBfp({p + "self_attn.q_proj.weight",
                         p + "self_attn.k_proj.weight",
                         p + "self_attn.v_proj.weight"});
      e.qkv_bias = Vector(p + "self_attn.q_proj.bias");
      for (const char* s : {"self_attn.k_proj.bias", "self_attn.v_proj.bias"}) {
        const auto b = Vector(p + s);
        e.qkv_bias.insert(e.qkv_bias.end(), b.begin(), b.end());
      }
      e.out = UploadBfp({p + "self_attn.out_proj.weight"});
      e.out_bias = Vector(p + "self_attn.out_proj.bias");
      e.fc1 = UploadBfp({p + "fc1.weight"});
      e.fc1_bias = Vector(p + "fc1.bias");
      e.fc2 = UploadBfp({p + "fc2.weight"});
      e.fc2_bias = Vector(p + "fc2.bias");
      e.ln1_w = Vector(p + "self_attn_layer_norm.weight");
      e.ln1_b = Vector(p + "self_attn_layer_norm.bias");
      e.ln2_w = Vector(p + "final_layer_norm.weight");
      e.ln2_b = Vector(p + "final_layer_norm.bias");
      enc.push_back(std::move(e));
    }
    ln_post_w = Vector(a + "ln_post.weight");
    ln_post_b = Vector(a + "ln_post.bias");
    proj1 = UploadBfp({a + "proj1.weight"});
    proj1_b = Vector(a + "proj1.bias");
    proj2 = UploadBfp({a + "proj2.weight"});
    proj2_b = Vector(a + "proj2.bias");

    embed = reinterpret_cast<const std::uint16_t*>(
        Find("thinker.model.embed_tokens.weight").data);
    using Fused = gufo::npu::Qwen3Decoder;
    const char* fused_env = std::getenv("GUFO_NPU_FUSED");
    const bool fits =
        !(fused_env && std::string(fused_env) == "0") &&
        text->num_hidden_layers == std::uint32_t(Fused::kLayers) &&
        text->hidden_size == std::uint32_t(Fused::kHidden) &&
        text->num_attention_heads == std::uint32_t(Fused::kHeads) &&
        text->num_key_value_heads == std::uint32_t(Fused::kKvHeads) &&
        text->head_dim == std::uint32_t(Fused::kHeadDim) &&
        text->intermediate_size == std::uint32_t(Fused::kFfn) &&
        text->vocab_size <= std::uint32_t(Fused::kMaxHeadRows);
    if (fits) {
      try {
        fused = Fused::Open(device, gufo::npu::FindDesign("qwen3dec"));
        if (std::size_t(fused.capacity()) < capacity)
          fused = Fused();
      } catch (const gufo::npu::Error&) {
        fused = Fused();
      }
    }
    if (!fused)
      q8 = gufo::npu::Q8Gemm::Open(device, gufo::npu::FindDesign("q8gemm"),
                                   std::size_t(2040) << 20);
    for (std::uint32_t l = 0; l < text->num_hidden_layers; ++l) {
      const std::string p = "thinker.model.layers." + std::to_string(l) + ".";
      DecoderLayer d;
      const std::initializer_list<std::string> qkv = {
          p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight",
          p + "self_attn.v_proj.weight"};
      const std::initializer_list<std::string> gu = {p + "mlp.gate_proj.weight",
                                                     p + "mlp.up_proj.weight"};
      d.qkv_b = UploadBfp(qkv);
      d.o_b = UploadBfp({p + "self_attn.o_proj.weight"});
      d.gate_up_b = UploadBfp(gu);
      d.down_b = UploadBfp({p + "mlp.down_proj.weight"});
      if (!fused) {
        d.qkv_q = UploadSplit(qkv);
        d.o_q = UploadSplit({p + "self_attn.o_proj.weight"});
        d.gate_up_q = UploadSplit(gu);
        d.down_q = UploadSplit({p + "mlp.down_proj.weight"});
      }
      d.input_norm = Vector(p + "input_layernorm.weight");
      d.post_norm = Vector(p + "post_attention_layernorm.weight");
      d.q_norm = Vector(p + "self_attn.q_norm.weight");
      d.k_norm = Vector(p + "self_attn.k_norm.weight");
      if (fused) {
        auto bf = [&](const std::string& n) {
          return reinterpret_cast<const std::uint16_t*>(Find(n).data);
        };
        fused.SetLayer(
            int(l),
            {bf(p + "self_attn.q_proj.weight"),
             bf(p + "self_attn.k_proj.weight"),
             bf(p + "self_attn.v_proj.weight"),
             bf(p + "self_attn.o_proj.weight"), bf(p + "mlp.gate_proj.weight"),
             bf(p + "mlp.up_proj.weight"), bf(p + "mlp.down_proj.weight"),
             d.input_norm.data(), d.post_norm.data(), d.q_norm.data(),
             d.k_norm.data()});
      }
      dec.push_back(std::move(d));
    }
    final_norm = Vector("thinker.model.norm.weight");
    if (fused) {
      fused.SetHead(reinterpret_cast<const std::uint16_t*>(
                        Find("thinker.lm_head.weight").data),
                    int(text->vocab_size), final_norm.data());
      fused.SetRopeTheta(text->rope_theta);
    } else {
      lm_head = UploadSplit({"thinker.lm_head.weight"});
    }
    lane = std::make_unique<Lane>();
    const std::size_t kv =
        std::size_t(text->num_key_value_heads) * text->head_dim;
    k_cache.assign(std::size_t(text->num_hidden_layers) * capacity * kv, 0.0f);
    v_cache.assign(k_cache.size(), 0.0f);
    model.ReleaseHostPages();
  }

  // ---- Audio encoder ----------------------------------------------------

  // Scratch kept across requests, so large buffers are not page-faulted in
  // again per call.
  std::vector<float> conv_a, conv_b, conv_cols, conv_y;

  // h = bf16(h + bf16(y + bias)) over rows of the encoder width.
  static void Residual(float* h, const float* y, const float* bias,
                       std::size_t rows) {
    host::ParallelFor(rows, 8, [&](std::size_t b, std::size_t e) {
      for (std::size_t r = b; r < e; ++r)
        for (std::size_t i = 0; i < kHidden; ++i)
          h[r * kHidden + i] =
              Round(h[r * kHidden + i] + Round(y[r * kHidden + i] + bias[i]));
    });
  }

  static float GeluRound(float v) {
    return Round(host::Activate(host::Activation::kGeluErf, v));
  }

  // conv1: 1 -> 480 channels, 3x3 stride 2 pad 1 over [chunk][128][100],
  // then bias + GELU, as BF16 [chunk][480][64][50].
  void Conv1(const std::vector<float>& in, std::size_t chunks,
             std::vector<float>* out) {
    const std::size_t H = kMelBins, W = kChunkFrames, OH = 64, OW = 50;
    out->resize(chunks * kChannels * OH * OW);
    // Zero-padded input rows [chunk][H + 2][W + 2] remove the edge checks.
    const std::size_t PH = H + 2, PW = W + 2;
    std::vector<float> pad(chunks * PH * PW, 0.0f);
    for (std::size_t c = 0; c < chunks; ++c)
      for (std::size_t y = 0; y < H; ++y)
        std::memcpy(&pad[(c * PH + y + 1) * PW + 1], &in[(c * H + y) * W],
                    W * sizeof(float));
    host::ParallelFor(chunks * kChannels, 4, [&](std::size_t b, std::size_t e) {
      for (std::size_t i = b; i < e; ++i) {
        const std::size_t c = i / kChannels, ch = i % kChannels;
        const float* x = pad.data() + c * PH * PW;
        const float* w = conv1_w.data() + ch * 9;
        float* o = out->data() + i * OH * OW;
        for (std::size_t y = 0; y < OH; ++y) {
          const float* r0 = x + (2 * y) * PW;
          const float* r1 = r0 + PW;
          const float* r2 = r1 + PW;
          for (std::size_t xx = 0; xx < OW; ++xx) {
            const std::size_t s = 2 * xx;
            const float acc =
                r0[s] * w[0] + r0[s + 1] * w[1] + r0[s + 2] * w[2] +
                r1[s] * w[3] + r1[s + 1] * w[4] + r1[s + 2] * w[5] +
                r2[s] * w[6] + r2[s + 1] * w[7] + r2[s + 2] * w[8];
            o[y * OW + xx] = GeluRound(Round(Round(acc) + conv1_b[ch]));
          }
        }
      }
    });
  }

  // 480 -> 480 channels, 3x3 stride 2 pad 1: im2col rows [chunk][oy][ox] by
  // [ci][ky][kx], one NPU product, then bias + GELU back to NCHW.
  void Conv(const std::vector<float>& in, std::size_t chunks, std::size_t H,
            std::size_t W, int id, const std::vector<float>& bias,
            std::vector<float>* out) {
    const std::size_t OH = (H + 1) / 2, OW = (W + 1) / 2, K = kChannels * 9;
    const std::size_t rows = chunks * OH * OW;
    const auto t_conv = Clock::now();
    conv_cols.resize(rows * K);
    host::ParallelFor(rows, 16, [&](std::size_t b, std::size_t e) {
      for (std::size_t r = b; r < e; ++r) {
        const std::size_t c = r / (OH * OW), y = r / OW % OH, xx = r % OW;
        float* dst = conv_cols.data() + r * K;
        long sy[3], sx[3];
        for (int t = 0; t < 3; ++t) {
          sy[t] = long(2 * y) + t - 1;
          sx[t] = long(2 * xx) + t - 1;
        }
        const bool inner =
            sy[0] >= 0 && sy[2] < long(H) && sx[0] >= 0 && sx[2] < long(W);
        for (std::size_t ci = 0; ci < kChannels; ++ci, dst += 9) {
          const float* x = in.data() + (c * kChannels + ci) * H * W;
          if (inner) {
            const float* p = x + std::size_t(sy[0]) * W + std::size_t(sx[0]);
            dst[0] = p[0], dst[1] = p[1], dst[2] = p[2];
            dst[3] = p[W], dst[4] = p[W + 1], dst[5] = p[W + 2];
            dst[6] = p[2 * W], dst[7] = p[2 * W + 1], dst[8] = p[2 * W + 2];
            continue;
          }
          for (int ky = 0; ky < 3; ++ky)
            for (int kx = 0; kx < 3; ++kx)
              dst[ky * 3 + kx] =
                  (sy[ky] < 0 || sy[ky] >= long(H) || sx[kx] < 0 ||
                   sx[kx] >= long(W))
                      ? 0.0f
                      : x[std::size_t(sy[ky]) * W + std::size_t(sx[kx])];
        }
      }
    });
    const double t_cols = Ms(t_conv);
    auto t_g = Clock::now();
    conv_y.resize(rows * kChannels);
    bfp.Run(id, conv_cols.data(), int(rows), conv_y.data());
    const double t_gemm = Ms(t_g);
    t_g = Clock::now();
    out->resize(chunks * kChannels * OH * OW);
    host::ParallelFor(chunks * kChannels, 8, [&](std::size_t b, std::size_t e) {
      for (std::size_t i = b; i < e; ++i) {
        const std::size_t c = i / kChannels, ch = i % kChannels;
        float* o = out->data() + i * OH * OW;
        const float* src = conv_y.data() + c * OH * OW * kChannels + ch;
        for (std::size_t sp = 0; sp < OH * OW; ++sp)
          o[sp] = GeluRound(Round(Round(src[sp * kChannels]) + bias[ch]));
      }
    });
    if (profile)
      std::fprintf(stderr,
                   "  conv rows %zu: im2col %.1f gemm %.1f epilogue %.1f ms\n",
                   rows, t_cols, t_gemm, Ms(t_g));
  }

  std::size_t Encode(std::span<const float> mel, std::size_t frames,
                     std::vector<float>* output,
                     const CancellationCheck& cancelled) {
    if (frames == 0 || mel.size() != kMelBins * frames)
      throw std::invalid_argument(
          "Qwen3-ASR log-mel tensor must have shape [128, frames]");
    const auto t0 = Clock::now();
    const std::size_t chunks = (frames + kChunkFrames - 1) / kChunkFrames;
    std::vector<float> in(chunks * kMelBins * kChunkFrames);
    for (std::size_t c = 0; c < chunks; ++c)
      for (std::size_t m = 0; m < kMelBins; ++m)
        for (std::size_t t = 0; t < kChunkFrames; ++t) {
          const std::size_t src = c * kChunkFrames + t;
          in[(c * kMelBins + m) * kChunkFrames + t] =
              src < frames ? Round(mel[m * frames + src]) : 0.0f;
        }
    auto tc = Clock::now();
    Conv1(in, chunks, &conv_a);
    const auto& c1 = conv_a;
    const double t_c1 = Ms(tc);
    tc = Clock::now();
    Conv(c1, chunks, 64, 50, conv2, conv2_b, &conv_b);
    const auto& c2 = conv_b;
    const double t_c2 = Ms(tc);
    tc = Clock::now();
    std::vector<float> c3;
    Conv(c2, chunks, 32, 25, conv3, conv3_b, &c3);
    const double t_c3 = Ms(tc);
    if (profile)
      std::fprintf(stderr,
                   "qwen3-asr npu: conv1 %.1f conv2 %.1f conv3 %.1f ms\n", t_c1,
                   t_c2, t_c3);
    // [chunk][time] rows of [channel][frequency].
    const std::size_t rows = chunks * kConvTime;
    std::vector<float> flat(rows * kFlattened);
    for (std::size_t r = 0; r < rows; ++r) {
      const std::size_t c = r / kConvTime, t = r % kConvTime;
      for (std::size_t ch = 0; ch < kChannels; ++ch)
        for (std::size_t f = 0; f < kConvFrequency; ++f)
          flat[r * kFlattened + ch * kConvFrequency + f] =
              c3[((c * kChannels + ch) * kConvFrequency + f) * kConvTime + t];
    }
    std::vector<float> projected(rows * kHidden);
    bfp.Run(conv_out, flat.data(), int(rows), projected.data());
    // Keep each chunk's valid times, adding the sinusoidal positions.
    std::vector<float> h;
    for (std::size_t c = 0; c < chunks; ++c) {
      const std::size_t cf = std::min(kChunkFrames, frames - c * kChunkFrames);
      const std::size_t valid = (cf + 7) / 8;
      for (std::size_t t = 0; t < valid; ++t)
        for (std::size_t col = 0; col < kHidden; ++col) {
          const float inc = 9.210340371976184f / 511.0f;
          const std::size_t fq = col % (kHidden / 2);
          const float phase = float(t) * std::exp(-inc * float(fq));
          const float pos =
              Round(col < kHidden / 2 ? std::sin(phase) : std::cos(phase));
          h.push_back(Round(
              Round(projected[(c * kConvTime + t) * kHidden + col]) + pos));
        }
    }
    const double t_front = Ms(t0);
    const std::size_t tokens = h.size() / kHidden;
    double enc_gemm = 0, op_ms[5] = {};
    const auto t_layers = Clock::now();
    std::vector<float> xn(tokens * kHidden), qkv(tokens * 3 * kHidden),
        att(tokens * kHidden), proj(tokens * kHidden), ffn(tokens * kFfn);
    for (const EncoderLayer& w : enc) {
      if (cancelled && cancelled())
        throw std::runtime_error("Qwen3-ASR transcription cancelled");
      {
        const auto to = Clock::now();
        host::LayerNormRows(h.data(), xn.data(), tokens, kHidden,
                            w.ln1_w.data(), w.ln1_b.data(), kLayerNormEps);
        op_ms[0] += Ms(to);
      }
      {
        const auto tb = Clock::now();
        bfp.Run(w.qkv, xn.data(), int(tokens), qkv.data());
        enc_gemm += Ms(tb);
      }
      host::ParallelFor(tokens, 4, [&](std::size_t b, std::size_t e) {
        for (std::size_t r = b; r < e; ++r)
          for (std::size_t i = 0; i < 3 * kHidden; ++i)
            qkv[r * 3 * kHidden + i] =
                Round(qkv[r * 3 * kHidden + i] + w.qkv_bias[i]);
      });
      {
        const auto to = Clock::now();
        host::Attention({.q = qkv.data(),
                         .q_stride = 3 * kHidden,
                         .k = qkv.data() + kHidden,
                         .k_stride = 3 * kHidden,
                         .v = qkv.data() + 2 * kHidden,
                         .v_stride = 3 * kHidden,
                         .out = att.data(),
                         .out_stride = kHidden,
                         .rows = tokens,
                         .heads = kHeads,
                         .kv_heads = kHeads,
                         .dim = kHeadDim,
                         .dim_v = kHeadDim,
                         .scale = 0.125f},
                        [&](std::size_t, std::size_t* b, std::size_t* e) {
                          *b = 0;
                          *e = tokens;
                        });
        op_ms[1] += Ms(to);
      }
      {
        const auto tb = Clock::now();
        bfp.Run(w.out, att.data(), int(tokens), proj.data());
        enc_gemm += Ms(tb);
      }
      {
        const auto to = Clock::now();
        Residual(h.data(), proj.data(), w.out_bias.data(), tokens);
        op_ms[3] += Ms(to);
      }
      {
        const auto to = Clock::now();
        host::LayerNormRows(h.data(), xn.data(), tokens, kHidden,
                            w.ln2_w.data(), w.ln2_b.data(), kLayerNormEps);
        op_ms[0] += Ms(to);
      }
      {
        const auto tb = Clock::now();
        bfp.Run(w.fc1, xn.data(), int(tokens), ffn.data());
        enc_gemm += Ms(tb);
      }
      {
        const auto to = Clock::now();
        host::ActivateRows(ffn.data(), tokens, kFfn, w.fc1_bias.data(),
                           host::Activation::kGeluErf);
        op_ms[2] += Ms(to);
      }
      {
        const auto tb = Clock::now();
        bfp.Run(w.fc2, ffn.data(), int(tokens), proj.data());
        enc_gemm += Ms(tb);
      }
      {
        const auto to = Clock::now();
        Residual(h.data(), proj.data(), w.fc2_bias.data(), tokens);
        op_ms[3] += Ms(to);
      }
    }
    if (profile)
      std::fprintf(
          stderr,
          "qwen3-asr npu: encoder layers %.1f ms, gemm %.1f ms, ln %.1f "
          "attn %.1f act %.1f res %.1f bias %.1f\n",
          Ms(t_layers), enc_gemm, op_ms[0], op_ms[1], op_ms[2], op_ms[3],
          op_ms[4]);
    host::LayerNormRows(h.data(), xn.data(), tokens, kHidden, ln_post_w.data(),
                        ln_post_b.data(), kLayerNormEps);
    bfp.Run(proj1, xn.data(), int(tokens), proj.data());
    host::ActivateRows(proj.data(), tokens, kHidden, proj1_b.data(),
                       host::Activation::kGeluErf);
    output->resize(tokens * kOutput);
    bfp.Run(proj2, proj.data(), int(tokens), output->data());
    for (std::size_t i = 0; i < output->size(); ++i)
      (*output)[i] = Round((*output)[i] + proj2_b[i % kOutput]);
    if (profile)
      std::fprintf(
          stderr,
          "qwen3-asr npu: encoder %zu frames -> %zu tokens, front %.1f "
          "ms, total %.1f ms\n",
          frames, tokens, t_front, Ms(t0));
    return tokens;
  }

  // ---- Text decoder -----------------------------------------------------

  std::size_t KvDim() const {
    return std::size_t(text->num_key_value_heads) * text->head_dim;
  }

  // q/k norms and RoPE on rows of fused qkv at positions [start, start + n),
  // then the keys and values into the layer's cache.
  void QkvPost(std::size_t layer, float* qkv, std::size_t n,
               std::size_t start) {
    const DecoderLayer& w = dec[layer];
    const std::size_t hd = text->head_dim, heads = text->num_attention_heads,
                      kvh = text->num_key_value_heads;
    const std::size_t qw = heads * hd, kw = kvh * hd, row = qw + 2 * kw;
    const float eps = text->rms_norm_eps;
    host::ParallelFor(n, 8, [&](std::size_t b, std::size_t e) {
      for (std::size_t r = b; r < e; ++r) {
        float* q = qkv + r * row;
        host::RmsNormRows(q, q, heads, hd, w.q_norm.data(), eps);
        host::RmsNormRows(q + qw, q + qw, kvh, hd, w.k_norm.data(), eps);
      }
    });
    std::vector<std::int32_t> pos(n);
    for (std::size_t r = 0; r < n; ++r)
      pos[r] = std::int32_t(start + r);
    host::RopeNeox(qkv, n, row, heads + kvh, hd, hd, pos.data(),
                   text->rope_theta);
    const std::size_t kv = KvDim();
    float* kc = k_cache.data() + layer * capacity * kv;
    float* vc = v_cache.data() + layer * capacity * kv;
    for (std::size_t r = 0; r < n; ++r) {
      std::memcpy(kc + (start + r) * kv, qkv + r * row + qw, kv * 4);
      std::memcpy(vc + (start + r) * kv, qkv + r * row + qw + kw, kv * 4);
    }
  }

  void Attend(std::size_t layer, const float* qkv, std::size_t n,
              std::size_t start, float* out) {
    const std::size_t hd = text->head_dim, heads = text->num_attention_heads,
                      kvh = text->num_key_value_heads;
    const std::size_t row = (heads + 2 * kvh) * hd, kv = KvDim();
    host::Attention({.q = qkv,
                     .q_stride = row,
                     .k = k_cache.data() + layer * capacity * kv,
                     .k_stride = kv,
                     .v = v_cache.data() + layer * capacity * kv,
                     .v_stride = kv,
                     .out = out,
                     .out_stride = heads * hd,
                     .rows = n,
                     .heads = heads,
                     .kv_heads = kvh,
                     .dim = hd,
                     .dim_v = hd,
                     .scale = 1.0f / std::sqrt(float(hd))},
                    [&](std::size_t r, std::size_t* b, std::size_t* e) {
                      *b = 0;
                      *e = start + r + 1;
                    });
  }

  // One pass of n rows at positions [start, start + n) through the layers;
  // `x` [n][hidden] becomes the final hidden states.
  void Layers(float* x, std::size_t n, std::size_t start, bool prefill) {
    const std::size_t hidden = text->hidden_size,
                      inner = text->intermediate_size;
    const std::size_t hd = text->head_dim, heads = text->num_attention_heads,
                      kvh = text->num_key_value_heads;
    const std::size_t row = (heads + 2 * kvh) * hd;
    const float eps = text->rms_norm_eps;
    std::vector<float> xn(n * hidden), qkv(n * row), att(n * heads * hd),
        tmp(n * hidden), gu(n * 2 * inner), act(n * inner);
    auto run = [&](int bid, const std::unique_ptr<SplitQ8>& q, const float* in,
                   float* out) {
      const auto t = Clock::now();
      if (prefill)
        bfp.Run(bid, in, int(n), out);
      else
        q->Run(*lane, in, out);
      gemm_ms += Ms(t);
    };
    for (std::size_t l = 0; l < dec.size(); ++l) {
      const DecoderLayer& w = dec[l];
      host::RmsNormRows(x, xn.data(), n, hidden, w.input_norm.data(), eps);
      run(w.qkv_b, w.qkv_q, xn.data(), qkv.data());
      QkvPost(l, qkv.data(), n, start);
      Attend(l, qkv.data(), n, start, att.data());
      run(w.o_b, w.o_q, att.data(), tmp.data());
      for (std::size_t i = 0; i < n * hidden; ++i)
        x[i] += tmp[i];
      host::RmsNormRows(x, xn.data(), n, hidden, w.post_norm.data(), eps);
      run(w.gate_up_b, w.gate_up_q, xn.data(), gu.data());
      host::GatedRows(gu.data(), 2 * inner, gu.data() + inner, 2 * inner,
                      act.data(), inner, n, inner, host::Activation::kSilu);
      run(w.down_b, w.down_q, act.data(), tmp.data());
      for (std::size_t i = 0; i < n * hidden; ++i)
        x[i] += tmp[i];
    }
  }

  // The greedy token of bf16-rounded logits (vocab_size when none is finite).
  std::uint32_t Argmax(std::span<const float> logits) const {
    std::uint32_t best = text->vocab_size;
    float top = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < text->vocab_size; ++i) {
      const float v = Round(logits[i]);
      if (std::isfinite(v) && v > top) {
        top = v;
        best = std::uint32_t(i);
      }
    }
    return best;
  }

  std::uint32_t Select(const float* last) {
    const std::size_t hidden = text->hidden_size;
    std::vector<float> xn(hidden), logits(text->vocab_size);
    host::RmsNormRows(last, xn.data(), 1, hidden, final_norm.data(),
                      text->rms_norm_eps);
    const auto t = Clock::now();
    lm_head->Run(*lane, xn.data(), logits.data());
    head_ms += Ms(t);
    std::uint32_t best = text->vocab_size;
    float top = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < logits.size(); ++i) {
      const float v = Round(logits[i]);
      if (std::isfinite(v) && v > top) {
        top = v;
        best = std::uint32_t(i);
      }
    }
    return best;
  }

  void Embed(std::uint32_t token, float* out) const {
    const std::size_t hidden = text->hidden_size;
    for (std::size_t i = 0; i < hidden; ++i)
      out[i] = host::Bf16ToFloat(embed[std::size_t(token) * hidden + i]);
  }

  void Generate(
      std::span<const std::uint32_t> prompt, std::span<const float> audio,
      std::size_t audio_tokens, std::size_t max_new,
      std::vector<std::uint32_t>* generated,
      const std::function<bool(std::span<const std::uint32_t>)>& on_tokens,
      const CancellationCheck& cancelled) {
    const std::size_t hidden = text->hidden_size;
    if (prompt.empty() || prompt.size() + max_new > capacity ||
        audio.size() != audio_tokens * hidden)
      throw std::invalid_argument("Qwen3-ASR text prompt shape is invalid");
    const auto t0 = Clock::now();
    const std::size_t n = prompt.size();
    std::vector<float> x(n * hidden);
    std::size_t audio_at = n, found = 0;
    for (std::size_t i = 0; i < n; ++i) {
      if (prompt[i] >= text->vocab_size)
        throw std::out_of_range("Qwen3-ASR prompt token is out of range");
      if (prompt[i] == model.config.audio_token_id) {
        if (found == 0)
          audio_at = i;
        ++found;
      }
      Embed(prompt[i], x.data() + i * hidden);
    }
    if (found != audio_tokens)
      throw std::invalid_argument(
          "Qwen3-ASR prompt/audio embedding count differs");
    for (std::size_t i = 0; i < audio.size(); ++i)
      x[audio_at * hidden + i] = Round(audio[i]);
    std::uint32_t token;
    if (fused) {
      // Prefill all but the last prompt row, which the fused program decodes
      // for the first token's logits.
      if (n > 1)
        Layers(x.data(), n - 1, 0, true);
      const std::size_t kv = KvDim();
      for (std::size_t l = 0; l < dec.size(); ++l)
        fused.WriteKv(int(l), 0, int(n - 1), k_cache.data() + l * capacity * kv,
                      kv, v_cache.data() + l * capacity * kv, kv);
      fused.Start(x.data() + (n - 1) * hidden, int(n - 1));
      fused.Wait();
      token = Argmax(fused.logits());
    } else {
      Layers(x.data(), n, 0, true);
      token = Select(x.data() + (n - 1) * hidden);
    }
    const double t_prefill = Ms(t0);
    const auto t1 = Clock::now();
    gemm_ms = head_ms = 0;
    generated->clear();
    auto push = [&](std::uint32_t t) {
      if (t >= text->vocab_size)
        throw std::runtime_error(
            "Qwen3-ASR decoder produced non-finite logits");
      generated->push_back(t);
      if (on_tokens && !on_tokens(*generated))
        throw std::runtime_error("Qwen3-ASR generation cancelled");
    };
    push(token);
    std::vector<float> h(hidden);
    std::size_t pos = n;
    while (token != kEndOfText && token != kImEnd &&
           generated->size() < max_new) {
      if (cancelled && cancelled())
        throw std::runtime_error("Qwen3-ASR transcription cancelled");
      Embed(token, h.data());
      if (fused) {
        const auto t = Clock::now();
        fused.Start(h.data(), int(pos++));
        fused.Wait();
        gemm_ms += Ms(t);
        token = Argmax(fused.logits());
      } else {
        Layers(h.data(), 1, pos++, false);
        token = Select(h.data());
      }
      push(token);
    }
    if (profile)
      std::fprintf(
          stderr,
          "qwen3-asr npu: prefill %zu tokens %.1f ms, %zu decode steps "
          "%.1f ms (%.1f ms/token; layer gemm %.1f, head %.1f ms)\n",
          n, t_prefill, generated->size() - 1, Ms(t1),
          generated->size() > 1 ? Ms(t1) / double(generated->size() - 1) : 0.0,
          gemm_ms, head_ms);
  }
};

Runtime::Runtime() : impl_(std::make_unique<Impl>()) {}
Runtime::~Runtime() = default;

std::unique_ptr<Runtime> Runtime::Create(const std::string& model_root,
                                         std::size_t maximum_context_tokens) {
  std::unique_ptr<Runtime> r(new Runtime());
  r->impl_->Load(model_root, maximum_context_tokens);
  return r;
}

std::size_t Runtime::Encode(std::span<const float> log_mel, std::size_t frames,
                            std::vector<float>* embeddings,
                            const CancellationCheck& is_cancelled) {
  return impl_->Encode(log_mel, frames, embeddings, is_cancelled);
}

void Runtime::Generate(
    std::span<const std::uint32_t> prompt_ids,
    std::span<const float> audio_embeddings, std::size_t audio_tokens,
    std::size_t maximum_new_tokens, std::vector<std::uint32_t>* generated,
    const std::function<bool(std::span<const std::uint32_t>)>& on_tokens,
    const CancellationCheck& is_cancelled) {
  impl_->Generate(prompt_ids, audio_embeddings, audio_tokens,
                  maximum_new_tokens, generated, on_tokens, is_cancelled);
}

}  // namespace gufo::models::qwen3_asr::npu
