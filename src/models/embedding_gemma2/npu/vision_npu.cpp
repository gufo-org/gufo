#include "src/models/embedding_gemma2/npu/vision_npu.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "src/models/embedding_gemma2/loader.hpp"
#include "src/models/embedding_gemma2/tensor.hpp"

namespace gufo::models::embedding_gemma2::npu {
namespace host = gufo::npu::host;

namespace {

constexpr std::size_t kWidth = 768, kHeads = 12, kHeadDim = 64, kFfn = 3072;
constexpr std::size_t kLayers = 16, kText = 512, kTableRows = 10240;
constexpr float kEps = 1e-6f, kRopeTheta = 100.0f;
// Attention scores held at once (floats): bounds the heads per pass.
constexpr std::size_t kScoreBudget = std::size_t(96) << 20;
constexpr std::size_t kScratchBytes = std::size_t(64) << 20;

std::vector<float> ToFloat(const Tensor& t) {
  const std::size_t n = std::size_t(*t.NumElements());
  std::vector<float> out(n);
  host::Bf16ToFloat(reinterpret_cast<const std::uint16_t*>(t.data), out.data(),
                    n);
  return out;
}

struct Layer {
  int qkv, o, gate_up, down;
  std::vector<float> input_norm, post_attn_norm, pre_ffn_norm, post_ffn_norm,
      q_norm, k_norm;
};

}  // namespace

struct VisionTower::Impl {
  gufo::npu::BfpGemm& gemm;
  gufo::npu::BfpGemm::Scratch scratch;
  int input, projection;
  std::vector<std::uint16_t> table;  // [2][10240][768] bf16
  std::vector<Layer> layers;
  // Request buffers, kept across requests.
  std::vector<float> pixels, state, normed, qkv, q, k, v, context, tmp, gate_up,
      act, scores, head_in, head_out, pooled, features;

  explicit Impl(gufo::npu::BfpGemm& g) : gemm(g) {}

  void Load(const std::string& model_dir) {
    auto loaded = LoadModelDirectory(model_dir, true, false);
    if (!loaded.ok)
      throw std::runtime_error(loaded.error);
    const TensorStore& store = *loaded.store;
    auto find = [&](const std::string& name) -> const Tensor& {
      const Tensor* t = store.Find(name);
      if (!t || t->dtype != DType::kBF16)
        throw std::runtime_error("missing BF16 tensor " + name);
      return *t;
    };
    auto upload = [&](std::initializer_list<std::string> names) {
      std::vector<std::uint16_t> rows;
      int n = 0, k = 0;
      for (const std::string& name : names) {
        const Tensor& t = find(name);
        n += int(t.shape[0]);
        k = int(t.shape[1]);
        const auto* p = reinterpret_cast<const std::uint16_t*>(t.data);
        rows.insert(rows.end(), p, p + t.shape[0] * t.shape[1]);
      }
      return gemm.UploadBf16(n, k, rows);
    };
    input = upload({"vision_tower.patch_embedder.input_proj.weight"});
    projection = upload({"embed_vision.embedding_projection.weight"});
    const Tensor& t =
        find("vision_tower.patch_embedder.position_embedding_table");
    const auto* tp = reinterpret_cast<const std::uint16_t*>(t.data);
    table.assign(tp, tp + std::size_t(*t.NumElements()));
    if (table.size() != 2 * kTableRows * kWidth)
      throw std::runtime_error("vision position table shape");
    for (std::size_t i = 0; i < kLayers; ++i) {
      const std::string p =
          "vision_tower.encoder.layers." + std::to_string(i) + ".";
      Layer l;
      l.qkv = upload({p + "self_attn.q_proj.linear.weight",
                      p + "self_attn.k_proj.linear.weight",
                      p + "self_attn.v_proj.linear.weight"});
      l.o = upload({p + "self_attn.o_proj.linear.weight"});
      l.gate_up = upload(
          {p + "mlp.gate_proj.linear.weight", p + "mlp.up_proj.linear.weight"});
      l.down = upload({p + "mlp.down_proj.linear.weight"});
      l.input_norm = ToFloat(find(p + "input_layernorm.weight"));
      l.post_attn_norm = ToFloat(find(p + "post_attention_layernorm.weight"));
      l.pre_ffn_norm = ToFloat(find(p + "pre_feedforward_layernorm.weight"));
      l.post_ffn_norm = ToFloat(find(p + "post_feedforward_layernorm.weight"));
      l.q_norm = ToFloat(find(p + "self_attn.q_norm.weight"));
      l.k_norm = ToFloat(find(p + "self_attn.k_norm.weight"));
      layers.push_back(std::move(l));
    }
    scratch = gemm.NewScratch(kScratchBytes);
  }

  static void Grow(std::vector<float>& v, std::size_t n) {
    if (v.size() < n)
      v.resize(n);
  }

  // Bidirectional attention over the image's patches, scale 1: per head,
  // scores = Q K^T and context = P V on the NPU, the softmax on the CPU.
  void Attention(std::size_t n) {
    const std::size_t per_head = n * n;
    // Scratch bytes per head: keys (n outputs x 64) and values (64 x n), as
    // BfpGemm pads them (1024-output groups, 512-input blocks).
    const std::size_t tile = std::size_t(2304) * 4 * 8;
    const std::size_t head_bytes =
        (n + 1023) / 1024 * tile * 8 + (n + 511) / 512 * 8 * tile;
    const std::size_t group =
        std::max<std::size_t>(1, std::min({kHeads, kScoreBudget / per_head,
                                           kScratchBytes / head_bytes}));
    Grow(scores, group * per_head);
    Grow(head_in, group * n * kHeadDim);
    Grow(head_out, group * n * kHeadDim);
    for (std::size_t h0 = 0; h0 < kHeads; h0 += group) {
      const std::size_t hs = std::min(group, kHeads - h0);
      scratch.Reset();
      std::vector<int> keys(hs), values(hs);
      for (std::size_t j = 0; j < hs; ++j) {
        const float* kh = k.data() + (h0 + j) * kHeadDim;
        const float* vh = v.data() + (h0 + j) * kHeadDim;
        keys[j] = scratch.Upload(int(n), int(kHeadDim), kh, kWidth, 1);
        values[j] = scratch.Upload(int(kHeadDim), int(n), vh, 1, kWidth);
      }
      host::ParallelFor(hs * n, 64, [&](std::size_t b, std::size_t e) {
        for (std::size_t i = b; i < e; ++i) {
          const std::size_t j = i / n, r = i % n;
          std::memcpy(head_in.data() + (j * n + r) * kHeadDim,
                      q.data() + r * kWidth + (h0 + j) * kHeadDim,
                      kHeadDim * sizeof(float));
        }
      });
      std::vector<gufo::npu::BfpGemm::Call> calls;
      for (std::size_t j = 0; j < hs; ++j)
        calls.push_back({keys[j], head_in.data() + j * n * kHeadDim, int(n),
                         scores.data() + j * per_head});
      gemm.Run(calls);
      host::ParallelFor(hs * n, 16, [&](std::size_t b, std::size_t e) {
        for (std::size_t i = b; i < e; ++i) {
          float* row = scores.data() + i * n;
          float top = -INFINITY;
          for (std::size_t t = 0; t < n; ++t)
            top = std::max(top, row[t]);
          float sum = 0;
          host::ExpInPlace(row, n, top, &sum);
          const float inv = 1.0f / sum;
          for (std::size_t t = 0; t < n; ++t)
            row[t] *= inv;
        }
      });
      calls.clear();
      for (std::size_t j = 0; j < hs; ++j)
        calls.push_back({values[j], scores.data() + j * per_head, int(n),
                         head_out.data() + j * n * kHeadDim});
      gemm.Run(calls);
      host::ParallelFor(hs * n, 64, [&](std::size_t b, std::size_t e) {
        for (std::size_t i = b; i < e; ++i) {
          const std::size_t j = i / n, r = i % n;
          float* dst = context.data() + r * kWidth + (h0 + j) * kHeadDim;
          std::memcpy(dst, head_out.data() + (j * n + r) * kHeadDim,
                      kHeadDim * sizeof(float));
          host::RoundBf16(dst, kHeadDim);
        }
      });
    }
  }

  std::vector<float> Run(const ImagePatches& image,
                         const std::function<bool()>& cancelled) {
    const std::size_t n = std::size_t(image.rows) * image.columns;
    const std::size_t columns = image.columns;
    if (n == 0 || image.pixels.size() != n * kWidth || image.rows % 3 ||
        columns % 3)
      throw std::invalid_argument("embedding_gemma2: invalid image patches");
    Grow(pixels, n * kWidth);
    Grow(state, n * kWidth);
    Grow(normed, n * kWidth);
    Grow(qkv, n * 3 * kWidth);
    Grow(q, n * kWidth);
    Grow(k, n * kWidth);
    Grow(v, n * kWidth);
    Grow(context, n * kWidth);
    Grow(tmp, n * kWidth);
    Grow(gate_up, n * 2 * kFfn);
    Grow(act, n * kFfn);

    // Patch embedding: bf16(2 (p - 0.5)) x input_proj + x/y position rows.
    host::ParallelFor(n, 16, [&](std::size_t b, std::size_t e) {
      for (std::size_t i = b * kWidth; i < e * kWidth; ++i)
        pixels[i] = 2.0f * (image.pixels[i] - 0.5f);
      host::RoundBf16(pixels.data() + b * kWidth, (e - b) * kWidth);
    });
    gemm.Run(input, pixels.data(), int(n), state.data());
    host::ParallelFor(n, 16, [&](std::size_t b, std::size_t e) {
      for (std::size_t p = b; p < e; ++p) {
        const std::uint16_t* tx = table.data() + (p % columns) * kWidth;
        const std::uint16_t* ty =
            table.data() + (kTableRows + p / columns) * kWidth;
        float* s = state.data() + p * kWidth;
        for (std::size_t c = 0; c < kWidth; ++c)
          s[c] += host::Bf16ToFloat(tx[c]) + host::Bf16ToFloat(ty[c]);
      }
    });
    std::vector<std::int32_t> pos_x(n), pos_y(n);
    for (std::size_t p = 0; p < n; ++p) {
      pos_x[p] = std::int32_t(p % columns);
      pos_y[p] = std::int32_t(p / columns);
    }

    for (const Layer& l : layers) {
      if (cancelled && cancelled())
        throw std::runtime_error("embedding_gemma2: request cancelled");
      host::RmsNormRows(state.data(), normed.data(), n, kWidth,
                        l.input_norm.data(), kEps);
      host::RoundBf16(normed.data(), n * kWidth);
      gemm.Run(l.qkv, normed.data(), int(n), qkv.data());
      // Per-head norms (q, k weighted; v unweighted), then axial RoPE: the
      // first 32 channels of a head rotate by x, the last 32 by y.
      host::ParallelFor(n, 16, [&](std::size_t b, std::size_t e) {
        for (std::size_t r = b; r < e; ++r) {
          const float* src = qkv.data() + r * 3 * kWidth;
          host::RmsNormRows(src, q.data() + r * kWidth, kHeads, kHeadDim,
                            l.q_norm.data(), kEps);
          host::RmsNormRows(src + kWidth, k.data() + r * kWidth, kHeads,
                            kHeadDim, l.k_norm.data(), kEps);
          host::RmsNormRows(src + 2 * kWidth, v.data() + r * kWidth, kHeads,
                            kHeadDim, nullptr, kEps);
        }
      });
      for (float* x : {q.data(), k.data()}) {
        host::RopeNeox(x, n, kWidth, kHeads, kHeadDim, 32, pos_x.data(),
                       kRopeTheta);
        host::RopeNeox(x + 32, n, kWidth, kHeads, kHeadDim, 32, pos_y.data(),
                       kRopeTheta);
      }
      host::RoundBf16(q.data(), n * kWidth);
      host::RoundBf16(k.data(), n * kWidth);
      host::RoundBf16(v.data(), n * kWidth);
      Attention(n);
      gemm.Run(l.o, context.data(), int(n), tmp.data());
      host::AddNormRows(state.data(), tmp.data(), n, kWidth,
                        l.post_attn_norm.data(), kEps, 1.0f);
      host::RmsNormRows(state.data(), normed.data(), n, kWidth,
                        l.pre_ffn_norm.data(), kEps);
      host::RoundBf16(normed.data(), n * kWidth);
      gemm.Run(l.gate_up, normed.data(), int(n), gate_up.data());
      host::GatedRows(gate_up.data(), 2 * kFfn, gate_up.data() + kFfn, 2 * kFfn,
                      act.data(), kFfn, n, kFfn, host::Activation::kGeluTanh);
      host::RoundBf16(act.data(), n * kFfn);
      gemm.Run(l.down, act.data(), int(n), tmp.data());
      host::AddNormRows(state.data(), tmp.data(), n, kWidth,
                        l.post_ffn_norm.data(), kEps, 1.0f);
    }

    // 3x3 average pool, x sqrt(768), unweighted RMS norm, bf16, projection.
    const std::size_t tokens = image.soft_tokens(), across = columns / 3;
    Grow(pooled, tokens * kWidth);
    host::ParallelFor(tokens, 8, [&](std::size_t b, std::size_t e) {
      for (std::size_t t = b; t < e; ++t) {
        const std::size_t x = t % across * 3, y = t / across * 3;
        float* o = pooled.data() + t * kWidth;
        for (std::size_t c = 0; c < kWidth; ++c) {
          float total = 0;
          for (std::size_t dy = 0; dy < 3; ++dy)
            for (std::size_t dx = 0; dx < 3; ++dx)
              total += state[((y + dy) * columns + x + dx) * kWidth + c] / 9.0f;
          o[c] = total * std::sqrt(float(kWidth));
        }
      }
    });
    host::RmsNormRows(pooled.data(), pooled.data(), tokens, kWidth, nullptr,
                      kEps);
    host::RoundBf16(pooled.data(), tokens * kWidth);
    std::vector<float> out(tokens * kText);
    gemm.Run(projection, pooled.data(), int(tokens), out.data());
    host::RoundBf16(out.data(), out.size());
    return out;
  }
};

VisionTower::VisionTower(gufo::npu::BfpGemm& gemm, const std::string& model_dir)
    : impl_(std::make_unique<Impl>(gemm)) {
  impl_->Load(model_dir);
}

VisionTower::~VisionTower() = default;

std::vector<float> VisionTower::Run(const ImagePatches& image,
                                    const std::function<bool()>& cancelled) {
  return impl_->Run(image, cancelled);
}

}  // namespace gufo::models::embedding_gemma2::npu
