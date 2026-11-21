#include "staging/models/glm5_next/vision/encoder.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "staging/models/glm5_next/kernels/rocm/ops.hpp"
#include "staging/models/glm5_next/vision/kernels.hpp"
#include "staging/models/glm5_next/vision/weights.hpp"

namespace g5r = gufo::models::glm5_next::rocm;

namespace gufo::models::glm5_next::vision {
namespace {

constexpr std::uint32_t kPatchVec = 588;  // 14 * 14 * 3

struct DBlock {
  float* ln1{nullptr};
  float* qkv{nullptr};        ///< [3072 x 1024]; q/k/v are consecutive slices.
  float* qkv_bias{nullptr};   ///< [3072]
  float* q_norm{nullptr};
  float* k_norm{nullptr};
  float* attn_out{nullptr};
  float* attn_bias{nullptr};
  float* ln2{nullptr};
  float* ffn_up{nullptr};
  float* ffn_up_bias{nullptr};
  float* ffn_gate{nullptr};
  float* ffn_gate_bias{nullptr};
  float* ffn_down{nullptr};
  float* ffn_down_bias{nullptr};
};

}  // namespace

struct Encoder::Impl {
  Config config;
  std::uint32_t proj_ffn{0};  ///< Projector FFN hidden width (mm.up rows).
  float* patch_weight{nullptr};
  float* patch_bias{nullptr};
  float* post_ln{nullptr};
  float* inv_freq{nullptr};
  std::vector<DBlock> blocks;
  float* merger{nullptr};
  float* merger_bias{nullptr};
  float* fc{nullptr};
  float* ln_weight{nullptr};
  float* ln_bias{nullptr};
  float* up{nullptr};
  float* gate{nullptr};
  float* down{nullptr};
  std::vector<void*> allocs;
  std::size_t bytes{0};

  ~Impl() {
    for (void* p : allocs) {
      (void)hipFree(p);
    }
  }
};

namespace {

[[nodiscard]] float HalfToFloat(std::uint16_t h) noexcept {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000U) << 16;
  const std::uint32_t expo = (h >> 10) & 0x1FU;
  const std::uint32_t mant = h & 0x3FFU;
  std::uint32_t bits = sign;
  if (expo == 0) {
    if (mant != 0) {
      std::uint32_t m = mant;
      std::uint32_t e = 0x7FU - 0x0FU;
      while ((m & 0x400U) == 0) {
        m <<= 1;
        --e;
      }
      m &= 0x3FFU;
      bits |= (e << 23) | (m << 13);
    }
  } else if (expo == 0x1FU) {
    bits |= 0x7F800000U | (mant << 13);
  } else {
    bits |= ((expo + 0x7FU - 0x0FU) << 23) | (mant << 13);
  }
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

struct Uploader {
  std::vector<void*>& allocs;
  std::size_t& bytes;
  std::string* error;
  bool ok{true};

  void Fail(const std::string& msg) {
    if (ok && error != nullptr) {
      *error = msg;
    }
    ok = false;
  }

  float* CopyRaw(const float* src, std::size_t elems, const std::string& what) {
    if (!ok) {
      return nullptr;
    }
    float* ptr = nullptr;
    const std::size_t size = elems * sizeof(float);
    if (hipMalloc(&ptr, size) != hipSuccess) {
      Fail("hipMalloc failed for " + what);
      return nullptr;
    }
    allocs.push_back(ptr);
    bytes += size;
    if (hipMemcpy(ptr, src, size, hipMemcpyHostToDevice) != hipSuccess) {
      Fail("upload failed for " + what);
      return nullptr;
    }
    return ptr;
  }

  float* Floats(const std::vector<float>& host, const std::string& what) {
    return CopyRaw(host.data(), host.size(), what);
  }

  /// Uploads an F16 or F32 tensor as F32 (the F16 mmproj weights are upcast
  /// host-side so the tower computes in full precision, matching the CPU
  /// oracle; it is tiny next to the text model).
  float* Proj(const TensorRef& t) {
    if (!ok || t.empty()) {
      return nullptr;
    }
    const std::size_t elems = static_cast<std::size_t>(t.ElementCount());
    if (t.type == core::GgmlType::kF32) {
      return CopyRaw(static_cast<const float*>(t.data), elems,
                     std::string(t.name));
    }
    if (t.type == core::GgmlType::kF16) {
      const auto* bits = static_cast<const std::uint16_t*>(t.data);
      std::vector<float> host(elems);
      for (std::size_t i = 0; i < elems; ++i) {
        host[i] = HalfToFloat(bits[i]);
      }
      return CopyRaw(host.data(), elems, std::string(t.name));
    }
    Fail("unsupported vision tensor type for " + std::string(t.name));
    return nullptr;
  }
};

}  // namespace

Encoder::Encoder(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Encoder::~Encoder() = default;

std::unique_ptr<Encoder> Encoder::Open(const std::string& mmproj_path,
                                       std::string* error_msg) {
  const auto reader = core::GgufReader::OpenFile(mmproj_path, error_msg);
  if (reader == nullptr) {
    return nullptr;
  }
  const auto weights = VisionWeights::Bind(*reader, error_msg);
  if (!weights.has_value()) {
    return nullptr;
  }
  auto impl = std::make_unique<Impl>();
  impl->config = weights->config;
  impl->proj_ffn = static_cast<std::uint32_t>(weights->up.rows);
  Uploader up{impl->allocs, impl->bytes, error_msg};
  impl->patch_weight = up.Floats(weights->patch_weight, "patch_weight");
  impl->patch_bias = up.Proj(weights->patch_bias);
  impl->post_ln = up.Proj(weights->post_ln);
  impl->blocks.reserve(weights->blocks.size());
  for (const BlockWeights& b : weights->blocks) {
    if (!up.ok) {
      break;
    }
    DBlock d;
    d.ln1 = up.Proj(b.ln1);
    d.qkv = up.Proj(b.qkv);
    d.qkv_bias = up.Proj(b.qkv_bias);
    d.q_norm = up.Proj(b.q_norm);
    d.k_norm = up.Proj(b.k_norm);
    d.attn_out = up.Proj(b.attn_out);
    d.attn_bias = up.Proj(b.attn_bias);
    d.ln2 = up.Proj(b.ln2);
    d.ffn_up = up.Proj(b.ffn_up);
    d.ffn_up_bias = up.Proj(b.ffn_up_bias);
    d.ffn_gate = up.Proj(b.ffn_gate);
    d.ffn_gate_bias = up.Proj(b.ffn_gate_bias);
    d.ffn_down = up.Proj(b.ffn_down);
    d.ffn_down_bias = up.Proj(b.ffn_down_bias);
    impl->blocks.push_back(d);
  }
  impl->merger = up.Proj(weights->merger);
  impl->merger_bias = up.Proj(weights->merger_bias);
  impl->fc = up.Proj(weights->fc);
  impl->ln_weight = up.Proj(weights->ln_weight);
  impl->ln_bias = up.Proj(weights->ln_bias);
  impl->up = up.Proj(weights->up);
  impl->gate = up.Proj(weights->gate);
  impl->down = up.Proj(weights->down);
  // Vision RoPE inverse frequencies: the pair stride is d_head/2, so
  // theta_scale^p = theta^(-2p/d_head) for p in [0, d_head/4) (mtmd
  // GGML_ROPE_TYPE_VISION with sections d_head/4).
  const std::uint32_t quarter = impl->config.RopeHalfDim();
  if (quarter == 0 || !up.ok) {
    return nullptr;
  }
  std::vector<float> inv(quarter);
  const float scale = static_cast<float>(
      std::pow(static_cast<double>(Config::kRopeTheta),
               -2.0 / static_cast<double>(2 * quarter)));
  float acc = 1.0F;
  for (std::uint32_t p = 0; p < quarter; ++p) {
    inv[p] = acc;
    acc *= scale;
  }
  impl->inv_freq = up.CopyRaw(inv.data(), inv.size(), "inv_freq");
  if (!up.ok || impl->inv_freq == nullptr) {
    return nullptr;
  }
  return std::unique_ptr<Encoder>(new Encoder(std::move(impl)));
}

bool Encoder::Encode(const float* pixels, std::uint32_t nx,
                     std::uint32_t ny, std::vector<float>& out,
                     std::string* error_msg) {
  Impl& m = *impl_;
  const Config& c = m.config;
  const std::uint32_t patch = c.patch_size;
  const std::uint32_t merge = c.merge_size;
  const std::uint32_t emb = c.embedding_length;
  const std::uint32_t heads = c.head_count;
  const std::uint32_t hd = c.HeadDim();
  const std::uint32_t ff = c.feed_forward_length;
  const std::uint32_t nout = c.projection_dim;
  const float eps = c.layer_norm_epsilon;
  const std::uint32_t align = patch * merge;
  if (nx % align != 0 || ny % align != 0) {
    if (error_msg != nullptr) {
      *error_msg = "image canvas is not a multiple of patch * merge";
    }
    return false;
  }
  const std::uint32_t n_px = nx / patch;
  const std::uint32_t n_py = ny / patch;
  const std::uint32_t gw = n_px / merge;
  const std::uint32_t n_patches = n_px * n_py;
  const std::uint32_t n_tokens = n_patches / (merge * merge);

  std::vector<void*> scratch;
  auto alloc = [&](std::size_t floats) -> float* {
    float* p = nullptr;
    if (hipMalloc(&p, floats * sizeof(float)) != hipSuccess) {
      return nullptr;
    }
    scratch.push_back(p);
    return p;
  };
  const std::size_t row = static_cast<std::size_t>(n_patches) * emb;
  const std::size_t frow = static_cast<std::size_t>(n_patches) * ff;
  const std::size_t token_row = static_cast<std::size_t>(n_tokens) * nout;
  const std::size_t pff = static_cast<std::size_t>(n_tokens) * m.proj_ffn;
  float* pix_dev = alloc(static_cast<std::size_t>(3) * ny * nx);
  float* patchmat = alloc(static_cast<std::size_t>(n_patches) * kPatchVec);
  float* h = alloc(row);
  float* xn = alloc(row);
  float* q = alloc(row);
  float* k = alloc(row);
  float* v = alloc(row);
  float* attn = alloc(row);
  float* delta = alloc(row);
  float* upb = alloc(frow);
  float* gateb = alloc(frow);
  float* act = alloc(frow);
  // Merger im2col gathers four emb-wide patch rows per merged token, so it
  // needs n_tokens * 4 * emb = n_patches * emb floats: patchmat is only
  // kPatchVec wide and cannot back it.
  float* mergerin = alloc(row);
  float* mrow = alloc(token_row);
  float* mrow2 = alloc(token_row);
  float* pf1 = alloc(pff);
  float* pf2 = alloc(pff);
  float* pff_act = alloc(pff);
  float* out_dev = alloc(token_row);
  auto release = [&]() {
    for (void* p : scratch) {
      (void)hipFree(p);
    }
  };
  if (pix_dev == nullptr || patchmat == nullptr || h == nullptr ||
      xn == nullptr || q == nullptr || k == nullptr || v == nullptr ||
      attn == nullptr || delta == nullptr || upb == nullptr ||
      gateb == nullptr || act == nullptr || mergerin == nullptr ||
      mrow == nullptr ||
      mrow2 == nullptr || pf1 == nullptr || pf2 == nullptr ||
      pff_act == nullptr || out_dev == nullptr) {
    release();
    if (error_msg != nullptr) {
      *error_msg = "vision scratch allocation failed";
    }
    return false;
  }

  hipStream_t s = nullptr;
  if (hipMemcpyAsync(pix_dev, pixels,
                     static_cast<std::size_t>(3) * ny * nx * sizeof(float),
                     hipMemcpyHostToDevice, s) != hipSuccess) {
    release();
    if (error_msg != nullptr) {
      *error_msg = "pixel upload failed";
    }
    return false;
  }

  // Folded Conv3d patch embedding + bias.
  PatchFoldIm2Col(pix_dev, patchmat, nx, ny, gw, n_patches, s);
  VisionGemm(m.patch_weight, patchmat, h, emb, kPatchVec, n_patches, s);
  g5r::AddBias(h, m.patch_bias, static_cast<std::uint32_t>(row), emb, s);

  const float qk_scale =
      1.0F / std::sqrt(static_cast<float>(hd));
  const std::uint32_t kv = 2 * emb;
  for (const DBlock& b : m.blocks) {
    g5r::RmsNorm(h, b.ln1, xn, n_patches, emb, emb, emb, eps, s);
    // qkv is fused as rows [q | k | v]; run the three sliced GEMMs into
    // consecutive buffers so per-head norms see plain [n_patches x 1024].
    VisionGemm(b.qkv, xn, q, emb, emb, n_patches, s);
    VisionGemm(b.qkv + static_cast<std::size_t>(emb) * emb, xn, k, emb, emb,
               n_patches, s);
    VisionGemm(b.qkv + 2 * static_cast<std::size_t>(emb) * emb, xn, v, emb, emb,
               n_patches, s);
    g5r::AddBias(q, b.qkv_bias, static_cast<std::uint32_t>(row), emb, s);
    g5r::AddBias(k, b.qkv_bias + emb, static_cast<std::uint32_t>(row), emb, s);
    g5r::AddBias(v, b.qkv_bias + kv, static_cast<std::uint32_t>(row), emb, s);
    g5r::RmsNorm(q, b.q_norm, q, n_patches * heads, hd, hd, hd, eps, s);
    g5r::RmsNorm(k, b.k_norm, k, n_patches * heads, hd, hd, hd, eps, s);
    RopeVision(q, m.inv_freq, gw, n_patches, heads, hd, emb, s);
    RopeVision(k, m.inv_freq, gw, n_patches, heads, hd, emb, s);
    VisionAttention(q, k, v, attn, n_patches, heads, hd, emb, qk_scale, s);
    VisionGemm(b.attn_out, attn, delta, emb, emb, n_patches, s);
    g5r::AddBias(delta, b.attn_bias, static_cast<std::uint32_t>(row), emb, s);
    g5r::Add(h, delta, h, static_cast<std::uint32_t>(row), s);

    g5r::RmsNorm(h, b.ln2, xn, n_patches, emb, emb, emb, eps, s);
    VisionGemm(b.ffn_up, xn, upb, ff, emb, n_patches, s);
    VisionGemm(b.ffn_gate, xn, gateb, ff, emb, n_patches, s);
    g5r::AddBias(upb, b.ffn_up_bias, static_cast<std::uint32_t>(frow), ff, s);
    g5r::AddBias(gateb, b.ffn_gate_bias, static_cast<std::uint32_t>(frow), ff,
                 s);
    g5r::SwigluClamp(gateb, upb, act, n_patches, ff, ff, ff, ff,
                     c.swiglu_clamp, c.swiglu_clamp, s);
    VisionGemm(b.ffn_down, act, delta, emb, ff, n_patches, s);
    g5r::AddBias(delta, b.ffn_down_bias, static_cast<std::uint32_t>(row), emb,
                 s);
    g5r::Add(h, delta, h, static_cast<std::uint32_t>(row), s);
  }

  g5r::RmsNorm(h, m.post_ln, xn, n_patches, emb, emb, emb, eps, s);

  // 2x2 merge-window conv, then fc + LayerNorm(1e-5) + gelu_erf.
  MergerIm2Col(xn, mergerin, n_tokens, emb, s);
  VisionGemm(m.merger, mergerin, mrow, nout, 4 * emb, n_tokens, s);
  g5r::AddBias(mrow, m.merger_bias, static_cast<std::uint32_t>(token_row), nout,
               s);
  VisionGemm(m.fc, mrow, mrow2, nout, nout, n_tokens, s);
  LayerNormRows(mrow2, m.ln_weight, m.ln_bias, mrow, n_tokens, nout,
                kLayerNormEps, s);
  GeluErf(mrow, mrow2, token_row, s);

  // Projector FFN (swiglu clamp, no biases).
  VisionGemm(m.up, mrow2, pf1, m.proj_ffn, nout, n_tokens, s);
  VisionGemm(m.gate, mrow2, pf2, m.proj_ffn, nout, n_tokens, s);
  g5r::SwigluClamp(pf2, pf1, pff_act, n_tokens, m.proj_ffn, m.proj_ffn,
                   m.proj_ffn, m.proj_ffn, c.swiglu_clamp, c.swiglu_clamp, s);
  VisionGemm(m.down, pff_act, out_dev, nout, m.proj_ffn, n_tokens, s);

  const hipError_t sync = hipStreamSynchronize(s);
  if (sync != hipSuccess) {
    release();
    if (error_msg != nullptr) {
      *error_msg = "vision stream synchronize failed";
    }
    return false;
  }
  out.assign(token_row, 0.0F);
  const hipError_t copy =
      hipMemcpy(out.data(), out_dev, token_row * sizeof(float),
                hipMemcpyDeviceToHost);
  release();
  if (copy != hipSuccess) {
    if (error_msg != nullptr) {
      *error_msg = "vision output download failed";
    }
    return false;
  }
  return true;
}

const Config& Encoder::config() const noexcept {
  return impl_->config;
}

std::size_t Encoder::resident_bytes() const noexcept {
  return impl_->bytes;
}

}  // namespace gufo::models::glm5_next::vision
