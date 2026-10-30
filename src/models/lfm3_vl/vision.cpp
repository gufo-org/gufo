#include "src/models/lfm3_vl/vision.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <utility>

#include "src/core/gguf_identity.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/lfm3/hip/ops.hpp"
#include "src/models/lfm3_audio/hip/ops.hpp"
#include "src/models/lfm3_vl/hip/ops.hpp"
#include "src/models/qwen/hip/ops/norm_residual.hpp"
namespace gufo::models::lfm3_vl {
namespace {
using core::GgmlType;
using lfm3::Tensor;
void Check(hipError_t s) {
  if (s != hipSuccess)
    throw std::runtime_error(hipGetErrorString(s));
}
struct Buffer {
  void* data{};
  Buffer() = default;
  explicit Buffer(std::size_t bytes) {
    if (bytes)
      Check(hipMalloc(&data, bytes));
  }
  ~Buffer() {
    if (data)
      (void)hipFree(data);
  }
  Buffer(Buffer&& b) noexcept : data(std::exchange(b.data, nullptr)) {}
  Buffer& operator=(Buffer&& b) noexcept {
    std::swap(data, b.data);
    return *this;
  }
  float* f() const { return static_cast<float*>(data); }
};
struct Stream {
  hipStream_t s{};
  Stream() { Check(hipStreamCreateWithFlags(&s, hipStreamNonBlocking)); }
  ~Stream() {
    if (s)
      (void)hipStreamDestroy(s);
  }
};
struct Block {
  Tensor norm, ffn_norm, gate, up, down, conv, in, out, q, k, v, qn, kn;
};
const float* F(const Tensor& t) {
  return static_cast<const float*>(t.data);
}
}  // namespace
struct Vision::Impl {
  std::string identity;
  std::once_flag identity_once;
  std::unique_ptr<core::GgufReader> reader;
  std::vector<Buffer> weights;
  std::vector<Block> blocks;
  Tensor patch, norm, proj_norm, proj_bias, proj1, bias1, proj2, bias2;
  Tensor Get(const std::string& name, unsigned cols, unsigned rows) {
    const auto* t = reader->FindTensor(name);
    std::size_t count = 1;
    if (t)
      for (auto d : t->dimensions)
        count *= d;
    if (!t || count != std::size_t(cols) * rows ||
        (t->type != GgmlType::kF32 && t->type != GgmlType::kF16 &&
         t->type != GgmlType::kBF16 && t->type != GgmlType::kQ8_0))
      throw std::runtime_error("invalid or unsupported LFM3 vision tensor: " +
                               name);
    const auto bytes = quant::EncodedSizeBytes(t->type, count);
    if (!bytes || bytes != t->size_bytes ||
        (t->type == GgmlType::kQ8_0 && cols % 32))
      throw std::runtime_error("invalid d1 vision tensor bytes: " + name);
    weights.emplace_back(bytes);
    Check(
        hipMemcpy(weights.back().data, t->data, bytes, hipMemcpyHostToDevice));
    return {weights.back().data, t->type, cols, rows, bytes};
  }
  Tensor Norm(const std::string& name, unsigned cols) {
    auto t = Get(name, cols, 1);
    if (t.type != GgmlType::kF32)
      throw std::runtime_error("vision norm must be F32: " + name);
    return t;
  }
};
Vision::Vision() : impl_(std::make_unique<Impl>()) {}
Vision::~Vision() = default;
std::unique_ptr<Vision> Vision::Load(const std::string& path) {
  auto v = std::unique_ptr<Vision>(new Vision);
  auto& m = *v->impl_;
  std::string error;
  m.reader = core::GgufReader::OpenFile(path, &error);
  if (!m.reader)
    throw std::runtime_error(error);
  auto& r = *m.reader;
  if (r.GetMetadataString("clip.vision.projector_type") != "lfm3vl" ||
      r.GetMetadataUint32("clip.vision.embedding_length") != 1024 ||
      r.GetMetadataUint32("clip.vision.block_count") != 14 ||
      r.GetMetadataUint32("clip.vision.attention.head_count") != 16 ||
      r.GetMetadataUint32("clip.vision.attention.head_count_kv") != 8 ||
      r.GetMetadataUint32("clip.vision.patch_size") != 16 ||
      r.GetMetadataUint32("clip.vision.spatial_merge_size") != 2 ||
      r.GetMetadataUint32("clip.vision.projection_dim") != 2048)
    throw std::runtime_error("unsupported LFM3 vision geometry");
  // Static images repeat their frame twice. Add the two temporal slices of
  // the patchifier in F32, retaining their exact stored F16 values.
  std::vector<float> patch(1024 * 768), row(768), second(768);
  const auto* p0 = r.FindTensor("v.patch_embd.weight");
  const auto* p1 = r.FindTensor("v.patch_embd.weight.1");
  if (!p0 || !p1)
    throw std::runtime_error("missing temporal patchifier slices");
  auto p = m.Get("v.patch_embd.weight", 768, 1024);
  auto pnext = m.Get("v.patch_embd.weight.1", 768, 1024);
  p.data = p0->data;
  pnext.data = p1->data;
  for (int i = 0; i < 1024; ++i) {
    lfm3::DecodeRow(p, i, row);
    lfm3::DecodeRow(pnext, i, second);
    for (int j = 0; j < 768; ++j)
      patch[i * 768 + j] = row[j] + second[j];
  }
  m.weights.emplace_back(patch.size() * 4);
  Check(hipMemcpy(m.weights.back().data, patch.data(), patch.size() * 4,
                  hipMemcpyHostToDevice));
  m.patch = {m.weights.back().data, GgmlType::kF32, 768, 1024,
             patch.size() * 4};
  m.norm = m.Norm("v.embedding_norm.weight", 1024);
  for (int i = 0; i < 14; ++i) {
    const auto prefix = "v.blk." + std::to_string(i) + ".";
    Block b;
    b.norm = m.Norm(prefix + "ln1.weight", 1024);
    b.ffn_norm = m.Norm(prefix + "ln2.weight", 1024);
    b.gate = m.Get(prefix + "ffn_gate.weight", 1024, 2560);
    b.up = m.Get(prefix + "ffn_up.weight", 1024, 2560);
    b.down = m.Get(prefix + "ffn_down.weight", 2560, 1024);
    if (r.FindTensor(prefix + "attn_q.weight")) {
      b.q = m.Get(prefix + "attn_q.weight", 1024, 1024);
      b.k = m.Get(prefix + "attn_k.weight", 1024, 512);
      b.v = m.Get(prefix + "attn_v.weight", 1024, 512);
      b.qn = m.Norm(prefix + "attn_q_norm.weight", 64);
      b.kn = m.Norm(prefix + "attn_k_norm.weight", 64);
      b.out = m.Get(prefix + "attn_out.weight", 1024, 1024);
    } else {
      b.conv = m.Get(prefix + "conv2d_spatial.weight", 9, 1024);
      if (b.conv.type != GgmlType::kF32)
        throw std::runtime_error("vision convolution must be F32");
      b.in = m.Get(prefix + "conv2d_in_proj.weight", 1024, 3072);
      b.out = m.Get(prefix + "conv2d_out_proj.weight", 1024, 1024);
    }
    m.blocks.push_back(b);
  }
  m.proj_norm = m.Norm("mm.input_norm.weight", 1024);
  m.proj_bias = m.Norm("mm.input_norm.bias", 1024);
  m.proj1 = m.Get("mm.0.weight", 4096, 4096);
  m.bias1 = m.Norm("mm.0.bias", 4096);
  m.proj2 = m.Get("mm.2.weight", 4096, 2048);
  m.bias2 = m.Norm("mm.2.bias", 2048);
  return v;
}
const std::string& Vision::Identity() const {
  std::call_once(impl_->identity_once, [&] {
    impl_->identity = core::GgufIdentityHex(*impl_->reader);
  });
  return impl_->identity;
}
std::vector<float> Vision::Encode(const Picture& p) {
  if (p.height <= 0 || p.width <= 0 || p.height % 2 || p.width % 2 ||
      std::size_t(p.height) * p.width > 65536 ||
      p.patches.size() != std::size_t(p.height) * p.width * 768)
    throw std::invalid_argument("invalid LFM3 picture patches");
  const int n = p.height * p.width;
  auto& m = *impl_;
  Stream stream;
  const auto s = stream.s;
  Buffer x(std::size_t(n) * 1024 * 4), z(std::size_t(n) * 4096 * 4),
      o(std::size_t(n) * 1024 * 4), mix(std::size_t(n) * 3072 * 4),
      q(std::size_t(n) * 1024 * 4), k(std::size_t(n) * 512 * 4),
      v(std::size_t(n) * 512 * 4), a(std::size_t(n) * 1024 * 4),
      gate(std::size_t(n) * 2560 * 4), up(std::size_t(n) * 2560 * 4),
      workspace(lfm3::gpu::DenseWorkspaceBytes(n, 4096));
  auto dense = [&](const Tensor& t, const float* in, float* out, int rows) {
    if (t.type == GgmlType::kBF16)
      lfm3_audio::gpu::SgemmBf16(in, t.data, nullptr, out, rows, t.rows, t.cols,
                                 lfm3_audio::gpu::Activation::kNone, s);
    else
      lfm3::gpu::Dense(t, in, out, rows, s, {workspace.data});
  };
  auto norm = [&](const float* in, const Tensor& weight, float* out) {
    gufo::hip::LaunchBatchedRMSNorm(in, F(weight), out, nullptr, n, 1024, 1e-5F,
                                    s);
  };
  auto residual = [&] {
    gufo::hip::LaunchBatchedResidualAdd(x.f(), o.f(), x.f(), n, 1024, s);
  };
  Check(hipMemcpyAsync(z.data, p.patches.data(), p.patches.size() * 4,
                       hipMemcpyHostToDevice, s));
  dense(m.patch, z.f(), x.f(), n);
  norm(x.f(), m.norm, x.f());
  for (const auto& b : m.blocks) {
    norm(x.f(), b.norm, z.f());
    if (b.q.data) {
      dense(b.q, z.f(), q.f(), n);
      dense(b.k, z.f(), k.f(), n);
      dense(b.v, z.f(), v.f(), n);
      gufo::hip::LaunchBatchedPerHeadRMSNorm(q.f(), F(b.qn), q.f(), n, 16, 64,
                                             1e-5F, s);
      gufo::hip::LaunchBatchedPerHeadRMSNorm(k.f(), F(b.kn), k.f(), n, 8, 64,
                                             1e-5F, s);
      gpu::AxialRope(q.f(), p.height, p.width, 16, s);
      gpu::AxialRope(k.f(), p.height, p.width, 8, s);
      lfm3_audio::gpu::Attention(q.f(), k.f(), v.f(), nullptr, nullptr, nullptr,
                                 a.f(), n, 16, 8, 64, 0, n, 0, false, s);
    } else {
      dense(b.in, z.f(), mix.f(), n);
      gpu::SpatialConv(mix.f(), F(b.conv), a.f(), p.height, p.width, 1024, s);
    }
    dense(b.out, a.f(), o.f(), n);
    residual();
    norm(x.f(), b.ffn_norm, z.f());
    dense(b.gate, z.f(), gate.f(), n);
    dense(b.up, z.f(), up.f(), n);
    lfm3_audio::gpu::SwiGlu(gate.f(), up.f(), gate.f(), std::size_t(n) * 2560,
                            s);
    dense(b.down, gate.f(), o.f(), n);
    residual();
  }
  lfm3_audio::gpu::LayerNorm(x.f(), F(m.proj_norm), F(m.proj_bias), z.f(), n,
                             1024, 1e-6F, s);
  dense(m.proj1, z.f(), mix.f(), n / 4);
  lfm3_audio::gpu::BiasAct(mix.f(), F(m.bias1), n / 4, 4096,
                           lfm3_audio::gpu::Activation::kGelu, s);
  dense(m.proj2, mix.f(), o.f(), n / 4);
  lfm3_audio::gpu::BiasAct(o.f(), F(m.bias2), n / 4, 2048,
                           lfm3_audio::gpu::Activation::kNone, s);
  std::vector<float> out(std::size_t(n / 4) * 2048);
  Check(hipMemcpyAsync(out.data(), o.data, out.size() * 4,
                       hipMemcpyDeviceToHost, s));
  Check(hipStreamSynchronize(s));
  return out;
}
}  // namespace gufo::models::lfm3_vl
