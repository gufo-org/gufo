#include "staging/models/glm5_next/vision/weights.hpp"

#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

namespace gufo::models::glm5_next::vision {
namespace {

using core::GgmlType;

[[nodiscard]] std::size_t ElementBytes(GgmlType type) noexcept {
  switch (type) {
    case GgmlType::kF32:
      return 4;
    case GgmlType::kF16:
    case GgmlType::kBF16:
      return 2;
    default:
      return 0;
  }
}

/// IEEE binary16 -> binary32 without <cuda_fp16.h>: this tier stays
/// host-only and free of backend headers beyond the GGUF reader.
[[nodiscard]] float HalfToFloat(std::uint16_t h) noexcept {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000U) << 16;
  const std::uint32_t expo = (h >> 10) & 0x1FU;
  const std::uint32_t mant = h & 0x3FFU;
  std::uint32_t bits = sign;
  if (expo == 0) {
    if (mant == 0) {
      bits = sign;  // signed zero
    } else {
      // Subnormal: renormalize by shifting the leading mantissa bit out.
      std::uint32_t m = mant;
      std::uint32_t e = 0x7FU - 0x0FU;  // exponent(2^-14) in F32 bias
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
  __builtin_memcpy(&value, &bits, sizeof(value));
  return value;
}

struct Binder {
  const core::GgufReader& reader;
  std::string* error;
  bool ok{true};

  void Fail(const std::string& message) {
    if (ok && error != nullptr) {
      *error = message;
    }
    ok = false;
  }

  /// Binds `name` with the exact dimension list and accepted formats,
  /// folding into a row-major [rows x cols] `TensorRef` (ne[0] -> cols
  /// after folding, ne[1] -> rows; ne[3] must be 1).
  TensorRef Get(const std::string& name,
                std::initializer_list<std::uint64_t> dims,
                std::initializer_list<GgmlType> types) {
    TensorRef t;
    const auto* info = reader.FindTensor(name);
    if (info == nullptr) {
      Fail("missing tensor " + name);
      return t;
    }
    const auto& d = info->dimensions;
    if (d.size() > 4) {
      Fail("tensor " + name + " has too many dimensions");
      return t;
    }
    const std::vector<std::uint64_t> want(dims);
    if (want.size() != d.size()) {
      Fail("tensor " + name + " has rank " + std::to_string(d.size()));
      return t;
    }
    for (std::size_t i = 0; i < want.size(); ++i) {
      if (want[i] != d[i]) {
        Fail("tensor " + name + " shape mismatch at dim " + std::to_string(i));
        return t;
      }
    }
    bool type_ok = false;
    for (auto type : types) {
      type_ok = type_ok || info->type == type;
    }
    if (!type_ok || ElementBytes(info->type) == 0) {
      Fail("tensor " + name + " has unsupported format " +
           std::string(core::ToString(info->type)));
      return t;
    }
    t.data = info->data;
    t.type = info->type;
    t.cols = d.size() > 0 ? d[0] : 1;
    t.rows = d.size() > 1 ? d[1] : 1;
    t.experts = d.size() > 2 ? d[2] : 1;
    t.name = info->name;
    const auto regions = reader.GetMappedRegions();
    const auto address = reinterpret_cast<std::uintptr_t>(info->data);
    bool inside = false;
    for (std::uint32_t i = 0; i < regions.size(); ++i) {
      const auto base = reinterpret_cast<std::uintptr_t>(regions[i].data);
      if (address >= base && address < base + regions[i].size) {
        t.shard = i;
        t.file_offset = address - base;
        inside = t.ElementCount() * ElementBytes(info->type) <=
                 regions[i].size - t.file_offset;
        break;
      }
    }
    if (!inside) {
      Fail("tensor " + name + " is truncated");
      return TensorRef{};
    }
    return t;
  }

  /// Reads one name as a flat F32 host vector.
  std::vector<float> Floats(const std::string& name, std::uint64_t count) {
    std::vector<float> out;
    const auto* info = reader.FindTensor(name);
    if (info == nullptr) {
      Fail("missing tensor " + name);
      return out;
    }
    out.resize(static_cast<std::size_t>(count));
    if (info->type == GgmlType::kF32) {
      const auto* src = static_cast<const float*>(info->data);
      for (std::uint64_t i = 0; i < count; ++i) {
        out[i] = src[i];
      }
    } else if (info->type == GgmlType::kF16) {
      const auto* src = static_cast<const std::uint16_t*>(info->data);
      for (std::uint64_t i = 0; i < count; ++i) {
        out[i] = HalfToFloat(src[i]);
      }
    } else {
      Fail("tensor " + name + " has unsupported format");
      out.clear();
    }
    return out;
  }

  BlockWeights Block(std::uint32_t il) {
    BlockWeights b;
    const std::string p = "v.blk." + std::to_string(il) + ".";
    const std::uint64_t hidden = 1024;
    const std::uint64_t ff = 4096;
    const std::uint64_t head_dim = 64;
    const auto dense = {GgmlType::kF16};
    const auto norm = {GgmlType::kF32};
    b.ln1 = Get(p + "ln1.weight", {hidden}, norm);
    b.qkv = Get(p + "attn_qkv.weight", {hidden, 3 * hidden}, dense);
    b.qkv_bias = Get(p + "attn_qkv.bias", {3 * hidden}, norm);
    b.q_norm = Get(p + "attn_q_norm.weight", {head_dim}, norm);
    b.k_norm = Get(p + "attn_k_norm.weight", {head_dim}, norm);
    b.attn_out = Get(p + "attn_out.weight", {hidden, hidden}, dense);
    b.attn_bias = Get(p + "attn_out.bias", {hidden}, norm);
    b.ln2 = Get(p + "ln2.weight", {hidden}, norm);
    b.ffn_up = Get(p + "ffn_up.weight", {hidden, ff}, dense);
    b.ffn_up_bias = Get(p + "ffn_up.bias", {ff}, norm);
    b.ffn_gate = Get(p + "ffn_gate.weight", {hidden, ff}, dense);
    b.ffn_gate_bias = Get(p + "ffn_gate.bias", {ff}, norm);
    b.ffn_down = Get(p + "ffn_down.weight", {ff, hidden}, dense);
    b.ffn_down_bias = Get(p + "ffn_down.bias", {hidden}, norm);
    return b;
  }
};

}  // namespace

std::optional<VisionWeights> VisionWeights::Bind(const core::GgufReader& reader,
                                                 std::string* error_msg) {
  auto config = Config::FromGguf(reader, error_msg);
  if (!config.has_value()) {
    return std::nullopt;
  }
  Binder b{reader, error_msg};
  VisionWeights w;
  w.config = *config;

  // The temporal-2 Conv3d splits into `.weight` (frame 0) and `.weight.1`
  // (frame 1) with identical flat layout [kx=14][ky=14][cin=3][cout=1024].
  // A decoded single-frame video passes through both kernels and sums, so
  // fold w0 + w1 into one [cout][196*cin*kxy] row-major F32 conv2d weight.
  constexpr std::uint64_t kElems = 14ULL * 14 * 3 * 1024;
  std::vector<float> w0 = b.Floats("v.patch_embd.weight", kElems);
  std::vector<float> w1 = b.Floats("v.patch_embd.weight.1", kElems);
  if (!b.ok) {
    return std::nullopt;
  }
  w.patch_weight.resize(kElems);
  for (std::uint64_t i = 0; i < kElems; ++i) {
    w.patch_weight[i] = w0[i] + w1[i];
  }
  w.patch_bias = b.Get("v.patch_embd.bias", {1024}, {GgmlType::kF32});
  w.post_ln = b.Get("v.post_ln.weight", {1024}, {GgmlType::kF32});

  w.blocks.reserve(w.config.block_count);
  for (std::uint32_t il = 0; il < w.config.block_count; ++il) {
    w.blocks.push_back(b.Block(il));
  }

  // Merger conv [kx=2, ky=2, cin=1024, cout=4096]: the flat order is already
  // row-major [cout][kx + 2*ky + 4*cin], so only the axes fold.
  TensorRef merger =
      b.Get("mm.patch_merger.weight", {2, 2, 1024, 4096}, {GgmlType::kF16});
  merger.cols = 4096;
  merger.rows = 4096;
  merger.experts = 1;
  w.merger = merger;
  w.merger_bias = b.Get("mm.patch_merger.bias", {4096}, {GgmlType::kF32});
  w.fc = b.Get("mm.model.fc.weight", {4096, 4096}, {GgmlType::kF16});
  w.ln_weight = b.Get("mm.post_norm.weight", {4096}, {GgmlType::kF32});
  w.ln_bias = b.Get("mm.post_norm.bias", {4096}, {GgmlType::kF32});
  w.up = b.Get("mm.up.weight", {4096, 10240}, {GgmlType::kF16});
  w.gate = b.Get("mm.gate.weight", {4096, 10240}, {GgmlType::kF16});
  w.down = b.Get("mm.down.weight", {10240, 4096}, {GgmlType::kF16});
  if (!b.ok) {
    return std::nullopt;
  }
  return w;
}

}  // namespace gufo::models::glm5_next::vision
