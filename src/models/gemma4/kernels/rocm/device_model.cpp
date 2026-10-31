#include "src/models/gemma4/kernels/rocm/device_model.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/hip/managed_alloc.hpp"
#include "src/core/hip/weight_upload.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/gemma4/kernels/rocm/gemv.hpp"
#include "src/models/gemma4/kernels/rocm/kernels.hpp"
#include "src/models/gemma4/kernels/rocm/moe.hpp"

namespace gufo::models::gemma4::rocm {
namespace {

std::optional<ExpertFormat> PrefillFormat(core::GgmlType type) {
  switch (type) {
    case core::GgmlType::kQ4_K:
      return ExpertFormat::kQ4_K;
    case core::GgmlType::kQ5_K:
      return ExpertFormat::kQ5_K;
    case core::GgmlType::kQ6_K:
      return ExpertFormat::kQ6_K;
    case core::GgmlType::kQ4_0:
      return ExpertFormat::kQ4_0;
    case core::GgmlType::kQ5_0:
      return ExpertFormat::kQ5_0;
    case core::GgmlType::kQ8_0:
      return ExpertFormat::kQ8_0;
    default:
      return std::nullopt;
  }
}

/// Bytes of a grouped-layout format's matrix (256-value super-blocks).
std::size_t GroupedBytes(const DeviceTensor& t) {
  const std::size_t super = t.type == core::GgmlType::kQ5_K   ? 176
                            : t.type == core::GgmlType::kQ6_K ? 210
                            : t.type == core::GgmlType::kQ8_0 ? 272
                                                              : 144;
  return std::size_t{t.rows} * (t.cols / 256) * super;
}

/// The prefill copy of a dense projection with k inputs under `mode`.
PrefillLayout ChoosePrefillLayout(PrefillWeights mode, ExpertFormat format,
                                  std::uint32_t k) {
  // Binary16 copies pay only on the 31B's 5376-wide inputs (pp2048,
  // 2026-10-21). From 8192 the blocks sharing a weight tile drift apart along
  // k and each reads it from DRAM (down, o: -29 to -62%); at the 26B-A4B's
  // 2816 they lose 1-2% (Flash-Next's tuned Q8_0 GEMM, Q4_0). Q8_0 takes
  // none: with GeGLU in Flash-Next's [gate | up] epilogue its GEMMs match
  // the copies' (UD-Q8_K_XL pp2048 610.0 vs 602.8 tok/s for +32.4 GiB).
  const bool decoded =
      PrefillLayoutSupports(format, PrefillLayout::kDecoded, k) &&
      format != ExpertFormat::kQ8_0 && k > 4096 && k < 8192;
  if (mode == PrefillWeights::kDecoded && decoded) {
    return PrefillLayout::kDecoded;
  }
  if (mode != PrefillWeights::kOff &&
      PrefillLayoutSupports(format, PrefillLayout::kCompact, k)) {
    return PrefillLayout::kCompact;
  }
  return PrefillLayout::kNone;
}

/// Quantized GEMM kernels read whole 256-element k-iterations and may
/// over-read the last row; every upload carries a zeroed tail so those reads
/// stay inside the allocation.
constexpr std::size_t kTailMargin = 4096;

struct Uploader {
  hip::WeightUpload& stager;
  const hip::WeightAllocator& weights;
  std::vector<void*>& allocations;
  std::size_t& bytes;
  std::size_t& max_cols;
  std::string* error;
  std::uint32_t shard_base{0};
  bool ok{true};
  /// BF16 layer tensors to rewrite as binary16 once uploaded.
  struct HalfConversion {
    void* data;
    std::size_t count;
    std::string name;
  };
  std::vector<HalfConversion> halves;
  /// Widest reduction of a binary16 dense projection.
  std::size_t max_half_cols{0};

  void Fail(const std::string& message) {
    if (ok && error != nullptr) {
      *error = message;
    }
    ok = false;
  }

  void* Allocate(std::size_t size, const std::string& what) {
    void* ptr = nullptr;
    if (weights.Allocate(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("device allocation failed for " + what + " (" +
           std::to_string(size) + " bytes)");
      return nullptr;
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0, kTailMargin,
                         nullptr);
    return ptr;
  }

  DeviceTensor Copy(const TensorRef& t) {
    DeviceTensor d;
    if (t.empty() || !ok) {
      return d;
    }
    const std::size_t size = t.SizeBytes();
    void* ptr = Allocate(size, std::string(t.name));
    if (ptr == nullptr) {
      return d;
    }
    if (!stager.Copy(shard_base + t.shard, t.file_offset, size, ptr, error)) {
      Fail("upload failed for " + std::string(t.name));
      return d;
    }
    d.data = ptr;
    d.type = t.type;
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = static_cast<std::uint32_t>(t.rows);
    d.experts = static_cast<std::uint32_t>(t.experts);
    if (t.type != core::GgmlType::kF32) {
      max_cols = std::max<std::size_t>(max_cols, t.cols);
    }
    return d;
  }

  float* Floats(const std::vector<float>& values, const char* what) {
    const std::size_t size = values.size() * sizeof(float);
    auto* ptr = static_cast<float*>(Allocate(size, what));
    if (ptr != nullptr && hipMemcpy(ptr, values.data(), size,
                                    hipMemcpyHostToDevice) != hipSuccess) {
      Fail(std::string("copy failed for ") + what);
    }
    return ptr;
  }

  /// Uploads `parts` (same format and width) into one allocation, rows
  /// back to back; `views` receive each part. Returns the whole tensor, or
  /// an empty one (nothing uploaded) when the parts differ.
  DeviceTensor Fuse(std::initializer_list<const TensorRef*> parts,
                    std::initializer_list<DeviceTensor*> views) {
    DeviceTensor d;
    const TensorRef& first = **parts.begin();
    std::size_t size = 0;
    std::uint64_t rows = 0;
    for (const TensorRef* t : parts) {
      if (t->empty() || t->type != first.type || t->cols != first.cols ||
          t->experts != 1 || t->type == core::GgmlType::kF32) {
        return d;
      }
      size += t->SizeBytes();
      rows += t->rows;
    }
    if (!ok) {
      return d;
    }
    auto* ptr = static_cast<std::uint8_t*>(
        Allocate(size, std::string(first.name) + " (fused)"));
    if (ptr == nullptr) {
      return d;
    }
    auto view = views.begin();
    std::size_t at = 0;
    for (const TensorRef* t : parts) {
      if (!stager.Copy(shard_base + t->shard, t->file_offset, t->SizeBytes(),
                       ptr + at, error)) {
        Fail("upload failed for " + std::string(t->name));
        return d;
      }
      DeviceTensor& v = **view++;
      v.data = ptr + at;
      v.type = t->type;
      v.cols = static_cast<std::uint32_t>(t->cols);
      v.rows = static_cast<std::uint32_t>(t->rows);
      at += t->SizeBytes();
    }
    max_cols = std::max<std::size_t>(max_cols, first.cols);
    d.data = ptr;
    d.type = first.type;
    d.cols = static_cast<std::uint32_t>(first.cols);
    d.rows = static_cast<std::uint32_t>(rows);
    return d;
  }

  DeviceLayer Layer(const LayerWeights& l) {
    DeviceLayer d;
    d.attn_norm = Copy(l.attn_norm);
    // Fused projections share one launch in decode and one GEMM in prefill.
    if (!l.attn_k.empty()) {
      d.attn_qkv = l.attn_v.empty()
                       ? Fuse({&l.attn_q, &l.attn_k}, {&d.attn_q, &d.attn_k})
                       : Fuse({&l.attn_q, &l.attn_k, &l.attn_v},
                              {&d.attn_q, &d.attn_k, &d.attn_v});
      if (d.attn_qkv.empty() && !l.attn_v.empty()) {
        d.attn_qkv = Fuse({&l.attn_q, &l.attn_k}, {&d.attn_q, &d.attn_k});
        if (!d.attn_qkv.empty()) {
          d.attn_v = Copy(l.attn_v);
        }
      }
    }
    if (d.attn_qkv.empty()) {
      d.attn_q = Copy(l.attn_q);
      d.attn_k = Copy(l.attn_k);
      d.attn_v = Copy(l.attn_v);
    }
    d.attn_q_norm = Copy(l.attn_q_norm);
    d.attn_k_norm = Copy(l.attn_k_norm);
    d.attn_output = Copy(l.attn_output);
    d.post_attn_norm = Copy(l.post_attn_norm);
    d.ffn_norm = Copy(l.ffn_norm);
    d.ffn_gate_up = Fuse({&l.ffn_gate, &l.ffn_up}, {&d.ffn_gate, &d.ffn_up});
    if (d.ffn_gate_up.empty()) {
      d.ffn_gate = Copy(l.ffn_gate);
      d.ffn_up = Copy(l.ffn_up);
    }
    d.ffn_down = Copy(l.ffn_down);
    d.post_ffn_norm = Copy(l.post_ffn_norm);
    d.output_scale = l.output_scale;
    d.router = Copy(l.router);
    d.router_scale = Copy(l.router_scale);
    d.pre_ffn_norm_2 = Copy(l.pre_ffn_norm_2);
    d.post_ffn_norm_1 = Copy(l.post_ffn_norm_1);
    d.post_ffn_norm_2 = Copy(l.post_ffn_norm_2);
    d.gate_up_exps = Copy(l.gate_up_exps);
    d.down_exps = Copy(l.down_exps);
    d.down_exps_scale = Copy(l.down_exps_scale);
    // Binary16 kernels serve the BF16 projections and experts: fused tensors
    // convert once with their views.
    Half(d.attn_qkv, l.attn_q.name, {&d.attn_q, &d.attn_k, &d.attn_v});
    Half(d.ffn_gate_up, l.ffn_gate.name, {&d.ffn_gate, &d.ffn_up});
    Half(d.attn_q, l.attn_q.name);
    Half(d.attn_k, l.attn_k.name);
    Half(d.attn_v, l.attn_v.name);
    Half(d.attn_output, l.attn_output.name);
    Half(d.ffn_gate, l.ffn_gate.name);
    Half(d.ffn_up, l.ffn_up.name);
    Half(d.ffn_down, l.ffn_down.name);
    Half(d.gate_up_exps, l.gate_up_exps.name);
    Half(d.down_exps, l.down_exps.name);
    for (const DeviceTensor* t :
         {&d.attn_qkv, &d.attn_q, &d.attn_k, &d.attn_v, &d.attn_output,
          &d.ffn_gate_up, &d.ffn_gate, &d.ffn_up, &d.ffn_down}) {
      if (t->type == core::GgmlType::kF16 && t->experts == 1) {
        max_half_cols = std::max<std::size_t>(max_half_cols, t->cols);
      }
    }
    return d;
  }

  /// Queues a BF16 tensor's conversion to binary16 and retypes it and the
  /// `views` into it.
  void Half(DeviceTensor& t, std::string_view name,
            std::initializer_list<DeviceTensor*> views = {}) {
    if (t.empty() || t.type != core::GgmlType::kBF16) {
      return;
    }
    halves.push_back(
        {t.data, std::size_t{t.rows} * t.cols * t.experts, std::string(name)});
    t.type = core::GgmlType::kF16;
    for (DeviceTensor* v : views) {
      if (!v->empty()) {
        v->type = core::GgmlType::kF16;
      }
    }
  }
};

}  // namespace

DeviceModel::~DeviceModel() {
  for (void* ptr : allocations_) {
    (void)hipFree(ptr);
  }
}

std::unique_ptr<DeviceModel> DeviceModel::Upload(
    const ModelWeights& weights, const core::GgufReader& reader,
    const DraftWeights* draft, const core::GgufReader* draft_reader,
    std::span<const std::int32_t> draft_order, PrefillWeights prefill_weights,
    std::string* error_msg) {
  std::unique_ptr<DeviceModel> m(new DeviceModel());
  m->config_ = weights.config;
  m->vocab_ = weights.vocab_size;
  const auto regions = reader.GetMappedRegions();
  std::vector<core::GgufMappedRegion> shards(regions.begin(), regions.end());
  const auto shard_count = static_cast<std::uint32_t>(shards.size());
  if (draft != nullptr) {
    if (draft_reader == nullptr) {
      if (error_msg != nullptr) {
        *error_msg = "draft weights require their bound reader";
      }
      return nullptr;
    }
    const auto extra = draft_reader->GetMappedRegions();
    shards.insert(shards.end(), extra.begin(), extra.end());
  }
  auto stager = hip::WeightUpload::Create(shards, error_msg);
  if (!stager) {
    return nullptr;
  }
  // The placement is decided once, from the mapped files: they hold every
  // tensor this upload copies.
  std::uint64_t file_bytes = 0;
  for (const auto& shard : shards) {
    file_bytes += shard.size;
  }
  const hip::WeightAllocator placement(file_bytes);
  Uploader up{*stager,   placement,    m->allocations_,
              m->bytes_, m->max_cols_, error_msg};
  m->token_embd_ = up.Copy(weights.token_embd);
  m->output_ = weights.TiedOutput() ? m->token_embd_ : up.Copy(weights.output);
  m->output_norm_ = up.Copy(weights.output_norm);
  m->rope_factors_ = up.Floats(weights.rope_factors.values, "rope factors");
  {
    // A pair whose angle stays below 1e-12 rad at 2^24 positions rounds to
    // the identity in binary16 keys.
    const auto& factors = weights.rope_factors.values;
    const double scale =
        std::pow(static_cast<double>(m->config_.rope_theta_global),
                 -2.0 / m->config_.head_dim_global);
    for (std::size_t i = 0; i < factors.size(); ++i) {
      const double angle =
          16777216.0 * std::pow(scale, static_cast<double>(i)) / factors[i];
      if (angle >= 1e-12) {
        m->global_rope_pairs_ = static_cast<std::uint32_t>(i + 1);
      }
    }
  }
  m->layers_.reserve(weights.layers.size());
  for (const auto& l : weights.layers) {
    m->layers_.push_back(up.Layer(l));
    if (!up.ok) {
      return nullptr;
    }
  }
  if (draft != nullptr) {
    up.shard_base = shard_count;
    auto& d = m->draft_;
    d.config = draft->config;
    d.pre_projection = up.Copy(draft->pre_projection);
    d.post_projection = up.Copy(draft->post_projection);
    d.token_embd = up.Copy(draft->token_embd);
    d.output_norm = up.Copy(draft->output_norm);
    d.rope_factors =
        up.Floats(draft->rope_factors.values, "draft rope factors");
    for (const auto& l : draft->layers) {
      d.layers.push_back(up.Layer(l));
    }
    m->has_draft_ = true;
  }
  if (!up.ok || !stager->Finish(error_msg)) {
    return nullptr;
  }
  m->max_half_cols_ = up.max_half_cols;
  m->half_prefill_ = m->config_.HasExperts();
  if (!m->half_prefill_) {
    // A dense model takes binary16 prefill when every projection has its
    // GEMM; each projection input is then staged as binary16.
    bool all = true;
    std::size_t widest = 0;
    for (const DeviceLayer& l : m->layers_) {
      for (const DeviceTensor* t :
           {&l.attn_qkv, &l.attn_q, &l.attn_k, &l.attn_v, &l.attn_output,
            &l.ffn_gate_up, &l.ffn_gate, &l.ffn_up, &l.ffn_down}) {
        if (!t->empty()) {
          all = all && HalfPrefillFormat(t->type);
          widest = std::max<std::size_t>(widest, t->cols);
        }
      }
    }
    m->half_prefill_ = all;
    if (all) {
      m->max_half_cols_ = std::max(m->max_half_cols_, widest);
    }
  }
  if (m->half_prefill_ && prefill_weights != PrefillWeights::kOff) {
    const auto pack = [&](DeviceTensor& t) {
      const auto format = PrefillFormat(t.type);
      if (t.empty() || !format || !up.ok) {
        return;
      }
      const PrefillLayout layout =
          ChoosePrefillLayout(prefill_weights, *format, t.cols);
      if (layout == PrefillLayout::kNone) {
        return;
      }
      t.prefill =
          up.Allocate(PrefillWeightBytes(*format, layout, t.rows, t.cols),
                      "prefill weights");
      if (t.prefill != nullptr) {
        t.prefill_layout = layout;
        t.prefill_format = *format;
        up.ok = PackPrefillWeights(*format, layout, t.data, t.prefill, t.rows,
                                   t.cols, nullptr);
      }
    };
    // The tensors prefill projects: fused ones, not their views.
    for (DeviceLayer& l : m->layers_) {
      if (!l.attn_qkv.empty()) {
        pack(l.attn_qkv);
        if (l.attn_qkv.rows == l.attn_q.rows + l.attn_k.rows) {
          pack(l.attn_v);
        }
      } else {
        pack(l.attn_q);
        pack(l.attn_k);
        pack(l.attn_v);
      }
      pack(l.attn_output);
      if (!l.ffn_gate_up.empty()) {
        pack(l.ffn_gate_up);
      } else {
        pack(l.ffn_gate);
        pack(l.ffn_up);
      }
      pack(l.ffn_down);
      // Expert [gate | up] rows stack per expert: one compact matrix, whole
      // 16-row groups per expert.
      DeviceTensor& e = l.gate_up_exps;
      const auto format = PrefillFormat(e.type);
      if (!e.empty() && format && up.ok && e.rows % 16 == 0 &&
          PrefillLayoutSupports(*format, PrefillLayout::kCompact, e.cols)) {
        const std::uint32_t rows = e.rows * e.experts;
        e.prefill = up.Allocate(
            PrefillWeightBytes(*format, PrefillLayout::kCompact, rows, e.cols),
            "expert prefill weights");
        if (e.prefill != nullptr) {
          e.prefill_layout = PrefillLayout::kCompact;
          e.prefill_format = *format;
          up.ok = PackPrefillWeights(*format, PrefillLayout::kCompact, e.data,
                                     e.prefill, rows, e.cols, nullptr);
        }
      }
    }
    if (!up.ok || hipDeviceSynchronize() != hipSuccess) {
      up.Fail("prefill weight packing failed");
      return nullptr;
    }
  }
  // With a drafter every decode width runs the WMMA projections, which read
  // the grouped layout 2-10% faster; prefill's GEMMs (and the copies above,
  // packed from GGUF rows) read it too. Only projections WmmaProjection
  // takes at every width are reordered, and a fused tensor only when each
  // view starts a 128-row group, so the views stay grouped matrices.
  if (draft != nullptr && m->half_prefill_) {
    const auto eligible = [](const DeviceTensor& t) {
      return !t.empty() && t.experts == 1 && t.cols >= 2048 &&
             GroupedLayoutSupports(t.type, t.rows, t.cols);
    };
    std::size_t widest = 0;
    for (const DeviceLayer& l : m->layers_) {
      for (const DeviceTensor* t :
           {&l.attn_qkv, &l.attn_q, &l.attn_k, &l.attn_v, &l.attn_output,
            &l.ffn_gate_up, &l.ffn_gate, &l.ffn_up, &l.ffn_down}) {
        if (eligible(*t)) {
          widest = std::max(widest, GroupedBytes(*t));
        }
      }
    }
    void* scratch = nullptr;
    if (widest != 0 && hipMalloc(&scratch, widest) != hipSuccess) {
      up.Fail("grouped weight scratch allocation failed");
      return nullptr;
    }
    const auto group = [&](DeviceTensor& t,
                           std::initializer_list<DeviceTensor*> views) {
      if (!eligible(t)) {
        return;
      }
      for (const DeviceTensor* v : views) {
        if (!v->empty() && v->rows % 128 != 0) {
          return;
        }
      }
      const std::size_t bytes = GroupedBytes(t);
      if (!GroupWeights(t.type, t.data, scratch, t.rows, t.cols, nullptr) ||
          hipMemcpyAsync(t.data, scratch, bytes, hipMemcpyDeviceToDevice,
                         nullptr) != hipSuccess) {
        up.ok = false;
        return;
      }
      t.grouped = true;
      for (DeviceTensor* v : views) {
        v->grouped = !v->empty();
      }
    };
    for (DeviceLayer& l : m->layers_) {
      if (!l.attn_qkv.empty()) {
        const bool separate_v =
            l.attn_qkv.rows == l.attn_q.rows + l.attn_k.rows;
        if (separate_v) {
          group(l.attn_qkv, {&l.attn_q, &l.attn_k});
          group(l.attn_v, {});
        } else {
          group(l.attn_qkv, {&l.attn_q, &l.attn_k, &l.attn_v});
        }
      } else {
        group(l.attn_q, {});
        group(l.attn_k, {});
        group(l.attn_v, {});
      }
      group(l.attn_output, {});
      if (!l.ffn_gate_up.empty()) {
        group(l.ffn_gate_up, {&l.ffn_gate, &l.ffn_up});
      } else {
        group(l.ffn_gate, {});
        group(l.ffn_up, {});
      }
      group(l.ffn_down, {});
    }
    if (hipDeviceSynchronize() != hipSuccess || !up.ok) {
      (void)hipFree(scratch);
      up.Fail("grouping weights failed");
      return nullptr;
    }
    (void)hipFree(scratch);
  }
  if (!up.halves.empty()) {
    // Every BF16 weight from 2^-17 to 65504 in magnitude is a binary16 value;
    // one past the binary16 range would become infinite.
    std::uint32_t* overflow = nullptr;
    std::vector<std::uint32_t> counts(up.halves.size());
    bool converted =
        hip::AllocateDevice(&overflow, counts.size() * sizeof(std::uint32_t)) ==
            hipSuccess &&
        hipMemset(overflow, 0, counts.size() * sizeof(std::uint32_t)) ==
            hipSuccess;
    for (std::size_t i = 0; converted && i < up.halves.size(); ++i) {
      ConvertBf16ToHalf(up.halves[i].data, up.halves[i].count, overflow + i,
                        nullptr);
    }
    converted = converted && hipMemcpy(counts.data(), overflow,
                                       counts.size() * sizeof(std::uint32_t),
                                       hipMemcpyDeviceToHost) == hipSuccess;
    (void)hipFree(overflow);
    if (!converted) {
      if (error_msg != nullptr) {
        *error_msg = "BF16 to binary16 weight conversion failed";
      }
      return nullptr;
    }
    for (std::size_t i = 0; i < counts.size(); ++i) {
      if (counts[i] != 0) {
        if (error_msg != nullptr) {
          *error_msg = "tensor " + up.halves[i].name + " has " +
                       std::to_string(counts[i]) +
                       " BF16 values outside the binary16 range";
        }
        return nullptr;
      }
    }
  }
  const hipError_t status = hipDeviceSynchronize();
  if (status != hipSuccess) {
    if (error_msg != nullptr) {
      *error_msg =
          std::string("weight upload failed: ") + hipGetErrorString(status);
    }
    return nullptr;
  }
  // Routing reads the router with its input scale folded in.
  for (DeviceLayer& l : m->layers_) {
    if (!l.router.empty()) {
      ScaleRouter(static_cast<float*>(l.router.data), l.router_scale.f32(),
                  l.router.rows, l.router.cols, nullptr);
    }
  }
  if (m->has_draft_) {
    // The drafter's vocabulary head only proposes tokens; verification
    // decides every emitted one. Read as Q4_K it moves half the bytes per
    // draft step.
    DeviceTensor& head = m->draft_.token_embd;
    if (head.type == core::GgmlType::kQ8_0 && head.cols % 256 == 0) {
      const std::size_t size = std::size_t{head.rows} * head.cols / 256 * 144;
      void* q4 = up.Allocate(size, "draft vocabulary head (Q4_K)");
      if (q4 == nullptr) {
        return nullptr;
      }
      RepackQ8_0AsQ4K(head.data, q4, head.rows, head.cols, nullptr);
      if (hipDeviceSynchronize() != hipSuccess) {
        if (error_msg != nullptr) {
          *error_msg = "draft vocabulary head repack failed";
        }
        return nullptr;
      }
      auto& allocations = m->allocations_;
      allocations.erase(
          std::find(allocations.begin(), allocations.end(), head.data));
      (void)hipFree(head.data);
      m->bytes_ -= std::size_t{head.rows} * head.cols / 32 * 34 + kTailMargin;
      head.data = q4;
      head.type = core::GgmlType::kQ4_K;
    }
    // Rows in `draft_order`, so a draft step can score a leading range.
    const std::size_t row_bytes =
        quant::QuantizedRowBytes(head.type, head.cols);
    if (draft_order.size() == head.rows && row_bytes != 0 &&
        row_bytes % 16 == 0) {
      const std::size_t size = row_bytes * head.rows;
      void* ordered = up.Allocate(size, "draft vocabulary head (ordered)");
      if (ordered == nullptr) {
        return nullptr;
      }
      std::vector<CopyRun> runs(head.rows);
      m->draft_head_rows_.assign(head.rows, 0);
      for (std::uint32_t row = 0; row < head.rows; ++row) {
        const auto token = static_cast<std::size_t>(draft_order[row]);
        runs[row] = {
            static_cast<const std::uint8_t*>(head.data) + token * row_bytes,
            static_cast<std::uint8_t*>(ordered) + row * row_bytes, row_bytes};
        m->draft_head_rows_[token] = row;
      }
      void* device_runs = nullptr;
      if (hip::AllocateDevice(&device_runs, runs.size() * sizeof(CopyRun)) !=
              hipSuccess ||
          hipMemcpy(device_runs, runs.data(), runs.size() * sizeof(CopyRun),
                    hipMemcpyHostToDevice) != hipSuccess) {
        (void)hipFree(device_runs);
        if (error_msg != nullptr) {
          *error_msg = "draft vocabulary head reorder failed";
        }
        return nullptr;
      }
      CopyRuns(static_cast<const CopyRun*>(device_runs), head.rows, row_bytes,
               nullptr);
      const bool copied = hipDeviceSynchronize() == hipSuccess;
      (void)hipFree(device_runs);
      if (!copied) {
        if (error_msg != nullptr) {
          *error_msg = "draft vocabulary head reorder failed";
        }
        return nullptr;
      }
      auto& allocations = m->allocations_;
      allocations.erase(
          std::find(allocations.begin(), allocations.end(), head.data));
      (void)hipFree(head.data);
      m->bytes_ -= size + kTailMargin;
      head.data = ordered;
      m->draft_head_tokens_.assign(draft_order.begin(), draft_order.end());
      m->draft_row_bytes_ = row_bytes;
    }
  }
  if (hipDeviceSynchronize() != hipSuccess) {
    if (error_msg != nullptr) {
      *error_msg = "router scaling failed";
    }
    return nullptr;
  }
  return m;
}

}  // namespace gufo::models::gemma4::rocm
