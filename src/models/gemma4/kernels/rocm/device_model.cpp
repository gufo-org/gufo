#include "src/models/gemma4/kernels/rocm/device_model.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/hip/weight_upload.hpp"
#include "src/models/gemma4/kernels/rocm/gemv.hpp"
#include "src/models/gemma4/kernels/rocm/moe.hpp"

namespace gufo::models::gemma4::rocm {
namespace {

/// Quantized GEMM kernels read whole 256-element k-iterations and may
/// over-read the last row; every upload carries a zeroed tail so those reads
/// stay inside the allocation.
constexpr std::size_t kTailMargin = 4096;

struct Uploader {
  hip::WeightUpload& stager;
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
    if (hipMalloc(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("hipMalloc failed for " + what + " (" + std::to_string(size) +
           " bytes)");
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
  Uploader up{*stager, m->allocations_, m->bytes_, m->max_cols_, error_msg};
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
  if (!up.halves.empty()) {
    // Every BF16 weight from 2^-17 to 65504 in magnitude is a binary16 value;
    // one past the binary16 range would become infinite.
    std::uint32_t* overflow = nullptr;
    std::vector<std::uint32_t> counts(up.halves.size());
    bool converted =
        hipMalloc(&overflow, counts.size() * sizeof(std::uint32_t)) ==
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
