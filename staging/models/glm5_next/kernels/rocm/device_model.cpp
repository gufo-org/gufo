#include "staging/models/glm5_next/kernels/rocm/device_model.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "src/core/hip/weight_upload.hpp"

namespace gufo::models::glm5_next::rocm {
namespace {

/// Streams one tensor's payload from disk into a fresh device allocation. The
/// GEMV tier reads exactly `row_bytes` per row, so no tail margin is needed.
/// Every stored format (F32, Q4_K, Q5_K, Q6_K, Q8_0, Q2_K, Q3_K, IQ2_XXS,
/// IQ1_M, IQ3_XXS, IQ4_XS) decodes on the fly, so nothing is transformed
/// during upload.
struct Uploader {
  hip::WeightUpload& stager;
  std::vector<void*>& allocations;
  std::size_t& bytes;
  std::string* error;
  bool ok{true};

  void Fail(const std::string& message) {
    if (ok && error != nullptr) {
      *error = message;
    }
    ok = false;
  }

  DeviceTensor Copy(const TensorRef& t) {
    DeviceTensor d;
    if (t.empty() || !ok) {
      return d;
    }
    const std::size_t size = t.SizeBytes();
    void* ptr = nullptr;
    if (hipMalloc(&ptr, size) != hipSuccess) {
      Fail("hipMalloc failed for " + std::string(t.name) + " (" +
           std::to_string(size) + " bytes)");
      return d;
    }
    allocations.push_back(ptr);
    bytes += size;
    if (!stager.Copy(t.shard, t.file_offset, size, ptr, error)) {
      Fail("upload failed for " + std::string(t.name) +
           (error != nullptr ? ": " + *error : std::string()));
      return d;
    }
    d.data = ptr;
    d.type = t.type;
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = static_cast<std::uint32_t>(t.rows);
    d.experts = static_cast<std::uint32_t>(t.experts);
    d.row_bytes = t.RowBytes();
    return d;
  }

  DeviceLayer Layer(const LayerWeights& l) {
    DeviceLayer d;
    d.dsa = l.dsa;
    d.hc_attn_fn = Copy(l.hc_attn_fn);
    d.hc_attn_base = Copy(l.hc_attn_base);
    d.hc_attn_scale = Copy(l.hc_attn_scale);
    d.hc_ffn_fn = Copy(l.hc_ffn_fn);
    d.hc_ffn_base = Copy(l.hc_ffn_base);
    d.hc_ffn_scale = Copy(l.hc_ffn_scale);
    d.attn_norm = Copy(l.attn_norm);
    d.ffn_norm = Copy(l.ffn_norm);
    if (l.dsa) {
      d.attn_q_a = Copy(l.attn_q_a);
      d.attn_q_a_norm = Copy(l.attn_q_a_norm);
      d.attn_q_b = Copy(l.attn_q_b);
      d.attn_kv_a = Copy(l.attn_kv_a);
      d.attn_kv_a_norm = Copy(l.attn_kv_a_norm);
      d.attn_k_b = Copy(l.attn_k_b);
      d.attn_v_b = Copy(l.attn_v_b);
      d.attn_out = Copy(l.attn_out);
      d.idx_q_b = Copy(l.idx_q_b);
      d.idx_k = Copy(l.idx_k);
      d.idx_k_norm = Copy(l.idx_k_norm);
      d.idx_k_bias = Copy(l.idx_k_bias);
      d.idx_proj = Copy(l.idx_proj);
      d.idx_comp_gate = Copy(l.idx_comp_gate);
      d.idx_comp_ape = Copy(l.idx_comp_ape);
    } else {
      d.ssm_q = Copy(l.ssm_q);
      d.ssm_k = Copy(l.ssm_k);
      d.ssm_v = Copy(l.ssm_v);
      d.ssm_conv_q = Copy(l.ssm_conv_q);
      d.ssm_conv_k = Copy(l.ssm_conv_k);
      d.ssm_conv_v = Copy(l.ssm_conv_v);
      d.ssm_f_a = Copy(l.ssm_f_a);
      d.ssm_f_b = Copy(l.ssm_f_b);
      d.ssm_g_a = Copy(l.ssm_g_a);
      d.ssm_g_b = Copy(l.ssm_g_b);
      d.ssm_beta = Copy(l.ssm_beta);
      d.ssm_a = Copy(l.ssm_a);
      d.ssm_dt = Copy(l.ssm_dt);
      d.ssm_norm = Copy(l.ssm_norm);
      d.ssm_out = Copy(l.ssm_out);
    }
    d.ffn_gate = Copy(l.ffn_gate);
    d.ffn_up = Copy(l.ffn_up);
    d.ffn_down = Copy(l.ffn_down);
    d.router = Copy(l.router);
    d.exp_probs_b = Copy(l.exp_probs_b);
    d.ffn_gate_exps = Copy(l.ffn_gate_exps);
    d.ffn_up_exps = Copy(l.ffn_up_exps);
    d.ffn_down_exps = Copy(l.ffn_down_exps);
    d.shexp_gate = Copy(l.shexp_gate);
    d.shexp_up = Copy(l.shexp_up);
    d.shexp_down = Copy(l.shexp_down);
    d.nextn_enorm = Copy(l.nextn_enorm);
    d.nextn_hnorm = Copy(l.nextn_hnorm);
    d.nextn_eh_proj = Copy(l.nextn_eh_proj);
    d.nextn_shared_head_norm = Copy(l.nextn_shared_head_norm);
    return d;
  }
};

}  // namespace

DeviceModel::~DeviceModel() {
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
}

std::unique_ptr<DeviceModel> DeviceModel::Upload(const ModelWeights& w,
                                                 const core::GgufReader& reader,
                                                 std::string* error_msg) {
  std::unique_ptr<DeviceModel> m(new DeviceModel());
  m->config_ = w.config;
  const auto regions = reader.GetMappedRegions();
  std::vector<core::GgufMappedRegion> shards(regions.begin(), regions.end());
  auto stager = hip::WeightUpload::Create(shards, error_msg);
  if (!stager) {
    return nullptr;
  }
  Uploader up{*stager, m->allocations_, m->bytes_, error_msg};
  m->token_embd_ = up.Copy(w.token_embd);
  m->output_ =
      w.output.data == w.token_embd.data ? m->token_embd_ : up.Copy(w.output);
  m->output_norm_ = up.Copy(w.output_norm);
  m->layers_.reserve(w.layers.size());
  for (const auto& l : w.layers) {
    m->layers_.push_back(up.Layer(l));
    if (!up.ok) {
      return nullptr;
    }
  }
  if (!up.ok || !stager->Finish(error_msg)) {
    return nullptr;
  }
  return m;
}

}  // namespace gufo::models::glm5_next::rocm
