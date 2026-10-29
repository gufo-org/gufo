#include "src/models/nemotron35_bolt/kernels/rocm/device_model.hpp"

#include <hip/hip_runtime.h>

#include <functional>
#include <utility>

#include "src/core/hip/managed_alloc.hpp"
#include "src/core/hip/weight_upload.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/nemotron35_bolt/kernels/rocm/gemv.hpp"

namespace gufo::models::nemotron35_bolt::rocm {
namespace {

using core::GgmlType;

/// Walks every tensor twice: once to measure, once to allocate and queue the
/// copies, so the placement is decided from the total.
struct Walker {
  const ModelWeights& w;
  std::function<bool(const TensorRef&, DeviceTensor&)> visit;

  bool Tensor(const TensorRef& t, DeviceTensor& out) {
    out.type = t.type;
    out.rows = static_cast<int>(t.rows * t.experts);
    out.cols = static_cast<int>(t.cols);
    out.scale = t.scale;
    return visit(t, out);
  }

  bool Model(DeviceTensor& embd, DeviceTensor& norm, DeviceTensor& output,
             std::vector<DeviceLayer>& layers) {
    bool ok = Tensor(w.token_embd, embd) && Tensor(w.output_norm, norm) &&
              Tensor(w.output, output);
    layers.resize(w.layers.size());
    for (std::size_t il = 0; ok && il < w.layers.size(); ++il) {
      const LayerWeights& l = w.layers[il];
      DeviceLayer& d = layers[il];
      d.type = l.type;
      ok = Tensor(l.norm, d.norm);
      switch (l.type) {
        case BlockType::kMamba:
          ok = ok && Tensor(l.ssm_in, d.ssm_in) &&
               Tensor(l.ssm_conv1d, d.ssm_conv1d) &&
               Tensor(l.ssm_conv1d_b, d.ssm_conv1d_b) &&
               Tensor(l.ssm_dt_b, d.ssm_dt_b) && Tensor(l.ssm_a, d.ssm_a) &&
               Tensor(l.ssm_d, d.ssm_d) && Tensor(l.ssm_norm, d.ssm_norm) &&
               Tensor(l.ssm_out, d.ssm_out);
          break;
        case BlockType::kAttention:
          ok = ok && Tensor(l.attn_q, d.attn_q) && Tensor(l.attn_k, d.attn_k) &&
               Tensor(l.attn_v, d.attn_v) && Tensor(l.attn_out, d.attn_out);
          break;
        case BlockType::kMoe:
          d.num_experts = static_cast<int>(l.num_experts);
          d.experts_used = static_cast<int>(w.config.experts_used[il]);
          d.expert_ff = static_cast<int>(w.config.expert_ff[il]);
          ok = ok && Tensor(l.router, d.router) &&
               Tensor(l.router_bias, d.router_bias) &&
               Tensor(l.latent_down, d.latent_down) &&
               Tensor(l.latent_up, d.latent_up) &&
               Tensor(l.up_exps, d.up_exps) &&
               Tensor(l.down_exps, d.down_exps) &&
               Tensor(l.up_shexp, d.up_shexp) &&
               Tensor(l.down_shexp, d.down_shexp);
          break;
      }
    }
    return ok;
  }
};

}  // namespace

DeviceModel::~DeviceModel() {
  for (void* p : allocations_)
    static_cast<void>(hipFree(p));
}

std::unique_ptr<DeviceModel> DeviceModel::Upload(const ModelWeights& weights,
                                                 const core::GgufReader& reader,
                                                 std::string* error_msg) {
  auto fail = [&](std::string message) -> std::unique_ptr<DeviceModel> {
    if (error_msg != nullptr)
      *error_msg = std::move(message);
    return nullptr;
  };
  // Only Mamba projections carry a per-tensor scale; the forward pass applies
  // it there and nowhere else.
  for (const LayerWeights& l : weights.layers) {
    for (const TensorRef* t :
         {&l.attn_q, &l.attn_k, &l.attn_v, &l.attn_out, &l.latent_down,
          &l.latent_up, &l.up_exps, &l.down_exps, &l.up_shexp, &l.down_shexp}) {
      if (!t->empty() && t->scale != 1.0F)
        return fail("unexpected per-tensor scale on " + std::string(t->name));
    }
  }
  if (weights.token_embd.type != GgmlType::kQ8_0 ||
      weights.output.scale != 1.0F)
    return fail("the embedding must be Q8_0 and the head unscaled");
  for (const LayerWeights& l : weights.layers) {
    for (const TensorRef* t :
         {&l.up_exps, &l.down_exps, &l.up_shexp, &l.down_shexp}) {
      if (!t->empty() && t->type != GgmlType::kIQ2_XXS)
        return fail("experts must be IQ2_XXS: " + std::string(t->name));
    }
    for (const TensorRef* t : {&l.ssm_in, &l.ssm_out}) {
      if (!t->empty() && t->type != GgmlType::kNVFP4)
        return fail("Mamba projections must be NVFP4: " + std::string(t->name));
    }
  }

  std::unique_ptr<DeviceModel> m(new DeviceModel());
  {
    std::uint32_t grid[kIq2xxsGridWords];
    Iq2xxsGridBf16(quant::Iq2xxsGrid(), grid);
    void* p = nullptr;
    if (hipMalloc(&p, sizeof(grid)) != hipSuccess ||
        hipMemcpy(p, grid, sizeof(grid), hipMemcpyHostToDevice) != hipSuccess) {
      static_cast<void>(hipFree(p));
      return fail("could not upload the IQ2_XXS grid");
    }
    m->allocations_.push_back(p);
    m->iq2xxs_grid_ = static_cast<std::uint32_t*>(p);
  }
  std::size_t total = 0;
  {
    Walker measure{weights, [&](const TensorRef& t, DeviceTensor&) {
                     total += t.SizeBytes();
                     return true;
                   }};
    DeviceTensor a, b, c;
    std::vector<DeviceLayer> layers;
    measure.Model(a, b, c, layers);
  }
  const auto regions = reader.GetMappedRegions();
  auto stager = hip::WeightUpload::Create(regions, error_msg);
  if (!stager)
    return nullptr;
  const hip::WeightAllocator allocator(total);
  std::string error;
  Walker upload{
      weights, [&](const TensorRef& t, DeviceTensor& out) {
        const std::size_t bytes = t.SizeBytes();
        if (allocator.Allocate(&out.data, bytes) != hipSuccess) {
          error = "device allocation failed for " + std::string(t.name);
          return false;
        }
        m->allocations_.push_back(out.data);
        m->bytes_ += bytes;
        return stager->Copy(t.shard, t.file_offset, bytes, out.data, &error);
      }};
  const bool ok =
      upload.Model(m->token_embd_, m->output_norm_, m->output_, m->layers_);
  if (!stager->Finish(ok ? &error : nullptr) || !ok)
    return fail(error.empty() ? "weight upload failed" : error);
  return m;
}

DeviceDrafter::~DeviceDrafter() {
  for (void* p : allocations)
    static_cast<void>(hipFree(p));
}

std::unique_ptr<DeviceDrafter> DeviceDrafter::Upload(
    const DrafterWeights& w, const core::GgufReader& reader,
    std::string* error_msg) {
  auto fail = [&](std::string message) -> std::unique_ptr<DeviceDrafter> {
    if (error_msg != nullptr)
      *error_msg = std::move(message);
    return nullptr;
  };
  auto d = std::make_unique<DeviceDrafter>();
  std::vector<std::pair<const TensorRef*, DeviceTensor*>> tensors = {
      {&w.fc, &d->fc},
      {&w.enc_norm, &d->enc_norm},
      {&w.output_norm, &d->output_norm},
      {&w.token_proj, &d->token_proj},
      {&w.token_proj_b, &d->token_proj_b},
      {&w.pass_proj, &d->pass_proj},
      {&w.pass_proj_b, &d->pass_proj_b},
      {&w.context_proj, &d->context_proj},
      {&w.context_proj_b, &d->context_proj_b},
      {&w.feature_norm, &d->feature_norm},
      {&w.feature_norm_b, &d->feature_norm_b},
      {&w.feature_up, &d->feature_up},
      {&w.feature_up_b, &d->feature_up_b},
      {&w.feature_down, &d->feature_down},
      {&w.feature_down_b, &d->feature_down_b},
      {&w.slot_embd, &d->slot_embd},
      {&w.rank_embd, &d->rank_embd},
      {&w.attention_bias, &d->attention_bias},
      {&w.lc_output_norm, &d->lc_output_norm},
      {&w.anchor_norm, &d->anchor_norm},
      {&w.factor_in, &d->factor_in},
      {&w.factor_in_b, &d->factor_in_b},
      {&w.out_head, &d->out_head},
      {&w.out_head_b, &d->out_head_b},
      {&w.in_head, &d->in_head},
      {&w.in_head_b, &d->in_head_b},
      {&w.anchor_head, &d->anchor_head},
      {&w.anchor_head_b, &d->anchor_head_b}};
  d->layers.resize(w.layers.size());
  for (std::size_t i = 0; i < w.layers.size(); ++i) {
    const DrafterLayerWeights& l = w.layers[i];
    DeviceDrafterLayer& o = d->layers[i];
    for (auto [src, dst] :
         {std::pair{&l.attn_norm, &o.attn_norm},
          std::pair{&l.attn_conv_base, &o.attn_conv_base},
          std::pair{&l.attn_conv_proj, &o.attn_conv_proj},
          std::pair{&l.attn_q, &o.attn_q}, std::pair{&l.attn_k, &o.attn_k},
          std::pair{&l.attn_v, &o.attn_v},
          std::pair{&l.attn_q_norm, &o.attn_q_norm},
          std::pair{&l.attn_k_norm, &o.attn_k_norm},
          std::pair{&l.attn_out, &o.attn_out},
          std::pair{&l.attn_sinks, &o.attn_sinks},
          std::pair{&l.ffn_norm, &o.ffn_norm},
          std::pair{&l.ffn_conv_base, &o.ffn_conv_base},
          std::pair{&l.ffn_conv_proj, &o.ffn_conv_proj},
          std::pair{&l.ffn_gate, &o.ffn_gate}, std::pair{&l.ffn_up, &o.ffn_up},
          std::pair{&l.ffn_down, &o.ffn_down}})
      tensors.emplace_back(src, dst);
  }
  d->lc_layers.resize(w.lc_layers.size());
  for (std::size_t i = 0; i < w.lc_layers.size(); ++i) {
    const LiliCorrLayerWeights& l = w.lc_layers[i];
    DeviceLiliCorrLayer& o = d->lc_layers[i];
    for (auto [src, dst] :
         {std::pair{&l.attn_norm, &o.attn_norm},
          std::pair{&l.mlp_norm, &o.mlp_norm},
          std::pair{&l.in_proj, &o.in_proj},
          std::pair{&l.in_proj_b, &o.in_proj_b},
          std::pair{&l.out_proj, &o.out_proj},
          std::pair{&l.out_proj_b, &o.out_proj_b},
          std::pair{&l.mlp_up, &o.mlp_up}, std::pair{&l.mlp_up_b, &o.mlp_up_b},
          std::pair{&l.mlp_down, &o.mlp_down},
          std::pair{&l.mlp_down_b, &o.mlp_down_b}})
      tensors.emplace_back(src, dst);
  }
  std::size_t total = 0;
  for (auto [src, dst] : tensors)
    total += src->SizeBytes();
  auto stager = hip::WeightUpload::Create(reader.GetMappedRegions(), error_msg);
  if (!stager)
    return nullptr;
  const hip::WeightAllocator allocator(total);
  std::string error;
  bool ok = true;
  for (auto [src, dst] : tensors) {
    dst->type = src->type;
    dst->rows = static_cast<int>(src->rows * src->experts);
    dst->cols = static_cast<int>(src->cols);
    dst->scale = src->scale;
    const std::size_t bytes = src->SizeBytes();
    if (allocator.Allocate(&dst->data, bytes) != hipSuccess) {
      error = "device allocation failed for " + std::string(src->name);
      ok = false;
      break;
    }
    d->allocations.push_back(dst->data);
    d->bytes += bytes;
    if (!stager->Copy(src->shard, src->file_offset, bytes, dst->data, &error)) {
      ok = false;
      break;
    }
  }
  if (!stager->Finish(ok ? &error : nullptr) || !ok)
    return fail(error.empty() ? "drafter upload failed" : error);
  return d;
}

}  // namespace gufo::models::nemotron35_bolt::rocm
