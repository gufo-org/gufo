#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "staging/models/glm5_next/kernels/rocm/device_model.hpp"
#include "staging/tests/models/glm5_next/hip_test.hpp"

// Upload contract for the real GLM-5.3-Flash artifact: every bound tensor
// arrives on the device with its type, shape, encoding stride and payload
// bytes intact. Metadata fields are compared for every layer; all F32
// tensors and byte windows of three quantized tensors (one per shard family
// of offsets: embedding, expert stack, nextn) are checked bit-exactly
// against the mapped file. Skips (77) unless GUFO_GLM5_NEXT_GGUF points at
// the metadata shard of the split artifact. Needs the GPU exclusively: the
// resident set is the artifact's own ~91 GB.
namespace {

namespace q = gufo::models::glm5_next::rocm;
namespace t = gufo::tests::glm5_next;

int failures = 0;

void Report(const std::string& tag, bool ok) {
  std::printf("%-42s %s\n", tag.c_str(), ok ? "OK" : "FAIL");
  if (!ok) {
    ++failures;
  }
}

void CompareTensor(const std::string& tag, const q::DeviceTensor& d,
                   const gufo::models::glm5_next::TensorRef& w) {
  bool ok = (d.empty() == w.empty());
  if (ok && !w.empty()) {
    ok = d.type == w.type && d.cols == static_cast<std::uint32_t>(w.cols) &&
         d.rows == static_cast<std::uint32_t>(w.rows) &&
         d.experts == static_cast<std::uint32_t>(w.experts) &&
         d.row_bytes == w.RowBytes();
  }
  Report("fields " + tag, ok);
}

void CompareLayer(std::size_t il, const q::DeviceLayer& d,
                  const gufo::models::glm5_next::LayerWeights& w) {
  const std::string p = "layer " + std::to_string(il) + " ";
  CompareTensor(p + "hc_attn_fn", d.hc_attn_fn, w.hc_attn_fn);
  CompareTensor(p + "hc_attn_base", d.hc_attn_base, w.hc_attn_base);
  CompareTensor(p + "hc_attn_scale", d.hc_attn_scale, w.hc_attn_scale);
  CompareTensor(p + "hc_ffn_fn", d.hc_ffn_fn, w.hc_ffn_fn);
  CompareTensor(p + "hc_ffn_base", d.hc_ffn_base, w.hc_ffn_base);
  CompareTensor(p + "hc_ffn_scale", d.hc_ffn_scale, w.hc_ffn_scale);
  CompareTensor(p + "attn_norm", d.attn_norm, w.attn_norm);
  CompareTensor(p + "ffn_norm", d.ffn_norm, w.ffn_norm);
  if (w.dsa) {
    CompareTensor(p + "attn_q_a", d.attn_q_a, w.attn_q_a);
    CompareTensor(p + "attn_q_a_norm", d.attn_q_a_norm, w.attn_q_a_norm);
    CompareTensor(p + "attn_q_b", d.attn_q_b, w.attn_q_b);
    CompareTensor(p + "attn_kv_a", d.attn_kv_a, w.attn_kv_a);
    CompareTensor(p + "attn_kv_a_norm", d.attn_kv_a_norm, w.attn_kv_a_norm);
    CompareTensor(p + "attn_k_b", d.attn_k_b, w.attn_k_b);
    CompareTensor(p + "attn_v_b", d.attn_v_b, w.attn_v_b);
    CompareTensor(p + "attn_out", d.attn_out, w.attn_out);
    CompareTensor(p + "idx_q_b", d.idx_q_b, w.idx_q_b);
    CompareTensor(p + "idx_k", d.idx_k, w.idx_k);
    CompareTensor(p + "idx_k_norm", d.idx_k_norm, w.idx_k_norm);
    CompareTensor(p + "idx_k_bias", d.idx_k_bias, w.idx_k_bias);
    CompareTensor(p + "idx_proj", d.idx_proj, w.idx_proj);
    CompareTensor(p + "idx_comp_gate", d.idx_comp_gate, w.idx_comp_gate);
    CompareTensor(p + "idx_comp_ape", d.idx_comp_ape, w.idx_comp_ape);
  } else {
    CompareTensor(p + "ssm_q", d.ssm_q, w.ssm_q);
    CompareTensor(p + "ssm_k", d.ssm_k, w.ssm_k);
    CompareTensor(p + "ssm_v", d.ssm_v, w.ssm_v);
    CompareTensor(p + "ssm_conv_q", d.ssm_conv_q, w.ssm_conv_q);
    CompareTensor(p + "ssm_conv_k", d.ssm_conv_k, w.ssm_conv_k);
    CompareTensor(p + "ssm_conv_v", d.ssm_conv_v, w.ssm_conv_v);
    CompareTensor(p + "ssm_f_a", d.ssm_f_a, w.ssm_f_a);
    CompareTensor(p + "ssm_f_b", d.ssm_f_b, w.ssm_f_b);
    CompareTensor(p + "ssm_g_a", d.ssm_g_a, w.ssm_g_a);
    CompareTensor(p + "ssm_g_b", d.ssm_g_b, w.ssm_g_b);
    CompareTensor(p + "ssm_beta", d.ssm_beta, w.ssm_beta);
    CompareTensor(p + "ssm_a", d.ssm_a, w.ssm_a);
    CompareTensor(p + "ssm_dt", d.ssm_dt, w.ssm_dt);
    CompareTensor(p + "ssm_norm", d.ssm_norm, w.ssm_norm);
    CompareTensor(p + "ssm_out", d.ssm_out, w.ssm_out);
  }
  Report(p + "dsa flag matches config", d.dsa == w.dsa);
}

bool BitExactFloats(const float* device, const float* host, std::size_t count,
                    const std::string& tag) {
  std::vector<float> got(count);
  t::CheckHip(hipMemcpy(got.data(), device, count * sizeof(float),
                        hipMemcpyDeviceToHost),
              (tag + " download").c_str());
  return std::memcmp(got.data(), host, count * sizeof(float)) == 0;
}

bool BitExactWindow(const void* device, const std::uint8_t* host,
                    std::size_t size) {
  std::vector<std::uint8_t> got(size);
  t::CheckHip(hipMemcpy(got.data(), device, size, hipMemcpyDeviceToHost),
              "download");
  return std::memcmp(got.data(), host, size) == 0;
}

}  // namespace

int main() {
  const char* path = std::getenv("GUFO_GLM5_NEXT_GGUF");
  if (path == nullptr || path[0] == '\0') {
    std::printf("SKIP: set GUFO_GLM5_NEXT_GGUF\n");
    return 77;
  }
  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(path, &error);
  if (reader == nullptr) {
    std::printf("FAIL: open %s: %s\n", path, error.c_str());
    return 1;
  }
  const auto weights =
      gufo::models::glm5_next::ModelWeights::Bind(*reader, &error);
  if (!weights.has_value()) {
    std::printf("FAIL: bind: %s\n", error.c_str());
    return 1;
  }
  const auto model = q::DeviceModel::Upload(*weights, *reader, &error);
  if (model == nullptr) {
    std::printf("FAIL: upload: %s\n", error.c_str());
    return 1;
  }

  // Every bound payload reaches the device exactly once (output aliasing
  // token_embd shares one allocation), mirrored in resident_bytes().
  std::size_t expected = weights->token_embd.SizeBytes();
  if (weights->output.data != weights->token_embd.data) {
    expected += weights->output.SizeBytes();
  }
  expected += weights->output_norm.SizeBytes();
  for (const auto& l : weights->layers) {
    const gufo::models::glm5_next::TensorRef* refs[] = {
        &l.hc_attn_fn,
        &l.hc_attn_base,
        &l.hc_attn_scale,
        &l.hc_ffn_fn,
        &l.hc_ffn_base,
        &l.hc_ffn_scale,
        &l.attn_norm,
        &l.ffn_norm,
        &l.attn_q_a,
        &l.attn_q_a_norm,
        &l.attn_q_b,
        &l.attn_kv_a,
        &l.attn_kv_a_norm,
        &l.attn_k_b,
        &l.attn_v_b,
        &l.attn_out,
        &l.idx_q_b,
        &l.idx_k,
        &l.idx_k_norm,
        &l.idx_k_bias,
        &l.idx_proj,
        &l.idx_comp_gate,
        &l.idx_comp_ape,
        &l.ssm_q,
        &l.ssm_k,
        &l.ssm_v,
        &l.ssm_conv_q,
        &l.ssm_conv_k,
        &l.ssm_conv_v,
        &l.ssm_f_a,
        &l.ssm_f_b,
        &l.ssm_g_a,
        &l.ssm_g_b,
        &l.ssm_beta,
        &l.ssm_a,
        &l.ssm_dt,
        &l.ssm_norm,
        &l.ssm_out,
        &l.ffn_gate,
        &l.ffn_up,
        &l.ffn_down,
        &l.router,
        &l.exp_probs_b,
        &l.ffn_gate_exps,
        &l.ffn_up_exps,
        &l.ffn_down_exps,
        &l.shexp_gate,
        &l.shexp_up,
        &l.shexp_down,
        &l.nextn_enorm,
        &l.nextn_hnorm,
        &l.nextn_eh_proj,
        &l.nextn_shared_head_norm};
    for (const auto* r : refs) {
      if (!r->empty()) {
        expected += r->SizeBytes();
      }
    }
  }
  Report("resident_bytes == summed tensors",
         model->resident_bytes() == expected);
  Report("layer count == num_layers_all",
         model->layers().size() == weights->config.num_layers_all);
  bool dsa_pattern = true;
  for (std::uint32_t il = 0; il < weights->config.num_layers_all; ++il) {
    if (model->layers()[il].dsa != weights->config.IsDsaLayer(il)) {
      dsa_pattern = false;
    }
  }
  Report("dsa flags match IsDsaLayer", dsa_pattern);
  Report("nextn() is the last layer",
         &model->nextn() == &model->layers().back());
  Report("nextn fuses hidden+embedding", !model->nextn().nextn_eh_proj.empty());

  CompareTensor("token_embd", model->token_embd(), weights->token_embd);
  CompareTensor("output", model->output(), weights->output);
  CompareTensor("output_norm", model->output_norm(), weights->output_norm);
  for (std::size_t il = 0; il < weights->layers.size(); ++il) {
    CompareLayer(il, model->layers()[il], weights->layers[il]);
  }

  // Payload integrity: every F32 tensor (norms, convs, per-head scalars,
  // sinkhorn bases, router/bias vectors, APE table) is byte-transferred, and
  // byte windows of three quantized tensors cover the per-shard offset
  // resolution of the split-file uploader.
  std::size_t f32_ok = 0;
  std::size_t f32_total = 0;
  auto check_f32 = [&](const q::DeviceTensor& d,
                       const gufo::models::glm5_next::TensorRef& w) {
    if (d.empty() || d.type != gufo::core::GgmlType::kF32) {
      return;
    }
    ++f32_total;
    if (BitExactFloats(d.f32(), static_cast<const float*>(w.data),
                       static_cast<std::size_t>(d.cols) * d.rows * d.experts,
                       "f32")) {
      ++f32_ok;
    }
  };
  check_f32(model->output_norm(), weights->output_norm);
  for (std::size_t il = 0; il < weights->layers.size(); ++il) {
    const auto& d = model->layers()[il];
    const auto& w = weights->layers[il];
    check_f32(d.attn_norm, w.attn_norm);
    check_f32(d.ffn_norm, w.ffn_norm);
    if (w.dsa) {
      check_f32(d.attn_q_a_norm, w.attn_q_a_norm);
      check_f32(d.attn_kv_a_norm, w.attn_kv_a_norm);
      check_f32(d.idx_k_norm, w.idx_k_norm);
      check_f32(d.idx_k_bias, w.idx_k_bias);
      check_f32(d.idx_comp_ape, w.idx_comp_ape);
    } else {
      check_f32(d.ssm_conv_q, w.ssm_conv_q);
      check_f32(d.ssm_conv_k, w.ssm_conv_k);
      check_f32(d.ssm_conv_v, w.ssm_conv_v);
      check_f32(d.ssm_a, w.ssm_a);
      check_f32(d.ssm_dt, w.ssm_dt);
      check_f32(d.ssm_norm, w.ssm_norm);
    }
    check_f32(d.hc_attn_base, w.hc_attn_base);
    check_f32(d.hc_attn_scale, w.hc_attn_scale);
    check_f32(d.hc_ffn_base, w.hc_ffn_base);
    check_f32(d.hc_ffn_scale, w.hc_ffn_scale);
    check_f32(d.router, w.router);
    check_f32(d.exp_probs_b, w.exp_probs_b);
    check_f32(d.nextn_enorm, w.nextn_enorm);
    check_f32(d.nextn_hnorm, w.nextn_hnorm);
    check_f32(d.nextn_shared_head_norm, w.nextn_shared_head_norm);
  }
  Report("f32 payload bit-exact " + std::to_string(f32_ok) + "/" +
             std::to_string(f32_total),
         f32_ok == f32_total && f32_total > 0);

  auto window_ok = [&](const q::DeviceTensor& d,
                       const gufo::models::glm5_next::TensorRef& w,
                       const char* tag) {
    constexpr std::size_t kWindow = 1U << 20U;
    const std::size_t size = w.SizeBytes();
    const std::size_t off = size > 2 * kWindow ? (size / 2U) & ~127U : 0;
    Report(std::string("window ") + tag + " bit-exact",
           !d.empty() && size > 0 &&
               BitExactWindow(static_cast<const std::uint8_t*>(d.data) + off,
                              static_cast<const std::uint8_t*>(w.data) + off,
                              size - off < kWindow ? size - off : kWindow));
  };
  window_ok(model->token_embd(), weights->token_embd, "token_embd");
  window_ok(weights->layers[3].dsa ? model->layers()[3].attn_q_b
                                   : model->layers()[3].ssm_q,
            weights->layers[3].dsa ? weights->layers[3].attn_q_b
                                   : weights->layers[3].ssm_q,
            "layer3 projection");
  window_ok(model->layers()[weights->config.num_layers].ffn_down_exps,
            weights->layers[weights->config.num_layers].ffn_down_exps,
            "nextn expert stack");

  std::printf("resident %.1f GiB\n",
              static_cast<double>(model->resident_bytes()) /
                  (1024.0 * 1024.0 * 1024.0));
  return failures == 0 ? 0 : 1;
}
