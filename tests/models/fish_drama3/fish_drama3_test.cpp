#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/fish_drama3/config.hpp"
#include "src/models/fish_drama3/hip/backbone_ops.hpp"
#include "src/models/fish_drama3/hip/codec_runtime.hpp"
#include "src/models/fish_drama3/hip/synthesis_runtime.hpp"
#include "src/models/fish_drama3/loader.hpp"
#include "src/models/fish_drama3/prompt.hpp"
#include "src/models/fish_drama3/tokenizer.hpp"
#include "src/models/qwen3_tts/audio.hpp"

namespace {

void Require(bool cond, const std::string& msg) {
  if (!cond) {
    std::cerr << "[FAIL] " << msg << "\n";
    std::exit(1);
  }
}

float Bf16BitsToFloat(std::uint16_t bits) {
  std::uint32_t u = static_cast<std::uint32_t>(bits) << 16U;
  float f = 0.0F;
  std::memcpy(&f, &u, sizeof(float));
  return f;
}

float Fp8E4m3ToFloatHost(std::uint8_t b) {
  const std::uint32_t sign = (b >> 7U) & 1U;
  const std::uint32_t exp = (b >> 3U) & 0xFU;
  const std::uint32_t mant = b & 0x7U;
  if (exp == 0U) {
    if (mant == 0U)
      return sign ? -0.0F : 0.0F;
    const float v = static_cast<float>(mant) * (1.0F / 512.0F);
    return sign ? -v : v;
  }
  if (exp == 15U && mant == 7U)
    return 0.0F;
  const std::uint32_t f_bits =
      (sign << 31U) | ((exp + 120U) << 23U) | (mant << 20U);
  float res = 0.0F;
  std::memcpy(&res, &f_bits, sizeof(float));
  return res;
}

}  // namespace

int main(int argc, char** argv) {
#if !defined(ENGINE_ENABLE_HIP)
  (void)argc;
  (void)argv;
  return 77;
#else
  using namespace gufo::models::fish_drama3;
  const std::string model_dir =
      (argc > 1) ? argv[1] : "/var/home/rikaaa0928/models/drama-3";

  std::cout << "=== [Stage 1] Testing Fish Drama-3 Config ===\n";
  Require(LooksLikeFishDrama3(model_dir),
          "LooksLikeFishDrama3 returned false for " + model_dir);
  const auto cfg_opt = LoadModelConfigFromPath(model_dir);
  Require(cfg_opt.has_value(), "LoadModelConfigFromPath returned nullopt");
  const auto& cfg = *cfg_opt;
  Require(IsSupportedModelConfig(cfg), "IsSupportedModelConfig returned false");
  Require(cfg.backbone.num_hidden_layers == 40, "Expected 40 backbone layers");
  Require(cfg.backbone.num_experts == 256, "Expected 256 routed experts");
  Require(cfg.backbone.num_experts_per_tok == 8, "Expected top-8 experts");
  std::cout << "[PASS] Config validated: 40 layers, 256 experts (top-8), "
               "hidden="
            << cfg.backbone.hidden_size << "\n";

  std::cout << "=== [Stage 2] Testing Drama-3 Tokenizer & Prompt Builder ===\n";
  const Drama3Tokenizer tok(model_dir);
  const auto ids =
      tok.Encode("<|im_start|>user\n<|speaker:0|>Hello world!<|im_end|>\n");
  Require(!ids.empty(), "Tokenizer returned empty ids");
  std::cout << "Encoded tokens (" << ids.size() << "): ";
  for (auto id : ids)
    std::cout << id << " ";
  std::cout << "\n";
  Require(ids.front() == 248045, "Expected <|im_start|> == 248045");

  FishDacCodes dummy_codes;
  dummy_codes.codebooks = 10;
  dummy_codes.frames = 4;
  dummy_codes.codes.assign(4 * 10, 0);
  for (std::size_t f = 0; f < 4; ++f) {
    dummy_codes.codes[0 * 4 + f] = static_cast<std::int32_t>(100 + f);
  }
  Drama3Take ref_take{.text = "<|speaker:0|>Reference voice.",
                      .codes = dummy_codes};
  const auto prompt_frames = BuildDrama3Prompt(
      cfg, tok, "<|speaker:0|>Hello from Fish Drama 3 on AMD gfx1151!",
      {ref_take});
  const auto prefix_frames = BuildDrama3PromptPrefix(cfg, tok, {ref_take});
  const auto suffix_frames = BuildDrama3PromptSuffix(
      tok, "<|speaker:0|>Hello from Fish Drama 3 on AMD gfx1151!");
  Require(!prompt_frames.empty(), "Prompt frames should not be empty");
  Require(prefix_frames.size() + suffix_frames.size() == prompt_frames.size(),
          "Prefix + suffix frame count must match full prompt");
  std::cout << "[PASS] Built prompt with " << prompt_frames.size()
            << " frames\n";

  std::cout << "=== [Stage 3] Testing Native FP8 (F8_E4M3) Safetensors Shard "
               "Loader ===\n";
  const auto shard1_path =
      std::filesystem::path(model_dir) / "model-00001-of-00008.safetensors";
  if (std::filesystem::exists(shard1_path) &&
      !std::filesystem::exists(shard1_path.string() + ".aria2")) {
    auto shard1 = LoadSingleSafetensorsFile(shard1_path.string());
    Require(shard1 != nullptr, "Failed to load shard 1");
    const auto& exp_w = shard1->Require(
        "model.language_model.layers.0.mlp.experts.0.gate_proj.weight");
    const auto& exp_s = shard1->Require(
        "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale_"
        "inv");
    Require(exp_w.dtype == DType::kF8_E4M3,
            "Expected F8_E4M3 for expert weight");
    Require(exp_w.shape.size() == 2 && exp_w.shape[0] == 512 &&
                exp_w.shape[1] == 2048,
            "Unexpected expert weight shape");
    Require(exp_s.dtype == DType::kF32, "Expected F32 for weight_scale_inv");
    Require(
        exp_s.shape.size() == 2 && exp_s.shape[0] == 4 && exp_s.shape[1] == 16,
        "Unexpected weight_scale_inv shape");
    std::cout << "[PASS] Verified native F8_E4M3 [512, 2048] + F32 scale_inv "
                 "[4, 16] in shard 1 ("
              << shard1->Tensors().size() << " tensors)\n";

    std::cout << "=== [Stage 3.5] Validating Native FP8 MoE & Gated DeltaNet "
                 "Kernels on gfx1151 ===\n";
    {
      const std::size_t hidden = 2048;
      const std::size_t num_experts = 256;
      const std::size_t top_k = 8;
      const std::size_t moe_inter = 512;

      std::vector<std::uint8_t> h_gate_fp8(num_experts * moe_inter * hidden);
      std::vector<float> h_gate_scale(num_experts * 64);
      std::vector<std::uint8_t> h_up_fp8(num_experts * moe_inter * hidden);
      std::vector<float> h_up_scale(num_experts * 64);
      std::vector<std::uint8_t> h_down_fp8(num_experts * hidden * moe_inter);
      std::vector<float> h_down_scale(num_experts * 64);

      for (std::size_t e = 0; e < num_experts; ++e) {
        const std::string ep =
            "model.language_model.layers.0.mlp.experts." + std::to_string(e);
        const auto* gw = shard1->Find(ep + ".gate_proj.weight");
        const auto* gs = shard1->Find(ep + ".gate_proj.weight_scale_inv");
        const auto* uw = shard1->Find(ep + ".up_proj.weight");
        const auto* us = shard1->Find(ep + ".up_proj.weight_scale_inv");
        const auto* dw = shard1->Find(ep + ".down_proj.weight");
        const auto* ds = shard1->Find(ep + ".down_proj.weight_scale_inv");
        Require(gw && gs && uw && us && dw && ds, "Missing expert tensor");
        std::memcpy(h_gate_fp8.data() + e * moe_inter * hidden, gw->data,
                    moe_inter * hidden);
        std::memcpy(h_gate_scale.data() + e * 64, gs->data, 64 * sizeof(float));
        std::memcpy(h_up_fp8.data() + e * moe_inter * hidden, uw->data,
                    moe_inter * hidden);
        std::memcpy(h_up_scale.data() + e * 64, us->data, 64 * sizeof(float));
        std::memcpy(h_down_fp8.data() + e * hidden * moe_inter, dw->data,
                    hidden * moe_inter);
        std::memcpy(h_down_scale.data() + e * 64, ds->data, 64 * sizeof(float));
      }

      const auto& router_t =
          shard1->Require("model.language_model.layers.0.mlp.gate.weight");
      const auto& sh_gate_w_t = shard1->Require(
          "model.language_model.layers.0.mlp.shared_expert_gate.weight");
      const auto& sh_gate_t = shard1->Require(
          "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight");
      const auto& sh_up_t = shard1->Require(
          "model.language_model.layers.0.mlp.shared_expert.up_proj.weight");
      const auto& sh_down_t = shard1->Require(
          "model.language_model.layers.0.mlp.shared_expert.down_proj.weight");

      std::vector<float> h_x(hidden);
      for (std::size_t i = 0; i < hidden; ++i) {
        h_x[i] = std::sin(static_cast<float>(i) * 0.031F) * 0.5F;
      }

      float *d_x = nullptr, *d_out = nullptr, *d_inter = nullptr,
            *d_top_w = nullptr, *d_sh_scale = nullptr;
      float *d_g_sc = nullptr, *d_u_sc = nullptr, *d_d_sc = nullptr;
      std::uint8_t *d_g_fp8 = nullptr, *d_u_fp8 = nullptr, *d_d_fp8 = nullptr;
      std::uint16_t *d_router = nullptr, *d_sh_gw = nullptr, *d_sh_g = nullptr,
                    *d_sh_u = nullptr, *d_sh_d = nullptr;
      std::int32_t* d_top_idx = nullptr;

      Require(hipMalloc(&d_x, hidden * sizeof(float)) == hipSuccess,
              "hipMalloc");
      Require(hipMalloc(&d_out, hidden * sizeof(float)) == hipSuccess,
              "hipMalloc");
      Require(hipMalloc(&d_inter, (top_k + 1) * moe_inter * sizeof(float)) ==
                  hipSuccess,
              "hipMalloc");
      Require(hipMalloc(&d_top_w, top_k * sizeof(float)) == hipSuccess,
              "hipMalloc");
      Require(hipMalloc(&d_sh_scale, sizeof(float)) == hipSuccess, "hipMalloc");
      Require(hipMalloc(&d_top_idx, top_k * sizeof(std::int32_t)) == hipSuccess,
              "hipMalloc");

      Require(hipMalloc(&d_g_fp8, h_gate_fp8.size()) == hipSuccess,
              "hipMalloc");
      Require(
          hipMalloc(&d_g_sc, h_gate_scale.size() * sizeof(float)) == hipSuccess,
          "hipMalloc");
      Require(hipMalloc(&d_u_fp8, h_up_fp8.size()) == hipSuccess, "hipMalloc");
      Require(
          hipMalloc(&d_u_sc, h_up_scale.size() * sizeof(float)) == hipSuccess,
          "hipMalloc");
      Require(hipMalloc(&d_d_fp8, h_down_fp8.size()) == hipSuccess,
              "hipMalloc");
      Require(
          hipMalloc(&d_d_sc, h_down_scale.size() * sizeof(float)) == hipSuccess,
          "hipMalloc");

      Require(hipMalloc(&d_router, router_t.byte_size) == hipSuccess,
              "hipMalloc");
      Require(hipMalloc(&d_sh_gw, sh_gate_w_t.byte_size) == hipSuccess,
              "hipMalloc");
      Require(hipMalloc(&d_sh_g, sh_gate_t.byte_size) == hipSuccess,
              "hipMalloc");
      Require(hipMalloc(&d_sh_u, sh_up_t.byte_size) == hipSuccess, "hipMalloc");
      Require(hipMalloc(&d_sh_d, sh_down_t.byte_size) == hipSuccess,
              "hipMalloc");

      Require(hipMemcpy(d_x, h_x.data(), hidden * sizeof(float),
                        hipMemcpyHostToDevice) == hipSuccess,
              "hipMemcpy");
      Require(hipMemset(d_out, 0, hidden * sizeof(float)) == hipSuccess,
              "hipMemset");
      Require(hipMemcpy(d_g_fp8, h_gate_fp8.data(), h_gate_fp8.size(),
                        hipMemcpyHostToDevice) == hipSuccess,
              "hipMemcpy");
      Require(hipMemcpy(d_g_sc, h_gate_scale.data(),
                        h_gate_scale.size() * sizeof(float),
                        hipMemcpyHostToDevice) == hipSuccess,
              "hipMemcpy");
      Require(hipMemcpy(d_u_fp8, h_up_fp8.data(), h_up_fp8.size(),
                        hipMemcpyHostToDevice) == hipSuccess,
              "hipMemcpy");
      Require(hipMemcpy(d_u_sc, h_up_scale.data(),
                        h_up_scale.size() * sizeof(float),
                        hipMemcpyHostToDevice) == hipSuccess,
              "hipMemcpy");
      Require(hipMemcpy(d_d_fp8, h_down_fp8.data(), h_down_fp8.size(),
                        hipMemcpyHostToDevice) == hipSuccess,
              "hipMemcpy");
      Require(hipMemcpy(d_d_sc, h_down_scale.data(),
                        h_down_scale.size() * sizeof(float),
                        hipMemcpyHostToDevice) == hipSuccess,
              "hipMemcpy");

      Require(hipMemcpy(d_router, router_t.data, router_t.byte_size,
                        hipMemcpyHostToDevice) == hipSuccess,
              "hipMemcpy");
      Require(hipMemcpy(d_sh_gw, sh_gate_w_t.data, sh_gate_w_t.byte_size,
                        hipMemcpyHostToDevice) == hipSuccess,
              "hipMemcpy");
      Require(hipMemcpy(d_sh_g, sh_gate_t.data, sh_gate_t.byte_size,
                        hipMemcpyHostToDevice) == hipSuccess,
              "hipMemcpy");
      Require(hipMemcpy(d_sh_u, sh_up_t.data, sh_up_t.byte_size,
                        hipMemcpyHostToDevice) == hipSuccess,
              "hipMemcpy");
      Require(hipMemcpy(d_sh_d, sh_down_t.data, sh_down_t.byte_size,
                        hipMemcpyHostToDevice) == hipSuccess,
              "hipMemcpy");

      hip::LaunchBackboneMoERouter(d_router, d_sh_gw, d_x, d_top_idx, d_top_w,
                                   d_sh_scale, 1, num_experts, top_k, hidden,
                                   nullptr);
      hip::LaunchBackboneMoEGateUpSwiGlu(
          d_g_fp8, d_g_sc, d_u_fp8, d_u_sc, d_sh_g, d_sh_u, d_x, d_top_idx,
          d_inter, 1, top_k, moe_inter, hidden, nullptr);
      hip::LaunchBackboneMoEDownCombine(d_d_fp8, d_d_sc, d_sh_d, d_inter,
                                        d_top_idx, d_top_w, d_sh_scale, d_out,
                                        1, top_k, moe_inter, hidden, nullptr);
      Require(hipDeviceSynchronize() == hipSuccess, "hipDeviceSynchronize");

      std::vector<std::int32_t> h_top_idx(top_k);
      std::vector<float> h_top_w(top_k);
      float h_sh_scale = 0.0F;
      std::vector<float> h_out_gpu(hidden);
      Require(
          hipMemcpy(h_top_idx.data(), d_top_idx, top_k * sizeof(std::int32_t),
                    hipMemcpyDeviceToHost) == hipSuccess,
          "hipMemcpy");
      Require(hipMemcpy(h_top_w.data(), d_top_w, top_k * sizeof(float),
                        hipMemcpyDeviceToHost) == hipSuccess,
              "hipMemcpy");
      Require(hipMemcpy(&h_sh_scale, d_sh_scale, sizeof(float),
                        hipMemcpyDeviceToHost) == hipSuccess,
              "hipMemcpy");
      Require(hipMemcpy(h_out_gpu.data(), d_out, hidden * sizeof(float),
                        hipMemcpyDeviceToHost) == hipSuccess,
              "hipMemcpy");

      // CPU FP32 reference for the 8 routed FP8 experts + 1 BF16 shared expert
      std::vector<float> h_out_ref(hidden, 0.0F);
      for (std::size_t k = 0; k < top_k; ++k) {
        const int e = h_top_idx[k];
        const float ew = h_top_w[k];
        std::vector<float> inter(moe_inter);
        const std::uint8_t* g_base =
            h_gate_fp8.data() +
            static_cast<std::size_t>(e) * moe_inter * hidden;
        const float* g_sc =
            h_gate_scale.data() + static_cast<std::size_t>(e) * 64;
        const std::uint8_t* u_base =
            h_up_fp8.data() + static_cast<std::size_t>(e) * moe_inter * hidden;
        const float* u_sc =
            h_up_scale.data() + static_cast<std::size_t>(e) * 64;
        for (std::size_t r = 0; r < moe_inter; ++r) {
          float g = 0.0F, u = 0.0F;
          for (std::size_t c = 0; c < hidden; ++c) {
            const float sg = g_sc[(r / 128) * 16 + (c / 128)];
            const float su = u_sc[(r / 128) * 16 + (c / 128)];
            g += Fp8E4m3ToFloatHost(g_base[r * hidden + c]) * sg * h_x[c];
            u += Fp8E4m3ToFloatHost(u_base[r * hidden + c]) * su * h_x[c];
          }
          inter[r] = (g / (1.0F + std::exp(-g))) * u;
        }
        const std::uint8_t* d_base =
            h_down_fp8.data() +
            static_cast<std::size_t>(e) * hidden * moe_inter;
        const float* d_sc =
            h_down_scale.data() + static_cast<std::size_t>(e) * 64;
        for (std::size_t r = 0; r < hidden; ++r) {
          float d = 0.0F;
          for (std::size_t c = 0; c < moe_inter; ++c) {
            const float sd = d_sc[(r / 128) * 4 + (c / 128)];
            d += Fp8E4m3ToFloatHost(d_base[r * moe_inter + c]) * sd * inter[c];
          }
          h_out_ref[r] += ew * d;
        }
      }
      // Shared expert (BF16)
      {
        const auto* sg_ptr =
            reinterpret_cast<const std::uint16_t*>(sh_gate_t.data);
        const auto* su_ptr =
            reinterpret_cast<const std::uint16_t*>(sh_up_t.data);
        const auto* sd_ptr =
            reinterpret_cast<const std::uint16_t*>(sh_down_t.data);
        std::vector<float> sh_inter(moe_inter);
        for (std::size_t r = 0; r < moe_inter; ++r) {
          float g = 0.0F, u = 0.0F;
          for (std::size_t c = 0; c < hidden; ++c) {
            g += Bf16BitsToFloat(sg_ptr[r * hidden + c]) * h_x[c];
            u += Bf16BitsToFloat(su_ptr[r * hidden + c]) * h_x[c];
          }
          sh_inter[r] = (g / (1.0F + std::exp(-g))) * u;
        }
        for (std::size_t r = 0; r < hidden; ++r) {
          float d = 0.0F;
          for (std::size_t c = 0; c < moe_inter; ++c) {
            d += Bf16BitsToFloat(sd_ptr[r * moe_inter + c]) * sh_inter[c];
          }
          h_out_ref[r] += h_sh_scale * d;
        }
      }

      float max_diff = 0.0F;
      double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
      for (std::size_t i = 0; i < hidden; ++i) {
        const float a = h_out_gpu[i];
        const float b = h_out_ref[i];
        max_diff = std::max(max_diff, std::fabs(a - b));
        dot += static_cast<double>(a) * static_cast<double>(b);
        norm_a += static_cast<double>(a) * static_cast<double>(a);
        norm_b += static_cast<double>(b) * static_cast<double>(b);
      }
      const double cos_sim =
          dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-12);
      std::cout
          << "[PASS] Native FP8 MoE + BF16 Shared Expert Layer 0 (256->top-8=["
          << h_top_idx[0] << "," << h_top_idx[1] << "," << h_top_idx[2]
          << "...]): " << "cosine_sim=" << cos_sim
          << ", max_abs_diff=" << max_diff << "\n";
      Require(cos_sim > 0.999999, "FP8 MoE cosine similarity below threshold");

      const int iters = 200;
      const auto t_moe0 = std::chrono::steady_clock::now();
      for (int it = 0; it < iters; ++it) {
        hip::LaunchBackboneMoERouter(d_router, d_sh_gw, d_x, d_top_idx, d_top_w,
                                     d_sh_scale, 1, num_experts, top_k, hidden,
                                     nullptr);
        hip::LaunchBackboneMoEGateUpSwiGlu(
            d_g_fp8, d_g_sc, d_u_fp8, d_u_sc, d_sh_g, d_sh_u, d_x, d_top_idx,
            d_inter, 1, top_k, moe_inter, hidden, nullptr);
        hip::LaunchBackboneMoEDownCombine(d_d_fp8, d_d_sc, d_sh_d, d_inter,
                                          d_top_idx, d_top_w, d_sh_scale, d_out,
                                          1, top_k, moe_inter, hidden, nullptr);
      }
      Require(hipDeviceSynchronize() == hipSuccess, "hipDeviceSynchronize");
      const double moe_us = std::chrono::duration<double, std::micro>(
                                std::chrono::steady_clock::now() - t_moe0)
                                .count() /
                            iters;
      std::cout << "[PASS] Native FP8 MoE single-step latency on gfx1151: "
                << moe_us << " us/layer\n";

      (void)hipFree(d_x);
      (void)hipFree(d_out);
      (void)hipFree(d_inter);
      (void)hipFree(d_top_w);
      (void)hipFree(d_sh_scale);
      (void)hipFree(d_top_idx);
      (void)hipFree(d_g_fp8);
      (void)hipFree(d_g_sc);
      (void)hipFree(d_u_fp8);
      (void)hipFree(d_u_sc);
      (void)hipFree(d_d_fp8);
      (void)hipFree(d_d_sc);
      (void)hipFree(d_router);
      (void)hipFree(d_sh_gw);
      (void)hipFree(d_sh_g);
      (void)hipFree(d_sh_u);
      (void)hipFree(d_sh_d);

      // Validate Register-Resident Gated DeltaNet Recurrent + Gated RMSNorm
      // kernel
      const std::size_t k_heads = 16, v_heads = 32, head_dim = 128, steps = 4;
      const std::size_t qkv_dim = (2 * k_heads + v_heads) * head_dim;  // 8192
      const std::size_t v_dim = v_heads * head_dim;                    // 4096
      const float eps = 1e-6F;

      const auto& a_log_t =
          shard1->Require("model.language_model.layers.0.linear_attn.A_log");
      const auto& dt_bias_t =
          shard1->Require("model.language_model.layers.0.linear_attn.dt_bias");
      const auto& norm_w_t = shard1->Require(
          "model.language_model.layers.0.linear_attn.norm.weight");

      std::vector<float> h_neg_exp_a(v_heads), h_dt_bias(v_heads),
          h_norm_w(head_dim);
      for (std::size_t h = 0; h < v_heads; ++h) {
        float al = 0.0F;
        std::memcpy(&al, static_cast<const std::uint8_t*>(a_log_t.data) + h * 4,
                    4);
        h_neg_exp_a[h] = -std::exp(al);
        std::uint16_t b_bits = 0;
        std::memcpy(&b_bits,
                    static_cast<const std::uint8_t*>(dt_bias_t.data) + h * 2,
                    2);
        h_dt_bias[h] = Bf16BitsToFloat(b_bits);
      }
      for (std::size_t d = 0; d < head_dim; ++d) {
        if (norm_w_t.dtype == DType::kF32) {
          std::memcpy(&h_norm_w[d],
                      static_cast<const std::uint8_t*>(norm_w_t.data) + d * 4,
                      4);
        } else {
          std::uint16_t w_bits = 0;
          std::memcpy(&w_bits,
                      static_cast<const std::uint8_t*>(norm_w_t.data) + d * 2,
                      2);
          h_norm_w[d] = Bf16BitsToFloat(w_bits);
        }
      }

      std::vector<float> h_qkv(steps * qkv_dim), h_z(steps * v_dim),
          h_b(steps * v_heads), h_a(steps * v_heads);
      for (std::size_t i = 0; i < h_qkv.size(); ++i)
        h_qkv[i] = std::sin(i * 0.017F) * 0.3F;
      for (std::size_t i = 0; i < h_z.size(); ++i)
        h_z[i] = std::cos(i * 0.023F) * 0.4F;
      for (std::size_t i = 0; i < h_b.size(); ++i)
        h_b[i] = std::sin(i * 0.29F) * 0.4F;
      for (std::size_t i = 0; i < h_a.size(); ++i)
        h_a[i] = std::cos(i * 0.13F) * 0.2F;

      float *d_qkv = nullptr, *d_z = nullptr, *d_b = nullptr, *d_a = nullptr;
      float *d_dt_bias = nullptr, *d_neg_exp_a = nullptr, *d_norm_w = nullptr,
            *d_state = nullptr, *d_gdn_out = nullptr;
      Require(hipMalloc(&d_qkv, h_qkv.size() * 4) == hipSuccess, "hipMalloc");
      Require(hipMalloc(&d_z, h_z.size() * 4) == hipSuccess, "hipMalloc");
      Require(hipMalloc(&d_b, h_b.size() * 4) == hipSuccess, "hipMalloc");
      Require(hipMalloc(&d_a, h_a.size() * 4) == hipSuccess, "hipMalloc");
      Require(hipMalloc(&d_dt_bias, v_heads * 4) == hipSuccess, "hipMalloc");
      Require(hipMalloc(&d_neg_exp_a, v_heads * 4) == hipSuccess, "hipMalloc");
      Require(hipMalloc(&d_norm_w, head_dim * 4) == hipSuccess, "hipMalloc");
      Require(
          hipMalloc(&d_state, v_heads * head_dim * head_dim * 4) == hipSuccess,
          "hipMalloc");
      Require(hipMalloc(&d_gdn_out, steps * v_dim * 4) == hipSuccess,
              "hipMalloc");

      Require(hipMemcpy(d_qkv, h_qkv.data(), h_qkv.size() * 4,
                        hipMemcpyHostToDevice) == hipSuccess,
              "hipMemcpy");
      Require(hipMemcpy(d_z, h_z.data(), h_z.size() * 4,
                        hipMemcpyHostToDevice) == hipSuccess,
              "hipMemcpy");
      Require(hipMemcpy(d_b, h_b.data(), h_b.size() * 4,
                        hipMemcpyHostToDevice) == hipSuccess,
              "hipMemcpy");
      Require(hipMemcpy(d_a, h_a.data(), h_a.size() * 4,
                        hipMemcpyHostToDevice) == hipSuccess,
              "hipMemcpy");
      Require(hipMemcpy(d_dt_bias, h_dt_bias.data(), v_heads * 4,
                        hipMemcpyHostToDevice) == hipSuccess,
              "hipMemcpy");
      Require(hipMemcpy(d_neg_exp_a, h_neg_exp_a.data(), v_heads * 4,
                        hipMemcpyHostToDevice) == hipSuccess,
              "hipMemcpy");
      Require(hipMemcpy(d_norm_w, h_norm_w.data(), head_dim * 4,
                        hipMemcpyHostToDevice) == hipSuccess,
              "hipMemcpy");
      Require(hipMemset(d_state, 0, v_heads * head_dim * head_dim * 4) ==
                  hipSuccess,
              "hipMemset");

      hip::LaunchBackboneGatedDeltaNetRecurrent(
          d_qkv, d_z, d_b, d_a, d_dt_bias, d_neg_exp_a, d_norm_w, d_state,
          d_gdn_out, steps, k_heads, v_heads, head_dim, eps, nullptr);
      Require(hipDeviceSynchronize() == hipSuccess, "hipDeviceSynchronize");

      std::vector<float> h_gdn_gpu(steps * v_dim);
      Require(hipMemcpy(h_gdn_gpu.data(), d_gdn_out, steps * v_dim * 4,
                        hipMemcpyDeviceToHost) == hipSuccess,
              "hipMemcpy");

      // CPU FP32 reference
      std::vector<float> h_state_ref(v_heads * head_dim * head_dim, 0.0F);
      std::vector<float> h_gdn_ref(steps * v_dim, 0.0F);
      const float q_scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
      for (std::size_t t = 0; t < steps; ++t) {
        for (std::size_t vh = 0; vh < v_heads; ++vh) {
          const std::size_t kh = vh / (v_heads / k_heads);
          const float* q_ptr = h_qkv.data() + t * qkv_dim + kh * head_dim;
          const float* k_ptr =
              h_qkv.data() + t * qkv_dim + (k_heads + kh) * head_dim;
          const float* v_ptr = h_qkv.data() + t * qkv_dim +
                               2 * k_heads * head_dim + vh * head_dim;
          float q_norm = 0.0F, k_norm = 0.0F;
          for (std::size_t d = 0; d < head_dim; ++d) {
            q_norm += q_ptr[d] * q_ptr[d];
            k_norm += k_ptr[d] * k_ptr[d];
          }
          q_norm = 1.0F / std::sqrt(q_norm + eps);
          k_norm = 1.0F / std::sqrt(k_norm + eps);

          const float a_val = h_a[t * v_heads + vh] + h_dt_bias[vh];
          const float sp = a_val > 20.0F ? a_val : std::log1pf(std::exp(a_val));
          const float decay = std::exp(sp * h_neg_exp_a[vh]);
          const float beta = 1.0F / (1.0F + std::exp(-h_b[t * v_heads + vh]));

          float* st = h_state_ref.data() + vh * head_dim * head_dim;
          std::vector<float> y_head(head_dim, 0.0F);
          float y_sq = 0.0F;
          for (std::size_t vi = 0; vi < head_dim; ++vi) {
            float kv_mem = 0.0F;
            for (std::size_t ki = 0; ki < head_dim; ++ki) {
              st[ki * head_dim + vi] *= decay;
              kv_mem += st[ki * head_dim + vi] * (k_ptr[ki] * k_norm);
            }
            const float delta = (v_ptr[vi] - kv_mem) * beta;
            float out_v = 0.0F;
            for (std::size_t ki = 0; ki < head_dim; ++ki) {
              st[ki * head_dim + vi] += (k_ptr[ki] * k_norm) * delta;
              out_v += st[ki * head_dim + vi] * (q_ptr[ki] * q_norm * q_scale);
            }
            y_head[vi] = out_v;
            y_sq += out_v * out_v;
          }
          const float inv_rms =
              1.0F / std::sqrt(y_sq / static_cast<float>(head_dim) + eps);
          for (std::size_t vi = 0; vi < head_dim; ++vi) {
            const float zv = h_z[(t * v_heads + vh) * head_dim + vi];
            const float silu_z = zv / (1.0F + std::exp(-zv));
            h_gdn_ref[(t * v_heads + vh) * head_dim + vi] =
                y_head[vi] * inv_rms * h_norm_w[vi] * silu_z;
          }
        }
      }

      double gdn_dot = 0.0, gdn_na = 0.0, gdn_nb = 0.0;
      float gdn_max_diff = 0.0F;
      for (std::size_t i = 0; i < h_gdn_ref.size(); ++i) {
        const float a = h_gdn_gpu[i];
        const float b = h_gdn_ref[i];
        gdn_max_diff = std::max(gdn_max_diff, std::fabs(a - b));
        gdn_dot += static_cast<double>(a) * static_cast<double>(b);
        gdn_na += static_cast<double>(a) * static_cast<double>(a);
        gdn_nb += static_cast<double>(b) * static_cast<double>(b);
      }
      const double gdn_cos =
          gdn_dot / (std::sqrt(gdn_na) * std::sqrt(gdn_nb) + 1e-12);
      std::cout
          << "[PASS] Register-Resident Gated DeltaNet Recurrent (4 steps): "
          << "cosine_sim=" << gdn_cos << ", max_abs_diff=" << gdn_max_diff
          << "\n";
      Require(gdn_cos > 0.999999,
              "Gated DeltaNet cosine similarity below threshold");

      const auto t_gdn0 = std::chrono::steady_clock::now();
      for (int it = 0; it < iters; ++it) {
        hip::LaunchBackboneGatedDeltaNetRecurrent(
            d_qkv, d_z, d_b, d_a, d_dt_bias, d_neg_exp_a, d_norm_w, d_state,
            d_gdn_out, 1, k_heads, v_heads, head_dim, eps, nullptr);
      }
      Require(hipDeviceSynchronize() == hipSuccess, "hipDeviceSynchronize");
      const double gdn_us = std::chrono::duration<double, std::micro>(
                                std::chrono::steady_clock::now() - t_gdn0)
                                .count() /
                            iters;
      std::cout << "[PASS] Register-Resident Gated DeltaNet single-step "
                   "latency on gfx1151: "
                << gdn_us << " us/layer\n";

      (void)hipFree(d_qkv);
      (void)hipFree(d_z);
      (void)hipFree(d_b);
      (void)hipFree(d_a);
      (void)hipFree(d_dt_bias);
      (void)hipFree(d_neg_exp_a);
      (void)hipFree(d_norm_w);
      (void)hipFree(d_state);
      (void)hipFree(d_gdn_out);
    }
  }

  std::cout
      << "=== [Stage 4] Testing Fish DAC Codec HIP Runtime on gfx1151 ===\n";
  std::string err;
  auto codec = hip::FishDacCodecHipRuntime::Create(model_dir, cfg.codec, &err);
  Require(codec != nullptr, "FishDacCodecHipRuntime::Create failed: " + err);

  const std::size_t test_frames = 16;
  const std::size_t test_samples = test_frames * cfg.codec.frame_length;
  std::vector<float> synth_in(test_samples);
  for (std::size_t i = 0; i < test_samples; ++i) {
    const float t = static_cast<float>(i) / 44100.0F;
    synth_in[i] = 0.25F * std::sin(2.0F * 3.14159265F * 220.0F * t) +
                  0.15F * std::sin(2.0F * 3.14159265F * 440.0F * t);
  }

  FishDacCodes encoded;
  const auto t0 = std::chrono::steady_clock::now();
  Require(codec->Encode(synth_in, &encoded, &err),
          "Codec Encode failed: " + err);
  const auto t1 = std::chrono::steady_clock::now();
  const double enc_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  Require(encoded.frames == static_cast<std::int64_t>(test_frames),
          "Unexpected encoded frames count");
  std::cout << "[PASS] Codec Encode " << encoded.frames << " frames in "
            << enc_ms << " ms (semantic[0]=" << encoded.codes[0] << ")\n";

  std::vector<float> decoded_pcm;
  const auto t2 = std::chrono::steady_clock::now();
  Require(codec->Decode(encoded, &decoded_pcm, &err),
          "Codec Decode failed: " + err);
  const auto t3 = std::chrono::steady_clock::now();
  const double dec_ms =
      std::chrono::duration<double, std::milli>(t3 - t2).count();
  Require(decoded_pcm.size() == test_samples,
          "Unexpected decoded sample count");
  const double audio_dur_ms =
      static_cast<double>(decoded_pcm.size()) * 1000.0 / 44100.0;
  std::cout << "[PASS] Codec Decode " << encoded.frames << " frames ("
            << audio_dur_ms << " ms audio) in " << dec_ms
            << " ms (Codec RTF = " << (dec_ms / audio_dur_ms) << ")\n";

  codec->ResetStream();
  std::vector<float> stream_pcm;
  for (std::size_t f = 0; f < static_cast<std::size_t>(encoded.frames);
       f += 4) {
    const std::size_t n =
        std::min<std::size_t>(4, static_cast<std::size_t>(encoded.frames) - f);
    FishDacCodes chunk_codes;
    chunk_codes.codebooks = 10;
    chunk_codes.frames = static_cast<std::int64_t>(n);
    chunk_codes.codes.resize(10 * n);
    for (std::size_t b = 0; b < 10; ++b) {
      for (std::size_t k = 0; k < n; ++k) {
        chunk_codes.codes[b * n + k] =
            encoded.codes[b * static_cast<std::size_t>(encoded.frames) + f + k];
      }
    }
    std::vector<float> chunk;
    Require(codec->DecodeStreamChunk(chunk_codes, &chunk, &err),
            "Codec DecodeStreamChunk failed: " + err);
    stream_pcm.insert(stream_pcm.end(), chunk.begin(), chunk.end());
  }
  Require(stream_pcm.size() == test_samples,
          "Streaming decoded sample count mismatch");
  float max_stream_diff = 0.0F;
  for (std::size_t i = 0; i < test_samples; ++i) {
    max_stream_diff =
        std::max(max_stream_diff, std::fabs(stream_pcm[i] - decoded_pcm[i]));
  }
  std::cout << "[PASS] Codec 4-frame Streaming Decode verified ("
            << stream_pcm.size()
            << " samples, max_diff_vs_offline=" << max_stream_diff << ")\n";

  bool all_shards_ready = true;
  for (int s = 1; s <= 8; ++s) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "model-0000%d-of-00008.safetensors", s);
    const auto sp = std::filesystem::path(model_dir) / buf;
    if (!std::filesystem::exists(sp) ||
        std::filesystem::exists(sp.string() + ".aria2")) {
      all_shards_ready = false;
      break;
    }
  }

  if (all_shards_ready) {
    std::cout << "=== [Stage 5] Full End-to-End Fish Drama-3 Synthesis on "
                 "gfx1151 ===\n";
    auto synth = hip::SynthesisHipRuntime::Create(model_dir, 4096, &err);
    Require(synth != nullptr, "SynthesisHipRuntime::Create failed: " + err);

    SynthesisRequest req;
    req.text =
        "Hello! This is Fish Audio S2 Pro Drama 3 running natively on AMD "
        "gfx1151.";
    req.sampling.max_new_frames = 96;
    req.sampling.temperature = 0.7F;
    req.sampling.top_k = 30;
    req.sampling.top_p = 0.8F;
    req.sampling.seed = 42;

    SynthesisResult res;
    Require(synth->Generate(req, {}, &res, &err), "Generate failed: " + err);
    const double gen_audio_s =
        static_cast<double>(res.samples.size()) / res.sample_rate;
    const double total_s = res.total_ms / 1000.0;
    std::cout << "[PASS] Full E2E Synthesis succeeded!\n"
              << "  Generated frames : " << res.codes.frames << "\n"
              << "  Audio duration   : " << gen_audio_s << " s ("
              << res.samples.size() << " samples @ " << res.sample_rate
              << " Hz)\n"
              << "  Prefill time     : "
              << (res.backbone_prefill_ms + res.tower_prefill_ms)
              << " ms (backbone=" << res.backbone_prefill_ms
              << " ms, tower=" << res.tower_prefill_ms << " ms)\n"
              << "  Decode time      : "
              << (res.backbone_decode_ms + res.tower_decode_ms)
              << " ms (backbone=" << res.backbone_decode_ms
              << " ms, tower=" << res.tower_decode_ms << " ms; "
              << (res.codes.frames * 1000.0 /
                  std::max(1.0, res.backbone_decode_ms + res.tower_decode_ms))
              << " frames/s)\n"
              << "  Codec time       : " << res.codec_decode_ms << " ms\n"
              << "  End-to-End RTF   : "
              << (total_s / std::max(1e-6, gen_audio_s)) << "\n";

    const std::string wav_out = "/tmp/drama3_e2e_test.wav";
    FILE* fp = std::fopen(wav_out.c_str(), "wb");
    if (fp != nullptr) {
      const std::uint32_t data_bytes =
          static_cast<std::uint32_t>(res.samples.size() * 2U);
      const std::uint32_t riff_size = 36U + data_bytes;
      const std::uint32_t byte_rate = res.sample_rate * 2U;
      const std::uint16_t fmt_pcm = 1, channels = 1, block_align = 2, bits = 16;
      const std::uint32_t fmt_len = 16;
      std::fwrite("RIFF", 1, 4, fp);
      std::fwrite(&riff_size, 4, 1, fp);
      std::fwrite("WAVEfmt ", 1, 8, fp);
      std::fwrite(&fmt_len, 4, 1, fp);
      std::fwrite(&fmt_pcm, 2, 1, fp);
      std::fwrite(&channels, 2, 1, fp);
      std::fwrite(&res.sample_rate, 4, 1, fp);
      std::fwrite(&byte_rate, 4, 1, fp);
      std::fwrite(&block_align, 2, 1, fp);
      std::fwrite(&bits, 2, 1, fp);
      std::fwrite("data", 1, 4, fp);
      std::fwrite(&data_bytes, 4, 1, fp);
      for (float s : res.samples) {
        const float clamped = std::max(-1.0F, std::min(1.0F, s));
        const auto pcm16 =
            static_cast<std::int16_t>(std::lrintf(clamped * 32767.0F));
        std::fwrite(&pcm16, 2, 1, fp);
      }
      std::fclose(fp);
      std::cout << "  Saved synthesized audio to " << wav_out << " ("
                << (44U + data_bytes) << " bytes)\n";
    }

    // Verify Async Dual-Stream Streaming Synthesis (4 frames / 185.8 ms chunk)
    SynthesisRequest stream_req = req;
    stream_req.sampling.stream_chunk_frames = 4;
    std::vector<float> streamed_samples;
    streamed_samples.reserve(res.samples.size());
    double ttfa_ms = 0.0;
    std::size_t chunk_count = 0;
    const auto stream_start = std::chrono::steady_clock::now();
    stream_req.on_audio = [&](std::span<const float> pcm) -> bool {
      if (chunk_count == 0) {
        ttfa_ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - stream_start)
                      .count();
      }
      ++chunk_count;
      streamed_samples.insert(streamed_samples.end(), pcm.begin(), pcm.end());
      return true;
    };
    SynthesisResult stream_res;
    Require(synth->Generate(stream_req, {}, &stream_res, &err),
            "Streaming Generate failed: " + err);
    Require(streamed_samples.size() == res.samples.size(),
            "Streaming sample count mismatch vs offline");
    float max_e2e_stream_diff = 0.0F;
    for (std::size_t i = 0; i < streamed_samples.size(); ++i) {
      max_e2e_stream_diff = std::max(
          max_e2e_stream_diff, std::fabs(streamed_samples[i] - res.samples[i]));
    }
    std::cout << "[PASS] Async Dual-Stream Streaming Synthesis verified!\n"
              << "  Stream chunks    : " << chunk_count
              << " (4 frames / 185.8 ms audio per chunk)\n"
              << "  First-chunk TTFA : " << ttfa_ms << " ms\n"
              << "  Codec sync wait  : " << stream_res.codec_decode_ms
              << " ms\n"
              << "  Max diff vs E2E  : " << max_e2e_stream_diff << "\n";
  } else {
    std::cout << "[INFO] Shards still downloading in background; skipping "
                 "Stage 5 full synthesis for now.\n";
  }

  return 0;
#endif
}
