// A wide copy verification (Executor::ForwardWide) must give every row the
// logits of consecutive verification passes of at most eight rows, and a
// rollback to any kept prefix must continue exactly as those passes cut back
// at the same token. A small synthetic model keeps the production layer
// geometry and weight formats (Q8_0 projections, Q4_K/Q5_1 experts, BF16
// indexer, F16 router, PLE and the Q8_0 MTP block) with four trunk layers
// and a reduced vocabulary and expert count, so no model file is loaded.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/platform/fs.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"
#include "src/models/qwen38_flash_next/ngram.hpp"
#include "tests/support/temporary_directory.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

using core::GgmlType;

struct DeviceModelTestAccess {
  DeviceModel& model;
  std::uint64_t state;

  std::uint64_t Next() {
    // splitmix64
    std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  float Uniform() {  // [-1, 1)
    return static_cast<float>(Next() >> 40) / 8388608.0F - 1.0F;
  }

  void* Upload(const std::vector<std::uint8_t>& bytes) {
    void* p = nullptr;
    constexpr std::size_t kTail = 4096;
    if (hipMalloc(&p, bytes.size() + kTail) != hipSuccess ||
        hipMemcpy(p, bytes.data(), bytes.size(), hipMemcpyHostToDevice) !=
            hipSuccess ||
        hipMemset(static_cast<std::uint8_t*>(p) + bytes.size(), 0, kTail) !=
            hipSuccess)
      throw std::runtime_error("synthetic weight upload failed");
    model.allocations_.push_back(p);
    return p;
  }
  static void PutHalf(std::uint8_t* dst, float value) {
    const auto h = static_cast<_Float16>(value);
    std::memcpy(dst, &h, 2);
  }

  DeviceTensor Make(GgmlType type, std::uint32_t rows, std::uint32_t cols,
                    std::vector<std::uint8_t> bytes, std::uint32_t experts) {
    DeviceTensor t;
    t.data = Upload(bytes);
    t.type = type;
    t.rows = rows;
    t.cols = cols;
    t.experts = experts;
    if (type == GgmlType::kF16 || type == GgmlType::kBF16)
      model.max_half_cols_ = std::max<std::size_t>(model.max_half_cols_, cols);
    if (type == GgmlType::kQ8_0 && experts == 1)
      model.max_q8_cols_ = std::max<std::size_t>(model.max_q8_cols_, cols);
    return t;
  }
  DeviceTensor F32(std::uint32_t rows, std::uint32_t cols, float center,
                   float spread) {
    std::vector<std::uint8_t> bytes(std::size_t{rows} * cols * 4);
    for (std::size_t i = 0; i < bytes.size() / 4; ++i) {
      const float v = center + spread * Uniform();
      std::memcpy(bytes.data() + 4 * i, &v, 4);
    }
    return Make(GgmlType::kF32, rows, cols, std::move(bytes), 1);
  }
  DeviceTensor Half(GgmlType type, std::uint32_t rows, std::uint32_t cols,
                    float spread) {
    std::vector<std::uint8_t> bytes(std::size_t{rows} * cols * 2);
    for (std::size_t i = 0; i < bytes.size() / 2; ++i) {
      const float v = spread * Uniform();
      if (type == GgmlType::kF16) {
        PutHalf(bytes.data() + 2 * i, v);
      } else {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &v, 4);
        const auto bf = static_cast<std::uint16_t>(bits >> 16);
        std::memcpy(bytes.data() + 2 * i, &bf, 2);
      }
    }
    return Make(type, rows, cols, std::move(bytes), 1);
  }
  /// Uniform int8 quants have a standard deviation of about 73.
  DeviceTensor Q8(std::uint32_t rows, std::uint32_t cols,
                  std::uint32_t experts = 1, float spread = 0.0F) {
    if (spread == 0.0F)
      spread = 1.0F / std::sqrt(static_cast<float>(cols));
    const std::size_t blocks = std::size_t{experts} * rows * (cols / 32);
    std::vector<std::uint8_t> bytes(blocks * 34);
    for (std::size_t b = 0; b < blocks; ++b) {
      auto* block = bytes.data() + b * 34;
      PutHalf(block, spread / 73.0F * (1.0F + 0.25F * Uniform()));
      for (int j = 0; j < 32; j += 8) {
        auto bits = Next();
        for (int k = 0; k < 8; ++k, bits >>= 8)
          block[2 + j + k] = static_cast<std::uint8_t>(
              std::max<std::int8_t>(static_cast<std::int8_t>(bits), -127));
      }
    }
    return Make(GgmlType::kQ8_0, rows, cols, std::move(bytes), experts);
  }
  /// Random 6-bit scales and minimums; dmin = 7.5 d centres the weights.
  DeviceTensor Q4K(std::uint32_t rows, std::uint32_t cols,
                   std::uint32_t experts) {
    const float d = 1.0F / std::sqrt(static_cast<float>(cols)) / 260.0F;
    const std::size_t blocks = std::size_t{experts} * rows * (cols / 256);
    std::vector<std::uint8_t> bytes(blocks * 144);
    for (std::size_t b = 0; b < blocks; ++b) {
      auto* block = bytes.data() + b * 144;
      PutHalf(block, d);
      PutHalf(block + 2, 7.5F * d);
      for (int j = 4; j < 144; j += 8) {
        auto bits = Next();
        for (int k = 0; k < 8 && j + k < 144; ++k, bits >>= 8)
          block[j + k] = static_cast<std::uint8_t>(bits);
      }
    }
    return Make(GgmlType::kQ4_K, rows, cols, std::move(bytes), experts);
  }
  DeviceTensor Q51(std::uint32_t rows, std::uint32_t cols,
                   std::uint32_t experts) {
    const float d = 1.0F / std::sqrt(static_cast<float>(cols)) / 9.2F;
    const std::size_t blocks = std::size_t{experts} * rows * (cols / 32);
    std::vector<std::uint8_t> bytes(blocks * 24);
    for (std::size_t b = 0; b < blocks; ++b) {
      auto* block = bytes.data() + b * 24;
      PutHalf(block, d);
      PutHalf(block + 2, -15.5F * d);
      for (int j = 4; j < 24; j += 8) {
        auto bits = Next();
        for (int k = 0; k < 8 && j + k < 24; ++k, bits >>= 8)
          block[j + k] = static_cast<std::uint8_t>(bits);
      }
    }
    return Make(GgmlType::kQ5_1, rows, cols, std::move(bytes), experts);
  }

  DeviceMixer Mixer(const Config& c, bool inject, bool q8_inject) {
    DeviceMixer m;
    m.norm = F32(1, c.HcDim(), 1.0F, 0.1F);
    m.down = Q8(c.hc_low_rank, c.HcDim());
    m.up = Q8(c.HcDim(), c.hc_low_rank);
    if (inject)
      m.inject = q8_inject ? Q8(c.hc_count, c.HcDim())
                           : F32(c.hc_count, c.HcDim(), 0.0F, 0.01F);
    return m;
  }
  void Ffn(const Config& c, DeviceLayer& l, bool q8_experts) {
    const auto e = c.num_experts;
    l.router = Half(GgmlType::kF16, e + 1, c.hidden_size, 0.05F);
    if (q8_experts) {
      l.ffn_gate_exps = Q8(c.expert_ff, c.hidden_size, e);
      l.ffn_up_exps = Q8(c.expert_ff, c.hidden_size, e);
      l.ffn_down_exps = Q8(c.hidden_size, c.expert_ff, e);
    } else {
      l.ffn_gate_exps = Q4K(c.expert_ff, c.hidden_size, e);
      l.ffn_up_exps = Q4K(c.expert_ff, c.hidden_size, e);
      l.ffn_down_exps = Q51(c.hidden_size, c.expert_ff, e);
    }
    l.shexp_gate = Q8(c.shared_expert_ff, c.hidden_size);
    l.shexp_up = Q8(c.shared_expert_ff, c.hidden_size);
    l.shexp_down = Q8(c.hidden_size, c.shared_expert_ff);
  }
  void Attention(const Config& c, DeviceLayer& l) {
    l.attn_qkv =
        Q8(2 * c.AttentionQDim() + 2 * c.AttentionKvDim(), c.hidden_size);
    l.attn_out = Q8(c.hidden_size, c.AttentionQDim());
    l.attn_q_norm = F32(1, c.head_dim, 1.0F, 0.1F);
    l.attn_k_norm = F32(1, c.head_dim, 1.0F, 0.1F);
    l.indexer_q = Half(GgmlType::kBF16, c.indexer_heads * c.indexer_head_dim,
                       c.hidden_size, 0.05F);
    l.indexer_k =
        Half(GgmlType::kBF16, c.indexer_head_dim, c.hidden_size, 0.05F);
    l.indexer_q_norm = F32(1, c.indexer_head_dim, 1.0F, 0.1F);
    l.indexer_k_norm = F32(1, c.indexer_head_dim, 1.0F, 0.1F);
  }

  static std::unique_ptr<DeviceModel> Build(const Config& c) {
    std::unique_ptr<DeviceModel> m(new DeviceModel());
    DeviceModelTestAccess b{*m, 0x5EED};
    m->config_ = c;
    m->token_embd_ = b.Q8(c.vocab_size, c.hidden_size, 1, 0.05F);
    m->output_ = b.Q8(c.vocab_size, c.hidden_size);
    m->output_bytes_ = std::size_t{c.vocab_size} * (c.hidden_size / 32 * 34);
    m->hc_head_ = b.Mixer(c, false, false);
    for (std::uint32_t il = 0; il < c.num_layers; ++il) {
      DeviceLayer l;
      l.linear = c.IsLinearLayer(il);
      l.hc_attn = b.Mixer(c, true, false);
      l.hc_ffn = b.Mixer(c, true, false);
      if (l.linear) {
        l.ssm_in = b.Q8(c.SsmConvChannels() + c.SsmValueDim(), c.hidden_size);
        l.ssm_alpha_beta =
            b.Half(GgmlType::kF16, 2 * c.ssm_num_v_heads, c.hidden_size, 0.05F);
        l.ssm_conv1d =
            b.F32(c.SsmConvChannels(), c.ssm_conv_kernel, 0.0F, 0.5F);
        l.ssm_dt = b.F32(1, c.ssm_num_v_heads, 0.5F, 0.5F);
        l.ssm_a = b.F32(1, c.ssm_num_v_heads, -1.25F, 0.75F);
        l.ssm_norm = b.F32(1, c.ssm_head_dim, 1.0F, 0.1F);
        l.ssm_out = b.Q8(c.hidden_size, c.SsmValueDim());
      } else {
        b.Attention(c, l);
      }
      if (c.IsPleLayer(il)) {
        l.ple_key = b.Q8(c.HcDim(), c.PleEmbeddingDim());
        l.ple_value = b.Q8(c.hidden_size, c.PleEmbeddingDim());
        l.ple_norm_key = b.F32(1, c.HcDim(), 1.0F, 0.1F);
        l.ple_norm_query = b.F32(1, c.HcDim(), 1.0F, 0.1F);
        l.ple_norm_conv = b.F32(1, c.HcDim(), 1.0F, 0.1F);
        l.ple_conv1d = b.F32(c.HcDim(), c.ple_conv_kernel, 0.0F, 0.5F);
      }
      b.Ffn(c, l, il == 2);  // one Q8_0 expert layer, as the artifact has
      m->layers_.push_back(l);
    }
    DeviceLayer& mtp = m->mtp_;
    mtp.hc_attn = b.Mixer(c, true, true);
    mtp.hc_ffn = b.Mixer(c, true, true);
    b.Attention(c, mtp);
    b.Ffn(c, mtp, true);
    mtp.nextn_enorm = b.F32(1, c.hidden_size, 1.0F, 0.1F);
    mtp.nextn_hnorm = b.F32(1, c.HcDim(), 1.0F, 0.1F);
    mtp.nextn_fc_embedding = b.Q8(c.hidden_size, c.hidden_size);
    mtp.nextn_fc_hidden = b.Q8(c.hidden_size, c.hidden_size);
    mtp.nextn_head = b.Mixer(c, false, false);
    m->has_mtp_ = true;
    if (hipDeviceSynchronize() != hipSuccess)
      throw std::runtime_error("synthetic model upload failed");
    return m;
  }
};

// Test-only access: production has no switch for disabling graphs or overlap.
struct ExecutorTestAccess {
  static void EagerNext(Session& session) { session.warmed_.clear(); }
  static std::size_t Graphs(const Session& session) {
    return session.graphs_.size();
  }
  static bool Capturing(const Executor& exec) { return exec.capturing_; }
  /// Forces a decode layout; returns the one Create calibrated.
  static bool SetDecodeOverlap(const Executor& exec, bool overlap) {
    return std::exchange(const_cast<Executor&>(exec).decode_overlap_, overlap);
  }
  static void RequireIdle(const Executor& exec) {
    for (const auto stream : {exec.stream_, exec.side_stream_}) {
      hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
      if (hipStreamIsCapturing(stream, &status) != hipSuccess ||
          status != hipStreamCaptureStatusNone)
        throw std::runtime_error("failed pass left a HIP capture active");
    }
    if (exec.capturing_ || exec.side_pending_)
      throw std::runtime_error("failed pass left capture or side work pending");
  }
  static void PoisonShared(const Executor& exec, std::uint32_t rows) {
    if (hipMemsetAsync(exec.s_.shexp_out, 0xff,
                       std::size_t{rows} * exec.config().hidden_size * 4,
                       exec.stream_) != hipSuccess)
      throw std::runtime_error("shared-expert poison failed");
  }
  static void CheckMoeCapture(const Executor& exec, std::uint32_t rows,
                              bool throw_in_capture = false) {
    std::string error;
    auto session =
        exec.CreateSession(core::SessionMode::kAutoregressive, 128, &error);
    if (!session)
      throw std::runtime_error(error);
    const std::size_t count = std::size_t{rows} * exec.config().hidden_size;
    std::vector<float> input(count);
    for (std::size_t i = 0; i < count; ++i)
      input[i] = 0.1F * std::sin(static_cast<float>(i) * 0.13F);
    if (hipMemcpy(exec.s_.mixed, input.data(), count * 4,
                  hipMemcpyHostToDevice) != hipSuccess)
      throw std::runtime_error("MoE input upload failed");
    bool checked = false;
    const auto body = [&] {
      exec.DropReadyQ8();
      exec.q8t_src_ = nullptr;
      exec.half_src_ = nullptr;
      if (!exec.Moe(exec.model_->layers().front(), exec.s_.mixed,
                    exec.s_.block_out, rows, &error))
        throw std::runtime_error(error);
      if (exec.capturing_) {
        hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
        if (hipStreamIsCapturing(exec.side_stream_, &status) != hipSuccess ||
            status != (rows <= 8 && exec.decode_overlap_
                           ? hipStreamCaptureStatusActive
                           : hipStreamCaptureStatusNone))
          throw std::runtime_error("unexpected MoE overlap shape");
        checked = true;
        if (throw_in_capture)
          throw std::runtime_error("injected capture failure");
      }
      return true;
    };
    const auto read = [&](std::vector<float>& out) {
      if (hipStreamSynchronize(exec.stream_) != hipSuccess ||
          hipMemcpy(out.data(), exec.s_.block_out, count * 4,
                    hipMemcpyDeviceToHost) != hipSuccess)
        throw std::runtime_error("MoE output download failed");
    };
    // The first pass warms the shape eagerly; the second captures and
    // launches; the third replays. Comparing the eager and replayed block
    // output catches a graph that dropped the shared expert, which poisoning
    // its scratch makes visible.
    constexpr std::uint64_t key = std::uint64_t{1} << 63;
    std::vector<float> expected(count), actual(count);
    if (!exec.Run(*session, key, true, body, &error, false))
      throw std::runtime_error(error);
    read(expected);
    const auto main = exec.stream_;
    bool threw = false;
    try {
      if (!exec.Run(*session, key, true, body, &error, false))
        throw std::runtime_error(error);
    } catch (const std::runtime_error& e) {
      if (!throw_in_capture ||
          std::string_view(e.what()) != "injected capture failure")
        throw;
      threw = true;
    }
    if (!checked || threw != throw_in_capture || exec.stream_ != main)
      throw std::runtime_error("capture failure was not exercised or restored");
    RequireIdle(exec);
    if (throw_in_capture) {
      throw_in_capture = false;
      if (Graphs(*session) != 0 ||
          !exec.Run(*session, key, true, body, &error, false))
        throw std::runtime_error("capture did not recover after exception");
    }
    if (Graphs(*session) != 1)
      throw std::runtime_error("MoE capture did not install a usable graph");
    PoisonShared(exec, rows);
    if (!exec.Run(*session, key, true, body, &error, false))
      throw std::runtime_error(error);
    read(actual);
    if (std::memcmp(expected.data(), actual.data(), count * 4) != 0)
      throw std::runtime_error("MoE graph replay changed the block output");
    RequireIdle(exec);
  }
};

}  // namespace gufo::models::qwen38_flash_next::rocm

namespace {

namespace qfn = gufo::models::qwen38_flash_next;
namespace rocm = qfn::rocm;
using rocm::Executor;
using rocm::Session;

qfn::Config SyntheticConfig() {
  qfn::Config c;
  c.num_layers = 4;  // three DeltaNet layers, then attention
  c.num_layers_all = 5;
  c.nextn_layers = 1;
  c.hidden_size = 2560;
  c.vocab_size = 8192;
  c.context_length = 4096;
  c.rms_eps = 1e-6F;
  c.hc_count = 4;
  c.hc_low_rank = 320;
  c.full_attention_interval = 4;
  c.num_heads = 24;
  c.num_kv_heads = 2;
  c.head_dim = 256;
  c.rotary_dim = 64;
  c.rope_theta = 1e7F;
  c.rope_sections = {11, 11, 10, 0};
  c.indexer_heads = 4;
  c.indexer_head_dim = 128;
  c.indexer_top_k = 2048;
  c.compress_ratio = 4;
  c.ssm_conv_kernel = 4;
  c.ssm_head_dim = 128;
  c.ssm_num_k_heads = 16;
  c.ssm_num_v_heads = 48;
  c.ssm_inner_size = 6144;
  c.num_experts = 16;
  c.num_experts_used = 10;
  c.expert_ff = 640;
  c.shared_expert_ff = 640;
  c.ple_layer = 1;  // after layer 0: graphs split into prefix and suffix
  c.ple_ngram_size = 3;
  c.ple_heads_per_ngram = 8;
  c.ple_heads = 16;
  c.ple_head_dim = 160;
  c.ple_conv_kernel = 4;
  c.ple_eos_token = 8191;
  c.ple_multipliers = {1000003ULL, 998244353ULL, 1000000007ULL};
  for (std::uint32_t h = 0; h < c.ple_heads; ++h) {
    c.ple_head_offsets[h] = h * 1024;
    c.ple_head_vocab[h] = 1021;
  }
  c.ple_rows = 16 * 1024;
  return c;
}

[[noreturn]] void Throw(const std::string& what, const std::string& error) {
  throw std::runtime_error(what + ": " + error);
}

std::int32_t Argmax(std::span<const float> row) {
  return static_cast<std::int32_t>(std::max_element(row.begin(), row.end()) -
                                   row.begin());
}

void RequireExact(std::span<const float> expected,
                  std::span<const float> actual, const std::string& what) {
  const auto finite = [](float v) { return std::isfinite(v); };
  if (expected.size() != actual.size() ||
      !std::all_of(expected.begin(), expected.end(), finite) ||
      !std::all_of(actual.begin(), actual.end(), finite) ||
      std::memcmp(expected.data(), actual.data(), expected.size_bytes()) != 0)
    throw std::runtime_error(what);
}

// Each width warms, captures and replays twice with different token histories.
// Poisoning the shared output prevents an earlier eager/reference pass from
// concealing a graph that omits the shared expert. The oracle is forced eager
// without adding a production dispatch switch.
void DecodeGraphChecks(const Executor& exec) {
  using Access = rocm::ExecutorTestAccess;
  const auto& c = exec.config();
  std::string error;
  // Both decode layouts must replay exactly, whichever one Create picked.
  const bool calibrated = Access::SetDecodeOverlap(exec, false);
  std::cout << "calibrated decode overlap=" << calibrated << '\n';
  for (const bool overlap : {false, true}) {
    Access::SetDecodeOverlap(exec, overlap);
    for (const auto prompt_size : {4U, c.indexer_top_k + 8}) {
      for (std::uint32_t width = 1; width <= 8; ++width) {
        auto eager = exec.CreateSession(gufo::core::SessionMode::kSpeculative,
                                        128, &error);
        auto replay = exec.CreateSession(gufo::core::SessionMode::kSpeculative,
                                         128, &error);
        if (!eager || !replay)
          Throw("graph sessions", error);
        std::vector<std::int32_t> prompt(prompt_size);
        for (std::uint32_t i = 0; i < prompt_size; ++i)
          prompt[i] = static_cast<std::int32_t>((i * 17 + 3) % c.vocab_size);
        for (auto* session : {eager.get(), replay.get()}) {
          if (!exec.Forward(*session, prompt, 0, nullptr,
                            Executor::ForwardMode::kPrefill, &error))
            Throw("graph prefill", error);
        }
        std::size_t graphs = 0;
        for (std::uint32_t pass = 0; pass < 4; ++pass) {
          std::vector<std::int32_t> tokens(width);
          for (std::uint32_t i = 0; i < width; ++i)
            tokens[i] = static_cast<std::int32_t>(
                (pass * 53 + i * 23 + prompt_size) % c.vocab_size);
          std::vector<float> expected(std::size_t{width} * c.vocab_size);
          std::vector<float> actual(expected.size());
          const auto mode = width == 1 ? Executor::ForwardMode::kDecode
                                       : Executor::ForwardMode::kVerify;
          Access::EagerNext(*eager);
          if (!exec.Forward(*eager, tokens, width, expected.data(), mode,
                            &error))
            Throw("eager graph oracle", error);
          Access::PoisonShared(exec, width);
          if (!exec.Forward(*replay, tokens, width, actual.data(), mode,
                            &error))
            Throw("graph candidate", error);
          RequireExact(
              expected, actual,
              "eager/replay logits differ: width=" + std::to_string(width) +
                  " pass=" + std::to_string(pass));
          const auto count = Access::Graphs(*replay);
          if ((pass == 0 && count != 0) || (pass == 1 && count == 0) ||
              (pass > 1 && count != graphs))
            throw std::runtime_error("fixture did not warm/capture/replay");
          graphs = count;
        }
        std::cout << "decode graph width=" << width
                  << " sparse=" << (prompt_size > c.indexer_top_k)
                  << " overlap=" << overlap << " eager/replay logits exact\n";
      }
    }
    for (const auto rows : {1U, 8U, 9U, 32U})
      Access::CheckMoeCapture(exec, rows);
    Access::CheckMoeCapture(exec, 1, true);
  }

  // Cancel the suffix capture after at least one layer has recorded MoE side
  // work (the overlapped layout is still forced). Prefix capture and
  // eager-before-capture execution stay enabled.
  auto failed =
      exec.CreateSession(gufo::core::SessionMode::kAutoregressive, 128, &error);
  auto peer =
      exec.CreateSession(gufo::core::SessionMode::kAutoregressive, 128, &error);
  if (!failed || !peer)
    Throw("capture cancellation sessions", error);
  const std::array<std::int32_t, 4> prompt{3, 7, 11, 19};
  std::vector<float> expected(c.vocab_size), actual(c.vocab_size);
  const std::int32_t token = 23;
  if (!exec.Forward(*failed, prompt, 0, nullptr,
                    Executor::ForwardMode::kPrefill, &error) ||
      !exec.Forward(*failed, std::span(&token, 1), 1, actual.data(),
                    Executor::ForwardMode::kDecode, &error))
    Throw("capture cancellation warmup", error);
  unsigned captured_checks = 0;
  failed->SetCancellationCheck(
      [&] { return Access::Capturing(exec) && ++captured_checks == 3; });
  if (exec.Forward(*failed, std::span(&token, 1), 1, actual.data(),
                   Executor::ForwardMode::kDecode, &error) ||
      captured_checks != 3)
    throw std::runtime_error("fixture did not cancel during capture");
  Access::RequireIdle(exec);
  failed->SetCancellationCheck({});
  failed->Reset();
  for (auto* session : {peer.get(), failed.get()}) {
    if (!exec.Forward(*session, prompt, 0, nullptr,
                      Executor::ForwardMode::kPrefill, &error))
      Throw("post-cancellation prefill", error);
  }
  if (!exec.Forward(*peer, std::span(&token, 1), 1, expected.data(),
                    Executor::ForwardMode::kDecode, &error) ||
      !exec.Forward(*failed, std::span(&token, 1), 1, actual.data(),
                    Executor::ForwardMode::kDecode, &error))
    Throw("post-cancellation decode", error);
  RequireExact(expected, actual, "cancelled capture changed reused executor");
  Access::RequireIdle(exec);
  Access::SetDecodeOverlap(exec, calibrated);
  std::cout << "capture exception/cancellation recovery exact\n";
}

/// A wide copy verification (ForwardWide) must give every row the logits of
/// consecutive verification passes of at most eight rows, and a rollback to
/// any kept prefix must continue exactly as those passes cut back at the
/// same token. Chains cross segment boundaries, end mid-segment and, from
/// the longer prompt, cross the sparse attention budget.
int WideChecks(const Executor& exec) {
  const auto vocab = exec.config().vocab_size;
  std::string error;
  int failures = 0;
  const auto prefill = [&](std::uint32_t length) {
    auto session =
        exec.CreateSession(gufo::core::SessionMode::kSpeculative, 2304, &error);
    if (!session)
      Throw("wide session", error);
    std::vector<std::int32_t> prompt(length);
    for (std::uint32_t t = 0; t < length; ++t)
      prompt[t] =
          static_cast<std::int32_t>((t * 2246822519U + 29U) % (vocab - 2));
    std::vector<float> logits(vocab);
    if (!exec.Forward(*session, prompt, 1, logits.data(),
                      Executor::ForwardMode::kPrefill, &error))
      Throw("wide prefill", error);
    return session;
  };
  // Three decode steps after the kept prefix, greedy from its last row.
  const auto continue_from = [&](Session& session, std::int32_t token) {
    std::vector<float> out;
    std::vector<float> logits(vocab);
    for (int step = 0; step < 3; ++step) {
      if (!exec.Forward(session, std::span<const std::int32_t>(&token, 1), 1,
                        logits.data(), Executor::ForwardMode::kDecode, &error))
        Throw("wide continuation", error);
      out.insert(out.end(), logits.begin(), logits.end());
      token = Argmax(logits);
    }
    return out;
  };
  for (const std::uint32_t prompt : {300U, 2030U}) {
    for (const std::uint32_t n : {64U, 20U, 9U}) {
      std::vector<std::int32_t> chain(n);
      for (std::uint32_t t = 0; t < n; ++t)
        chain[t] =
            static_cast<std::int32_t>((t * 40503U + prompt) % (vocab - 2));
      for (const std::uint32_t keep :
           {n, 1U, 7U, 8U, 9U, 16U, 17U, 40U, 56U, 57U, n - 1}) {
        if (keep == 0 || keep > n)
          continue;
        auto wide = prefill(prompt);
        std::vector<float> wide_rows(std::size_t{n} * vocab);
        if (!exec.ForwardWide(*wide, chain, &error) ||
            !exec.SelectBatchLogits(0, n, wide_rows.data(), &error))
          Throw("wide verification", error);
        if (keep < n && !exec.Rollback(*wide, keep, &error))
          Throw("wide rollback", error);
        // Consecutive passes of at most eight rows, the one holding the
        // last kept token cut back to it.
        auto reference = prefill(prompt);
        std::vector<float> rows;
        for (std::uint32_t first = 0; first < keep; first += 8) {
          const auto count = std::min(8U, n - first);
          std::vector<float> logits(std::size_t{count} * vocab);
          if (!exec.Forward(
                  *reference,
                  std::span<const std::int32_t>(chain).subspan(first, count),
                  count, logits.data(), Executor::ForwardMode::kVerify,
                  &error) ||
              !exec.Rollback(*reference, std::min(count, keep - first), &error))
            Throw("reference verification", error);
          rows.insert(rows.end(), logits.begin(), logits.end());
        }
        const bool same_rows =
            std::memcmp(
                rows.data(), wide_rows.data(),
                std::min(rows.size(), wide_rows.size()) * sizeof(float)) == 0;
        const auto next = Argmax(std::span<const float>(wide_rows).subspan(
            std::size_t{keep - 1} * vocab, vocab));
        const bool same_state =
            wide->position() == reference->position() &&
            continue_from(*wide, next) == continue_from(*reference, next);
        std::cout << "wide prompt=" << prompt << " rows=" << n
                  << " keep=" << keep << ": logits "
                  << (same_rows ? "exact" : "DIFFER") << ", continuation "
                  << (same_state ? "exact" : "DIFFERS") << '\n';
        failures += !same_rows + !same_state;
      }
    }
  }
  return failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  const bool graph_only =
      argc == 2 && std::string_view(argv[1]) == "--decode-graph-only";
  if (argc != 1 && !graph_only)
    return 2;
  try {
    auto config = SyntheticConfig();
    if (graph_only) {
      config.vocab_size = 256;
      config.ple_eos_token = 255;
      config.indexer_top_k = 64;
    }
    auto model = rocm::DeviceModelTestAccess::Build(config);
    std::string error;
    if (!graph_only && !model->SetDraftVocab(2048, {8000, 8001}, &error))
      Throw("draft vocabulary", error);
    // The n-gram rows come from a BF16 table file, as from the artifact.
    TemporaryDirectory temp("qfn-wide-verification");
    const auto path = temp.path() / "ngram.bin";
    {
      rocm::DeviceModelTestAccess rng{*model, 0xB16};
      std::vector<std::uint16_t> rows(config.ple_rows * config.ple_head_dim);
      for (auto& v : rows) {
        const float f = 0.1F * rng.Uniform();
        std::uint32_t bits = 0;
        std::memcpy(&bits, &f, 4);
        v = static_cast<std::uint16_t>(bits >> 16);
      }
      std::ofstream out(path, std::ios::binary);
      out.write(reinterpret_cast<const char*>(rows.data()),
                static_cast<std::streamsize>(rows.size() * 2));
      if (!out)
        throw std::runtime_error("n-gram table write");
    }
    const auto file = gufo::platform::OpenReadFile(path);
    auto ngram =
        qfn::NgramTable::Open(file, 0, config.ple_rows, config.ple_head_dim,
                              gufo::core::GgmlType::kBF16, &error);
    gufo::platform::CloseFile(file);
    if (!ngram)
      Throw("n-gram table", error);
    // A whole 2,030-token prompt in one pass needs the wide scratch. The
    // focused graph regression needs only short prompts around a small budget.
    auto exec = Executor::Create(*model, ngram.get(),
                                 {.max_batch = graph_only ? 128U : 2048U,
                                  .max_logit_rows = 8,
                                  .max_speculative = 8},
                                 &error);
    if (!exec)
      Throw("executor", error);
    if (graph_only) {
      DecodeGraphChecks(*exec);
      return 0;
    }
    if (WideChecks(*exec) != 0)
      return 1;
    std::cout << "wide verification matches eight-row passes\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
