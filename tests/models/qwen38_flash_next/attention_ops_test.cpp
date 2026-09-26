#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "src/models/qwen38_flash_next/cpu_ops.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/rope_scaling.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace qfn = gufo::models::qwen38_flash_next;
namespace {

// The model's full-attention geometry: 24 query heads over 2 KV heads,
// head_dim 256, 4-token selection blocks.
constexpr std::uint32_t kHeads = 24;
constexpr std::uint32_t kKvHeads = 2;
constexpr std::uint32_t kDim = 256;
constexpr std::uint32_t kRatio = 4;
constexpr std::uint32_t kQWidth = kHeads * kDim;
constexpr std::uint32_t kKvWidth = kKvHeads * kDim;
// The per-token kernel accumulates in FP32 over FP16 keys/values; its gated
// output (order one) stays within this of an FP64 evaluation.
constexpr double kFp64Limit = 1e-6;

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

template<typename T>
class HipBuffer {
public:
  explicit HipBuffer(std::size_t count) : count_(count) {
    void* allocation = nullptr;
    CheckHip(hipMalloc(&allocation, bytes()), "hipMalloc");
    CheckHip(hipMemset(allocation, 0, bytes()), "hipMemset");
    data_ = static_cast<T*>(allocation);
  }
  ~HipBuffer() {
    if (data_ != nullptr) {
      (void)hipFree(data_);
    }
  }

  HipBuffer(const HipBuffer&) = delete;
  HipBuffer& operator=(const HipBuffer&) = delete;
  HipBuffer(HipBuffer&&) = delete;
  HipBuffer& operator=(HipBuffer&&) = delete;

  [[nodiscard]] T* get() noexcept { return data_; }
  [[nodiscard]] std::size_t bytes() const noexcept {
    return count_ * sizeof(T);
  }

private:
  T* data_{nullptr};
  std::size_t count_{0};
};

std::uint32_t NextRandom(std::uint32_t* state) noexcept {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

std::vector<float> MakeValues(std::size_t count, std::uint32_t seed,
                              float scale) {
  std::vector<float> values(count);
  for (float& value : values) {
    value = scale *
            static_cast<float>(static_cast<int>(NextRandom(&seed) & 0xFFFFU) -
                               32768) /
            32768.0F;
  }
  return values;
}

template<typename T>
void Upload(HipBuffer<T>* destination, const std::vector<T>& source) {
  CheckHip(hipMemcpy(destination->get(), source.data(), destination->bytes(),
                     hipMemcpyHostToDevice),
           "upload");
}

std::vector<float> Download(HipBuffer<float>* source, std::size_t count) {
  std::vector<float> values(count);
  CheckHip(hipMemcpy(values.data(), source->get(), source->bytes(),
                     hipMemcpyDeviceToHost),
           "download");
  return values;
}

void CheckPreparation(std::uint32_t n, std::uint32_t start,
                      std::uint32_t rotary, const qfn::YarnRope& yarn = {}) {
  constexpr std::uint32_t stride = 2 * (kQWidth + kKvWidth);
  const std::size_t count = static_cast<std::size_t>(n) * kQWidth;
  const std::size_t cache_rows = start + n + 2;
  HipBuffer<float> packed(static_cast<std::size_t>(n) * stride);
  HipBuffer<float> q_gamma(kDim), k_gamma(kDim);
  HipBuffer<float> q_ref(count), gate_ref(count), q_out(count), gate_out(count);
  HipBuffer<float> k(static_cast<std::size_t>(n) * kKvWidth);
  HipBuffer<float> v(static_cast<std::size_t>(n) * kKvWidth);
  HipBuffer<__half> k_ref(cache_rows * kKvWidth), v_ref(cache_rows * kKvWidth);
  HipBuffer<__half> k_out(cache_rows * kKvWidth), v_out(cache_rows * kKvWidth);
  HipBuffer<std::uint32_t> pos(1);
  Upload(&packed, MakeValues(static_cast<std::size_t>(n) * stride, 17, 4.0F));
  auto qg = MakeValues(kDim, 37, 0.5F);
  auto kg = MakeValues(kDim, 91, 0.5F);
  for (auto& x : qg)
    x += 1.0F;
  for (auto& x : kg)
    x += 1.0F;
  Upload(&q_gamma, qg);
  Upload(&k_gamma, kg);

  struct Capture {
    hipStream_t stream{nullptr};
    hipGraph_t graph{nullptr};
    hipGraphExec_t exec{nullptr};
    Capture() {
      CheckHip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
               "preparation stream");
    }
    ~Capture() {
      if (exec)
        (void)hipGraphExecDestroy(exec);
      if (graph)
        (void)hipGraphDestroy(graph);
      (void)hipStreamDestroy(stream);
    }
  } capture;
  const auto same = [&](auto* expected, auto* actual, std::size_t size,
                        const char* name, std::uint32_t replay) {
    using T = std::remove_pointer_t<decltype(expected)>;
    std::vector<T> a(size), b(size);
    CheckHip(
        hipMemcpy(a.data(), expected, size * sizeof(T), hipMemcpyDeviceToHost),
        "preparation reference");
    CheckHip(
        hipMemcpy(b.data(), actual, size * sizeof(T), hipMemcpyDeviceToHost),
        "preparation result");
    for (std::size_t i = 0; i < size; ++i) {
      if (std::memcmp(&a[i], &b[i], sizeof(T)) == 0)
        continue;
      std::cerr << "preparation n=" << n << " start=" << start
                << " rotary=" << rotary << " replay=" << replay
                << " index=" << i << " expected=" << std::hexfloat
                << static_cast<float>(a[i])
                << " actual=" << static_cast<float>(b[i]) << std::defaultfloat
                << '\n';
      throw std::runtime_error(std::string("attention preparation changed ") +
                               name);
    }
  };
  for (std::uint32_t replay = 0; replay < 2; ++replay) {
    const std::vector<std::uint32_t> position{start + replay};
    Upload(&pos, position);
    q::UnpackQGate(packed.get(), stride, q_ref.get(), gate_ref.get(), k.get(),
                   v.get(), n, kHeads, kDim, kKvWidth, nullptr);
    q::RmsNormRows(q_ref.get(), q_gamma.get(), q_ref.get(), n * kHeads, kDim, 1,
                   1e-6F, nullptr);
    q::RmsNormRows(k.get(), k_gamma.get(), k.get(), n * kKvHeads, kDim, 1,
                   1e-6F, nullptr);
    q::Rope(q_ref.get(), n, kHeads, kDim, rotary, pos.get(), 1e7F, nullptr,
            nullptr, yarn);
    q::Rope(k.get(), n, kKvHeads, kDim, rotary, pos.get(), 1e7F, nullptr,
            nullptr, yarn);
    q::StoreKv(k.get(), k_ref.get(), n, kKvWidth, pos.get(), nullptr);
    q::StoreKv(v.get(), v_ref.get(), n, kKvWidth, pos.get(), nullptr);
    CheckHip(hipDeviceSynchronize(), "preparation reference ready");
    if (replay == 0) {
      CheckHip(
          hipStreamBeginCapture(capture.stream, hipStreamCaptureModeGlobal),
          "preparation capture");
      const bool supported = q::PrepareAttention(
          packed.get(), stride, q_gamma.get(), k_gamma.get(), q_out.get(),
          gate_out.get(), k_out.get(), v_out.get(), n, kHeads, kKvHeads, kDim,
          rotary, pos.get(), 1e7F, 1e-6F, capture.stream, nullptr, false, yarn);
      CheckHip(hipStreamEndCapture(capture.stream, &capture.graph),
               "preparation capture end");
      if (!supported)
        throw std::runtime_error("attention preparation rejected geometry");
      CheckHip(hipGraphInstantiate(&capture.exec, capture.graph, nullptr,
                                   nullptr, 0),
               "preparation instantiate");
    }
    CheckHip(hipGraphLaunch(capture.exec, capture.stream),
             "preparation replay");
    CheckHip(hipStreamSynchronize(capture.stream), "preparation ready");
    same(q_ref.get(), q_out.get(), count, "queries", replay);
    same(gate_ref.get(), gate_out.get(), count, "gates", replay);
    // Include the neighboring cache rows and replay at a new device position.
    const std::size_t first = start == 0 ? 0 : start - 1;
    const std::size_t window = (cache_rows - first) * kKvWidth;
    same(k_ref.get() + first * kKvWidth, k_out.get() + first * kKvWidth, window,
         "key cache", replay);
    same(v_ref.get() + first * kKvWidth, v_out.get() + first * kKvWidth, window,
         "value cache", replay);
  }
  if (n >= 32) {
    // Each small prefill piece must match the full preparation, including
    // normalization rounding and absolute rotary positions.
    for (const std::uint32_t chunk : {1U, 8U, 9U, 16U, 32U}) {
      for (std::uint32_t off = 0; off < n; off += chunk) {
        Upload(&pos, std::vector<std::uint32_t>{start + 1 + off});
        if (!q::PrepareAttention(
                packed.get() + std::size_t(off) * stride, stride, q_gamma.get(),
                k_gamma.get(), q_out.get() + std::size_t(off) * kQWidth,
                gate_out.get() + std::size_t(off) * kQWidth, k_out.get(),
                v_out.get(), std::min(chunk, n - off), kHeads, kKvHeads, kDim,
                rotary, pos.get(), 1e7F, 1e-6F, nullptr, nullptr, true, yarn))
          throw std::runtime_error("prefill preparation rejected a short tail");
      }
      CheckHip(hipDeviceSynchronize(), "short prefill preparation");
      same(q_ref.get(), q_out.get(), count, "prefill queries", chunk);
      same(gate_ref.get(), gate_out.get(), count, "prefill gates", chunk);
      same(k_ref.get(), k_out.get(), cache_rows * kKvWidth, "prefill keys",
           chunk);
      same(v_ref.get(), v_out.get(), cache_rows * kKvWidth, "prefill values",
           chunk);
    }
  }
  std::cout << "attention preparation n=" << n << " start=" << start
            << " rotary=" << rotary << " yarn=" << yarn.enabled
            << ": exact, including graph positions\n";
}

qfn::YarnRope Yarn(float factor) {
  return qfn::MakeYarnRope({.factor = factor, .original_context = 262144}, 64,
                           1e7F);
}

// HF frequency of pair i in double (same ramp, 1/factor blend).
double HfFrequency(float factor, std::uint32_t i) {
  const double extrapolation = std::pow(1e7, -2.0 * i / 64.0);
  if (factor == 1.0F)
    return extrapolation;
  const double ramp = std::clamp((double(i) - 14.0) / 8.0, 0.0, 1.0);
  return extrapolation * (1.0 - ramp + ramp / factor);
}

// Unit input per pair: the Rope kernel's output is mscale * (cos, sin) of
// the angle, so angle and scale are checked directly against HF.
void CheckPhase(const std::vector<float>& out, const std::vector<double>& pos,
                float factor, const char* label) {
  const double mscale = factor == 1.0F ? 1.0 : 0.1 * std::log(factor) + 1.0;
  for (std::uint32_t i = 0; i < 32; ++i) {
    const double angle = pos[i] * HfFrequency(factor, i);
    const double got = std::atan2(out[i + 32], out[i]);
    const double diff = std::remainder(got - angle, 2.0 * std::numbers::pi);
    const double magnitude = std::hypot(out[i], out[i + 32]);
    if (std::abs(diff) > 1e-6 * angle + 1e-4 ||
        std::abs(magnitude - mscale) > 1e-5 * mscale) {
      throw std::runtime_error(
          std::string(label) + ": pair " + std::to_string(i) + " angle error " +
          std::to_string(diff) + " magnitude " + std::to_string(magnitude));
    }
  }
  for (std::uint32_t i = 64; i < kDim; ++i)
    if (out[i] != 0.25F)
      throw std::runtime_error(std::string(label) + ": tail changed");
}

std::vector<float> UnitPairs() {
  std::vector<float> host(kDim, 0.0F);
  std::fill_n(host.begin(), 32, 1.0F);
  std::fill(host.begin() + 64, host.end(), 0.25F);
  return host;
}

void CheckYarnPhase(float factor, std::uint32_t position) {
  HipBuffer<float> x(kDim);
  HipBuffer<std::uint32_t> pos(1);
  Upload(&x, UnitPairs());
  Upload(&pos, std::vector<std::uint32_t>{position});
  q::Rope(x.get(), 1, 1, kDim, 64, pos.get(), 1e7F, nullptr, nullptr,
          factor == 1.0F ? qfn::YarnRope{} : Yarn(factor));
  CheckHip(hipDeviceSynchronize(), "phase rope");
  CheckPhase(Download(&x, kDim), std::vector<double>(32, position), factor,
             "rope phase");
  std::cout << "rope phase factor=" << factor << " pos=" << position
            << ": HF angle and mscale\n";
}

// mRoPE: pair i reads T, H or W exactly as qwen::vision::RopePosition does.
void CheckYarnVisionPhase(float factor) {
  const std::vector<std::int32_t> thw{300000, 200000, 250000};
  HipBuffer<std::int32_t> positions(3);
  CheckHip(hipMemcpy(positions.get(), thw.data(), 12, hipMemcpyHostToDevice),
           "vision positions");
  const gufo::models::qwen::vision::DeviceRope host_rope{positions.get(), 1, 0};
  HipBuffer<gufo::models::qwen::vision::DeviceRope> rope(1);
  CheckHip(hipMemcpy(rope.get(), &host_rope, sizeof(host_rope),
                     hipMemcpyHostToDevice),
           "vision rope");
  HipBuffer<float> x(kDim);
  HipBuffer<std::uint32_t> pos(1);
  Upload(&x, UnitPairs());
  Upload(&pos, std::vector<std::uint32_t>{0});
  q::Rope(x.get(), 1, 1, kDim, 64, pos.get(), 1e7F, nullptr, rope.get(),
          Yarn(factor));
  CheckHip(hipDeviceSynchronize(), "vision phase rope");
  std::vector<double> coordinate(32);
  for (std::uint32_t i = 0; i < 32; ++i) {
    const int axis = i % 3 == 1 && i < 33 ? 1 : i % 3 == 2 && i < 30 ? 2 : 0;
    coordinate[i] = thw[axis];
  }
  CheckPhase(Download(&x, kDim), coordinate, factor, "vision rope phase");
  std::cout << "vision rope phase factor=" << factor << ": per-axis YaRN\n";
}

// Relative RMS of the rotated dims, GPU Rope kernel versus cpu::Rope.
double OracleRms(float factor, std::uint32_t position) {
  const auto yarn = factor == 1.0F ? qfn::YarnRope{} : Yarn(factor);
  const auto input = MakeValues(std::size_t(kHeads) * kDim, 0x5EED, 1.0F);
  HipBuffer<float> x(input.size());
  HipBuffer<std::uint32_t> pos(1);
  Upload(&x, input);
  Upload(&pos, std::vector<std::uint32_t>{position});
  q::Rope(x.get(), 1, kHeads, kDim, 64, pos.get(), 1e7F, nullptr, nullptr,
          yarn);
  const auto gpu = Download(&x, input.size());
  auto cpu = input;
  qfn::cpu::Rope(cpu.data(), kHeads, kDim, 64, position, 1e7F, yarn);
  double num = 0.0, den = 0.0;
  for (std::uint32_t h = 0; h < kHeads; ++h)
    for (std::uint32_t i = 0; i < 64; ++i) {
      const auto k = std::size_t(h) * kDim + i;
      num += (double(gpu[k]) - cpu[k]) * (double(gpu[k]) - cpu[k]);
      den += double(cpu[k]) * cpu[k];
    }
  return std::sqrt(num / den);
}

void CheckOracleRms() {
  for (const std::uint32_t position : {1000U, 300000U, 650000U}) {
    const double off = OracleRms(1.0F, position);
    const double on = OracleRms(2.5F, position);
    std::cout << "rope vs CPU oracle pos=" << position << ": relative RMS off "
              << off << " on " << on << '\n';
    if ((position == 1000 && on > 1e-4) || on > 2.0 * off + 1e-5)
      throw std::runtime_error("YaRN rope drifts from the CPU oracle");
  }
}

/// FP64 gated attention over the keys each row can see: every key below
/// pos + 1, or with a mask the selected complete blocks plus the incomplete
/// tail block. An independent formula for the per-token kernel.
std::vector<double> Fp64Attention(
    const std::vector<float>& qv, const std::vector<float>& gate,
    const std::vector<__half>& kh, const std::vector<__half>& vh,
    const std::uint32_t* mask, std::uint32_t mask_words, std::uint32_t n_tokens,
    std::uint32_t start_pos, std::uint32_t ratio = kRatio) {
  std::vector<double> out(static_cast<std::size_t>(n_tokens) * kQWidth);
  const double scale = 1.0 / std::sqrt(static_cast<double>(kDim));
  std::vector<std::uint32_t> keys;
  std::vector<double> scores;
  for (std::uint32_t t = 0; t < n_tokens; ++t) {
    const std::uint32_t n_kv = start_pos + t + 1;
    const std::uint32_t tail_start = n_kv / ratio * ratio;
    keys.clear();
    for (std::uint32_t j = 0; j < n_kv; ++j) {
      const std::uint32_t b = j / ratio;
      if (mask == nullptr || j >= tail_start ||
          ((mask[static_cast<std::size_t>(t) * mask_words + b / 32] >>
            (b % 32)) &
           1U) != 0) {
        keys.push_back(j);
      }
    }
    for (std::uint32_t h = 0; h < kHeads; ++h) {
      const std::size_t row = (static_cast<std::size_t>(t) * kHeads + h) * kDim;
      const std::size_t kv_head = (h / (kHeads / kKvHeads)) * kDim;
      scores.assign(keys.size(), 0.0);
      double maximum = -INFINITY;
      for (std::size_t k = 0; k < keys.size(); ++k) {
        const std::size_t base = keys[k] * std::size_t{kKvWidth} + kv_head;
        double dot = 0.0;
        for (std::uint32_t d = 0; d < kDim; ++d) {
          dot += static_cast<double>(qv[row + d]) *
                 static_cast<double>(__half2float(kh[base + d]));
        }
        scores[k] = dot * scale;
        maximum = std::max(maximum, scores[k]);
      }
      double sum = 0.0;
      for (double& score : scores) {
        score = std::exp(score - maximum);
        sum += score;
      }
      for (std::uint32_t d = 0; d < kDim; ++d) {
        double acc = 0.0;
        for (std::size_t k = 0; k < keys.size(); ++k) {
          acc += scores[k] *
                 static_cast<double>(__half2float(
                     vh[keys[k] * std::size_t{kKvWidth} + kv_head + d]));
        }
        const double g = static_cast<double>(gate[row + d]);
        out[row + d] = acc / sum / (1.0 + std::exp(-g));
      }
    }
  }
  return out;
}

/// Runs the per-token reference and the fused WMMA route over the same
/// cache and reports the worst absolute error of the gated context.
double Compare(std::uint32_t n_tokens, std::uint32_t start_pos, bool masked,
               std::uint32_t seed, bool sparse = false, bool bounded = false) {
  const std::uint32_t n_kv = start_pos + n_tokens;
  const std::uint32_t max_blocks = (n_kv + kRatio - 1) / kRatio;
  const std::uint32_t mask_words = (max_blocks + 31) / 32;
  const std::size_t q_count = static_cast<std::size_t>(n_tokens) * kQWidth;
  const std::size_t kv_count = static_cast<std::size_t>(n_kv) * kKvWidth;

  const auto qv = MakeValues(q_count, seed, 4.0F);
  const auto gate = MakeValues(q_count, seed ^ 0x5555U, 3.0F);
  const auto kf = MakeValues(kv_count, seed ^ 0xAAAAU, 1.0F);
  const auto vf = MakeValues(kv_count, seed ^ 0x3333U, 1.0F);
  std::vector<__half> kh(kv_count);
  std::vector<__half> vh(kv_count);
  for (std::size_t i = 0; i < kv_count; ++i) {
    kh[i] = __float2half(kf[i]);
    vh[i] = __float2half(vf[i]);
  }
  std::vector<std::uint32_t> mask(static_cast<std::size_t>(n_tokens) *
                                  mask_words);
  std::uint32_t state = seed ^ 0x77777777U;
  for (std::uint32_t& word : mask) {
    word = NextRandom(&state) & NextRandom(&state);
    if (sparse) {
      // About one block in sixteen: most 16-key tiles are unselected for a
      // whole query block, which is what the tile skipping is for.
      word &= NextRandom(&state) & NextRandom(&state);
    }
  }
  if (bounded) {
    std::fill(mask.begin(), mask.end(), 0);
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      const auto complete = (start_pos + t + 1) / kRatio;
      std::uint32_t selected = 0;
      while (selected < std::min(512U, complete)) {
        const auto block = NextRandom(&state) % complete;
        auto& word =
            mask[static_cast<std::size_t>(t) * mask_words + block / 32];
        const auto bit = 1U << (block % 32);
        if ((word & bit) == 0) {
          word |= bit;
          ++selected;
        }
      }
    }
  }

  HipBuffer<float> d_q(q_count);
  HipBuffer<float> d_gate(q_count);
  HipBuffer<__half> d_k(kv_count);
  HipBuffer<__half> d_v(kv_count);
  HipBuffer<std::uint32_t> d_mask(mask.size());
  HipBuffer<std::uint32_t> d_pos(1);
  HipBuffer<float> d_ref(q_count);
  HipBuffer<float> d_wmma(q_count);
  Upload(&d_q, qv);
  Upload(&d_gate, gate);
  Upload(&d_k, kh);
  Upload(&d_v, vh);
  Upload(&d_mask, mask);
  Upload(&d_pos, std::vector<std::uint32_t>{start_pos});
  const std::uint32_t* mask_ptr = masked ? d_mask.get() : nullptr;

  q::Attention(d_q.get(), d_k.get(), d_v.get(), mask_ptr, mask_words,
               d_ref.get(), nullptr, 1, n_tokens, d_pos.get(), kHeads, kKvHeads,
               kDim, kRatio, nullptr);
  q::SigmoidMul(d_ref.get(), d_gate.get(), q_count, nullptr);
  // The split-key form of the reference must match its single-block form
  // (log-sum-exp merge), the way decode runs it.
  {
    constexpr std::uint32_t kSplits = 8;
    HipBuffer<float> d_split(q_count);
    HipBuffer<float> d_partials(static_cast<std::size_t>(n_tokens) * kHeads *
                                kSplits * (kDim + 2));
    q::Attention(d_q.get(), d_k.get(), d_v.get(), mask_ptr, mask_words,
                 d_split.get(), d_partials.get(), kSplits, n_tokens,
                 d_pos.get(), kHeads, kKvHeads, kDim, kRatio, nullptr);
    q::SigmoidMul(d_split.get(), d_gate.get(), q_count, nullptr);
    CheckHip(hipDeviceSynchronize(), "split attention");
    const auto a = Download(&d_ref, q_count);
    const auto b = Download(&d_split, q_count);
    double worst = 0.0;
    for (std::size_t i = 0; i < q_count; ++i) {
      worst = std::max(worst, std::abs(static_cast<double>(a[i] - b[i])));
    }
    std::cout << "  split-key reference worst absolute error " << worst << '\n';
    const auto exact =
        Fp64Attention(qv, gate, kh, vh, masked ? mask.data() : nullptr,
                      mask_words, n_tokens, start_pos);
    double single_error = 0.0;
    double split_error = 0.0;
    for (std::size_t i = 0; i < q_count; ++i) {
      single_error = std::max(single_error, std::abs(a[i] - exact[i]));
      split_error = std::max(split_error, std::abs(b[i] - exact[i]));
    }
    std::cout << "  FP64 worst absolute error single " << single_error
              << " split " << split_error << '\n';
    if (std::max(single_error, split_error) > kFp64Limit) {
      throw std::runtime_error("per-token attention exceeds its FP64 limit");
    }
    if (worst > 1e-4) {
      throw std::runtime_error("split-key attention disagrees");
    }
  }
  if (!q::WmmaCausalAttention(d_q.get(), d_gate.get(), d_k.get(), d_v.get(),
                              mask_ptr, mask_words, d_wmma.get(), n_tokens,
                              start_pos, kHeads, kKvHeads, kDim, kRatio,
                              nullptr)) {
    throw std::runtime_error("WMMA attention rejected the model geometry");
  }
  CheckHip(hipDeviceSynchronize(), "attention synchronization");

  const auto ref = Download(&d_ref, q_count);
  const auto out = Download(&d_wmma, q_count);
  if (!q::WmmaCausalAttention(d_q.get(), d_gate.get(), d_k.get(), d_v.get(),
                              mask_ptr, mask_words, d_wmma.get(), n_tokens,
                              start_pos, kHeads, kKvHeads, kDim, kRatio,
                              nullptr)) {
    throw std::runtime_error("WMMA attention replay rejected the geometry");
  }
  const auto replay = Download(&d_wmma, q_count);
  if (std::memcmp(out.data(), replay.data(), q_count * sizeof(float)) != 0) {
    throw std::runtime_error("WMMA attention replay changed the output");
  }
  {
    constexpr float poison = -12345.0F;
    Upload(&d_wmma, std::vector<float>(q_count, poison));
    if (!q::WmmaCausalAttention(d_q.get(), d_gate.get(), d_k.get(), d_v.get(),
                                mask_ptr, mask_words, d_wmma.get(), n_tokens,
                                start_pos, kHeads, kKvHeads, kDim, kRatio,
                                nullptr, true)) {
      throw std::runtime_error("last-tile attention rejected the geometry");
    }
    const auto tail = Download(&d_wmma, q_count);
    const std::uint32_t tile_rows = mask_ptr ? 4 : 16;
    const std::size_t begin =
        static_cast<std::size_t>((n_tokens - 1) / tile_rows * tile_rows) *
        kQWidth;
    for (std::size_t i = 0; i < q_count; ++i) {
      if (i < begin ? tail[i] != poison
                    : std::memcmp(&out[i], &tail[i], sizeof(float)) != 0) {
        throw std::runtime_error("last-tile attention changed a row or guard");
      }
    }
  }
  double worst = 0.0;
  for (std::size_t i = 0; i < q_count; ++i) {
    if (!std::isfinite(out[i])) {
      throw std::runtime_error("WMMA attention output is not finite");
    }
    worst = std::max(worst, std::abs(static_cast<double>(ref[i] - out[i])));
  }
  if (worst > 1e-2) {
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      for (std::uint32_t h = 0; h < kHeads; ++h) {
        double w = 0.0;
        for (std::uint32_t d = 0; d < kDim; ++d) {
          const std::size_t i =
              (static_cast<std::size_t>(t) * kHeads + h) * kDim + d;
          w = std::max(w, std::abs(static_cast<double>(ref[i] - out[i])));
        }
        if (w > 1e-2) {
          std::cout << "  query " << t << " head " << h << " err " << w << '\n';
        }
      }
    }
  }
  return worst;
}

/// Selected blocks for one query row, given its complete-block count.
using BlockPattern =
    std::function<void(std::uint32_t, std::vector<std::uint32_t>*)>;

/// The per-token kernel against FP64 on a structured selection, in its
/// single-block form and split 1, 3 and 8 ways. The masks are built so the
/// sparse path's 1024-block compaction windows and 64-block key tiles meet
/// their edge cases: tiles that straddle windows, a carried partial tile of
/// every size, a single partial tile, and a ninth tile that wraps a split.
void CheckStructured(const char* name, std::uint32_t n_tokens,
                     std::uint32_t start_pos, const BlockPattern& pattern,
                     std::uint32_t seed, std::uint32_t ratio = kRatio) {
  const std::uint32_t n_kv = start_pos + n_tokens;
  const std::uint32_t max_blocks = (n_kv + ratio - 1) / ratio;
  const std::uint32_t mask_words = (max_blocks + 31) / 32;
  const std::size_t q_count = static_cast<std::size_t>(n_tokens) * kQWidth;
  const std::size_t kv_count = static_cast<std::size_t>(n_kv) * kKvWidth;
  const auto qv = MakeValues(q_count, seed, 4.0F);
  const auto gate = MakeValues(q_count, seed ^ 0x5555U, 3.0F);
  std::vector<__half> kh(kv_count);
  std::vector<__half> vh(kv_count);
  {
    const auto kf = MakeValues(kv_count, seed ^ 0xAAAAU, 1.0F);
    const auto vf = MakeValues(kv_count, seed ^ 0x3333U, 1.0F);
    for (std::size_t i = 0; i < kv_count; ++i) {
      kh[i] = __float2half(kf[i]);
      vh[i] = __float2half(vf[i]);
    }
  }
  std::vector<std::uint32_t> mask(static_cast<std::size_t>(n_tokens) *
                                  mask_words);
  std::vector<std::uint32_t> blocks;
  std::size_t selected = 0;
  for (std::uint32_t t = 0; t < n_tokens; ++t) {
    const std::uint32_t complete = (start_pos + t + 1) / ratio;
    blocks.clear();
    pattern(complete, &blocks);
    for (const std::uint32_t b : blocks) {
      if (b >= complete) {
        throw std::runtime_error("structured mask selects past the context");
      }
      mask[static_cast<std::size_t>(t) * mask_words + b / 32] |= 1U << (b % 32);
    }
    selected += blocks.size();
  }
  const auto exact = Fp64Attention(qv, gate, kh, vh, mask.data(), mask_words,
                                   n_tokens, start_pos, ratio);

  HipBuffer<float> d_q(q_count);
  HipBuffer<float> d_gate(q_count);
  HipBuffer<__half> d_k(kv_count);
  HipBuffer<__half> d_v(kv_count);
  HipBuffer<std::uint32_t> d_mask(mask.size());
  HipBuffer<std::uint32_t> d_pos(1);
  HipBuffer<float> d_out(q_count);
  Upload(&d_q, qv);
  Upload(&d_gate, gate);
  Upload(&d_k, kh);
  Upload(&d_v, vh);
  Upload(&d_mask, mask);
  Upload(&d_pos, std::vector<std::uint32_t>{start_pos});
  std::cout << "structured " << name << " ratio=" << ratio << " n=" << n_tokens
            << " start=" << start_pos << " blocks/row=" << selected / n_tokens
            << " FP64 worst absolute error";
  double worst = 0.0;
  for (const std::uint32_t splits : {0U, 1U, 3U, 8U}) {
    HipBuffer<float> d_partials(static_cast<std::size_t>(n_tokens) * kHeads *
                                std::max(splits, 1U) * (kDim + 2));
    q::Attention(d_q.get(), d_k.get(), d_v.get(), d_mask.get(), mask_words,
                 d_out.get(), splits == 0 ? nullptr : d_partials.get(), splits,
                 n_tokens, d_pos.get(), kHeads, kKvHeads, kDim, ratio, nullptr);
    q::SigmoidMul(d_out.get(), d_gate.get(), q_count, nullptr);
    CheckHip(hipDeviceSynchronize(), "structured attention");
    const auto out = Download(&d_out, q_count);
    double error = 0.0;
    for (std::size_t i = 0; i < q_count; ++i) {
      if (!std::isfinite(out[i])) {
        throw std::runtime_error("structured attention output is not finite");
      }
      error = std::max(error, std::abs(out[i] - exact[i]));
    }
    std::cout << (splits == 0 ? " single "
                              : " split" + std::to_string(splits) + " ")
              << error;
    worst = std::max(worst, error);
  }
  std::cout << '\n';
  if (worst > kFp64Limit) {
    throw std::runtime_error("structured attention exceeds its FP64 limit");
  }
}

void CheckChunks(std::uint32_t n, std::uint32_t split) {
  const std::size_t count = std::size_t(n) * kQWidth;
  HipBuffer<float> queries(count), gates(count), full(count), chunked(count);
  HipBuffer<__half> keys(std::size_t(n) * kKvWidth),
      values(std::size_t(n) * kKvWidth);
  Upload(&queries, MakeValues(count, 412, 4.0F));
  Upload(&gates, MakeValues(count, 721, 3.0F));
  for (auto* destination : {&keys, &values}) {
    const auto f = MakeValues(std::size_t(n) * kKvWidth,
                              destination == &keys ? 891 : 347, 1.0F);
    std::vector<__half> half(f.size());
    std::transform(f.begin(), f.end(), half.begin(),
                   [](float x) { return __float2half(x); });
    Upload(destination, half);
  }
  const auto run = [&](std::uint32_t start, std::uint32_t rows, float* out) {
    const auto offset = std::size_t(start) * kQWidth;
    if (!q::WmmaCausalAttention(queries.get() + offset, gates.get() + offset,
                                keys.get(), values.get(), nullptr, 0,
                                out + offset, rows, start, kHeads, kKvHeads,
                                kDim, kRatio, nullptr)) {
      throw std::runtime_error("chunk attention rejected geometry");
    }
  };
  run(0, n, full.get());
  run(0, split, chunked.get());
  run(split, n - split, chunked.get());
  const auto a = Download(&full, count);
  const auto b = Download(&chunked, count);
  std::size_t differences = 0, first = count;
  float max_error = 0;
  for (std::size_t i = 0; i < count; ++i) {
    if (a[i] != b[i]) {
      ++differences;
      first = std::min(first, i);
      max_error = std::max(max_error, std::abs(a[i] - b[i]));
    }
  }
  std::cout << "chunk attention n=" << n << " split=" << split
            << " differing=" << differences << " max=" << max_error
            << " first_row=" << first / kQWidth << '\n';
  if (differences != 0)
    throw std::runtime_error("attention depends on prefill chunk boundary");
}

}  // namespace

int main() {
  try {
    CheckChunks(136, 94);
    CheckChunks(2048, 1025);
    CheckPreparation(1, 0, 64);
    CheckPreparation(8, 4096, 64);
    CheckPreparation(65, 131069, 64);
    CheckPreparation(7, 131069, 256);
    CheckPreparation(8, 4096, 64, Yarn(1.5625F));
    CheckPreparation(65, 300000, 64, Yarn(2.5F));
    CheckYarnPhase(1.0F, 262143);  // off: sincosf accuracy at native depth
    for (const float factor : {1.5625F, 2.5F, 4.0F})
      for (const std::uint32_t position : {1000U, 300000U, 650000U})
        CheckYarnPhase(factor, position);
    CheckYarnVisionPhase(2.5F);
    CheckOracleRms();
    // Structured selections against the sparse path's 1024-block compaction
    // windows and 64-block key tiles. At 131069 the last 512 blocks sit in
    // the final window; 262141 spans all 64 windows of the native context,
    // so 63 blocks per window carry every partial-tile size once.
    constexpr std::uint32_t kWindow = 1024;
    const auto spread = [](std::uint32_t count) {
      return [count](std::uint32_t complete, std::vector<std::uint32_t>* out) {
        for (std::uint32_t j = 0; j < count && j < complete; ++j) {
          out->push_back(static_cast<std::uint32_t>(
              static_cast<std::uint64_t>(j) * complete / count));
        }
      };
    };
    const auto per_window = [](std::uint32_t count) {
      return [count](std::uint32_t complete, std::vector<std::uint32_t>* out) {
        for (std::uint32_t w0 = 0; w0 < complete; w0 += kWindow) {
          for (std::uint32_t j = 0; j < count; ++j) {
            const std::uint32_t b = w0 + j * 16 + (w0 / kWindow) % 16;
            if (b < complete) {
              out->push_back(b);
            }
          }
        }
      };
    };
    const BlockPattern last_budget = [](std::uint32_t complete,
                                        std::vector<std::uint32_t>* out) {
      for (std::uint32_t b = complete - std::min(512U, complete); b < complete;
           ++b) {
        out->push_back(b);
      }
    };
    const BlockPattern one_window = [](std::uint32_t complete,
                                       std::vector<std::uint32_t>* out) {
      for (std::uint32_t j = 0; j < 64 && 7 * kWindow + j * 16 < complete;
           ++j) {
        out->push_back(7 * kWindow + j * 16);
      }
    };
    struct Structured {
      const char* name;
      std::uint32_t start_pos;
      BlockPattern pattern;
    };
    const Structured structured[] = {
        {"last-512", 131069, last_budget},
        {"63-per-window", 262141, per_window(63)},
        {"1-per-window", 262141, per_window(1)},
        {"40-total", 131069, spread(40)},
        {"64-in-one-window", 131069, one_window},
        {"513-total", 131069, spread(513)},
        {"576-total", 131069, spread(576)},
    };
    std::uint32_t structured_seed = 0x5EED0000U;
    for (const Structured& c : structured) {
      for (const std::uint32_t n : {1U, 3U}) {
        CheckStructured(c.name, n, c.start_pos, c.pattern, structured_seed++);
      }
    }
    // At ratio 1 a tile is 256 blocks, so partial tiles reach 231 blocks,
    // more than the room kept for a ratio-4 carry. Alternating 999 and 100
    // selected blocks per window attends some partial tiles in their window
    // and carries others.
    const BlockPattern alternating = [](std::uint32_t complete,
                                        std::vector<std::uint32_t>* out) {
      for (std::uint32_t b = 0; b < complete; ++b) {
        if (b % kWindow < ((b / kWindow) % 2 == 0 ? 999U : 100U)) {
          out->push_back(b);
        }
      }
    };
    for (const std::uint32_t n : {1U, 3U}) {
      CheckStructured("ratio-1-alternating", n, 8189, alternating,
                      structured_seed++, 1);
    }
    struct Case {
      std::uint32_t n_tokens;
      std::uint32_t start_pos;
      bool masked;
      bool sparse;
      bool bounded{false};
    };
    const Case cases[] = {
        {4, 4096, false, false},      {3, 9000, true, true},
        {100, 0, false, false},       {64, 37, false, false},
        {100, 0, true, false},        {77, 51, true, false},
        {96, 4096, true, true},       {70, 8000, true, true},
        {5, 32765, true, true, true}, {7, 65533, true, true, true},
        {5, 131069, true, true},      {7, 131069, true, true, true},
        {17, 2047, true, true, true}};
    bool ok = true;
    std::uint32_t seed = 0x1234ABCDU;
    for (const Case& c : cases) {
      const double worst = Compare(c.n_tokens, c.start_pos, c.masked, seed++,
                                   c.sparse, c.bounded);
      std::cout << "WMMA attention n=" << c.n_tokens << " start=" << c.start_pos
                << (c.masked ? (c.sparse ? " sparse" : " masked") : " dense")
                << " worst absolute error " << worst << '\n';
      // The reference accumulates FP32 probabilities; the WMMA route rounds
      // Q and P to FP16, so the contract is a small absolute envelope on
      // values of order one.
      ok = ok && worst < 2e-2;
    }
    return ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
