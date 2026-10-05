#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace {

// The model's linear-attention geometry: 16 key heads, 48 value heads,
// 128-wide state, kernel-4 causal conv.

constexpr std::uint32_t kKHeads = 16;
constexpr std::uint32_t kVHeads = 48;
constexpr std::uint32_t kDim = 128;
constexpr std::uint32_t kKernel = 4;
constexpr std::uint32_t kChannels = 2 * kKHeads * kDim + kVHeads * kDim;
constexpr std::uint32_t kZ = kVHeads * kDim;
constexpr float kEps = 1e-6F;

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
                              float scale, float offset = 0.0F) {
  std::vector<float> values(count);
  for (float& value : values) {
    value = offset +
            scale *
                static_cast<float>(
                    static_cast<int>(NextRandom(&seed) & 0xFFFFU) - 32768) /
                32768.0F;
  }
  return values;
}

void Upload(HipBuffer<float>* destination, const std::vector<float>& source) {
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

double WorstRelative(const std::vector<float>& reference,
                     const std::vector<float>& candidate) {
  double worst = 0.0;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    const double scale =
        std::max(1e-3, std::abs(static_cast<double>(reference[i])));
    worst = std::max(
        worst,
        std::abs(static_cast<double>(reference[i] - candidate[i])) / scale);
  }
  return worst;
}

/// Worst absolute difference over the reference's RMS: the chunked route
/// rounds its operands to F16, so near-zero elements carry the row's
/// rounding noise.
double WorstOverRms(const std::vector<float>& reference,
                    const std::vector<float>& candidate) {
  double ss = 0.0;
  double worst = 0.0;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    ss += static_cast<double>(reference[i]) * reference[i];
    worst = std::max(
        worst, std::abs(static_cast<double>(reference[i]) - candidate[i]));
  }
  return worst / std::sqrt(ss / static_cast<double>(reference.size()));
}

int RunCase(std::uint32_t kTokens, bool extreme_gates = false) {
  {
    const std::size_t kQkvCount = static_cast<std::size_t>(kTokens) * kChannels;
    const std::size_t kStateCount =
        static_cast<std::size_t>(kVHeads) * kDim * kDim;
    const std::size_t kConvState =
        static_cast<std::size_t>(kKernel - 1) * kChannels;
    const std::size_t kScratch =
        static_cast<std::size_t>(kTokens + kKernel) * kChannels;
    const std::size_t kQn = static_cast<std::size_t>(kTokens) * kKHeads * kDim;
    const std::size_t kRaw = static_cast<std::size_t>(kTokens) * kVHeads * kDim;
    const std::size_t kOut = static_cast<std::size_t>(kTokens) * kZ;

    const auto qkv = MakeValues(kQkvCount, 0x1234ABCDU, 1.0F);
    const auto z = MakeValues(kOut, 0x0BADF00DU, 2.0F);
    auto alpha_beta = MakeValues(
        static_cast<std::size_t>(kTokens) * 2 * kVHeads, 0xBADC0FFEU, 2.0F);
    const auto conv_w = MakeValues(
        static_cast<std::size_t>(kChannels) * kKernel, 0xDEADBEEFU, 0.5F);
    auto a = MakeValues(kVHeads, 0xC0FFEE11U, 1.0F, -1.5F);
    auto dt = MakeValues(kVHeads, 0xFEEDFACEU, 1.0F);
    if (extreme_gates) {
      // Finite gates on both sides of the softplus threshold and expf overflow.
      constexpr float gates[] = {-100.0F, 19.9F, 20.1F, 88.8F, 100.0F, 1000.0F};
      for (std::uint32_t h = 0; h < kVHeads; ++h) {
        a[h] = -0.01F;
        dt[h] = 0.0F;
        for (std::uint32_t t = 0; t < kTokens; ++t)
          alpha_beta[t * 2 * kVHeads + h] = gates[h % 6];
      }
    }
    const auto norm_w = MakeValues(kDim, 0x600DCAFEU, 0.5F, 1.0F);
    const auto conv_state = MakeValues(kConvState, 0x13579BDFU, 1.0F);
    const auto state = MakeValues(kStateCount, 0x2468ACE0U, 0.1F);

    HipBuffer<float> d_qkv(kQkvCount);
    HipBuffer<float> d_z(kOut);
    HipBuffer<float> d_alpha_beta(alpha_beta.size());
    HipBuffer<float> d_conv_w(conv_w.size());
    HipBuffer<float> d_a(kVHeads);
    HipBuffer<float> d_dt(kVHeads);
    HipBuffer<float> d_norm_w(kDim);
    HipBuffer<float> d_scratch(kScratch);
    HipBuffer<float> d_qn(kQn);
    HipBuffer<float> d_kn(kQn);
    HipBuffer<float> d_raw(kRaw);
    Upload(&d_qkv, qkv);
    Upload(&d_z, z);
    Upload(&d_alpha_beta, alpha_beta);
    Upload(&d_conv_w, conv_w);
    Upload(&d_a, a);
    Upload(&d_dt, dt);
    Upload(&d_norm_w, norm_w);

    std::vector<float> outs[2];
    std::vector<float> states[2];
    std::vector<float> raws[2];
    std::vector<float> conv_out;
    for (int route = 0; route < 2; ++route) {
      HipBuffer<float> d_conv_state(kConvState);
      HipBuffer<float> d_state(kStateCount);
      HipBuffer<float> d_out(kOut);
      Upload(&d_conv_state, conv_state);
      Upload(&d_state, state);
      q::GatedDeltaNet(
          d_qkv.get(), kChannels, d_z.get(), kZ, d_alpha_beta.get(),
          d_conv_w.get(), d_a.get(), d_dt.get(), d_norm_w.get(),
          d_conv_state.get(), d_scratch.get(), d_qn.get(), d_kn.get(),
          d_raw.get(), d_state.get(), d_out.get(), nullptr, {}, {}, kTokens,
          kKHeads, kVHeads, kDim, kKernel, route == 1, false, kEps, nullptr);
      CheckHip(hipDeviceSynchronize(), "GDN synchronization");
      outs[route] = Download(&d_out, kOut);
      states[route] = Download(&d_state, kStateCount);
      raws[route] = Download(&d_raw, kRaw);
      constexpr std::size_t kCheckpointGuard = 16;
      constexpr float kCheckpointSentinel = 12345.0F;
      HipBuffer<float> checkpoint_state(kStateCount + kCheckpointGuard);
      HipBuffer<float> checkpoint_conv(kConvState + kCheckpointGuard);
      const auto forward = [&](std::uint32_t count,
                               q::GdnCheckpoint checkpoint) {
        Upload(&d_conv_state, conv_state);
        Upload(&d_state, state);
        q::GatedDeltaNet(d_qkv.get(), kChannels, d_z.get(), kZ,
                         d_alpha_beta.get(), d_conv_w.get(), d_a.get(),
                         d_dt.get(), d_norm_w.get(), d_conv_state.get(),
                         d_scratch.get(), d_qn.get(), d_kn.get(), d_raw.get(),
                         d_state.get(), d_out.get(), nullptr, {}, {}, count,
                         kKHeads, kVHeads, kDim, kKernel, route == 1, false,
                         kEps, nullptr, nullptr, checkpoint);
      };
      for (const auto prefix : {1U, std::max(1U, kTokens / 2), kTokens}) {
        Upload(&checkpoint_state,
               std::vector<float>(kStateCount + kCheckpointGuard,
                                  kCheckpointSentinel));
        Upload(&checkpoint_conv,
               std::vector<float>(kConvState + kCheckpointGuard,
                                  kCheckpointSentinel));
        forward(kTokens,
                {checkpoint_state.get(), checkpoint_conv.get(), prefix});
        const auto final_state = Download(&d_state, kStateCount);
        const auto final_out = Download(&d_out, kOut);
        if (std::memcmp(final_state.data(), states[route].data(),
                        kStateCount * sizeof(float)) ||
            std::memcmp(final_out.data(), outs[route].data(),
                        kOut * sizeof(float)))
          throw std::runtime_error(
              "GDN checkpoint changed full-pass arithmetic");
        const auto captured =
            Download(&checkpoint_state, kStateCount + kCheckpointGuard);
        const auto history =
            Download(&checkpoint_conv, kConvState + kCheckpointGuard);
        forward(prefix, {});
        const auto expected_state = Download(&d_state, kStateCount);
        const auto expected_conv = Download(&d_conv_state, kConvState);
        if (std::memcmp(captured.data(), expected_state.data(),
                        kStateCount * sizeof(float)) ||
            std::memcmp(history.data(), expected_conv.data(),
                        kConvState * sizeof(float)))
          throw std::runtime_error("GDN checkpoint changed boundary state");
        for (std::size_t i = 0; i < kCheckpointGuard; ++i)
          if (captured[kStateCount + i] != kCheckpointSentinel ||
              history[kConvState + i] != kCheckpointSentinel)
            throw std::runtime_error(
                "GDN checkpoint wrote beyond its boundary");
      }
      // Direct F16 output must equal narrowing the independently checked
      // F32 epilogue and must not change the recurrent state.
      constexpr std::size_t kHalfGuard = 16;
      HipBuffer<__half> d_half(kOut + kHalfGuard);
      CheckHip(hipMemset(d_half.get(), 0x7F, d_half.bytes()), "half guard");
      Upload(&d_conv_state, conv_state);
      Upload(&d_state, state);
      q::GatedDeltaNet(d_qkv.get(), kChannels, d_z.get(), kZ,
                       d_alpha_beta.get(), d_conv_w.get(), d_a.get(),
                       d_dt.get(), d_norm_w.get(), d_conv_state.get(),
                       d_scratch.get(), d_qn.get(), d_kn.get(), d_raw.get(),
                       d_state.get(), nullptr, nullptr, {}, {}, kTokens,
                       kKHeads, kVHeads, kDim, kKernel, route == 1, false, kEps,
                       nullptr, d_half.get());
      std::vector<__half> half(kOut + kHalfGuard);
      CheckHip(hipMemcpy(half.data(), d_half.get(), d_half.bytes(),
                         hipMemcpyDeviceToHost),
               "half output");
      for (std::size_t i = 0; i < kOut; ++i) {
        const __half expected = __float2half_rn(outs[route][i]);
        if (std::memcmp(&expected, &half[i], sizeof(__half)) != 0)
          throw std::runtime_error("GDN F16 epilogue changed rounding");
      }
      for (std::size_t i = kOut; i < half.size(); ++i) {
        std::uint16_t bits;
        std::memcpy(&bits, &half[i], sizeof(bits));
        if (bits != 0x7F7F)
          throw std::runtime_error("GDN F16 epilogue wrote past its output");
      }
      const auto half_state = Download(&d_state, kStateCount);
      if (std::memcmp(half_state.data(), states[route].data(),
                      kStateCount * sizeof(float)) != 0)
        throw std::runtime_error("GDN F16 epilogue changed the state");
      // The causal conv's output (the scratch's first rows) against a CPU
      // reference over the same history.
      conv_out = Download(&d_scratch, kScratch);
      double worst_conv = 0.0;
      for (std::uint32_t t = 0; t < kTokens; ++t) {
        for (std::uint32_t c = 0; c < kChannels; ++c) {
          double acc = 0.0;
          for (std::uint32_t k = 0; k < kKernel; ++k) {
            const std::int32_t src = static_cast<std::int32_t>(t + k) -
                                     static_cast<std::int32_t>(kKernel - 1);
            const double v =
                src >= 0
                    ? qkv[static_cast<std::size_t>(src) * kChannels + c]
                    : conv_state[static_cast<std::size_t>(kKernel - 1 + src) *
                                     kChannels +
                                 c];
            acc += static_cast<double>(conv_w[c * kKernel + k]) * v;
          }
          const double ref = acc / (1.0 + std::exp(-acc));
          worst_conv = std::max(
              worst_conv,
              std::abs(ref -
                       conv_out[static_cast<std::size_t>(t) * kChannels + c]) /
                  std::max(1e-3, std::abs(ref)));
        }
      }
      std::cout << "GDN conv vs CPU worst relative error " << worst_conv
                << '\n';
      if (worst_conv > 1e-4) {
        return 1;
      }
    }
    if (kTokens <= 8) {
      // Every rollback prefix must match a fresh forward of that prefix.
      // The final state stays live; no snapshot may be written for it.
      constexpr std::size_t kGuard = 32;
      constexpr float kSentinel = 12345.0F;
      const std::size_t saved = kTokens - 1;
      std::vector<std::size_t> state_offsets(saved);
      std::size_t state_storage = kGuard;
      for (std::size_t i = saved; i-- > 0;) {
        state_offsets[i] = state_storage;
        state_storage +=
            q::GdnRollbackRowFloats(i, kKHeads, kVHeads, kDim) + kGuard;
      }
      HipBuffer<float> state_snaps(state_storage);
      HipBuffer<float> conv_snaps(saved * kConvState + kGuard);
      Upload(&state_snaps, std::vector<float>(state_storage, kSentinel));
      Upload(&conv_snaps,
             std::vector<float>(saved * kConvState + kGuard, kSentinel));
      HipBuffer<float> current_state(kStateCount), current_conv(kConvState);
      HipBuffer<float> current_out(kOut);
      HipBuffer<q::GdnBatchItem> batch_items(3);
      auto forward = [&](std::uint32_t n, float* ss, float* cs,
                         bool batch = false) {
        q::RollbackRows states, convs;
        // Reverse the physical rows to prove that the kernel honors independent
        // prefix addresses rather than relying on a contiguous allocation.
        for (std::size_t i = 0; i < saved; ++i) {
          states.rows[i] = ss ? ss + state_offsets[i] : nullptr;
          convs.rows[i] = cs ? cs + (saved - 1 - i) * kConvState : nullptr;
        }
        Upload(&current_state, state);
        Upload(&current_conv, conv_state);
        if (batch) {
          // Cancelled peers have no valid pointers. Every batch stage must
          // skip them before reading or mutating any request-local state.
          std::array<q::GdnBatchItem, 3> items{};
          items[1] = {d_qkv.get(),
                      d_z.get(),
                      d_alpha_beta.get(),
                      current_conv.get(),
                      d_scratch.get(),
                      d_qn.get(),
                      d_kn.get(),
                      d_raw.get(),
                      current_state.get(),
                      current_out.get(),
                      states,
                      convs,
                      n};
          CheckHip(hipMemcpy(batch_items.get(), items.data(),
                             batch_items.bytes(), hipMemcpyHostToDevice),
                   "batch descriptors");
          if (!q::GatedDeltaNetBatch(batch_items.get(), items.size(), kTokens,
                                     2U, kChannels, kZ, d_conv_w.get(),
                                     d_a.get(), d_dt.get(), d_norm_w.get(),
                                     kKHeads, kVHeads, kEps, nullptr))
            throw std::runtime_error("GDN batch launch failed");
          return;
        }
        q::GatedDeltaNet(d_qkv.get(), kChannels, d_z.get(), kZ,
                         d_alpha_beta.get(), d_conv_w.get(), d_a.get(),
                         d_dt.get(), d_norm_w.get(), current_conv.get(),
                         d_scratch.get(), d_qn.get(), d_kn.get(), d_raw.get(),
                         current_state.get(), current_out.get(), nullptr,
                         states, convs, n, kKHeads, kVHeads, kDim, kKernel,
                         false, false, kEps, nullptr);
      };
      forward(kTokens, state_snaps.get(), conv_snaps.get());
      const auto saved_states = Download(&state_snaps, state_storage);
      const auto saved_convs =
          Download(&conv_snaps, saved * kConvState + kGuard);
      const auto final_state = Download(&current_state, kStateCount);
      const auto final_out = Download(&current_out, kOut);
      if (std::memcmp(final_state.data(), states[0].data(),
                      kStateCount * sizeof(float)) ||
          std::memcmp(final_out.data(), outs[0].data(), kOut * sizeof(float))) {
        throw std::runtime_error("GDN snapshots changed the full forward");
      }
      forward(kTokens, state_snaps.get(), conv_snaps.get(), true);
      const auto exact = [](const auto& a, const auto& b) {
        return a.size() == b.size() &&
               std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
      };
      if (!exact(Download(&current_state, kStateCount), final_state) ||
          !exact(Download(&current_out, kOut), final_out) ||
          !exact(Download(&state_snaps, state_storage), saved_states) ||
          !exact(Download(&conv_snaps, saved * kConvState + kGuard),
                 saved_convs))
        throw std::runtime_error("GDN batch changed output or rollback state");
      for (std::size_t i = 0; i < kGuard; ++i) {
        if (saved_states[i] != kSentinel ||
            saved_convs[saved * kConvState + i] != kSentinel) {
          throw std::runtime_error("GDN wrote an unused final snapshot");
        }
      }
      for (std::size_t row = 0; row < saved; ++row) {
        const auto end = state_offsets[row] +
                         q::GdnRollbackRowFloats(row, kKHeads, kVHeads, kDim);
        for (std::size_t i = 0; i < kGuard; ++i)
          if (saved_states[end + i] != kSentinel)
            throw std::runtime_error("GDN overwrote a compact rollback row");
      }
      for (std::uint32_t keep = 1; keep < kTokens; ++keep) {
        forward(keep, nullptr, nullptr, true);
        const auto expected_state = Download(&current_state, kStateCount);
        const auto expected_conv = Download(&current_conv, kConvState);
        // An independent host FMA oracle fixes the rounded-product contract.
        // Comparing only two GPU executions misses compiler reassociation.
        const auto product = [](float a, float b) {
          volatile float rounded = a * b;
          return rounded;
        };
        for (std::size_t index = 0; index < kStateCount; ++index) {
          const auto h = index / (kDim * kDim);
          const auto j = (index / kDim) % kDim;
          const auto k = index % kDim;
          float value = saved_states[state_offsets[0] + index];
          for (std::uint32_t t = 1; t < keep; ++t) {
            const auto* update = saved_states.data() + state_offsets[t];
            const float key = update[(h % kKHeads) * kDim + k];
            const float decay = update[kKHeads * kDim + h];
            const float beta = update[kKHeads * kDim + kVHeads + h];
            const float error =
                update[kKHeads * kDim + 2 * kVHeads + h * kDim + j];
            const float correction = product(error, key);
            value = (k + 1) % 32 == 0
                        ? std::fma(value, decay, product(beta, correction))
                        : std::fma(beta, correction, product(value, decay));
          }
          if (std::memcmp(&value, &expected_state[index], sizeof(float)))
            throw std::runtime_error(
                "GDN changed its rounded-product contract");
        }
        q::RollbackRows replay;
        for (std::size_t i = 0; i < saved; ++i)
          replay.rows[i] = state_snaps.get() + state_offsets[i];
        q::RestoreGdnState(current_state.get(), replay, keep, kKHeads, kVHeads,
                           nullptr);
        const auto restored_state = Download(&current_state, kStateCount);
        if (std::memcmp(expected_state.data(), restored_state.data(),
                        kStateCount * sizeof(float)) ||
            std::memcmp(expected_conv.data(),
                        saved_convs.data() + (saved - keep) * kConvState,
                        kConvState * sizeof(float))) {
          throw std::runtime_error("GDN rollback prefix changed state");
        }
      }

      // The PLE convolution has a dilated history and the same prefix contract.
      constexpr std::uint32_t kDilation = 2;
      constexpr std::size_t kPleHistory = (kKernel - 1) * kDilation * kChannels;
      const auto ple_history = MakeValues(kPleHistory, 0x12873491U, 1.0F);
      HipBuffer<float> ple_current(kPleHistory), ple_scratch(kPleHistory);
      HipBuffer<float> ple_out(kQkvCount),
          ple_snaps(saved * kPleHistory + kGuard);
      Upload(&ple_snaps,
             std::vector<float>(saved * kPleHistory + kGuard, kSentinel));
      auto ple = [&](std::uint32_t n, float* snapshots) {
        q::RollbackRows rows;
        for (std::size_t i = 0; i < saved; ++i)
          rows.rows[i] =
              snapshots ? snapshots + (saved - 1 - i) * kPleHistory : nullptr;
        Upload(&ple_current, ple_history);
        q::PleConv(d_qkv.get(), d_conv_w.get(), ple_current.get(),
                   ple_scratch.get(), ple_out.get(), rows, n, kChannels,
                   kKernel, kDilation, nullptr);
      };
      ple(kTokens, ple_snaps.get());
      const auto saved_ple = Download(&ple_snaps, saved * kPleHistory + kGuard);
      const auto full_ple = Download(&ple_current, kPleHistory);
      const auto full_ple_out = Download(&ple_out, kQkvCount);
      for (std::size_t i = 0; i < kGuard; ++i) {
        if (saved_ple[saved * kPleHistory + i] != kSentinel)
          throw std::runtime_error("PLE wrote an unused final snapshot");
      }
      for (std::uint32_t keep = 1; keep <= kTokens; ++keep) {
        ple(keep, nullptr);
        const auto expected = Download(&ple_current, kPleHistory);
        const float* actual =
            keep == kTokens ? full_ple.data()
                            : saved_ple.data() + (saved - keep) * kPleHistory;
        if (std::memcmp(expected.data(), actual, kPleHistory * sizeof(float)))
          throw std::runtime_error("PLE rollback prefix changed history");
      }
      const auto plain_ple_out = Download(&ple_out, kQkvCount);
      if (std::memcmp(full_ple_out.data(), plain_ple_out.data(),
                      kQkvCount * sizeof(float)))
        throw std::runtime_error("PLE snapshots changed the full forward");
    }
    for (float v : outs[1]) {
      if (!std::isfinite(v)) {
        std::cerr << "row-split output is not finite\n";
        return 1;
      }
    }
    {
      // Double-precision recurrence over the conv output: the raw rows and
      // the final state of both routes against it.
      std::vector<float> raw_ref(kRaw);
      std::vector<float> state_ref(kStateCount);
      std::vector<double> S(static_cast<std::size_t>(kDim) * kDim);
      for (std::uint32_t h = 0; h < kVHeads; ++h) {
        const std::uint32_t kh = h % kKHeads;
        for (std::size_t i = 0; i < S.size(); ++i) {
          S[i] = state[h * S.size() + i];
        }
        for (std::uint32_t t = 0; t < kTokens; ++t) {
          const float* row = conv_out.data() + t * kChannels;
          const float* q = row + kh * kDim;
          const float* k = row + (kKHeads + kh) * kDim;
          const float* v = row + 2 * kKHeads * kDim + h * kDim;
          double qs = 0.0;
          double ks = 0.0;
          for (std::uint32_t i = 0; i < kDim; ++i) {
            qs += static_cast<double>(q[i]) * q[i];
            ks += static_cast<double>(k[i]) * k[i];
          }
          const double q_scale = 1.0 / std::sqrt(qs + kEps) / std::sqrt(128.0);
          const double inv_k = 1.0 / std::sqrt(ks + kEps);
          const double alpha = alpha_beta[t * 2 * kVHeads + h];
          const double beta_raw = alpha_beta[t * 2 * kVHeads + kVHeads + h];
          const double x = alpha + dt[h];
          const double softplus =
              std::max(x, 0.0) + std::log1p(std::exp(-std::abs(x)));
          const double g = std::exp(a[h] * softplus);
          const double beta = 1.0 / (1.0 + std::exp(-beta_raw));
          for (std::uint32_t r = 0; r < kDim; ++r) {
            double u = 0.0;
            for (std::uint32_t c = 0; c < kDim; ++c) {
              S[r * kDim + c] *= g;
              u += S[r * kDim + c] * k[c] * inv_k;
            }
            const double delta = (v[r] - u) * beta;
            double o = 0.0;
            for (std::uint32_t c = 0; c < kDim; ++c) {
              S[r * kDim + c] += delta * k[c] * inv_k;
              o += S[r * kDim + c] * q[c] * q_scale;
            }
            raw_ref[(t * kVHeads + h) * kDim + r] = static_cast<float>(o);
          }
        }
        for (std::size_t i = 0; i < S.size(); ++i) {
          state_ref[h * S.size() + i] = static_cast<float>(S[i]);
        }
      }
      // Per (token, head) row and per-head state: worst error over that
      // row's / head's own RMS (heads decay at very different rates).
      double worst_row[2] = {0.0, 0.0};
      double worst_head[2] = {0.0, 0.0};
      for (int route = 0; route < 2; ++route) {
        for (std::uint32_t tt = 0; tt < kTokens; ++tt) {
          for (std::uint32_t h = 0; h < kVHeads; ++h) {
            double ss = 0.0;
            double worst = 0.0;
            for (std::uint32_t v = 0; v < kDim; ++v) {
              const std::size_t idx = (tt * kVHeads + h) * kDim + v;
              ss += static_cast<double>(raw_ref[idx]) * raw_ref[idx];
              worst =
                  std::max(worst, std::abs(static_cast<double>(raw_ref[idx]) -
                                           raws[route][idx]));
            }
            worst_row[route] = std::max(worst_row[route],
                                        worst / std::sqrt(ss / kDim + 1e-30));
          }
        }
        for (std::uint32_t h = 0; h < kVHeads; ++h) {
          double ss = 0.0;
          double worst = 0.0;
          for (std::uint32_t i = 0; i < kDim * kDim; ++i) {
            const std::size_t idx = h * kDim * kDim + i;
            ss += static_cast<double>(state_ref[idx]) * state_ref[idx];
            worst =
                std::max(worst, std::abs(static_cast<double>(state_ref[idx]) -
                                         states[route][idx]));
          }
          worst_head[route] = std::max(
              worst_head[route], worst / std::sqrt(ss / (kDim * kDim) + 1e-30));
        }
      }
      std::cout << "  per row / head over own RMS: raw route0 " << worst_row[0]
                << " route1 " << worst_row[1] << "; state route0 "
                << worst_head[0] << " route1 " << worst_head[1] << '\n';
      // Both routes accumulate in F32: F64 agreement to accumulation order.
      if (std::max(std::max(worst_row[0], worst_row[1]),
                   std::max(worst_head[0], worst_head[1])) > 1e-3) {
        return 1;
      }
    }
    const double worst_out = WorstRelative(outs[0], outs[1]);
    const double worst_state = WorstRelative(states[0], states[1]);
    std::cout << "GDN row-split worst relative error: output " << worst_out
              << ", state " << worst_state << "; over RMS: output "
              << WorstOverRms(outs[0], outs[1]) << ", state "
              << WorstOverRms(states[0], states[1]) << '\n';
    if (worst_out > 1e-3 || worst_state > 1e-3) {
      return 1;
    }
    return 0;
  }
}

}  // namespace

int main() {
  try {
    // Short batches cover every rollback prefix and the no-snapshot case.
    // Longer batches exercise full and ragged recurrence windows.
    for (std::uint32_t n : {1u, 8u, 37u, 99u}) {
      std::cout << "n=" << n << '\n';
      if (RunCase(n) != 0) {
        return 1;
      }
    }
    if (RunCase(1, true) != 0 || RunCase(8, true) != 0)
      return 1;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
