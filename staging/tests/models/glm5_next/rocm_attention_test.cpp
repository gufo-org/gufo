#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "staging/models/glm5_next/kernels/rocm/attention.hpp"
#include "staging/tests/models/glm5_next/hip_test.hpp"

// Parity checks for the glm5_next absorbed-MLA gather attention against
// independent host double references: query absorption (attn_k_b), the
// gathered-latent score pass over the selection, the softmax-weighted
// latent combination and the per-head attn_v_b projection. The references
// mirror the kernel contracts (F32 double-accumulated score dots feeding
// the chain, softmax with exact max, latent sums in selection order); the
// WMMA F16 score dots sum in tile order, so the score deviation enters
// latent/out through the tolerances below, and the exact single-slot case
// locks the degenerate softmax. Synthetic inputs only.
namespace {

namespace q = gufo::models::glm5_next::rocm;
namespace t = gufo::tests::glm5_next;

constexpr std::uint32_t kHeads = 64U;
constexpr std::uint32_t kVbOut = 256U;
constexpr std::uint32_t kTile = 128U;
constexpr std::uint32_t kCapacity = 4096U;

bool Check(const char* what, double worst, double limit) {
  const bool ok = worst <= limit;
  std::printf("%-34s worst %.3e limit %.0e %s\n", what, worst, limit,
              ok ? "OK" : "FAIL");
  return ok;
}

std::uint16_t HalfBits(float value) {
  const __half bits = __float2half_rn(value);
  std::uint16_t out = 0U;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

const __half* HalfPtr(t::HipBuffer<std::uint16_t>* buffer) {
  return reinterpret_cast<const __half*>(buffer->get());
}

float HalfValue(std::uint16_t bits) {
  __half value{};
  std::memcpy(&value, &bits, sizeof(value));
  return __half2float(value);
}

std::vector<std::int32_t> MakeSelection(std::uint32_t n_sel,
                                        std::uint32_t capacity,
                                        std::uint32_t seed) {
  std::vector<std::int32_t> indices(capacity);
  std::mt19937 gen(seed);
  for (std::uint32_t i = 0; i < capacity; ++i) {
    indices[i] = static_cast<std::int32_t>(i);
  }
  for (std::uint32_t i = 0; i < n_sel; ++i) {
    std::uniform_int_distribution<std::uint32_t> pick(i, capacity - 1U);
    std::swap(indices[i], indices[pick(gen)]);
  }
  indices.resize(n_sel);
  return indices;
}

struct Chain {
  std::vector<float> q_abs_ref;   // [heads][512]
  std::vector<float> latent_ref;  // [heads][512]
  std::vector<float> out_ref;     // [heads][256]
};

// Doubles-only mirror of the full kernel chain (absorption excluded;
// `q_abs` supplies the absorbed queries), in the kernel summation order.
Chain ReferenceChain(const std::vector<float>& scores_ref,
                     const std::vector<float>& cache_ref,
                     const std::vector<std::int32_t>& sel, std::uint32_t n_sel,
                     std::uint32_t tile, const std::vector<float>& vb) {
  Chain chain;
  chain.latent_ref.assign(kHeads * q::kMlaKvRank, 0.0F);
  chain.out_ref.assign(kHeads * kVbOut, 0.0F);
  const std::uint32_t tiles = q::MlaGatherTiles(n_sel, tile);
  for (std::uint32_t head = 0; head < kHeads; ++head) {
    double max_s = -std::numeric_limits<double>::infinity();
    for (std::uint32_t j = 0; j < n_sel; ++j) {
      max_s = std::max(
          max_s, static_cast<double>(scores_ref[(j / tile) * (kHeads * tile) +
                                                head * tile + (j % tile)]));
    }
    std::vector<double> ex(n_sel);
    double sumexp = 0.0;
    for (std::uint32_t j = 0; j < n_sel; ++j) {
      ex[j] =
          std::exp(static_cast<double>(scores_ref[(j / tile) * (kHeads * tile) +
                                                  head * tile + (j % tile)]) -
                   max_s);
      sumexp += ex[j];
    }
    for (std::uint32_t i = 0; i < q::kMlaKvRank; ++i) {
      double acc = 0.0;
      for (std::uint32_t j = 0; j < n_sel; ++j) {
        acc += ex[j] *
               cache_ref[static_cast<std::size_t>(sel[j]) * q::kMlaKvRank + i];
      }
      chain.latent_ref[head * q::kMlaKvRank + i] =
          static_cast<float>(acc / sumexp);
    }
    for (std::uint32_t o = 0; o < kVbOut; ++o) {
      const std::size_t row = head * kVbOut + o;
      double acc = 0.0;
      for (std::uint32_t i = 0; i < q::kMlaKvRank; ++i) {
        acc += static_cast<double>(vb[row * q::kMlaKvRank + i]) *
               static_cast<double>(chain.latent_ref[head * q::kMlaKvRank + i]);
      }
      chain.out_ref[row] = static_cast<float>(acc);
    }
  }
  (void)tiles;
  return chain;
}

bool TestQabs() {
  const auto query = t::MakeValues(kHeads * kVbOut, 0xF00D1234U, 1.0F);
  const auto kb =
      t::MakeValues(kHeads * q::kMlaKvRank * kVbOut, 0x13579ACEU, 0.05F);
  const float scale = 1.0F / std::sqrt(static_cast<float>(q::kMlaKvRank));
  std::vector<float> ref(kHeads * q::kMlaKvRank, 0.0F);
  for (std::uint32_t head = 0; head < kHeads; ++head) {
    for (std::uint32_t i = 0; i < q::kMlaKvRank; ++i) {
      double acc = 0.0;
      for (std::uint32_t l = 0; l < kVbOut; ++l) {
        acc +=
            static_cast<double>(kb[(head * q::kMlaKvRank + i) * kVbOut + l]) *
            static_cast<double>(query[head * kVbOut + l]);
      }
      ref[head * q::kMlaKvRank + i] = static_cast<float>(acc) * scale;
    }
  }
  t::HipBuffer<float> d_q(query.size()), d_kb(kb.size()), d_out(ref.size());
  t::Upload(&d_q, query);
  t::Upload(&d_kb, kb);
  q::MlaQabs(d_q.get(), d_kb.get(), kHeads, scale, d_out.get(), nullptr);
  t::CheckHip(hipDeviceSynchronize(), "MlaQabs sync");
  return Check("mla qabs 64x512x256",
               t::WorstRelative(ref, t::Download(&d_out, ref.size())), 2e-5);
}

bool RunGatherCase(const char* name, std::uint32_t n_sel, std::uint32_t tile,
                   std::uint32_t seed) {
  const auto cache_f32 = t::MakeValues(
      static_cast<std::size_t>(kCapacity) * q::kMlaKvRank, seed + 1U, 1.0F);
  std::vector<std::uint16_t> cache_bits(cache_f32.size());
  std::vector<float> cache_ref(cache_f32.size());
  for (std::size_t i = 0; i < cache_f32.size(); ++i) {
    cache_bits[i] = HalfBits(cache_f32[i]);
    cache_ref[i] = HalfValue(cache_bits[i]);
  }
  const auto sel = MakeSelection(n_sel, kCapacity, seed + 2U);
  const auto q_abs = t::MakeValues(kHeads * q::kMlaKvRank, seed + 3U, 0.02F);
  const auto vb =
      t::MakeValues(kHeads * kVbOut * q::kMlaKvRank, seed + 4U, 0.05F);

  const std::uint32_t tiles = q::MlaGatherTiles(n_sel, tile);
  std::vector<float> scores_ref(static_cast<std::size_t>(tiles) * kHeads * tile,
                                0.0F);
  for (std::uint32_t tb = 0; tb < tiles; ++tb) {
    for (std::uint32_t head = 0; head < kHeads; ++head) {
      for (std::uint32_t l = 0; l < tile; ++l) {
        const std::uint32_t j = tb * tile + l;
        if (j >= n_sel) {
          continue;
        }
        double acc = 0.0;
        const float* qa =
            &q_abs[static_cast<std::size_t>(head) * q::kMlaKvRank];
        const float* row =
            &cache_ref[static_cast<std::size_t>(sel[j]) * q::kMlaKvRank];
        for (std::uint32_t i = 0; i < q::kMlaKvRank; ++i) {
          acc += static_cast<double>(qa[i]) * static_cast<double>(row[i]);
        }
        scores_ref[(static_cast<std::size_t>(tb) * kHeads + head) * tile + l] =
            static_cast<float>(acc);
      }
    }
  }
  const Chain chain =
      ReferenceChain(scores_ref, cache_ref, sel, n_sel, tile, vb);

  t::HipBuffer<float> d_qabs(q_abs.size()), d_latent(chain.latent_ref.size()),
      d_out(chain.out_ref.size()), d_scores(scores_ref.size()), d_vb(vb.size());
  t::HipBuffer<std::uint16_t> d_cache(cache_bits.size());
  t::HipBuffer<std::uint16_t> d_qpack(2U * kHeads * q::kMlaKvRank);
  t::HipBuffer<std::int32_t> d_sel(sel.size());
  t::Upload(&d_qabs, q_abs);
  t::Upload(&d_vb, vb);
  t::Upload(&d_scores, scores_ref);  // scratch; overwritten below
  t::CheckHip(
      hipMemcpy(d_sel.get(), sel.data(), sel.size() * sizeof(std::int32_t),
                hipMemcpyHostToDevice),
      "upload sel");
  t::CheckHip(hipMemcpy(d_cache.get(), cache_bits.data(),
                        cache_bits.size() * sizeof(std::uint16_t),
                        hipMemcpyHostToDevice),
              "upload cache");
  q::MlaGatherScore(d_qabs.get(),
                    reinterpret_cast<__half*>(d_qpack.get()), HalfPtr(&d_cache),
                    d_sel.get(), n_sel, tile, kHeads, d_scores.get(), nullptr);
  q::MlaGatherCombine(d_scores.get(), HalfPtr(&d_cache), d_sel.get(), tile,
                      n_sel, kHeads, d_latent.get(), nullptr);
  q::MlaApplyVb(d_latent.get(), d_vb.get(), kHeads, d_out.get(), nullptr);
  t::CheckHip(hipDeviceSynchronize(), "gather chain sync");
  const auto got_latent = t::Download(&d_latent, chain.latent_ref.size());
  const auto got_out = t::Download(&d_out, chain.out_ref.size());
  bool ok = Check((std::string(name) + " latent").c_str(),
                  t::WorstRelative(chain.latent_ref, got_latent), 2e-4);
  // WMMA split-q score dots leave a ~2^-22-of-q residual that the
  // softmax amplifies on near-zero latent elements; the head output
  // through the vb dot stays an order of magnitude below the 1e-3 gate.
  // The oracle out uses the oracle latent; float latent error (~1e-4)
  // legitimately propagates through the 512-term vb dot.
  ok &= Check((std::string(name) + " out").c_str(),
              t::WorstRelative(chain.out_ref, got_out), 1e-3);
  return ok;
}

bool TestGather() {
  bool ok = RunGatherCase("gather 2048/128", 2048U, kTile, 0xB0B5E000U);
  ok &= RunGatherCase("gather 777/128 tail", 777U, kTile, 0xC0FFEE11U);
  ok &= RunGatherCase("gather 128/1024", 128U, 1024U, 0xD1CE0B0EU);
  const std::uint32_t tiles_checks = (q::MlaGatherTiles(777U, 128U) == 7U &&
                                      q::MlaGatherTiles(2048U, 128U) == 16U &&
                                      q::MlaGatherTiles(1U, 128U) == 1U)
                                         ? 0U
                                         : 1U;
  ok &= Check("gather tiles", static_cast<double>(tiles_checks), 0.0);
  return ok;
}

// Exact single-slot case: softmax degenerates to p = 1 so the latent must
// match the half-precision cache row bit-for-bit.
bool TestGatherSingleExact() {
  const auto cache_f32 = t::MakeValues(
      static_cast<std::size_t>(kCapacity) * q::kMlaKvRank, 0x5EED1234U, 1.0F);
  std::vector<std::uint16_t> cache_bits(cache_f32.size());
  for (std::size_t i = 0; i < cache_f32.size(); ++i) {
    cache_bits[i] = HalfBits(cache_f32[i]);
  }
  const std::vector<std::int32_t> sel{17, 991, 2050, 4095};
  const std::uint32_t n_sel = static_cast<std::uint32_t>(sel.size());
  const std::uint32_t tile = 128U;
  // A query proportional to the first selected row with a huge gap: the
  // softmax exponentials of the other slots underflow expf to exactly 0,
  // so the latent must reproduce slot 0's half row bit for bit.
  std::vector<float> q_abs(static_cast<std::size_t>(kHeads) * q::kMlaKvRank);
  for (std::size_t i = 0; i < q_abs.size(); ++i) {
    q_abs[i] =
        100.0F * cache_f32[(static_cast<std::size_t>(sel[0]) * q::kMlaKvRank) +
                           (i % q::kMlaKvRank)];
  }
  const std::uint32_t tiles = q::MlaGatherTiles(n_sel, tile);

  const std::size_t scores_host_size =
      static_cast<std::size_t>(tiles) * kHeads * tile;
  std::vector<float> scores_host(scores_host_size, 0.0F);
  t::HipBuffer<float> d_qabs(q_abs.size()), d_scores(scores_host_size),
      d_latent(kHeads * q::kMlaKvRank);
  t::HipBuffer<std::uint16_t> d_cache(cache_bits.size());
  t::HipBuffer<std::uint16_t> d_qpack(2U * kHeads * q::kMlaKvRank);
  t::HipBuffer<std::int32_t> d_sel(sel.size());
  t::Upload(&d_qabs, q_abs);
  t::CheckHip(
      hipMemcpy(d_sel.get(), sel.data(), sel.size() * sizeof(std::int32_t),
                hipMemcpyHostToDevice),
      "upload sel");
  t::CheckHip(hipMemcpy(d_cache.get(), cache_bits.data(),
                        cache_bits.size() * sizeof(std::uint16_t),
                        hipMemcpyHostToDevice),
              "upload cache");
  q::MlaGatherScore(d_qabs.get(),
                    reinterpret_cast<__half*>(d_qpack.get()), HalfPtr(&d_cache),
                    d_sel.get(), n_sel, tile, kHeads, d_scores.get(), nullptr);
  q::MlaGatherCombine(d_scores.get(), HalfPtr(&d_cache), d_sel.get(), tile,
                      n_sel, kHeads, d_latent.get(), nullptr);
  t::CheckHip(hipDeviceSynchronize(), "gather single sync");
  const auto got = t::Download(&d_latent, kHeads * q::kMlaKvRank);
  std::vector<float> got_scores(scores_host.size());
  t::CheckHip(
      hipMemcpy(got_scores.data(), d_scores.get(),
                got_scores.size() * sizeof(float), hipMemcpyDeviceToHost),
      "download scores");
  std::uint32_t wrong = 0U;
  const std::uint16_t* row =
      &cache_bits[static_cast<std::size_t>(sel[0]) * q::kMlaKvRank];
  for (std::uint32_t head = 0; head < kHeads; ++head) {
    // Slot 0 must win its head's max outright by a margin that zeroes the
    // other exponentials (checked via the kernel's own score buffer).
    const float* sc = &got_scores[static_cast<std::size_t>(head) * tile];
    if (sc[0] <= sc[1] || sc[0] <= sc[2] || sc[0] <= sc[3]) {
      wrong += q::kMlaKvRank;
    }
    for (std::uint32_t i = 0; i < q::kMlaKvRank; ++i) {
      if (got[head * q::kMlaKvRank + i] != HalfValue(row[i])) {
        ++wrong;
      }
    }
  }
  return Check("gather single exact row", static_cast<double>(wrong), 0.0);
}

double Nmse(const std::vector<float>& ref, const std::vector<float>& got) {
  double num = 0.0;
  double den = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double d = static_cast<double>(ref[i]) - static_cast<double>(got[i]);
    num += d * d;
    den += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
  }
  return num / den;
}

// Batched fused attention against the doubles oracle: token t of a chunk
// selects kpool * min(top_pools, (pos+1)/kpool) + (use_tail ? (pos+1) %
// kpool : 0) distinct cache rows, matching the DsaSelCount contract the
// batch pair uses. The fused online softmax (base 2, F16 probabilities)
// reorders the sums, so the gate is NMSE-level, not the batch pair's
// per-element tolerance.
bool RunFusedCase(const char* name, std::uint32_t pos_first,
                  std::uint32_t kpool, std::uint32_t top_pools,
                  bool use_tail, std::uint32_t tokens, std::uint32_t seed) {
  const auto cache_f32 = t::MakeValues(
      static_cast<std::size_t>(kCapacity) * q::kMlaKvRank, seed + 1U, 1.0F);
  std::vector<std::uint16_t> cache_bits(cache_f32.size());
  std::vector<float> cache_ref(cache_f32.size());
  for (std::size_t i = 0; i < cache_f32.size(); ++i) {
    cache_bits[i] = HalfBits(cache_f32[i]);
    cache_ref[i] = HalfValue(cache_bits[i]);
  }
  const std::size_t qa = static_cast<std::size_t>(kHeads) * q::kMlaKvRank;
  const auto q_abs = t::MakeValues(tokens * qa, seed + 3U, 0.02F);
  const auto vb =
      t::MakeValues(kHeads * kVbOut * q::kMlaKvRank, seed + 4U, 0.05F);

  std::vector<std::uint32_t> n_sel(tokens);
  std::uint32_t n_max = 0U;
  for (std::uint32_t r = 0; r < tokens; ++r) {
    const std::uint32_t seen = pos_first + r + 1U;
    const std::uint32_t k =
        std::min(seen / kpool, top_pools);
    n_sel[r] =
        k * kpool + (use_tail ? seen % kpool : 0U);
    n_max = std::max(n_max, n_sel[r]);
  }
  std::vector<std::int32_t> sel(static_cast<std::size_t>(tokens) * n_max, -1);
  std::vector<float> latent_ref(tokens * qa, 0.0F);
  for (std::uint32_t r = 0; r < tokens; ++r) {
    const auto rows = MakeSelection(n_sel[r], kCapacity, seed + 10U + r);
    for (std::uint32_t j = 0; j < n_sel[r]; ++j) {
      sel[static_cast<std::size_t>(r) * n_max + j] = rows[j];
    }
    // Oracle in the [tiles][heads][tile] layout consumed by
    // ReferenceChain; the fused kernel never materializes scores.
    const std::uint32_t tiles = q::MlaGatherTiles(n_sel[r], kTile);
    std::vector<float> scores_ref(static_cast<std::size_t>(tiles) * kHeads *
                                      kTile,
                                  0.0F);
    const float* qa_t = &q_abs[static_cast<std::size_t>(r) * qa];
    for (std::uint32_t j = 0; j < n_sel[r]; ++j) {
      for (std::uint32_t head = 0; head < kHeads; ++head) {
        double acc = 0.0;
        const float* qh = &qa_t[static_cast<std::size_t>(head) * q::kMlaKvRank];
        const float* row =
            &cache_ref[static_cast<std::size_t>(rows[j]) * q::kMlaKvRank];
        for (std::uint32_t i = 0; i < q::kMlaKvRank; ++i) {
          acc += static_cast<double>(qh[i]) * static_cast<double>(row[i]);
        }
        scores_ref[(static_cast<std::size_t>(j / kTile) * kHeads + head) *
                       kTile +
                   (j % kTile)] = static_cast<float>(acc);
      }
    }
    const Chain chain =
        ReferenceChain(scores_ref, cache_ref, rows, n_sel[r], kTile, vb);
    std::copy(chain.latent_ref.begin(), chain.latent_ref.end(),
              latent_ref.begin() + static_cast<std::ptrdiff_t>(r * qa));
  }

  t::HipBuffer<float> d_qabs(q_abs.size()), d_latent(latent_ref.size());
  t::HipBuffer<std::uint16_t> d_cache(cache_bits.size());
  t::HipBuffer<std::uint16_t> d_qpack(2U * tokens * qa);
  t::HipBuffer<std::int32_t> d_sel(sel.size());
  t::Upload(&d_qabs, q_abs);
  t::CheckHip(hipMemcpy(d_sel.get(), sel.data(), sel.size() * sizeof(std::int32_t),
                        hipMemcpyHostToDevice),
              "upload fused sel");
  t::CheckHip(hipMemcpy(d_cache.get(), cache_bits.data(),
                        cache_bits.size() * sizeof(std::uint16_t),
                        hipMemcpyHostToDevice),
              "upload fused cache");
  q::MlaFusedAttend(d_qabs.get(), qa, reinterpret_cast<__half*>(d_qpack.get()),
                    HalfPtr(&d_cache), d_sel.get(), n_max, pos_first, kpool,
                    top_pools, use_tail ? 1U : 0U, tokens, kHeads,
                    d_latent.get(), qa, nullptr);
  t::CheckHip(hipDeviceSynchronize(), "fused sync");
  return Check(name, Nmse(latent_ref, t::Download(&d_latent, latent_ref.size())),
               1e-5);
}

bool TestFused() {
  bool ok = RunFusedCase("fused 5tok/8pools", 0U, 4U, 8U, true, 5U,
                         0xA5A50001U);
  ok &= RunFusedCase("fused 3tok/no-tail", 4U, 4U, 3U, false, 3U,
                     0xA5A50002U);
  ok &= RunFusedCase("fused 2tok/257sel", 256U, 4U, 64U, true, 2U,
                     0xA5A50003U);
  return ok;
}

bool TestApplyVb() {
  const auto latent = t::MakeValues(kHeads * q::kMlaKvRank, 0x0F0F1234U, 1.0F);
  const auto vb =
      t::MakeValues(kHeads * kVbOut * q::kMlaKvRank, 0x9A9A7777U, 0.05F);
  std::vector<float> ref(kHeads * kVbOut, 0.0F);
  for (std::uint32_t head = 0; head < kHeads; ++head) {
    for (std::uint32_t o = 0; o < kVbOut; ++o) {
      const std::size_t row = head * kVbOut + o;
      double acc = 0.0;
      for (std::uint32_t i = 0; i < q::kMlaKvRank; ++i) {
        acc += static_cast<double>(vb[row * q::kMlaKvRank + i]) *
               static_cast<double>(latent[head * q::kMlaKvRank + i]);
      }
      ref[row] = static_cast<float>(acc);
    }
  }
  t::HipBuffer<float> d_latent(latent.size()), d_vb(vb.size()),
      d_out(ref.size());
  t::Upload(&d_latent, latent);
  t::Upload(&d_vb, vb);
  q::MlaApplyVb(d_latent.get(), d_vb.get(), kHeads, d_out.get(), nullptr);
  t::CheckHip(hipDeviceSynchronize(), "MlaApplyVb sync");
  // Output rms ~1.13 with 512-term sums: near-cancelling rows dominate
  // the floor-1e-3 relative metric, so the floor tracks the rms scale.
  return Check("mla apply_vb",
               t::WorstRelative(ref, t::Download(&d_out, ref.size()), 1e-2),
               1e-4);
}

}  // namespace

int main() {
  try {
    bool ok = true;
    ok &= TestQabs();
    ok &= TestGather();
    ok &= TestGatherSingleExact();
    ok &= TestFused();
    ok &= TestApplyVb();
    return ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::printf("EXCEPTION: %s\n", error.what());
    return 2;
  }
}
