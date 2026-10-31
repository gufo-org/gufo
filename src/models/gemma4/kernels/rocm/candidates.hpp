#ifndef GUFO_MODELS_GEMMA4_KERNELS_ROCM_CANDIDATES_HPP_
#define GUFO_MODELS_GEMMA4_KERNELS_ROCM_CANDIDATES_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/core/sampling.hpp"

namespace gufo::models::gemma4::rocm {

/// Logits ranked per row by LogitCandidates, and the most extra tokens whose
/// logits it gathers beside them.
inline constexpr std::uint32_t kCandidateTop = 64;
inline constexpr std::uint32_t kMaxCandidateExtra = 1024;
/// A row reports kCandidateOverflow when more logits share its threshold
/// histogram bin than the workspace holds; it then needs its full logits.
inline constexpr std::uint32_t kCandidateOverflow = 0xFFFFFFFFU;

/// Device workspace for up to `rows` rows. Its histograms and counters must
/// start zeroed; each launch leaves them zeroed again.
struct CandidateWork {
  std::uint32_t* histogram;
  std::uint32_t* counts;
  std::uint32_t* sure;
  std::uint64_t* boundary;
};
[[nodiscard]] std::size_t CandidateWorkBytes(std::uint32_t rows) noexcept;
/// Carves a workspace from `bytes` (CandidateWorkBytes(rows), 8-byte
/// aligned) and zeroes it on `stream`.
CandidateWork CandidateWorkAt(void* bytes, std::uint32_t rows,
                              hipStream_t stream);

/// Per logit row r of [rows][vocab]: the `top` largest finite logits in
/// sampling order (value, then lower token id; -0 equals +0), in no
/// particular order, at out[r * stride, + counts[r]), followed by the logits
/// of `extra` tokens at out[r * stride + top, + n_extra), where stride = top +
/// n_extra. counts[r] is below `top` only when the row has fewer finite
/// logits, and kCandidateOverflow when the row needs its full logits.
/// Requires vocab % 4 == 0, top <= kCandidateTop + kMaxCandidateExtra and
/// n_extra <= kMaxCandidateExtra.
void LogitCandidates(const float* logits, std::uint32_t rows,
                     std::uint32_t vocab, std::uint32_t top,
                     const std::uint32_t* extra, std::uint32_t n_extra,
                     CandidateWork work, std::uint32_t* counts,
                     sampling::LogitCandidate* out, hipStream_t stream);

}  // namespace gufo::models::gemma4::rocm

#endif  // GUFO_MODELS_GEMMA4_KERNELS_ROCM_CANDIDATES_HPP_
