#ifndef GUFO_MODELS_NEMOTRON35_BOLT_KERNELS_ROCM_SAMPLE_HPP_
#define GUFO_MODELS_NEMOTRON35_BOLT_KERNELS_ROCM_SAMPLE_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

/// Device summaries of logits rows (sampling::LogitHead), so that the host
/// samples from a few candidates instead of passing over the vocabulary.
namespace gufo::models::nemotron35_bolt::rocm {

/// The filters a head covers (sampling::LogitHeadFilter): the `keep` best
/// tokens, or, when keep is zero, at least `minimum` tokens and top_p of the
/// softmax mass at `temperature`.
struct LogitHeadParams {
  float temperature;
  int keep;
  float top_p;
  int minimum;
};

/// Candidates one head holds at most; a row that needs more reports no head
/// and is sampled from its full row.
inline constexpr int kLogitHeadCandidates = 1024;

/// One row's head: the softmax sum over every finite logit, the largest
/// finite logit, and its candidate count (zero: no head).
struct LogitHeadRecord {
  double softmax_sum;
  float maximum;
  int count;
};

struct LogitHeadCandidate {
  std::uint32_t token;
  float value;
};

/// Device scratch LogitHeads needs for `rows` rows of `vocab` logits.
[[nodiscard]] std::size_t LogitHeadScratchBytes(int rows, int vocab);

/// Summarizes `rows` rows of `vocab` logits: records[r], and the head's
/// tokens in any order at candidates[r * kLogitHeadCandidates]. The head is
/// every finite token within a logit bin of the cut the filters need, so it
/// holds the cut's ties. The softmax terms are F64; the bins choosing the cut
/// only bound it, and the host checks that the head covers its filters.
hipError_t LogitHeads(const float* logits, int rows, int vocab,
                      const LogitHeadParams& params, void* scratch,
                      LogitHeadRecord* records, LogitHeadCandidate* candidates,
                      hipStream_t stream);

}  // namespace gufo::models::nemotron35_bolt::rocm

#endif  // GUFO_MODELS_NEMOTRON35_BOLT_KERNELS_ROCM_SAMPLE_HPP_
