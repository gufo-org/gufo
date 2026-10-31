#include "src/core/sampling_logit_row.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <functional>
#include <stdexcept>

namespace gufo::sampling {
namespace {

constexpr float kApproxFloor = -40.0F;
/// Blocks whose every term is below exp(kSkipExponent) are not summed; the
/// interval widens by their bound instead.
constexpr double kSkipExponent = -28.0;

/// (logit - maximum) * inverse, raised to kApproxFloor when below it or NaN.
inline float ClampedExponent(float logit, float maximum,
                             float inverse) noexcept {
  const float x = (logit - maximum) * inverse;
  return x > kApproxFloor ? x : kApproxFloor;
}

// The error figures below come from an exhaustive float sweep (exp_check) and
// hold only for codegen without FMA contraction or fast-math; CMake compiles
// this file with -ffp-contract=off.

/// Cephes expf for x in [kApproxFloor, 0]; relative error below 1e-7.
inline float AccurateExp(float x) noexcept {
  const int n = static_cast<int>(x * 1.44269504F - 0.5F);
  const float nf = static_cast<float>(n);
  float r = x - nf * 0.693359375F;
  r = r - nf * -2.12194440e-4F;
  float p = 1.9875691500e-4F;
  p = p * r + 1.3981999507e-3F;
  p = p * r + 8.3334519073e-3F;
  p = p * r + 4.1665795894e-2F;
  p = p * r + 1.6666665459e-1F;
  p = p * r + 5.0000001201e-1F;
  p = p * (r * r) + r + 1.0F;
  return p * std::bit_cast<float>(static_cast<std::int32_t>((n + 127) << 23));
}

/// exp(x) with 2^f replaced by 1 + f: the exponent field of the float built
/// from x * log2(e) is 2^n * (1 + f). For x in [kApproxFloor, 0] it is at most
/// 6.2% above and 1e-5 below exp(x).
inline float CoarseExp(float x) noexcept {
  return std::bit_cast<float>(
      static_cast<std::int32_t>((x * 1.44269504F + 127.0F) * 8388608.0F));
}

double LaneSum(const std::array<float, kSkipBlock>& terms) noexcept {
  constexpr std::size_t kLanes = 8;
  std::array<float, kLanes> lanes{};
  for (std::size_t i = 0; i < kSkipBlock; i += kLanes)
    for (std::size_t lane = 0; lane < kLanes; ++lane)
      lanes[lane] += terms[i + lane];
  double sum = 0.0;
  for (const float lane : lanes)
    sum += lane;
  return sum;
}

/// Sum of exp((logit - maximum) * inverse) over one block of logits, with
/// plain loops over a fixed block size so the compiler vectorizes them.
double AccurateBlockSum(const float* logits, float maximum,
                        float inverse) noexcept {
  std::array<float, kSkipBlock> terms;
  for (std::size_t i = 0; i < kSkipBlock; ++i)
    terms[i] = AccurateExp(ClampedExponent(logits[i], maximum, inverse));
  return LaneSum(terms);
}

double CoarseBlockSum(const float* logits, float maximum,
                      float inverse) noexcept {
  std::array<float, kSkipBlock> terms;
  for (std::size_t i = 0; i < kSkipBlock; ++i)
    terms[i] = CoarseExp(ClampedExponent(logits[i], maximum, inverse));
  return LaneSum(terms);
}

}  // namespace

LogitRow::LogitRow(std::span<const float> logits, const SamplingConfig& config,
                   std::span<const TokenPenalty> penalties,
                   LogitRowScratch* scratch)
    : logits_(logits),
      config_(config),
      penalties_(penalties),
      scratch_(scratch) {
  Scan();
}

void LogitRow::Collect(std::size_t limit, std::vector<Probability>* out,
                       bool sorted) const {
  out->clear();
  double threshold = -std::numeric_limits<double>::infinity();
  if (limit < finite_blocks_) {
    // Each of the `limit` strongest blocks holds a distinct element at least
    // as large as its maximum, so the limit-th best element is >= the
    // limit-th best block maximum. The scan below needs block_max_ in index
    // order, so the selection reorders a copy.
    auto& maxima = scratch_->block_maxima;
    maxima.assign(block_max_.begin(), block_max_.end());
    std::nth_element(maxima.begin(),
                     maxima.begin() + static_cast<std::ptrdiff_t>(limit - 1),
                     maxima.end(), std::greater<>());
    threshold = maxima[limit - 1];
  } else if (limit < logits_.size() / 2) {
    threshold = HistogramThreshold(limit);
  }
  auto penalty = penalties_.begin();
  for (std::size_t block = 0; block < block_max_.size(); ++block) {
    const double block_max = block_max_[block];
    if (block_max < threshold ||
        block_max == -std::numeric_limits<double>::infinity())
      continue;
    const std::size_t begin = block * kSkipBlock;
    const std::size_t end = std::min(begin + kSkipBlock, logits_.size());
    for (std::size_t index = begin; index < end; ++index) {
      if (!std::isfinite(logits_[index]))
        continue;
      const double value = Adjusted(index, penalty);
      if (value >= threshold)
        out->push_back({.token = static_cast<TokenId>(index), .value = value});
    }
  }
  if (out->size() > limit) {
    std::nth_element(out->begin(),
                     out->begin() + static_cast<std::ptrdiff_t>(limit),
                     out->end(), IsBetterProbability);
    out->resize(limit);
  }
  if (sorted)
    std::ranges::sort(*out, IsBetterProbability);
}

double LogitRow::ExactSoftmaxSum() const {
  double sum = 0.0;
  auto penalty = penalties_.begin();
  for (std::size_t index = 0; index < logits_.size(); ++index) {
    if (std::isfinite(logits_[index]))
      sum +=
          std::exp((Adjusted(index, penalty) - maximum_) / config_.temperature);
  }
  if (!(sum > 0.0) || !std::isfinite(sum))
    throw std::runtime_error("logit softmax normalization failed");
  return sum;
}

std::optional<LogitRow::Interval> LogitRow::ApproxSoftmaxSum() const {
  const float temperature = config_.temperature;
  // The bounds assume x = (logit - maximum) / T stays within float range.
  if (!(temperature >= 1e-4F && temperature <= 1e4F))
    return std::nullopt;
  // Unpenalized terms are formed against a float no smaller than the
  // maximum, so x <= 0, and rescaled by exp((maximum - maximum_) / T). The
  // clamp and skip bounds below assume that shift is negligible.
  float maximum = static_cast<float>(maximum_);
  if (static_cast<double>(maximum) < maximum_)
    maximum = std::nextafter(maximum, std::numeric_limits<float>::infinity());
  const double shift = (static_cast<double>(maximum) - maximum_) /
                       static_cast<double>(temperature);
  if (!(shift <= 1e-3))
    return std::nullopt;
  const double rescale = std::exp(shift);
  const float inverse =
      static_cast<float>(1.0 / static_cast<double>(temperature));
  // Blocks whose coarse sum is small are kept as the coarse sum; the rest
  // are recomputed accurately. Together with the head those are the only
  // terms that matter for a peaked row.
  constexpr double kCoarseBlockLimit = 1e-6;
  double accurate = 0.0;
  double coarse = 0.0;
  double exact = 0.0;
  std::size_t skipped = 0;
  auto penalty = penalties_.begin();
  for (std::size_t block = 0; block < block_max_.size(); ++block) {
    const double block_max = block_max_[block];
    if (block_max == -std::numeric_limits<double>::infinity())
      continue;
    const std::size_t begin = block * kSkipBlock;
    const std::size_t end = std::min(begin + kSkipBlock, logits_.size());
    if (!scalar_block_[block] && end - begin == kSkipBlock) {
      if ((block_max - maximum_) / static_cast<double>(temperature) <
          kSkipExponent) {
        ++skipped;
        continue;
      }
      const double estimate =
          CoarseBlockSum(logits_.data() + begin, maximum, inverse);
      if (estimate <= kCoarseBlockLimit)
        coarse += estimate;
      else
        accurate += AccurateBlockSum(logits_.data() + begin, maximum, inverse);
      continue;
    }
    for (std::size_t index = begin; index < end; ++index) {
      if (std::isfinite(logits_[index]))
        exact += std::exp((Adjusted(index, penalty) - maximum_) /
                          config_.temperature);
    }
  }
  // Relative error of the accurate terms, each bounded by u = 2^-24:
  // AccurateExp is within 1e-7 of exp (checked over every float in
  // [kApproxFloor, 0] by exp_check); x = (logit - maximum) * inverse carries
  // three roundings, 3u * |x|; and the |x|-weighted terms sum to at most
  // ln(N) * S, because they are probabilities of N tokens (Jensen, S >= 1).
  // Float lane sums of eight terms add 7u. The slack factor is 1.5.
  // Coarse terms lie within 6.2% of exp(x) before the same rounding.
  // Clamped terms add under 1e-12.
  constexpr double kUnit = 0x1p-24;
  const double accurate_error =
      1.5 * (1e-7 + 7 * kUnit +
             3 * kUnit * std::log(static_cast<double>(logits_.size())));
  constexpr double kCoarseError = 0.065;
  accurate *= rescale;
  coarse *= rescale;
  const double sum = accurate + coarse + exact;
  const double error = accurate_error * sum + kCoarseError * coarse;
  const double dropped =
      static_cast<double>(skipped * kSkipBlock) * std::exp(kSkipExponent);
  return Interval{.low = sum - error, .high = sum + error + dropped};
}

double LogitRow::Adjusted(std::size_t index, PenaltyIt& penalty) const {
  while (penalty != penalties_.end() && penalty->token < index)
    ++penalty;
  if (penalty == penalties_.end() || penalty->token != index)
    return static_cast<double>(logits_[index]);
  const double adjusted = Penalize(logits_[index], config_, *penalty);
  if (!std::isfinite(adjusted))
    throw std::runtime_error("sampling penalties produced a non-finite logit");
  return adjusted;
}

/// A value no larger than the limit-th best adjusted logit, found from a
/// histogram of the unpenalized logits' leading 16 bits. Penalized tokens are
/// left out of the count, which only lowers the bound.
double LogitRow::HistogramThreshold(std::size_t limit) const {
  constexpr std::size_t kBins = 1U << 16;
  auto& counts = scratch_->histogram;
  counts.assign(kBins, 0);
  std::size_t next = 0;
  const auto penalized = [&](std::size_t index) {
    while (next < penalties_.size() && penalties_[next].token < index)
      ++next;
    return next < penalties_.size() && penalties_[next].token == index;
  };
  for (std::size_t block = 0; block < block_max_.size(); ++block) {
    if (block_max_[block] == -std::numeric_limits<double>::infinity())
      continue;
    const std::size_t begin = block * kSkipBlock;
    const std::size_t end = std::min(begin + kSkipBlock, logits_.size());
    for (std::size_t index = begin; index < end; ++index) {
      const float value = logits_[index];
      if (!std::isfinite(value) || penalized(index))
        continue;
      // -0.0 and +0.0 compare equal, so they must share a bin.
      const auto bits = std::bit_cast<std::uint32_t>(value + 0.0F);
      const std::uint32_t key =
          (bits & 0x80000000U) != 0 ? ~bits : bits | 0x80000000U;
      ++counts[key >> 16];
    }
  }
  std::size_t seen = 0;
  for (std::size_t bin = kBins; bin-- > 0;) {
    seen += counts[bin];
    if (seen >= limit) {
      // Bin 0 holds only negative NaNs, which the counting loop skips.
      const std::uint32_t key = static_cast<std::uint32_t>(bin) << 16;
      const std::uint32_t bits =
          (key & 0x80000000U) != 0 ? key & 0x7fffffffU : ~key;
      return static_cast<double>(std::bit_cast<float>(bits));
    }
  }
  return -std::numeric_limits<double>::infinity();
}

void LogitRow::Scan() {
  const std::size_t blocks = (logits_.size() + kSkipBlock - 1) / kSkipBlock;
  block_max_.assign(blocks, -std::numeric_limits<double>::infinity());
  scalar_block_.assign(blocks, 0);
  auto penalty = penalties_.begin();
  for (std::size_t block = 0; block < blocks; ++block) {
    const std::size_t begin = block * kSkipBlock;
    const std::size_t end = std::min(begin + kSkipBlock, logits_.size());
    while (penalty != penalties_.end() && penalty->token < begin)
      ++penalty;
    const bool penalized = penalty != penalties_.end() && penalty->token < end;
    if (!penalized) {
      const float maximum = BlockMaximum(logits_.subspan(begin, end - begin));
      // +inf is not a finite logit; any other maximum is exact.
      if (maximum != std::numeric_limits<float>::infinity()) {
        block_max_[block] = static_cast<double>(maximum);
        continue;
      }
    }
    scalar_block_[block] = 1;
    double block_max = -std::numeric_limits<double>::infinity();
    for (std::size_t index = begin; index < end; ++index) {
      if (std::isfinite(logits_[index]))
        block_max = std::max(block_max, Adjusted(index, penalty));
    }
    block_max_[block] = block_max;
  }
  for (const double block_max : block_max_) {
    if (block_max != -std::numeric_limits<double>::infinity()) {
      ++finite_blocks_;
      maximum_ = std::max(maximum_, block_max);
    }
  }
  if (finite_blocks_ == 0)
    throw std::runtime_error("logit distribution contains no finite values");
}

}  // namespace gufo::sampling
