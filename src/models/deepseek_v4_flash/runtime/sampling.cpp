#include <gufo/ext/sampling.hpp>

#include <cmath>
#include <limits>

#include "native_internal.h"

int ds4_sample_argmax(const float* logits, uint32_t vocabulary_size) {
  if (logits == nullptr)
    return -1;
  int best = -1;
  float maximum = -std::numeric_limits<float>::infinity();
  for (uint32_t i = 0; i < vocabulary_size; ++i) {
    if (std::isfinite(logits[i]) && logits[i] > maximum) {
      maximum = logits[i];
      best = static_cast<int>(i);
    }
  }
  return best;
}

int ds4_sample_top_p_min_p(const float* logits, uint32_t vocabulary_size,
                           float temperature, int top_k, float top_p,
                           float min_p, uint64_t* rng_state) {
  if (logits == nullptr || vocabulary_size == 0)
    return -1;
  // Keep one definition of target sampling, including finite-value handling
  // and temperature-before-filtering. The native API reports failure as -1.
  try {
    const gufo::sampling::SamplingConfig config{.temperature = temperature,
                                                .top_k = top_k,
                                                .top_p = top_p,
                                                .min_p = min_p};
    return static_cast<int>(
        gufo::sampling::BuildDistribution({logits, vocabulary_size}, config)
            .Sample(rng_state));
  } catch (const std::exception&) {
    return -1;
  }
}
