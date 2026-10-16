#ifndef GUFO_TESTS_QWEN38_FLASH_NEXT_MTP_REPLAY_HPP_
#define GUFO_TESTS_QWEN38_FLASH_NEXT_MTP_REPLAY_HPP_

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/engine.hpp"

namespace gufo::test {

struct MtpReplay {
  std::vector<std::int32_t> tokens;
  std::vector<std::size_t> widths;
  std::uint64_t rng;
  std::uint64_t drafted;
  std::uint64_t accepted;
  bool operator==(const MtpReplay&) const = default;
};

inline MtpReplay DecodeMtpReplay(models::qwen38_flash_next::Session& session,
                                 sampling::SamplerState& sampler,
                                 std::size_t count) {
  MtpReplay result{};
  const auto before = session.Statistics();
  std::string error;
  while (result.tokens.size() < count) {
    models::qwen38_flash_next::Session::DecodeResult step;
    if (!session.DecodeStep(count - result.tokens.size(), sampler, &step,
                            &error, false))
      throw std::runtime_error(error);
    if (step.tokens.empty())
      throw std::runtime_error("MTP replay made no progress");
    result.tokens.insert(result.tokens.end(), step.tokens.begin(),
                         step.tokens.end());
    result.widths.push_back(step.tokens.size());
  }
  result.rng = sampler.rng_state();
  result.drafted = session.Statistics().drafted - before.drafted;
  result.accepted = session.Statistics().accepted - before.accepted;
  return result;
}

}  // namespace gufo::test

#endif
