#ifndef GUFO_TESTS_CACHE_PREFIX_INDEX_FIXTURE_HPP_
#define GUFO_TESTS_CACHE_PREFIX_INDEX_FIXTURE_HPP_

#include <array>
#include <memory>
#include <optional>
#include <vector>

#include "src/cache/prefix_index.hpp"

namespace gufo::cache::testing {
inline constexpr ResourceLimits kIndexLimits{1ULL << 32, 1ULL << 32, 1ULL << 32,
                                             1ULL << 32};
inline constexpr std::array<ComponentDescriptor, 3> kIndexComponents{{
    {{1}, 1, ComponentKind::kAppendRows, 4, 1024, 0},
    {{2}, 1, ComponentKind::kAppendRows, 2, 512, 0},
    {{3}, 1, ComponentKind::kPrivateState, 0, 0, 32},
}};
inline std::vector<ComponentAvailability> ResidentComponents() {
  return {{{1}, true, false}, {{2}, true, false}, {{3}, true, false}};
}
// Size-only opaque storage exercises real captures and resource accounting,
// without allocating the physical payload or asserting numerical restoration.
inline std::shared_ptr<const Checkpoint> IndexCheckpoint(
    ResourceLedger& ledger, std::span<const Token> tokens,
    Identity compatibility = {1}, Identity input = {},
    std::span<const ComponentDescriptor> descriptors = kIndexComponents,
    std::optional<Rows> private_boundary = {}) {
  auto history =
      ExecutionHistory::Cold(ledger, descriptors, std::move(compatibility));
  std::vector<ComponentPosition> positions;
  for (const auto& d : descriptors)
    positions.push_back({d.id, d.kind == ComponentKind::kPrivateState
                                   ? private_boundary.value_or(tokens.size())
                                   : tokens.size() / d.id.value});
  return history.Capture(
      {tokens, InputIdentity(tokens.size(), std::move(input)), positions,
       CheckpointPurpose::kPrompt, 1},
      [&](const PayloadRequest& request) {
        auto pool =
            ledger.Reserve(ResourceCategory::kBackingFree, request.bytes)
                .Convert();
        auto charge = pool.ReserveBacking(request.category).Convert();
        return Payload::Committed(std::move(charge), std::make_shared<int>(0));
      });
}
inline LiveFrontier IndexLive(std::vector<Token> tokens, SlotId slot = {1},
                              Identity input = {}) {
  const auto count = tokens.size();
  return {{slot, 1},
          std::move(tokens),
          {1},
          std::move(input),
          {{{1}, count}, {{2}, count / 2}, {{3}, count}},
          true};
}
}  // namespace gufo::cache::testing
#endif
