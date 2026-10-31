#ifndef GUFO_MODELS_GEMMA4_CONTINUATION_HOOKS_HPP_
#define GUFO_MODELS_GEMMA4_CONTINUATION_HOOKS_HPP_

#include <cstdint>
#include <span>

namespace gufo::models::gemma4 {
// Installed only in sessions owned by the component-cache adapter; legacy
// serving leaves the pointer null. Runs on the host, with the model mutex
// held, before a forward or key move queues device work.
class ContinuationHooks {
public:
  virtual ~ContinuationHooks() = default;
  // Positions [first, end) are about to be written. Sliding layers write
  // them into ring slots that held the positions one ring earlier.
  virtual void BeforeWrite(std::uint32_t first, std::uint32_t end) = 0;
};
// Installed in the executor by the component-cache adapter. Runs on the
// host, with the model mutex held, before the executor queues work that
// writes rows or frontier state (hidden state, logits) of the sessions with
// `writers`' hooks; every earlier write of other sessions is queued by then.
class ContinuationFences {
public:
  virtual ~ContinuationFences() = default;
  virtual void BeforeMutation(std::span<ContinuationHooks* const> writers) = 0;
};
}  // namespace gufo::models::gemma4
#endif
