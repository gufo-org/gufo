#ifndef GUFO_CACHE_EVENTS_HPP_
#define GUFO_CACHE_EVENTS_HPP_

#include <cstddef>
#include <cstdint>

#include "src/cache/types.hpp"

namespace gufo::cache {

enum class EventKind : std::uint8_t {
  kCapture,
  kRestore,
  kInvalidate,
  kTransferFailure,
};

// Bounded internal metadata only; no prompts, model pointers or user labels.
// Later policy/store cards add their own typed reasons and counters.
struct Event {
  EventKind kind{EventKind::kCapture};
  SlotId slot;
  ComponentId component;
  Rows rows{0};
  std::size_t bytes{0};
};

class EventSink {
public:
  EventSink() = default;
  virtual ~EventSink() = default;
  EventSink(const EventSink&) = delete;
  EventSink& operator=(const EventSink&) = delete;
  EventSink(EventSink&&) = delete;
  EventSink& operator=(EventSink&&) = delete;
  virtual void Emit(const Event&) noexcept = 0;
};

}  // namespace gufo::cache

#endif
