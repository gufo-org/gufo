#ifndef GUFO_CACHE_COMPLETION_HPP_
#define GUFO_CACHE_COMPLETION_HPP_

#include <cstdint>
#include <memory>
#include <optional>

namespace gufo::cache {

enum class TransferResult : std::uint8_t {
  kSucceeded,
  kFailed,
};

// Wait must settle the operation, including on failure: no buffer access may
// remain outstanding on return. Ready does not advance or synchronize work.
class CompletionSignal {
public:
  CompletionSignal() = default;
  virtual ~CompletionSignal() = default;
  CompletionSignal(const CompletionSignal&) = delete;
  CompletionSignal& operator=(const CompletionSignal&) = delete;
  CompletionSignal(CompletionSignal&&) = delete;
  CompletionSignal& operator=(CompletionSignal&&) = delete;
  [[nodiscard]] virtual bool Ready() const noexcept = 0;
  [[nodiscard]] virtual TransferResult Wait() noexcept = 0;
};

// Move-only ownership of one transfer. Destruction and move assignment drain
// existing work. Keep slots, buffers and stream alive until Wait or destruction
// completes; source data must remain immutable during the transfer.
// Draining alone does not handle a load failure: adapters must latch failures
// independently of this handle, so Validate rejects discarded errors too.
class Completion {
public:
  explicit Completion(std::unique_ptr<CompletionSignal>);
  ~Completion();
  Completion(Completion&&) noexcept;
  Completion& operator=(Completion&&) noexcept;
  Completion(const Completion&) = delete;
  Completion& operator=(const Completion&) = delete;

  [[nodiscard]] bool Ready() const noexcept;
  // Repeated waits retain the same result. A moved-from handle reports failure.
  [[nodiscard]] TransferResult Wait() noexcept;

private:
  std::unique_ptr<CompletionSignal> signal_;
  std::optional<TransferResult> result_;
};

}  // namespace gufo::cache

#endif
