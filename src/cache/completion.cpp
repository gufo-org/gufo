#include "src/cache/completion.hpp"

#include <stdexcept>
#include <utility>

namespace gufo::cache {

Completion::Completion(std::unique_ptr<CompletionSignal> signal)
    : signal_(std::move(signal)) {
  if (!signal_)
    throw std::invalid_argument("completion requires a signal");
}

Completion::~Completion() {
  (void)Wait();
}

Completion::Completion(Completion&& other) noexcept
    : signal_(std::move(other.signal_)),
      result_(std::exchange(other.result_, {})) {}

Completion& Completion::operator=(Completion&& other) noexcept {
  if (this != &other) {
    (void)Wait();
    signal_ = std::move(other.signal_);
    result_ = std::exchange(other.result_, {});
  }
  return *this;
}

bool Completion::Ready() const noexcept {
  return result_.has_value() || !signal_ || signal_->Ready();
}

TransferResult Completion::Wait() noexcept {
  if (!result_)
    result_ = signal_ ? signal_->Wait() : TransferResult::kFailed;
  return *result_;
}

}  // namespace gufo::cache
