#pragma once

#include <exception>
#include <utility>

namespace gufo::core::ffi {

// A noexcept C callback records the first exception here, then the caller
// rethrows it after Rust returns. Never unwind a C++ exception through Rust.
class CallbackFailure {
public:
  template<class T, class F>
  T Guard(T fallback, F&& call) noexcept {
    if (failure_)
      return fallback;
    try {
      return std::forward<F>(call)();
    } catch (...) {
      failure_ = std::current_exception();
      return fallback;
    }
  }

  template<class F>
  void Guard(F&& call) noexcept {
    Guard(false, [&] {
      std::forward<F>(call)();
      return true;
    });
  }

  void Rethrow() const {
    if (failure_)
      std::rethrow_exception(failure_);
  }

private:
  std::exception_ptr failure_;
};

}  // namespace gufo::core::ffi
