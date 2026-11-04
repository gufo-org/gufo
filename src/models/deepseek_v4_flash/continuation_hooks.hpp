#ifndef GUFO_MODELS_DEEPSEEK_V4_FLASH_CONTINUATION_HOOKS_HPP_
#define GUFO_MODELS_DEEPSEEK_V4_FLASH_CONTINUATION_HOOKS_HPP_

#include <cstdint>

namespace gufo::models::deepseek_v4_flash {

// Installed only in sessions owned by the continuation adapter; other sessions
// leave the pointer null. Every callback runs on the host before the session
// changes state or submits device work, and may throw to refuse it.
class ContinuationHooks {
public:
  virtual ~ContinuationHooks() = default;
  // Target tokens [first, end) are about to be written, including rows a
  // speculative verification may later roll back.
  virtual void BeforeWrite(std::uint32_t first, std::uint32_t end) = 0;
  virtual void BeforeReset() = 0;
  virtual void BeforeRestore() = 0;
};

}  // namespace gufo::models::deepseek_v4_flash

#endif
