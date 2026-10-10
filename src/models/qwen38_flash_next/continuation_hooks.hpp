#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_CONTINUATION_HOOKS_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_CONTINUATION_HOOKS_HPP_

#include <cstdint>

namespace gufo::models::qwen38_flash_next {
class ContinuationAdapter;
// Installed only in sessions owned by the new adapter. Legacy serving leaves
// the pointer null. These callbacks run on the host before state is changed or
// device work is submitted (including a captured graph replay).
class ContinuationHooks {
public:
  virtual ~ContinuationHooks() = default;
  virtual void BeforeExecution() = 0;
  virtual void BeforeWrite(bool draft, std::uint32_t first, std::uint32_t end,
                           std::uint32_t first_block,
                           std::uint32_t end_block) = 0;
  virtual void BeforeReset() = 0;
  virtual void AfterReset() = 0;
  virtual void BeforeRestore() = 0;
  virtual void BeforeRelease() noexcept = 0;
};
}  // namespace gufo::models::qwen38_flash_next
#endif
