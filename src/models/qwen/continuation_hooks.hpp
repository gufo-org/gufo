#ifndef GUFO_MODELS_QWEN_CONTINUATION_HOOKS_HPP_
#define GUFO_MODELS_QWEN_CONTINUATION_HOOKS_HPP_

#include <cstdint>
#include <memory>

namespace gufo::models::qwen {
namespace vision {
struct Prompt;
struct RopeLayout;
}  // namespace vision
class ContinuationAdapter;
// Null in legacy serving. Hooks run before host mutation or GPU submission,
// including graph replay, so independent preservation streams finish first.
class ContinuationHooks {
public:
  virtual ~ContinuationHooks() = default;
  virtual void BeforeExecution() = 0;
  virtual void BeforeWrite(std::uint32_t first, std::uint32_t end) = 0;
  virtual void AfterWrite(std::uint32_t end, bool logits) noexcept = 0;
  virtual void WriteFailed() noexcept = 0;
  virtual void BeforeReset() = 0;
  virtual void AfterReset() = 0;
  virtual void BeforeRestore() = 0;
  virtual void AfterRestore(std::uint32_t position) noexcept = 0;
  virtual void AfterSaveState() noexcept = 0;
  virtual void AfterFinishVerification() noexcept = 0;
  virtual void BeforeVision(const vision::Prompt*,
                            const vision::RopeLayout*) = 0;
  virtual void AfterLogits() noexcept = 0;
  virtual void BeforeRelease() noexcept = 0;
};

class ContinuationWrite {
public:
  ContinuationWrite() = default;
  ContinuationWrite(ContinuationHooks* hooks, std::uint32_t first,
                    std::uint32_t end) {
    Begin(hooks, first, end);
  }
  ContinuationWrite(const ContinuationWrite&) = delete;
  ContinuationWrite& operator=(const ContinuationWrite&) = delete;
  ~ContinuationWrite() {
    if (hooks_)
      hooks_->WriteFailed();
  }
  void Begin(ContinuationHooks* hooks, std::uint32_t first, std::uint32_t end) {
    if (hooks)
      hooks->BeforeWrite(first, end);
    hooks_ = hooks;
  }
  // Batched admission guards every peer before arming failure tracking.
  // A refusal before submission leaves all untouched peers valid.
  void Arm(ContinuationHooks* hooks) noexcept { hooks_ = hooks; }
  void Commit(std::uint32_t position, bool logits) noexcept {
    if (hooks_)
      hooks_->AfterWrite(position, logits);
    hooks_ = nullptr;
  }

private:
  ContinuationHooks* hooks_{};
};
class ContinuationRestore {
public:
  explicit ContinuationRestore(ContinuationHooks* hooks) : hooks_(hooks) {
    if (hooks_)
      hooks_->BeforeRestore();
  }
  ContinuationRestore(const ContinuationRestore&) = delete;
  ContinuationRestore& operator=(const ContinuationRestore&) = delete;
  ~ContinuationRestore() {
    if (hooks_)
      hooks_->WriteFailed();
  }
  void Commit(std::uint32_t position) noexcept {
    if (hooks_)
      hooks_->AfterRestore(position);
    hooks_ = nullptr;
  }

private:
  ContinuationHooks* hooks_;
};
}  // namespace gufo::models::qwen
#endif
