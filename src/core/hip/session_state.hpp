#ifndef GUFO_CORE_HIP_SESSION_STATE_HPP_
#define GUFO_CORE_HIP_SESSION_STATE_HPP_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "src/core/hip/state_region.hpp"
#include "src/mem/account.hpp"
#include "src/mem/state_declaration.hpp"

namespace gufo::mem {

/// The memory of one session's declared state (StateDeclaration), owned by
/// the core: kContext buffers are regions charged to the session's account,
/// kWhole buffers regions backed whole, kFixed buffers zeroed device
/// allocations (placed like other session state, hip::AllocateDevice).
///
/// Creation clears on the null stream and waits for it, so the buffers are
/// ready for any stream when Create() returns. Destruction releases every
/// buffer; regions others still share (checkpoints) live on until they let
/// go. Call it where no work that uses the buffers is in flight.
class SessionState {
public:
  /// Every buffer of `declaration`, or null with `error` set. The session
  /// has an account when it declares a kContext buffer and `charged`: a
  /// model whose sessions claim no state (DeepSeek V4 Flash when even
  /// shared rows do not fit) passes false, and its kContext regions map
  /// pages without charging any account.
  [[nodiscard]] static std::unique_ptr<SessionState> Create(
      const StateDeclaration& declaration, std::string* error,
      bool charged = true);
  ~SessionState();
  SessionState(const SessionState&) = delete;
  SessionState& operator=(const SessionState&) = delete;

  [[nodiscard]] void* data(StateBufferId id) const { return data_.at(id); }
  template<typename T>
  [[nodiscard]] T* data(StateBufferId id) const {
    return static_cast<T*>(data(id));
  }
  /// The region of a kContext or kWhole buffer; null for a kFixed one.
  [[nodiscard]] const std::shared_ptr<Region>& region(StateBufferId id) const {
    return regions_.at(id);
  }
  /// Charges the kContext regions' pages; null without kContext buffers or
  /// when created uncharged.
  [[nodiscard]] const std::shared_ptr<StateAccount>& account() const noexcept {
    return account_;
  }
  /// Bytes allocated: each region's size and each fixed buffer's bytes, as
  /// StateDeclaration::Bytes() counts them without demand mapping.
  [[nodiscard]] std::size_t allocated_bytes() const noexcept {
    return allocated_bytes_;
  }

private:
  SessionState() = default;

  std::shared_ptr<StateAccount> account_;
  std::vector<void*> data_;
  std::vector<std::shared_ptr<Region>> regions_;
  /// kFixed allocations, freed by the destructor.
  std::vector<void*> fixed_;
  std::size_t allocated_bytes_{0};
};

}  // namespace gufo::mem

#endif  // GUFO_CORE_HIP_SESSION_STATE_HPP_
