#include "src/core/hip/session_state.hpp"

#include <hip/hip_runtime.h>

#include <string>
#include <utility>

#include "src/core/hip/managed_alloc.hpp"

namespace gufo::mem {

namespace {

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

}  // namespace

std::unique_ptr<SessionState> SessionState::Create(
    const StateDeclaration& declaration, std::string* error, bool charged) {
  std::unique_ptr<SessionState> state(new SessionState());
  const auto& buffers = declaration.buffers();
  state->data_.resize(buffers.size(), nullptr);
  state->regions_.resize(buffers.size());
  if (charged && declaration.has_context())
    state->account_ =
        std::make_shared<StateAccount>(declaration.margin_tokens());
  bool cleared = false;
  for (StateBufferId id = 0; id < buffers.size(); ++id) {
    const StateBuffer& buffer = buffers[id];
    if (buffer.kind == StateKind::kFixed) {
      const std::size_t bytes = buffer.shape.row_bytes;
      void* p = nullptr;
      if (hip::AllocateDevice(&p, bytes) != hipSuccess || p == nullptr) {
        (void)hipGetLastError();
        SetError(error,
                 "allocation of " + std::to_string(bytes) + " bytes failed");
        return nullptr;
      }
      state->fixed_.push_back(p);
      if (hipMemset(p, 0, bytes) != hipSuccess) {
        (void)hipGetLastError();
        SetError(error, "session state clear failed");
        return nullptr;
      }
      cleared = true;
      state->data_[id] = p;
      state->allocated_bytes_ += bytes;
      continue;
    }
    const bool context = buffer.kind == StateKind::kContext;
    auto region =
        Region::Create(buffer.shape, error, context ? state->account_ : nullptr,
                       context ? buffer.tokens_per_row : 1);
    if (!region)
      return nullptr;
    if (!context && !region->Ensure(buffer.shape.rows, error))
      return nullptr;
    state->data_[id] = region->data();
    state->allocated_bytes_ += region->bytes();
    state->regions_[id] = std::move(region);
  }
  // The clears run on the null stream; nothing may use the buffers from
  // another stream until they have landed.
  if (cleared && hipStreamSynchronize(nullptr) != hipSuccess) {
    (void)hipGetLastError();
    SetError(error, "session state clear failed");
    return nullptr;
  }
  return state;
}

SessionState::~SessionState() {
  for (void* p : fixed_)
    (void)hipFree(p);
  regions_.clear();
  account_.reset();
}

}  // namespace gufo::mem
