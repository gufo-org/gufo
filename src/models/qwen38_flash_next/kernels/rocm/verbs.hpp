#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_VERBS_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_VERBS_HPP_

#include <cstdint>
#include <memory>
#include <string>

#include "src/models/qwen38_flash_next/kernels/rocm/communicator.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

struct IbrverbsConfig {
  std::uint32_t rank{0};
  std::uint32_t world_size{2};
  /// TCP bootstrap address used by rank 1. Rank 0 binds to all interfaces.
  std::string bootstrap_host;
  std::uint16_t bootstrap_port{18515};
  std::uint32_t device_index{0};
  std::uint32_t gid_index{0};
};

/// Creates a two-rank RC communicator. Each exchange is one RDMA write with
/// immediate data of a header and the partial into the peer's next receive
/// window; completions arrive through an event-driven completion channel,
/// and the bootstrap socket stays open only to detect a lost peer. Overlapped
/// exchanges run on a background thread. Both processes map the same fixed
/// send and receive windows. Callers must bind the same control scope before
/// model execution; every collective carries and validates its operation
/// ordinal.
[[nodiscard]] std::shared_ptr<Communicator> CreateIbrverbsCommunicator(
    const IbrverbsConfig& config, std::string* error_msg);

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_VERBS_HPP_
