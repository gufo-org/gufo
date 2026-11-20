#include "src/core/hip/copy_engine.hpp"

namespace gufo::hip {

// HIP on Windows exposes no HSA runtime: restores copy on HIP streams.
CopyEngine* DeviceCopyEngine() noexcept {
  return nullptr;
}

}  // namespace gufo::hip
