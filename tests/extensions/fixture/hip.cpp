#include "gufo/ext/generated/alpha_constants.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <gufo/ext/experimental/dispatch_telemetry.hpp>
#include <gufo/ext/experimental/gemv_dispatcher.hpp>
#include <gufo/ext/experimental/ggml_dequant.hpp>
#include <gufo/ext/experimental/ggml_gemm.hpp>
#include <gufo/ext/experimental/heap_placement.hpp>
#include <gufo/ext/experimental/hip_graph_decode_executor.hpp>
#include <gufo/ext/experimental/hip_utils.hpp>
#include <gufo/ext/experimental/hipblaslt_plan_database.hpp>
#include <gufo/ext/experimental/host_allocations.hpp>
#include <gufo/ext/experimental/managed_alloc.hpp>
#include <gufo/ext/experimental/snapshot_copy.hpp>
#include <gufo/ext/experimental/snapshot_transfer.hpp>
#include <gufo/ext/experimental/transfer_pool.hpp>
#include <gufo/ext/experimental/weight_upload.hpp>
#endif

// This symbol and registration function must be absent from a CPU executable.
int CreateUnavailableHipProvider() {
  return 7;
}
void alpha_hip_register(gufo::ext::Registry& registry) {
  if (CreateUnavailableHipProvider() == 7) {
    registry.AddProvider(
        gufo::ext::generated::extension_alpha::kProvider_alpha_hip);
  }
}
