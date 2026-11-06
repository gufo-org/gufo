# Admission memory

Publish `include/gufo/ext/experimental/admission_query.hpp`, forwarding
`src/core/hip/admission_query.hpp`:
- `gufo::hip::QueryAdmissionMemory()` returns an optional `gufo::hip::AdmissionMemory`,
  whose `free_bytes` is what a `TextRunnerResourceClaim::state_capacity_bytes` may
  count on. On Linux and on discrete GPUs it equals `hipMemGetInfo`'s figure.
- `gufo::hip::LogStateCapacity()` reports one such check.

The closure also publishes `gufo::hip::AdmissionFreeBytes`,
`gufo::hip::DxgiNodeFromMask` and `gufo::hip::kAdmissionHostMarginBytes`
(`src/core/hip/admission_memory.hpp`), and `gufo::platform::GpuLocalBudget` and
`gufo::platform::QueryGpuLocalBudget` (`src/core/platform/gpu_budget.hpp`). Additive.
