#pragma once

// Demand-mapped state memory: fixed-address regions backed by pooled 2 MiB
// pages as rows are written, pass scratch whose wide rows take memory only
// while a wide pass needs them, and the free-memory reading every capacity
// decision uses.
#include "src/core/hip/scratch_arena.hpp"
#include "src/core/hip/state_memory.hpp"
#include "src/core/hip/state_region.hpp"
#include "src/mem/ledger.hpp"
