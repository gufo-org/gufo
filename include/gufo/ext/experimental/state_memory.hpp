#pragma once

// Demand-mapped state memory: fixed-address regions backed by pooled 2 MiB
// pages as rows are written, and the free-memory reading every capacity
// decision uses.
#include "src/core/hip/state_memory.hpp"
#include "src/core/hip/state_region.hpp"
#include "src/mem/ledger.hpp"
