#pragma once

// Demand-mapped state memory: fixed-address regions backed by pooled 2 MiB
// pages as rows are written, the pages a checkpoint shares with the session
// that wrote them, pass scratch whose wide rows take memory only while a wide
// pass needs them, the free-memory reading every capacity decision uses, and
// the memory of a session's declared state (gufo/ext/state.hpp).
#include "src/core/hip/scratch_arena.hpp"
#include "src/core/hip/session_state.hpp"
#include "src/core/hip/state_memory.hpp"
#include "src/core/hip/state_region.hpp"
#include "src/core/hip/state_shared_page.hpp"
#include "src/mem/ledger.hpp"
#include "src/mem/remap_barrier.hpp"
#include "src/mem/shared_rows.hpp"
