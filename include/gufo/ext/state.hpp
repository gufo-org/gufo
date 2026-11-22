#pragma once

// Session state declarations: a model lists the buffers one session holds
// (growing with the context, backed whole, or fixed) and the core allocates
// them (mem::SessionState in gufo/ext/experimental/state_memory.hpp) and
// derives the runner's state claim from the same list.
#include "src/mem/state_declaration.hpp"
