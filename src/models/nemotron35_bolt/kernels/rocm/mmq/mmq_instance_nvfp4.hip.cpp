// SPDX-License-Identifier: MIT
// MMQ tiled-kernel instantiation for GGML NVFP4 weights. One weight format per
// translation unit keeps template builds parallel (see VENDOR.md).
#include "nb_mmq_prelude.h"

namespace nb_mmq {

#include "mmq.hip.hpp"

DECL_MMQ_CASE(GGML_TYPE_NVFP4);

}  // namespace nb_mmq
