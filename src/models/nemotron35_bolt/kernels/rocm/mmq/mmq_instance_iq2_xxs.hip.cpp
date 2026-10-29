// SPDX-License-Identifier: MIT
// MMQ tiled-kernel instantiation for GGML IQ2_XXS weights. One weight format per
// translation unit keeps template builds parallel (see VENDOR.md).
#include "nb_mmq_prelude.h"

namespace nb_mmq {

#include "mmq.hip.hpp"

DECL_MMQ_CASE(GGML_TYPE_IQ2_XXS);

}  // namespace nb_mmq
