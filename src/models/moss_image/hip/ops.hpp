#ifndef GUFO_MODELS_MOSS_IMAGE_HIP_OPS_HPP_
#define GUFO_MODELS_MOSS_IMAGE_HIP_OPS_HPP_

#include "src/models/qwen_image_21/hip/runtime.hpp"

// MOSS-Image operators on top of the Qwen-Image BF16 runtime. Every matrix is
// row-major BF16; `mod` rows hold consecutive [dim] parts of a modulation.
namespace gufo::models::moss_image::hip {

using qwen_image_21::hip::Matrix;
using qwen_image_21::hip::Runtime;

/// x * (1 + mod[scale_part]) + mod[shift_part], each operation rounded to
/// BF16 as in the PyTorch reference.
Matrix ShiftScale(Runtime& rt, const Matrix& x, const Matrix& mod,
                  int shift_part, int scale_part);

/// residual + y * mod[gate_part], with the product rounded first.
Matrix GatedResidual(Runtime& rt, const Matrix& residual, const Matrix& y,
                     const Matrix& mod, int gate_part);

/// MOSS's situ gate: (4 tanh(g / 4) sigmoid(g)) * (25 tanh(h / 25)). Hidden
/// and gate are `width` columns of their matrices starting at the given
/// column offsets.
Matrix Situ(Runtime& rt, const Matrix& hidden, int hidden_offset,
            const Matrix& gate, int gate_offset, int width);

/// GroupNorm over channels-last [pixels, channels] with an optional SiLU,
/// statistics accumulated in FP32/FP64 as in the FP32 reference VAE.
Matrix GroupNorm(Runtime& rt, const Matrix& x, const Matrix& weight,
                 const Matrix& bias, int groups, float epsilon, bool silu);

}  // namespace gufo::models::moss_image::hip
#endif
