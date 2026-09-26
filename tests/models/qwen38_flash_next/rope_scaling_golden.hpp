#ifndef GUFO_TESTS_QWEN38_FLASH_NEXT_ROPE_SCALING_GOLDEN_HPP_
#define GUFO_TESTS_QWEN38_FLASH_NEXT_ROPE_SCALING_GOLDEN_HPP_

// YaRN parameters against Hugging Face transformers 5.17.0
// `_compute_yarn_parameters` (Flash-Next: rotary 64 of 256, theta 1e7,
// original context 262144, beta 32/1, truncated correction range). Shared by
// the hosted golden-table check (rope_scaling_test.cpp) and the CPU-oracle
// check (rope_scaling_oracle_test.cpp) so the numbers have one source.
//
// Regeneration command (transformers 5.17.0, run 2026-09-26 on crow from
// ~/video-stack/.venv-comfy): calls the real
// `transformers.modeling_rope_utils._compute_yarn_parameters` on a config
// object with head_dim 256, partial_rotary_factor 0.25 (dim 64), rope_theta
// 1e7, original_max_position_embeddings 262144, rope_type yarn, no
// attention_factor/mscale/betas (so defaults 32/1, truncate true):
//
//   ~/video-stack/.venv-comfy/bin/python - <<'EOF'
//   from transformers.modeling_rope_utils import _compute_yarn_parameters
//   class Cfg:
//       def __init__(self, f):
//           self.hidden_size=2560; self.num_attention_heads=24
//           self.head_dim=256; self.max_position_embeddings=262144
//           self.rope_parameters={"rope_type":"yarn","rope_theta":1e7,
//               "partial_rotary_factor":0.25,"factor":f,
//               "original_max_position_embeddings":262144}
//       def standardize_rope_params(self): pass
//   for f in (1.5625, 2.5, 4.0):
//       inv, att = _compute_yarn_parameters(Cfg(f))
//       print(f, repr(att), ", ".join("%.9e" % float(x) for x in inv))
//   EOF
//
// Correction range raw 14.2410..21.1216, truncated to pairs low 14, high 22.

#include <array>

namespace gufo::models::qwen38_flash_next::testing {

struct Golden {
  float factor;
  double attention_factor;
  std::array<double, 32> inv_freq;
};

constexpr std::array<Golden, 3> kGolden{{
    {1.5625F,
     1.044628710262842,
     {
         1.000000000e+00, 6.042963862e-01, 3.651741445e-01, 2.206733972e-01,
         1.333521456e-01, 8.058421314e-02, 4.869675264e-02, 2.942727320e-02,
         1.778279431e-02, 1.074607857e-02, 6.493816618e-03, 3.924189601e-03,
         2.371373819e-03, 1.433012658e-03, 8.659643354e-04, 4.997506621e-04,
         2.877672960e-04, 1.652974315e-04, 9.469212091e-05, 5.408186553e-05,
         3.078384179e-05, 1.745583177e-05, 9.855529242e-06, 5.955661436e-06,
         3.598984449e-06, 2.174853307e-06, 1.314255996e-06, 7.942002185e-07,
         4.799323392e-07, 2.900213474e-07, 1.752588616e-07, 1.059082990e-07,
     }},
    {2.5F,
     1.0916290731874154,
     {
         1.000000000e+00, 6.042963862e-01, 3.651741445e-01, 2.206733972e-01,
         1.333521456e-01, 8.058421314e-02, 4.869675264e-02, 2.942727320e-02,
         1.778279431e-02, 1.074607857e-02, 6.493816618e-03, 3.924189601e-03,
         2.371373819e-03, 1.433012658e-03, 8.659643354e-04, 4.840516776e-04,
         2.687936067e-04, 1.480988576e-04, 8.083473222e-05, 4.361441097e-05,
         2.319330997e-05, 1.210440860e-05, 6.159706118e-06, 3.722288511e-06,
         2.249365252e-06, 1.359283260e-06, 8.214099694e-07, 4.963750939e-07,
         2.999576907e-07, 1.812633457e-07, 1.095367850e-07, 6.619268333e-08,
     }},
    {4.0F,
     1.138629436111989,
     {
         1.000000000e+00, 6.042963862e-01, 3.651741445e-01, 2.206733972e-01,
         1.333521456e-01, 8.058421314e-02, 4.869675264e-02, 2.942727320e-02,
         1.778279431e-02, 1.074607857e-02, 6.493816618e-03, 3.924189601e-03,
         2.371373819e-03, 1.433012658e-03, 8.659643354e-04, 4.742398160e-04,
         2.569350763e-04, 1.373497507e-04, 7.217386883e-05, 3.707224823e-05,
         1.844922372e-05, 8.759770026e-06, 3.849816039e-06, 2.326430149e-06,
         1.405853368e-06, 8.495520660e-07, 5.133812238e-07, 3.102344408e-07,
         1.874735602e-07, 1.132895946e-07, 6.846049416e-08, 4.137042708e-08,
     }},
}};

}  // namespace gufo::models::qwen38_flash_next::testing

#endif  // GUFO_TESTS_QWEN38_FLASH_NEXT_ROPE_SCALING_GOLDEN_HPP_
