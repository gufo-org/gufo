#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

#include "src/models/deepseek_v4_flash/runtime/model_data_internal.h"

namespace {
void Expect(bool ok, const char* message) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
  }
}

struct Constant {
  const char* key;
  float expected;
  float stored_by_0731;
};

constexpr Constant kConstants[] = {
    {"rope.freq_base", DS4_ROPE_FREQ_BASE, 10000.0F},
    {"rope.scaling.factor", DS4_ROPE_SCALE_FACTOR, 16.0F},
    {"rope.scaling.yarn_beta_fast", DS4_ROPE_YARN_BETA_FAST, 32.0F},
    {"rope.scaling.yarn_beta_slow", DS4_ROPE_YARN_BETA_SLOW, 1.0F},
    {"attention.compress_rope_freq_base", DS4_COMPRESS_ROPE_FREQ_BASE,
     160000.0F},
    {"expert_weights_scale", DS4_EXPERT_WEIGHT_SCALE, 1.5F},
    {"attention.layer_norm_rms_epsilon", DS4_RMS_EPS,
     static_cast<float>(9.999999974752427e-07)},
    {"hyper_connection.epsilon", DS4_HC_EPS,
     static_cast<float>(9.999999974752427e-07)},
    {"swiglu_clamp_exp", DS4_SWIGLU_CLAMP_EXP, 10.0F},
};
}  // namespace

int main() {
  for (const Constant& c : kConstants) {
    Expect(ds4_config_f32_matches(c.stored_by_0731, c.expected), c.key);
    Expect(!ds4_config_f32_matches(c.expected * 1.001F, c.expected), c.key);
    Expect(!ds4_config_f32_matches(c.expected * 0.999F, c.expected), c.key);
    Expect(!ds4_config_f32_matches(c.expected * 1.0e-6F, c.expected), c.key);
    Expect(!ds4_config_f32_matches(std::numeric_limits<float>::quiet_NaN(),
                                   c.expected),
           c.key);
    Expect(!ds4_config_f32_matches(std::numeric_limits<float>::infinity(),
                                   c.expected),
           c.key);
  }

  const float vision_exp_epsilon = static_cast<float>(9.999999682655225e-21);
  Expect(!ds4_config_f32_matches(vision_exp_epsilon, DS4_RMS_EPS),
         "Vision-Exp RMS epsilon 1e-20 is rejected");
  Expect(!ds4_config_f32_matches(vision_exp_epsilon, DS4_HC_EPS),
         "epsilon 1e-20 is rejected for the hyper-connection epsilon");
  Expect(!ds4_config_f32_matches(1.0e-5F, DS4_RMS_EPS),
         "epsilon 1e-5 is rejected");
  return 0;
}
