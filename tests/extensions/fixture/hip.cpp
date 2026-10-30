#include "gufo/ext/generated/alpha_constants.hpp"

// This symbol and registration function must be absent from a CPU executable.
int CreateUnavailableHipProvider() {
  return 7;
}
void alpha_hip_register(gufo::ext::Registry& registry) {
  if (CreateUnavailableHipProvider() == 7) {
    registry.AddProvider(
        gufo::ext::generated::extension_alpha::kProvider_alpha_hip);
  }
}
