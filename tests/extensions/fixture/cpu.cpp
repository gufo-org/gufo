#include "gufo/ext/generated/alpha_constants.hpp"

extern int ExtensionTestBase();
// References a symbol in a dependency archive to exercise normal static links.
int CreateAlpha() {
  return ExtensionTestBase();
}
void alpha_cpu_register(gufo::ext::Registry& registry) {
  if (CreateAlpha() == 42) {
    registry.AddProvider(
        gufo::ext::generated::extension_alpha::kProvider_alpha);
  }
}
