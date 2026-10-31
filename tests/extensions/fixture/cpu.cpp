#include <gufo/ext/adapter.hpp>
#include <gufo/ext/cache.hpp>
#include <gufo/ext/host.hpp>
#include <gufo/ext/json.hpp>
#include <gufo/ext/options.hpp>
#include <gufo/ext/platform/fs.hpp>
#include <gufo/ext/platform/os_info.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/version.hpp>

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
