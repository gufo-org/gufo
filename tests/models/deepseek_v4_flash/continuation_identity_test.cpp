#include "src/models/deepseek_v4_flash/continuation_identity.hpp"

#include <sstream>

#include "src/core/speculative/speculative_verifier.hpp"
#include "tests/support/continuation_identity.hpp"

namespace identity_test {
using namespace gufo;
using namespace gufo::server;
void DeepSeekCases() {
  Golden("dspark", DeepSeekCompatibilityIdentity(artifact, draft, 4096, 4));
  Golden("target-only", DeepSeekCompatibilityIdentity(artifact, "", 4096, 4));
  Golden("target-only-minimum",
         DeepSeekCompatibilityIdentity(artifact, "", 4096, 1));
  Golden("no-draft", DeepSeekCompatibilityIdentity(artifact, draft, 4096, 0));
  Golden("context", DeepSeekCompatibilityIdentity(artifact, draft, 131072, 4));
  Golden("draft-limit",
         DeepSeekCompatibilityIdentity(artifact, draft, 4096, 7));
  Golden("artifact", DeepSeekCompatibilityIdentity(draft, artifact, 4096, 4));
  Golden("empty-support", DeepSeekCompatibilityIdentity(artifact, "", 4096, 0));
}
void Validation() {
  for (const std::string& bad :
       {std::string{}, std::string(63, 'a'), std::string(64, 'A'),
        std::string(64, 'g'), std::string(65, 'a')}) {
    Reject([&] { DeepSeekCompatibilityIdentity(bad, draft, 4096, 4); },
           "DeepSeek disk cache requires an artifact fingerprint");
  }
  // Current writer validates only the main fingerprint. This fixture preserves
  // acceptance of empty support; changing validation is a separate behavior
  // fix.
}

}  // namespace identity_test
int main() {
  return identity_test::Run([] {
    identity_test::DeepSeekCases();
    identity_test::Validation();
  });
}
