#include "src/models/qwen38_flash_next/continuation_identity.hpp"

#include <sstream>

#include "src/core/speculative/speculative_verifier.hpp"
#include "tests/support/continuation_identity.hpp"

namespace identity_test {
using namespace gufo;
using namespace gufo::server;
void FlashCases() {
  for (bool mtp : {false, true}) {
    Golden("draft-" + std::to_string(mtp),
           QwenFlashNextCompatibilityIdentity(artifact, draft, mtp, 4096, 4, 1,
                                              "absolute-v1"));
    Golden("context-" + std::to_string(mtp),
           QwenFlashNextCompatibilityIdentity(artifact, draft, mtp, 131072, 4,
                                              1, "absolute-v1"));
    Golden("position-" + std::to_string(mtp),
           QwenFlashNextCompatibilityIdentity(
               artifact, draft, mtp, 4096, 4, 1,
               "absolute-yarn-v1;factor=0x1p+1;original=262144;beta_fast=0x1p+"
               "5;beta_slow=0x1p+0"));
    Golden("concurrency-" + std::to_string(mtp),
           QwenFlashNextCompatibilityIdentity(artifact, draft, mtp, 4096, 4, 8,
                                              "absolute-v1"));
    Golden("tokens-" + std::to_string(mtp),
           QwenFlashNextCompatibilityIdentity(artifact, draft, mtp, 4096, 7, 1,
                                              "absolute-v1"));
  }
  Golden("artifact", QwenFlashNextCompatibilityIdentity(
                         draft, artifact, true, 4096, 4, 1, "absolute-v1"));
}

void Validation() {
  for (const std::string& bad :
       {std::string{}, std::string(63, 'a'), std::string(64, 'A'),
        std::string(64, 'g'), std::string(65, 'a')}) {
    Reject(
        [&] {
          QwenFlashNextCompatibilityIdentity(bad, draft, false, 4096, 4, 1,
                                             "absolute-v1");
        },
        "Qwen3.8-Flash-Next disk cache requires an artifact fingerprint");
    Reject(
        [&] {
          QwenFlashNextCompatibilityIdentity(artifact, bad, true, 4096, 4, 1,
                                             "absolute-v1");
        },
        "Qwen3.8-Flash-Next MTP disk cache requires a draft artifact "
        "fingerprint");
  }
  Golden("draft-0", QwenFlashNextCompatibilityIdentity(
                        artifact, "ignored", false, 4096, 4, 1, "absolute-v1"));
  int model{}, other{};
  std::ostringstream pointer;
  pointer << &model;
  Require(Text(QwenFlashNextProcessIdentity(&model, 4096, true, 4, 8)) ==
              "schema=gufo-process-text-continuation-v1\nmodel_instance=" +
                  pointer.str() +
                  "\ncontext_tokens=4096\nmtp=1\ndraft_max_tokens=4\ndraft_"
                  "cost_concurrency=8\n",
          "process identity serialization changed");
  const auto baseline = QwenFlashNextProcessIdentity(&model, 4096, false, 4, 1);
  Require(Text(baseline) ==
              "schema=gufo-process-text-continuation-v1\nmodel_instance=" +
                  pointer.str() +
                  "\ncontext_tokens=4096\nmtp=0\ndraft_max_tokens=4\ndraft_"
                  "cost_concurrency=1\n",
          "disabled-MTP process identity serialization changed");
  Require(baseline != QwenFlashNextProcessIdentity(&other, 4096, false, 4, 1),
          "model lifetimes alias");
  Require(baseline != QwenFlashNextProcessIdentity(&model, 8192, false, 4, 1),
          "process contexts alias");
  Require(baseline != QwenFlashNextProcessIdentity(&model, 4096, true, 4, 1),
          "process MTP modes alias");
  Require(baseline != QwenFlashNextProcessIdentity(&model, 4096, false, 7, 1),
          "process draft lengths alias");
  Require(baseline != QwenFlashNextProcessIdentity(&model, 4096, false, 4, 8),
          "process concurrency aliases");
}

}  // namespace identity_test
int main() {
  return identity_test::Run([] {
    identity_test::FlashCases();
    identity_test::Validation();
  });
}
