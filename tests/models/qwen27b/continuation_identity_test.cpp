#include "src/models/qwen/continuation_identity.hpp"

#include <sstream>

#include "src/core/speculative/speculative_verifier.hpp"
#include "tests/support/continuation_identity.hpp"

namespace identity_test {
using namespace gufo;
using namespace gufo::server;
void QwenCases() {
  for (bool fp16 : {false, true}) {
    for (bool bf16 : {false, true}) {
      for (bool spec : {false, true}) {
        hip::QwenExecutionPolicy p;
        p.kv_cache_storage = fp16 ? hip::QwenKvCacheStorage::kFp16
                                  : hip::QwenKvCacheStorage::kFp32;
        p.recurrent_state_storage = bf16
                                        ? hip::QwenRecurrentStateStorage::kBf16
                                        : hip::QwenRecurrentStateStorage::kFp32;
        Golden("storage-" + std::to_string(fp16) + std::to_string(bf16) +
                   "-draft-" + std::to_string(spec),
               QwenCompatibilityIdentity(
                   artifact, draft, 4096, p, spec, {},
                   speculative::DFlashDraftPolicy::kAdaptive));
      }
    }
  }
  auto p = hip::QwenExecutionPolicy::Production();
  Golden("context",
         QwenCompatibilityIdentity(artifact, draft, 131072, p, true, {},
                                   speculative::DFlashDraftPolicy::kAdaptive));
  Golden("artifact",
         QwenCompatibilityIdentity(draft, artifact, 4096, p, true, {},
                                   speculative::DFlashDraftPolicy::kAdaptive));
  for (unsigned variant = 0; variant < 10; ++variant) {
    speculative::SpeculativeOptions o;
    auto draft_policy = speculative::DFlashDraftPolicy::kAdaptive;
    switch (variant) {
      case 0:
        draft_policy = speculative::DFlashDraftPolicy::kFixed;
        break;
      case 1:
        o.max_draft_tokens = 7;
        break;
      case 2:
        o.min_draft_tokens = 2;
        break;
      case 3:
        o.initial_draft_tokens = 2;
        break;
      case 4:
        o.rolling_window = 32;
        break;
      case 5:
        o.target_acceptance_rate = 0.81234568F;
        break;
      case 6:
        o.enable_adaptive_draft_length = false;
        break;
      case 7:
        o.use_batched_verification = true;
        break;
      case 8:
        o.retain_frontier_logits = true;
        break;
      case 9:
        p.fuse_qk_norm_rope_kv = false;
        break;
    }
    Golden("option-" + std::to_string(variant),
           QwenCompatibilityIdentity(artifact, draft, 4096, p, true, o,
                                     draft_policy));
  }
}

void Validation() {
  for (const std::string& bad :
       {std::string{}, std::string(63, 'a'), std::string(64, 'A'),
        std::string(64, 'g'), std::string(65, 'a')}) {
    Reject(
        [&] {
          QwenCompatibilityIdentity(bad, draft, 4096, {}, false, {},
                                    speculative::DFlashDraftPolicy::kAdaptive);
        },
        "Qwen disk cache requires an artifact fingerprint");
    Reject(
        [&] {
          QwenCompatibilityIdentity(artifact, bad, 4096, {}, true, {},
                                    speculative::DFlashDraftPolicy::kAdaptive);
        },
        "Qwen DFlash disk cache requires a draft artifact fingerprint");
  }
  Golden("storage-10-draft-0",
         QwenCompatibilityIdentity(artifact, "ignored", 4096, {}, false, {},
                                   speculative::DFlashDraftPolicy::kAdaptive));
  // Pointer spelling and null spelling are implementation/process specific.
  // Compare each address independently, never persist an ASLR-dependent golden.
  int model{}, other{}, draft_model{};
  std::ostringstream pointers;
  pointers << &model << '\n' << &draft_model;
  auto split = pointers.str().find('\n');
  const std::string expected =
      "schema=gufo-process-qwen-component-v2\nmodel_instance=" +
      pointers.str().substr(0, split) +
      "\ndraft_instance=" + pointers.str().substr(split + 1) +
      "\ncontext_tokens=4096\n";
  Require(Text(QwenProcessIdentity(&model, &draft_model, 4096)) == expected,
          "process identity serialization changed");
  std::ostringstream null_pointer;
  null_pointer << static_cast<const void*>(nullptr);
  Require(Text(QwenProcessIdentity(&model, nullptr, 4096)) ==
              "schema=gufo-process-qwen-component-v2\nmodel_instance=" +
                  pointers.str().substr(0, split) + "\ndraft_instance=" +
                  null_pointer.str() + "\ncontext_tokens=4096\n",
          "null-draft process identity serialization changed");
  Require(QwenProcessIdentity(&model, nullptr, 4096) !=
              QwenProcessIdentity(&other, nullptr, 4096),
          "model lifetimes alias");
  Require(QwenProcessIdentity(&model, &draft_model, 4096) !=
              QwenProcessIdentity(&model, nullptr, 4096),
          "draft lifetimes alias");
  Require(QwenProcessIdentity(&model, nullptr, 4096) !=
              QwenProcessIdentity(&model, nullptr, 8192),
          "process contexts alias");
}

}  // namespace identity_test
int main() {
  return identity_test::Run([] {
    identity_test::QwenCases();
    identity_test::Validation();
  });
}
