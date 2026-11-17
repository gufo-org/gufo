// Shared session state on real weights (#715): a reclaim releases a
// session's compressed rows through its state account (StateAccount::
// TrimTo); the session must back them again as it writes and compute exactly
// what it computed before. Also checks the account covers the demand-mapped
// rows the runner claims.
//
// ds4_state_trim_test [MODEL.gguf]
// Without an argument, GUFO_DEEPSEEK_V4_FLASH_MODEL names the weights; the
// test is skipped without them or without demand-mapped state memory.

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "src/mem/account.hpp"
#include "src/models/deepseek_v4_flash/engine.hpp"

namespace ds4 = gufo::models::deepseek_v4_flash;

namespace {

constexpr std::uint32_t kContext = 16384;
constexpr int kDecodeTokens = 16;

std::string Ledger(std::size_t entries) {
  std::string text = "Inventory ledger.\n";
  for (std::size_t i = 0; i < entries; ++i)
    text += "Entry " + std::to_string(i) + " holds " +
            std::to_string(i * 7919 % 1009) + " units of part " +
            std::to_string(i * 31 % 97) + ".\n";
  return text + "Which part does entry 42 hold? Answer briefly.";
}

struct Run {
  std::vector<float> logits;
  std::vector<int> tokens;
};

bool Generate(ds4::Session& session, const std::vector<int>& prompt, Run* run) {
  std::string error;
  if (!session.Sync(prompt, &error)) {
    std::cerr << "prefill failed: " << error << '\n';
    return false;
  }
  run->logits = session.CopyLogits(&error);
  run->tokens.clear();
  for (int i = 0; i < kDecodeTokens; ++i) {
    std::uint64_t rng = 0;
    const int token = session.SelectNext(0.0F, &rng);
    run->tokens.push_back(token);
    if (!session.Evaluate(token, &error)) {
      std::cerr << "decode failed: " << error << '\n';
      return false;
    }
  }
  return !run->logits.empty();
}

}  // namespace

int main(int argc, char** argv) {
  std::string path = argc > 1 ? argv[1] : "";
  if (path.empty()) {
    const char* env = std::getenv("GUFO_DEEPSEEK_V4_FLASH_MODEL");
    if (!env || !*env) {
      std::cerr << "GUFO_DEEPSEEK_V4_FLASH_MODEL is not set; skipping\n";
      return 77;
    }
    path = env;
  }
  if (!ds4_session_demand_mapped()) {
    std::cerr << "no demand-mapped state memory on this device; skipping\n";
    return 77;
  }
  std::string error;
  ds4::ModelOptions options;
  options.max_context = kContext;
  auto model = ds4::Model::Load(path, options, &error);
  if (!model) {
    std::cerr << "load failed: " << error << '\n';
    return 1;
  }
  auto session = model->CreateSession(gufo::core::SessionMode::kAutoregressive,
                                      kContext, &error);
  if (!session) {
    std::cerr << "session failed: " << error << '\n';
    return 1;
  }
  int failures = 0;
  const auto check = [&](bool ok, const char* what) {
    std::cout << what << ',' << (ok ? "PASS" : "FAIL") << '\n';
    failures += ok ? 0 : 1;
  };
  const auto account = session->StateAccount();
  check(account != nullptr, "account");
  if (!account)
    return 1;
  check(account->full_bytes() == model->DemandMappedBytes(kContext),
        "account_covers_claim");

  // Long enough that the compressed rows grow past their first pages.
  const auto prompt = model->EncodeChat("", Ledger(700));
  std::cout << "prompt_tokens=" << prompt.size() << '\n';
  Run before;
  if (!Generate(*session, prompt, &before))
    return 1;
  const std::size_t mapped = account->mapped_bytes();

  // What a reclaim does to an idle frontier: release every row, then the
  // slot is reused from scratch.
  session->Invalidate();
  const std::size_t released = account->TrimTo(0);
  check(released > 0 && account->mapped_bytes() < mapped, "rows_released");
  std::cout << "mapped_bytes=" << mapped << " released_bytes=" << released
            << '\n';

  Run after;
  if (!Generate(*session, prompt, &after))
    return 1;
  check(after.logits.size() == before.logits.size() &&
            std::memcmp(after.logits.data(), before.logits.data(),
                        before.logits.size() * sizeof(float)) == 0,
        "logits_identical");
  check(after.tokens == before.tokens, "tokens_identical");
  check(account->mapped_bytes() >= mapped / 2, "rows_backed_again");
  return failures == 0 ? 0 : 1;
}
