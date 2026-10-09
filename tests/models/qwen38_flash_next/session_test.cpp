#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstring>
#include <future>
#include <iostream>
#include <latch>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/cli/serve/inference_backend.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"
#include "tests/models/qwen27b/sampling_cases.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace sampling = gufo::sampling;

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

void RequireExact(std::span<const float> expected,
                  std::span<const float> actual, const std::string& message) {
  Require(expected.size() == actual.size() &&
              std::memcmp(expected.data(), actual.data(),
                          expected.size_bytes()) == 0,
          message);
}

void CheckSnapshotDuringGraphCapture(const std::shared_ptr<qfn::Model>& model,
                                     bool warm_storage,
                                     bool protect_rows = false) {
  std::string error;
  const auto mode = gufo::core::SessionMode::kAutoregressive;
  auto decoding = model->CreateSession(mode, 128, &error);
  auto reference = model->CreateSession(mode, 128, &error);
  auto frozen = model->CreateSession(mode, 128, &error);
  Require(decoding && reference && frozen, error);
  const auto prompt = model->Tokenize("Continue: red, blue, red, blue,");
  for (auto* session : {decoding.get(), reference.get(), frozen.get()})
    Require(session->Sync(prompt, &error), error);
  auto expected_snapshot =
      (warm_storage ? frozen : reference)->SaveSnapshot(&error);
  Require(expected_snapshot != nullptr, error);
  std::unique_ptr<qfn::SessionSnapshot> prefix_checkpoint;
  if (protect_rows) {
    prefix_checkpoint = frozen->SaveSnapshot(&error);
    Require(prefix_checkpoint != nullptr, error);
    auto extended = prompt;
    extended.push_back(prompt.front());
    Require(frozen->Sync(extended, &error), error);
    expected_snapshot =
        frozen->SaveSnapshot(&error, qfn::Session::SnapshotMode::kMaterialized);
    Require(expected_snapshot != nullptr, error);
  }

  // Warm the one-token shape; its next execution records the decode graph.
  Require(decoding->Evaluate(prompt.front(), &error) &&
              reference->Evaluate(prompt.front(), &error),
          error);
  std::unique_ptr<qfn::SessionSnapshot> concurrent_snapshot;
  std::string snapshot_error;
  unsigned checkpoints = 0;
  decoding->SetCancellationCheck([&] {
    // Forward checks once before capture and again at its first layer.
    if (++checkpoints == 2) {
      try {
        concurrent_snapshot = std::async(std::launch::async, [&] {
                                return frozen->SaveSnapshot(&snapshot_error);
                              }).get();
      } catch (const std::exception& e) {
        snapshot_error = e.what();
        throw;
      }
    }
    return false;
  });
  const bool evaluated = decoding->Evaluate(prompt.back(), &error);
  Require(evaluated, error + "; peer snapshot: " + snapshot_error);
  decoding->SetCancellationCheck({});
  Require(concurrent_snapshot != nullptr, snapshot_error);
  // Restore shares the executor's stream and follows its completed forward.
  // The captured checkpoint must preserve its suffix before the rewind.
  if (prefix_checkpoint)
    Require(frozen->RestoreSnapshot(*prefix_checkpoint, &error), error);
  Require(concurrent_snapshot->bytes().size() ==
                  expected_snapshot->bytes().size() &&
              std::memcmp(concurrent_snapshot->bytes().data(),
                          expected_snapshot->bytes().data(),
                          expected_snapshot->bytes().size()) == 0,
          "decode graph capture changed a peer snapshot");
  Require(reference->Evaluate(prompt.back(), &error), error);
  RequireExact(reference->Logits(), decoding->Logits(),
               "snapshot capture changed decode logits");
  Require(decoding->Evaluate(prompt.front(), &error) &&
              reference->Evaluate(prompt.front(), &error),
          error);
  RequireExact(reference->Logits(), decoding->Logits(),
               "snapshot capture changed graph replay logits");
  std::cout << "peer snapshot bytes, graph capture and replay logits exact\n";
}

void CheckFailureRecovery(const std::shared_ptr<qfn::Model>& model) {
  std::string error;
  auto session = model->CreateSession(
      model->HasMtp() ? gufo::core::SessionMode::kSpeculative
                      : gufo::core::SessionMode::kAutoregressive,
      4096, &error);
  Require(session != nullptr, error);
  const auto initial_bytes = session->AllocatedBytes();
  const auto& c = model->config();
  const auto checkpoint_bytes =
      std::size_t{c.num_layers - c.num_layers / c.full_attention_interval} *
      c.ssm_num_v_heads * c.ssm_head_dim * c.ssm_head_dim * sizeof(float);
  Require(model->SessionBytes(gufo::core::SessionMode::kSpeculative, 4096) >
              initial_bytes + checkpoint_bytes,
          "unused deep rollback state was allocated eagerly");
  const auto pattern =
      model->Tokenize("Explain virtual memory in a short sentence.");
  std::vector<std::int32_t> prompt(2048);
  for (std::size_t i = 0; i < prompt.size(); ++i)
    prompt[i] = pattern[i % pattern.size()];
  Require(session->Sync(prompt, &error), error);
  const std::vector<float> expected(session->Logits().begin(),
                                    session->Logits().end());
  const auto snapshot = session->SaveSnapshot(&error);
  Require(snapshot != nullptr, error);
  Require(snapshot->SizeBytes() < 200ULL * 1024 * 1024,
          "short prompt snapshot retains a full prefill hidden buffer");

  auto peer = model->CreateSession(gufo::core::SessionMode::kAutoregressive, 64,
                                   &error);
  Require(peer && peer->Sync(pattern, &error), error);
  const std::vector<float> peer_expected(peer->Logits().begin(),
                                         peer->Logits().end());
  unsigned checks = 0;
  for (const unsigned stop_at : {3U, 4U, 6U}) {
    peer->Reset();
    checks = 0;
    // Exercise both sides of PLE and reuse its mapped reader buffer in
    // another request immediately after cancellation.
    session->SetCancellationCheck([&] { return ++checks >= stop_at; });
    Require(!session->Evaluate(prompt.front(), &error) && !session->IsValid(),
            "cancelled mutation left a reusable session");
    Require(session->Tokens().empty() && session->Logits().empty() &&
                !session->SaveSnapshot(&error),
            "poisoned state exposed reusable tokens, logits or a snapshot");
    session->SetCancellationCheck({});
    Require(peer->Sync(pattern, &error), error);
    RequireExact(peer_expected, peer->Logits(),
                 "cancelled forward left pinned inputs in use");
    Require(session->Sync(prompt, &error) && session->IsValid(), error);
    RequireExact(expected, session->Logits(),
                 "Sync reused partially mutated state");
  }

  checks = 0;
  session->SetCancellationCheck([&] { return ++checks >= 3; });
  Require(!session->RestoreSnapshot(*snapshot, &error),
          "snapshot restoration ignored cancellation");
  session->SetCancellationCheck({});
  Require(session->Sync(prompt, &error), error);
  RequireExact(expected, session->Logits(),
               "cancelled restore leaked device state");
  Require(session->AllocatedBytes() == initial_bytes,
          "reset retained rollback allocations");
  std::cout
      << "cancelled forward/restore recover exactly; rollback starts empty\n";
}

void CheckRollbackReuse(const std::shared_ptr<qfn::Model>& model) {
  std::string error;
  const auto prompt = model->Tokenize("Continue: red, blue, red, blue,");
  const auto context = static_cast<std::uint32_t>(prompt.size() + 4);
  auto session = model->CreateSession(
      model->HasMtp() ? gufo::core::SessionMode::kSpeculative
                      : gufo::core::SessionMode::kAutoregressive,
      context, &error);
  Require(session != nullptr, error);
  const auto empty_bytes = session->AllocatedBytes();
  Require(session->Sync(prompt, &error), error);
  sampling::SamplerState sampler(
      {.seed = 73},
      std::vector<sampling::TokenId>(prompt.begin(), prompt.end()));
  qfn::Session::DecodeResult step;
  Require(session->DecodeStep(2, sampler, &step, &error, false), error);
  const auto warm_bytes = session->AllocatedBytes();
  Require(warm_bytes > empty_bytes &&
              warm_bytes <= model->SessionBytes(
                                gufo::core::SessionMode::kSpeculative, context),
          "rollback allocation exceeded its configured bound");
  session->ResetDraftPolicy();
  const auto warm = session->SaveSnapshot(&error);
  Require(warm && session->RestoreSnapshot(*warm, &error), error);
  Require(session->AllocatedBytes() == warm_bytes,
          "restore discarded reusable rollback or allocated deeper rows");
  auto replay_sampler = sampler;
  qfn::Session::DecodeResult expected, actual;
  Require(session->DecodeStep(2, sampler, &expected, &error, false), error);
  const std::vector<float> logits(session->Logits().begin(),
                                  session->Logits().end());
  Require(session->RestoreSnapshot(*warm, &error) &&
              session->DecodeStep(2, replay_sampler, &actual, &error, false),
          error);
  Require(actual.tokens == expected.tokens &&
              sampler.rng_state() == replay_sampler.rng_state(),
          "reused rollback changed continuation replay");
  RequireExact(logits, session->Logits(), "reused rollback changed logits");
  while (session->Position() < context)
    Require(session->Evaluate(prompt.front(), &error), error);
  const auto full = session->SaveSnapshot(&error);
  Require(full && session->RestoreSnapshot(*full, &error), error);
  Require(session->AllocatedBytes() == empty_bytes,
          "restore kept rollback that the restored context cannot use");
  session->Reset();
  Require(session->AllocatedBytes() == empty_bytes,
          "explicit reset retained rollback");
  std::cout
      << "rollback restore is bounded, reuses warm rows and replays exactly\n";
}

void CheckImageSnapshotAttachment(const std::shared_ptr<qfn::Model>& model) {
  namespace vision = gufo::models::qwen::vision;
  if (!model->VisionEncoder()) {
    std::cout
        << "image snapshot attachment: matching vision sidecar unavailable\n";
    return;
  }
  const auto image = [](bool blue) {
    auto prompt = std::make_shared<vision::Prompt>();
    const vision::ImageGrid grid{0, 2, 2};
    prompt->tokens = {vision::kImageToken, vision::kImageToken,
                      vision::kImageToken, vision::kImageToken, 42};
    prompt->rope.images.push_back(grid);
    gufo::core::Image pixels;
    pixels.width = pixels.height = 64;
    pixels.pixels.resize(64 * 64 * 3);
    for (std::size_t i = blue ? 2 : 0; i < pixels.pixels.size(); i += 3)
      pixels.pixels[i] = 255;
    prompt->images.push_back({std::move(pixels), grid});
    prompt->images.back().prefix_identity.fill(blue ? 2 : 1);
    prompt->cache_identity.assign(32, blue ? 2 : 1);
    return prompt;
  };
  const auto red = image(false), blue = image(true);
  const std::vector<std::int32_t> tokens(red->tokens.begin(),
                                         red->tokens.end());
  std::string error;
  auto origin = model->CreateSession(
      model->HasMtp() ? gufo::core::SessionMode::kSpeculative
                      : gufo::core::SessionMode::kAutoregressive,
      64, &error);
  auto restored = model->CreateSession(
      model->HasMtp() ? gufo::core::SessionMode::kSpeculative
                      : gufo::core::SessionMode::kAutoregressive,
      64, &error);
  auto text = model->CreateSession(
      model->HasMtp() ? gufo::core::SessionMode::kSpeculative
                      : gufo::core::SessionMode::kAutoregressive,
      64, &error);
  Require(origin && restored && text, error);
  origin->ConfigureVision(red);
  Require(origin->Sync(std::span(tokens).first(2), &error), error);
  const auto snapshot = origin->SaveSnapshot(&error);
  Require(snapshot != nullptr, error);
  Require(!restored->RestoreSnapshot(*snapshot, &error),
          "image snapshot restored without pixels");
  restored->ConfigureVision(blue);
  Require(restored->Sync(std::span(tokens).first(2), &error), error);
  Require(!restored->RestoreSnapshot(*snapshot, &error),
          "image snapshot inherited another image's pixels");
  restored->ConfigureVision(red);
  Require(restored->RestoreSnapshot(*snapshot, &error) &&
              restored->Sync(tokens, &error) && origin->Sync(tokens, &error),
          error);
  RequireExact(origin->Logits(), restored->Logits(),
               "attached image snapshot changed continuation logits");

  const std::array<std::int32_t, 2> plain{42, 43};
  Require(text->Sync(plain, &error), error);
  const auto text_snapshot = text->SaveSnapshot(&error);
  Require(text_snapshot && restored->RestoreSnapshot(*text_snapshot, &error),
          error);
  Require(text->Evaluate(44, &error) && restored->Evaluate(44, &error), error);
  RequireExact(text->Logits(), restored->Logits(),
               "text snapshot retained stale image input");
  std::cout
      << "image snapshot attachment, replacement and text restore exact\n";
}

void CheckPrefillChunks(const std::shared_ptr<qfn::Model>& model) {
  std::string error;
  const auto pattern = model->Tokenize(
      "Virtual memory maps pages to physical storage. "
      "A train travels sixty kilometers per hour. "
      "Continue red, green, blue, red, green, blue. ");
  Require(!pattern.empty(), "empty chunk fixture");
  for (const auto [length, boundary] :
       {std::pair{136U, 94U}, std::pair{136U, 103U}, std::pair{136U, 104U},
        std::pair{136U, 127U}, std::pair{136U, 128U}, std::pair{136U, 135U},
        std::pair{2048U, 1025U}, std::pair{4096U, 2048U}}) {
    std::vector<std::int32_t> tokens(length);
    for (std::size_t i = 0; i < tokens.size(); ++i)
      tokens[i] = pattern[i % pattern.size()];
    auto bulk = model->CreateSession(
        model->HasMtp() ? gufo::core::SessionMode::kSpeculative
                        : gufo::core::SessionMode::kAutoregressive,
        6145, &error);
    auto split = model->CreateSession(
        model->HasMtp() ? gufo::core::SessionMode::kSpeculative
                        : gufo::core::SessionMode::kAutoregressive,
        6145, &error);
    Require(bulk && split, error);
    Require(bulk->Sync(tokens, &error) &&
                split->Sync(std::span(tokens).first(boundary), &error) &&
                split->Sync(tokens, &error),
            error);
    std::vector<sampling::TokenId> history(tokens.begin(), tokens.end());
    for (unsigned step = 0; step < 4; ++step) {
      const auto logits = bulk->Logits();
      RequireExact(
          logits, split->Logits(),
          "prefill chunking changed logits: length=" + std::to_string(length) +
              " boundary=" + std::to_string(boundary) +
              " step=" + std::to_string(step));
      const auto token = static_cast<std::int32_t>(
          std::max_element(logits.begin(), logits.end()) - logits.begin());
      Require(bulk->Evaluate(token, &error) && split->Evaluate(token, &error),
              error);
      history.push_back(token);
    }
    if (model->HasMtp()) {
      const sampling::SamplingConfig config{
          .temperature = 0.8F, .top_k = 20, .top_p = 0.95F, .seed = 73};
      sampling::SamplerState a(config, history), b(config, history);
      for (unsigned generated = 0; generated < 16;) {
        qfn::Session::DecodeResult first, second;
        Require(
            bulk->DecodeStep(16 - generated, a, &first, &error, false) &&
                split->DecodeStep(16 - generated, b, &second, &error, false),
            error);
        Require(!first.tokens.empty() && first.tokens == second.tokens &&
                    a.rng_state() == b.rng_state(),
                "prefill chunking changed sampled MTP replay: length=" +
                    std::to_string(length) +
                    " boundary=" + std::to_string(boundary));
        RequireExact(bulk->Logits(), split->Logits(),
                     "prefill chunking changed MTP target logits");
        generated += first.tokens.size();
      }
    }
    std::cout << "prefill chunks: length=" << length << " boundary=" << boundary
              << " four logit rows and sampled MTP replay exact\n"
              << std::flush;
  }
}

void CheckMtpCacheReplay(const std::shared_ptr<qfn::Model>& model) {
  namespace qt = gufo::tokenization;
  std::string error;
  for (const auto seed : {1U, 2U}) {
    std::string system = "Case " + std::to_string(seed) + ". ";
    for (unsigned i = 0; i < 32; ++i)
      system += "Follow the user instruction carefully and answer accurately. ";
    const std::array messages{
        qt::ChatMessage{qt::ChatRole::kSystem, system},
        qt::ChatMessage{qt::ChatRole::kUser,
                        "Name three rivers in Europe and one fact about each."},
        qt::ChatMessage{qt::ChatRole::kAssistant,
                        "The Danube, the Rhine and the Loire."},
        qt::ChatMessage{qt::ChatRole::kUser,
                        "Now write a short story about one of them."}};
    const auto encoded = qt::QwenChatTemplate::RenderAndTokenize(
        model->tokenizer(), messages, {.enable_thinking = false}, &error);
    Require(encoded.has_value(), error);
    const std::vector<std::int32_t> prompt(encoded->begin(), encoded->end());
    Require(prompt.size() == 355, "MTP cache replay fixture changed");
    const sampling::SamplingConfig config{
        .temperature = 0.8F, .top_k = 20, .top_p = 0.95F, .seed = seed};
    const std::vector<sampling::TokenId> history(prompt.begin(), prompt.end());
    std::vector<std::int32_t> expected;
    std::vector<std::size_t> expected_widths;
    std::uint64_t expected_rng = 0;
    // The server's immutable frontier excludes the seven-token generation
    // suffix. Rebuilding the same prompt must not depend on whether a
    // previous turn already supplied its first 320 tokens.
    for (const auto boundary : {320U, 348U}) {
      auto session = model->CreateSession(gufo::core::SessionMode::kSpeculative,
                                          6145, &error);
      Require(session &&
                  session->Sync(std::span(prompt).first(boundary), &error) &&
                  session->Sync(prompt, &error),
              error);
      const auto snapshot = session->SaveSnapshot(&error);
      Require(snapshot != nullptr, error);
      for (unsigned restore = 0; restore < 2; ++restore) {
        if (restore)
          Require(session->RestoreSnapshot(*snapshot, &error), error);
        session->ResetDraftPolicy();
        sampling::SamplerState sampler(config, history);
        std::vector<std::int32_t> tokens;
        std::vector<std::size_t> widths;
        while (tokens.size() < 200) {
          qfn::Session::DecodeResult step;
          Require(session->DecodeStep(200 - tokens.size(), sampler, &step,
                                      &error, false),
                  error);
          Require(!step.tokens.empty(), "MTP replay made no progress");
          tokens.insert(tokens.end(), step.tokens.begin(), step.tokens.end());
          widths.push_back(step.tokens.size());
        }
        if (expected.empty()) {
          expected = tokens;
          expected_widths = widths;
          expected_rng = sampler.rng_state();
        } else {
          Require(tokens == expected && widths == expected_widths &&
                      sampler.rng_state() == expected_rng,
                  "MTP cache replay changed tokens, acceptance or RNG: seed=" +
                      std::to_string(seed) +
                      " boundary=" + std::to_string(boundary) +
                      " restore=" + std::to_string(restore));
        }
        std::cout << "MTP cache replay seed=" << seed
                  << " boundary=" << boundary << " restore=" << restore
                  << " tokens=200 exact=1\n"
                  << std::flush;
      }
    }
  }
}

void CheckBatchFailureIsolation(const std::shared_ptr<qfn::Model>& model) {
  std::string error;
  const auto prompt = model->Tokenize("Repeat red, blue, red, blue,");
  for (const auto mode : {gufo::core::SessionMode::kAutoregressive,
                          gufo::core::SessionMode::kSpeculative}) {
    auto first = model->CreateSession(mode, 256, &error);
    auto second = model->CreateSession(mode, 256, &error);
    Require(first && second && first->Sync(prompt, &error), error);
    const auto snapshot = first->SaveSnapshot(&error);
    Require(snapshot && second->RestoreSnapshot(*snapshot, &error), error);
    const auto token = prompt.front();
    const std::array baseline{
        qfn::Session::AdvanceRequest{first.get(), token},
        qfn::Session::AdvanceRequest{second.get(), token}};
    Require(qfn::Session::EvaluateBatch(baseline, &error), error);
    const std::vector<float> expected(second->Logits().begin(),
                                      second->Logits().end());
    Require(first->RestoreSnapshot(*snapshot, &error) &&
                second->RestoreSnapshot(*snapshot, &error),
            error);
    unsigned checks = 0;
    first->SetCancellationCheck([&] { return ++checks >= 12; });
    std::array<qfn::Session::BatchOutcome, 2> outcomes;
    const std::array cancelled{
        qfn::Session::AdvanceRequest{first.get(), token, &outcomes[0]},
        qfn::Session::AdvanceRequest{second.get(), token, &outcomes[1]}};
    Require(!qfn::Session::EvaluateBatch(cancelled, &error),
            "batch cancellation not reported");
    Require(!outcomes[0].completed && !outcomes[0].error.empty() &&
                outcomes[1].completed && second->IsValid() && !first->IsValid(),
            "cancellation must invalidate only the partially executed session");
    RequireExact(expected, second->Logits(),
                 "cancelled peer changed target logits");
    first->SetCancellationCheck({});
    Require(first->RestoreSnapshot(*snapshot, &error) &&
                second->RestoreSnapshot(*snapshot, &error),
            error);
    const std::array invalid{
        qfn::Session::AdvanceRequest{first.get(), -1, &outcomes[0]},
        qfn::Session::AdvanceRequest{second.get(), token, &outcomes[1]}};
    Require(!qfn::Session::EvaluateBatch(invalid, &error) && first->IsValid() &&
                second->IsValid() && outcomes[1].completed,
            "invalid preparation poisoned a peer or the untouched session");
    RequireExact(expected, second->Logits(),
                 "invalid peer changed target logits");
    Require(first->RestoreSnapshot(*snapshot, &error) &&
                second->RestoreSnapshot(*snapshot, &error),
            error);
    const sampling::SamplingConfig config{
        .temperature = 0.7F, .top_p = 0.9F, .seed = 73};
    sampling::SamplerState control(config), a(config), b(config);
    qfn::Session::DecodeResult reference, ignored, actual;
    Require(second->DecodeStep(3, control, &reference, &error, false), error);
    const std::vector<float> speculative_logits(second->Logits().begin(),
                                                second->Logits().end());
    Require(second->RestoreSnapshot(*snapshot, &error), error);
    const std::array invalid_decode{
        qfn::Session::DecodeRequest{first.get(), 0, &a, &ignored, false,
                                    &outcomes[0]},
        qfn::Session::DecodeRequest{second.get(), 3, &b, &actual, false,
                                    &outcomes[1]}};
    Require(!qfn::Session::DecodeBatch(invalid_decode, &error) &&
                first->IsValid() && outcomes[1].completed &&
                actual.tokens == reference.tokens &&
                b.rng_state() == control.rng_state(),
            "invalid decode peer changed sampling replay");
    RequireExact(speculative_logits, second->Logits(),
                 "invalid decode peer changed logits");
    if (mode == gufo::core::SessionMode::kSpeculative) {
      Require(first->RestoreSnapshot(*snapshot, &error) &&
                  second->RestoreSnapshot(*snapshot, &error),
              error);
      a = sampling::SamplerState(config);
      b = sampling::SamplerState(config);
      checks = 0;
      first->SetCancellationCheck([&] { return ++checks >= 6; });
      const std::array mtp_cancel{
          qfn::Session::DecodeRequest{first.get(), 3, &a, &ignored, false,
                                      &outcomes[0]},
          qfn::Session::DecodeRequest{second.get(), 3, &b, &actual, false,
                                      &outcomes[1]}};
      Require(!qfn::Session::DecodeBatch(mtp_cancel, &error) &&
                  !outcomes[0].completed && outcomes[1].completed &&
                  second->IsValid() && actual.tokens == reference.tokens &&
                  b.rng_state() == control.rng_state(),
              "cancelled MTP peer changed sampling replay");
      RequireExact(speculative_logits, second->Logits(),
                   "cancelled MTP peer changed logits");
    }
  }
  std::cout << "batch preparation and mid-execution cancellation isolate peers "
               "and sampling replay\n";
}

void CheckBatchedSessions(const std::shared_ptr<qfn::Model>& model) {
  std::string error;
  std::size_t sampled_rejections = 0;
  std::size_t sampled_acceptances = 0;
  std::vector<std::unique_ptr<qfn::Session>> serial, batched;
  std::vector<sampling::SamplerState> serial_samplers, batch_samplers;
  const auto cases = gufo::test::QwenSamplingCases();
  for (std::size_t i = 0; i < 8; ++i) {
    serial.push_back(model->CreateSession(
        model->HasMtp() ? gufo::core::SessionMode::kSpeculative
                        : gufo::core::SessionMode::kAutoregressive,
        6145, &error));
    batched.push_back(model->CreateSession(
        model->HasMtp() ? gufo::core::SessionMode::kSpeculative
                        : gufo::core::SessionMode::kAutoregressive,
        6145, &error));
    Require(serial.back() && batched.back(), error);
    auto prompt = model->Tokenize("Batch request " + std::to_string(i) +
                                  ": Continue red, blue, blue, red,");
    // One member crosses the sparse-attention boundary while the others
    // remain short; interleaving must not share positions or pooling state.
    if (i == 1) {
      const auto pattern = prompt;
      prompt.resize(4095);
      for (std::size_t t = pattern.size(); t < prompt.size(); ++t) {
        prompt[t] = pattern[t % pattern.size()];
      }
    }
    Require(serial.back()->Sync(prompt, &error), error);
    auto snapshot = serial.back()->SaveSnapshot(&error);
    Require(snapshot && batched.back()->RestoreSnapshot(*snapshot, &error),
            error);
    std::vector<sampling::TokenId> history(prompt.begin(), prompt.end());
    auto config = cases[(i * 3) % cases.size()].config;
    if (i % 2 == 0) {
      config.temperature = 0;
      config.repeat_penalty = i == 0 ? 0.7F : 1.3F;
      config.repeat_last_n = 3;
      config.frequency_penalty = i == 0 ? -0.4F : 0.2F;
      config.presence_penalty = i == 0 ? -1.5F : 1.5F;
    }
    serial_samplers.emplace_back(config, history);
    batch_samplers.emplace_back(config, history);
  }
  // Exercise the AR serving entrypoint, including a session crossing 4K.
  for (const std::size_t width : {2, 4, 6, 8}) {
    // Reuse cohorts with different tokens and positions in the same
    // allocations.
    for (unsigned replay = 0; replay < 2; ++replay) {
      std::vector<qfn::Session::AdvanceRequest> advances;
      for (std::size_t i = 0; i < width; ++i) {
        const auto token = static_cast<std::int32_t>(
            std::max_element(serial[i]->Logits().begin(),
                             serial[i]->Logits().end()) -
            serial[i]->Logits().begin());
        Require(serial[i]->Evaluate(token, &error), error);
        advances.push_back({batched[i].get(), token});
        serial_samplers[i].Accept(token);
        batch_samplers[i].Accept(token);
      }
      Require(qfn::Session::EvaluateBatch(advances, &error), error);
      for (std::size_t i = 0; i < width; ++i) {
        RequireExact(serial[i]->Logits(), batched[i]->Logits(),
                     "batched AR frontier differs at C" +
                         std::to_string(width) + " row " + std::to_string(i));
      }
    }
    // Different budgets create ragged chains and include ordinary one-token
    // decoding in the same batch as speculative verification.
    for (unsigned cycle = 0; cycle < 2; ++cycle) {
      std::vector<qfn::Session::DecodeResult> expected(width), actual(width);
      std::vector<qfn::Session::DecodeRequest> requests;
      std::vector<qfn::Session::SpeculativeStats> before;
      for (std::size_t i = 0; i < width; ++i) {
        const std::size_t budget = 1 + (i * 3 + cycle * 5 + 7) % 8;
        before.push_back(serial[i]->Statistics());
        Require(serial[i]->DecodeStep(budget, serial_samplers[i], &expected[i],
                                      &error, false),
                error);
        requests.push_back(
            {batched[i].get(), budget, &batch_samplers[i], &actual[i], false});
      }
      if (cycle % 2 != 0) {
        std::reverse(requests.begin(), requests.end());
      }
      Require(qfn::Session::DecodeBatch(requests, &error), error);
      for (std::size_t i = 0; i < width; ++i) {
        const auto a = serial[i]->Statistics();
        const auto b = batched[i]->Statistics();
        Require(expected[i].tokens == actual[i].tokens &&
                    expected[i].stop == actual[i].stop &&
                    serial_samplers[i].rng_state() ==
                        batch_samplers[i].rng_state() &&
                    a.cycles == b.cycles && a.drafted == b.drafted &&
                    a.accepted == b.accepted &&
                    std::equal(serial[i]->Tokens().begin(),
                               serial[i]->Tokens().end(),
                               batched[i]->Tokens().begin(),
                               batched[i]->Tokens().end()),
                "batched decode tokens, RNG or acceptance differ");
        RequireExact(serial[i]->Logits(), batched[i]->Logits(),
                     "batched MTP frontier differs at C" +
                         std::to_string(width) + " row " + std::to_string(i) +
                         " cycle " + std::to_string(cycle));
        Require(std::equal(serial_samplers[i].history().begin(),
                           serial_samplers[i].history().end(),
                           batch_samplers[i].history().begin(),
                           batch_samplers[i].history().end()),
                "batched sampling history differs");
        // Probe a copy so the test also compares pending residual draws at
        // the final cycle, without consuming either request's real state.
        auto serial_next = serial_samplers[i];
        auto batch_next = batch_samplers[i];
        Require(serial_next.Sample(serial[i]->Logits()) ==
                        batch_next.Sample(batched[i]->Logits()) &&
                    serial_next.rng_state() == batch_next.rng_state(),
                "batched next draw or deferred residual differs");
        if (serial_samplers[i].config().uses_random_sampling()) {
          const auto drafted = a.drafted - before[i].drafted;
          const auto accepted = a.accepted - before[i].accepted;
          sampled_acceptances += accepted;
          if (accepted < drafted) {
            ++sampled_rejections;
            // stop_at_eos is false: a sampled rejection always defers its
            // correction. Retrieving it must not draw again from p.
            Require(serial_next.rng_state() == serial_samplers[i].rng_state(),
                    "sampled rejection lost its deferred correction");
          }
        }
      }
    }
    std::cout << "batch C" << width << " AR_MTP_logits_RNG_exact=1\n"
              << std::flush;
  }
  Require(
      sampled_acceptances > 0 && sampled_rejections > 0,
      "batch test must exercise sampled acceptance and residual correction");
  std::cout << "batch sampled_accepted=" << sampled_acceptances
            << " rejected_cycles=" << sampled_rejections
            << " history_and_residual_exact=1\n"
            << std::flush;
  // A complete state comparison catches recurrent/hidden differences that a
  // short output comparison could miss.
  for (std::size_t i = 0; i < serial.size(); ++i) {
    const auto a = serial[i]->SaveSnapshot(&error);
    const auto b = batched[i]->SaveSnapshot(&error);
    Require(a && b &&
                std::equal(a->bytes().begin(), a->bytes().end(),
                           b->bytes().begin(), b->bytes().end()),
            "batched snapshot state differs: " + std::to_string(i));
  }
  std::cout << "batch independent_state_exact=1\n" << std::flush;
}

void CheckExecutionModes(const std::shared_ptr<qfn::Model>& model,
                         std::uint32_t prompt_tokens = 4096,
                         bool with_image = false) {
  using gufo::core::SessionMode;
  namespace vision = gufo::models::qwen::vision;
  std::string error;
  auto ar = model->CreateSession(SessionMode::kAutoregressive,
                                 prompt_tokens + 16, &error);
  auto mtp = model->CreateSession(SessionMode::kSpeculative, prompt_tokens + 16,
                                  &error);
  Require(ar && mtp, error);
  Require(ar->AllocatedBytes() < mtp->AllocatedBytes(),
          "AR allocated predictor state");
  const auto pattern = model->Tokenize(
      "Review this transaction log and identify the failed request.\n"
      "def retry(items):\n"
      "    return [item for item in items if item['status'] != 200]\n"
      "{\"request\":\"tools.read\",\"status\":503,\"retry_after\":2.5}\n"
      "The train travels 60 km in 45 minutes. Explain the calculation.\n"
      "Café, e\u0301, 日本語, العربية. Keep Unicode and literal <tags> "
      "intact.\n");
  Require(!pattern.empty(), "empty execution-mode fixture");
  std::vector<std::int32_t> prompt(prompt_tokens);
  for (std::size_t i = 0; i < prompt.size(); ++i)
    prompt[i] = pattern[i % pattern.size()];
  if (with_image) {
    Require(model->VisionEncoder() != nullptr && prompt_tokens >= 4,
            "image prefill requires its encoder and four image tokens");
    auto image = std::make_shared<vision::Prompt>();
    const vision::ImageGrid grid{0, 2, 2};
    std::fill_n(prompt.begin(), 4, vision::kImageToken);
    image->tokens.assign(prompt.begin(), prompt.end());
    image->rope.images.push_back(grid);
    gufo::core::Image pixels;
    pixels.width = pixels.height = 64;
    pixels.pixels.resize(64 * 64 * 3);
    for (std::size_t i = 0; i < pixels.pixels.size(); ++i)
      pixels.pixels[i] = static_cast<std::uint8_t>((i * 37 + i / 64) % 256);
    image->images.push_back({std::move(pixels), grid});
    image->images.back().prefix_identity.fill(3);
    image->cache_identity.assign(32, 3);
    ar->ConfigureVision(image);
    mtp->ConfigureVision(image);
  }
  Require(ar->Sync(prompt, &error) && mtp->Sync(prompt, &error), error);
  RequireExact(ar->Logits(), mtp->Logits(),
               "execution mode changes target prefill logits");
  const auto ar_snapshot = ar->SaveSnapshot(&error);
  const auto mtp_snapshot = mtp->SaveSnapshot(&error);
  Require(ar_snapshot && mtp_snapshot &&
              ar_snapshot->SizeBytes() < mtp_snapshot->SizeBytes(),
          "AR snapshot includes predictor state");
  const auto anchor = static_cast<std::int32_t>(
      std::max_element(ar->Logits().begin(), ar->Logits().end()) -
      ar->Logits().begin());
  const std::array<qfn::Session::AdvanceRequest, 2> mixed{
      {{ar.get(), anchor}, {mtp.get(), anchor}}};
  Require(qfn::Session::EvaluateBatch(mixed, &error), error);
  RequireExact(ar->Logits(), mtp->Logits(),
               "mixed execution modes contaminate target logits");
  const sampling::SamplingConfig config{
      .temperature = 0.8F, .top_k = 20, .top_p = 0.95F, .seed = 73};
  sampling::SamplerState sampler(config);
  qfn::Session::DecodeResult step;
  Require(ar->DecodeStep(8, sampler, &step, &error, false), error);
  Require(step.tokens.size() == 1 && ar->Statistics().drafted == 0,
          "AR session with a resident sidecar executed speculative decoding");
  const std::vector<float> expected(ar->Logits().begin(), ar->Logits().end());
  for (bool serialized : {false, true}) {
    if (serialized) {
      std::vector<std::uint8_t> bytes(ar_snapshot->SizeBytes());
      Require(ar_snapshot->CopyTo(bytes), "serialize AR frontier");
      Require(ar->RestoreSnapshot(bytes, &error), error);
    } else {
      Require(ar->RestoreSnapshot(*ar_snapshot, &error), error);
    }
    Require(ar->Evaluate(anchor, &error), error);
    sampling::SamplerState replay(config);
    qfn::Session::DecodeResult resumed;
    Require(ar->DecodeStep(8, replay, &resumed, &error, false), error);
    Require(resumed.tokens == step.tokens &&
                replay.rng_state() == sampler.rng_state(),
            "AR frontier changed sampled replay");
    RequireExact(expected, ar->Logits(),
                 "AR frontier changed resumed target logits");
  }
  Require(!mtp->RestoreSnapshot(*ar_snapshot, &error) &&
              !ar->RestoreSnapshot(*mtp_snapshot, &error),
          "snapshots crossed execution modes");
  std::cout << "execution_tokens=" << prompt_tokens << " image=" << with_image
            << " execution_modes=independent AR_predictor_bytes=0 "
               "mixed_batch_exact=1 RAM_serialized_sampled_replay=1\n"
            << std::flush;
}

void CheckServingEos(const std::shared_ptr<qfn::Model>& model) {
  namespace server = gufo::server;
  using Backend = server::InferenceBackend;
  Backend backend;
  std::string error;
  server::TextSpeculativeConfig options;
  options.backend = server::TextSpeculativeBackend::kMtp;
  Require(backend.load(model, &error, 128, 2, {}, {}, options), error);
  // Synthetic grammar pieces make the boundary deterministic: any ordinary
  // first token completes the empty JSON object; only stop tokens may follow.
  // Actual token IDs, EOS recognition, GPU execution and checkpoints stay real.
  auto constraint = std::make_shared<sampling::TokenConstraint>();
  constraint->grammar = sampling::JsonConstraint::Compile(
      gufo::json::parse(
          R"({"type":"object","properties":{},"additionalProperties":false})"),
      true);
  constraint->vocabulary = std::make_shared<sampling::ConstraintVocabulary>(
      model->VocabSize(), [&](std::uint32_t token) {
        return sampling::ConstraintVocabulary::Piece{
            .text = model->IsStopToken(token) ? "" : "{}",
            .stop = model->IsStopToken(token),
        };
      });
  sampling::SamplingConfig config;
  config.constraint = std::move(constraint);
  const std::array<std::string, 2> prompts{
      "EOS fixture one: continue red, blue,",
      "EOS fixture two: continue green, yellow,"};
  std::array<Backend::Result, 2> references;
  for (std::size_t row = 0; row < prompts.size(); ++row) {
    references[row] = backend.complete(prompts[row], 8, config);
    Require(
        references[row].tokens.size() == 1 &&
            !model->IsStopToken(references[row].tokens.front()),
        "scalar MTP executed EOS or trailing work after the forced boundary");
    const auto replay = backend.complete(prompts[row], 8, config);
    Require(
        replay.tokens == references[row].tokens && replay.prefill_tokens == 0,
        "scalar EOS checkpoint did not replay exactly without prefill");
  }
  std::array<std::future<Backend::Result>, 2> pending;
  std::latch ready(pending.size());
  for (std::size_t row = 0; row < pending.size(); ++row) {
    pending[row] = std::async(
        std::launch::async, [&backend, &ready, &prompts, &config, row] {
          ready.arrive_and_wait();
          return backend.complete(prompts[row], 8, config);
        });
  }
  std::size_t batched = 0;
  for (std::size_t row = 0; row < pending.size(); ++row) {
    const auto result = pending[row].get();
    Require(result.tokens == references[row].tokens,
            "batched EOS accounting differs from the scalar checkpoint");
    batched += result.physical_execution_width > 1;
  }
  Require(batched > 0, "EOS fixture did not exercise batched MTP");
  // Completions explicitly ignoring EOS must retain their requested budget.
  const auto ignored_request =
      backend.start_complete(prompts.front(), 8, config, {}, false, true);
  const auto ignored = ignored_request->Wait({}, {});
  Require(
      ignored.tokens.size() == 8 &&
          std::all_of(ignored.tokens.begin() + 1, ignored.tokens.end(),
                      [&](auto token) { return model->IsStopToken(token); }),
      "MTP ignored the request's explicit ignore_eos policy");
  std::latch mixed_ready(pending.size());
  for (std::size_t row = 0; row < pending.size(); ++row) {
    pending[row] = std::async(
        std::launch::async, [&backend, &mixed_ready, &prompts, &config, row] {
          mixed_ready.arrive_and_wait();
          const auto request = backend.start_complete(prompts[row], 8, config,
                                                      {}, false, row == 1);
          return request->Wait({}, {});
        });
  }
  const auto stopped = pending[0].get();
  const auto continuing = pending[1].get();
  Require(
      stopped.tokens == references[0].tokens && continuing.tokens.size() == 8 &&
          std::all_of(continuing.tokens.begin() + 1, continuing.tokens.end(),
                      [&](auto token) { return model->IsStopToken(token); }) &&
          (stopped.physical_execution_width > 1 ||
           continuing.physical_execution_width > 1),
      "mixed MTP batch did not respect each row's EOS policy");
  std::cout << "serving_eos scalar_and_batch_exact=1 checkpoint_replay=1 "
               "ignore_eos_budget=8 batched_requests="
            << batched << '\n'
            << std::flush;
}

void CheckServingSampling(const std::shared_ptr<qfn::Model>& model) {
  namespace server = gufo::server;
  using Backend = server::InferenceBackend;
  Backend ar, mtp;
  std::string error;
  Require(ar.load(model, &error, 6145, 2), error);
  server::TextSpeculativeConfig options;
  options.backend = server::TextSpeculativeBackend::kMtp;
  Require(mtp.load(model, &error, 6145, 2, {}, {}, options), error);
  struct Reference {
    gufo::test::SamplingCase test;
    std::string prompt;
    Backend::Result ar, mtp;
  };
  std::vector<Reference> references;
  for (const auto& test : gufo::test::QwenSamplingCases()) {
    const std::string prompt = "Sampling " + std::string(test.name) +
                               ": Continue red, blue, blue, red,";
    auto ordinary = ar.complete(prompt, 8, test.config);
    auto speculative = mtp.complete(prompt, 8, test.config);
    // Exercise the serving handoff against the model session directly.
    // Replaying HTTP alone would not detect a consistently dropped residual.
    auto direct = model->CreateSession(
        model->HasMtp() ? gufo::core::SessionMode::kSpeculative
                        : gufo::core::SessionMode::kAutoregressive,
        6145, &error);
    const auto encoded = model->Tokenize(prompt);
    Require(direct && direct->Sync(encoded, &error), error);
    const std::vector<sampling::TokenId> history(encoded.begin(),
                                                 encoded.end());
    sampling::SamplerState direct_sampler(test.config, history);
    std::vector<sampling::TokenId> direct_tokens;
    while (direct_tokens.size() < 8) {
      qfn::Session::DecodeResult step;
      Require(direct->DecodeStep(8 - direct_tokens.size(), direct_sampler,
                                 &step, &error),
              error);
      direct_tokens.insert(direct_tokens.end(), step.tokens.begin(),
                           step.tokens.end());
      if (step.stop)
        break;
      Require(!step.tokens.empty(),
              "direct serving reference made no progress");
    }
    const auto direct_stats = direct->Statistics();
    Require(direct_tokens == speculative.tokens &&
                direct_stats.drafted == speculative.draft_tokens &&
                direct_stats.accepted == speculative.draft_accepted_tokens,
            "serving lost sampling state: " + std::string(test.name));
    Require(!ordinary.tokens.empty() && ordinary.draft_tokens == 0 &&
                speculative.draft_tokens > 0,
            "sampling strategy did not exercise AR and MTP");
    Require(ordinary.physical_execution_width == 1 &&
                speculative.physical_execution_width == 1,
            "serial completion reported a batched execution");
    if (!test.config.uses_random_sampling()) {
      Require(ordinary.tokens == speculative.tokens,
              "serving greedy AR/MTP mismatch: " + std::string(test.name));
    }
    for (const bool use_mtp : {false, true}) {
      auto& backend = use_mtp ? mtp : ar;
      const auto& expected = use_mtp ? speculative : ordinary;
      const auto replay = backend.complete(prompt, 8, test.config);
      Require(
          replay.tokens == expected.tokens &&
              replay.draft_tokens == expected.draft_tokens &&
              replay.draft_accepted_tokens == expected.draft_accepted_tokens,
          "serving replay changed sampling or draft acceptance");
    }
    references.push_back(
        {test, prompt, std::move(ordinary), std::move(speculative)});
    std::cout << "serving sampling=" << test.name << " replay_exact=1\n"
              << std::flush;
  }
  // Every strategy also runs with shared projections in a concurrent pair.
  for (const bool use_mtp : {false, true}) {
    auto& backend = use_mtp ? mtp : ar;
    std::size_t batched_requests = 0;
    for (std::size_t offset = 0; offset < references.size(); offset += 2) {
      std::array<std::future<Backend::Result>, 2> pending;
      std::latch ready(pending.size());
      for (std::size_t row = 0; row < pending.size(); ++row) {
        const auto* saved = &references[(offset + row) % references.size()];
        pending[row] =
            std::async(std::launch::async, [&backend, &ready, saved] {
              ready.arrive_and_wait();
              return backend.complete(saved->prompt, 8, saved->test.config);
            });
      }
      for (std::size_t row = 0; row < pending.size(); ++row) {
        const auto& saved = references[(offset + row) % references.size()];
        const auto& expected = use_mtp ? saved.mtp : saved.ar;
        const auto actual = pending[row].get();
        Require(actual.tokens == expected.tokens &&
                    (!saved.test.config.uses_random_sampling() ||
                     (actual.draft_tokens == expected.draft_tokens &&
                      actual.draft_accepted_tokens ==
                          expected.draft_accepted_tokens)),
                "interleaving changed serving sampling or acceptance");
        if (actual.physical_execution_width > 1) {
          Require(actual.physical_execution_width == 2 &&
                      actual.execution_plan == "batched-w2",
                  "paired completion reported an incorrect execution plan");
          ++batched_requests;
        }
      }
    }
    Require(batched_requests > 0,
            "concurrent sampling checks did not report batched execution");
    // Greedy/AR prefixes are budget-independent. Sampled MTP consumes
    // proposal/rejection draws, so each budget must replay its own result.
    for (const auto index : {std::size_t{0}, references.size() - 2}) {
      const auto& saved = references[index];
      for (const auto budget : {1U, 2U, 3U}) {
        const auto actual =
            backend.complete(saved.prompt, budget, saved.test.config);
        if (use_mtp && saved.test.config.uses_random_sampling()) {
          const auto replay =
              backend.complete(saved.prompt, budget, saved.test.config);
          Require(
              actual.tokens.size() <= budget &&
                  actual.tokens == replay.tokens &&
                  actual.draft_tokens == replay.draft_tokens &&
                  actual.draft_accepted_tokens == replay.draft_accepted_tokens,
              "sampled MTP budget or replay differs");
        } else {
          Require(actual.tokens.size() == std::min<std::size_t>(
                                              budget, saved.ar.tokens.size()) &&
                      std::equal(actual.tokens.begin(), actual.tokens.end(),
                                 saved.ar.tokens.begin()),
                  "output budget changed the AR/greedy prefix");
        }
      }
    }
    std::cout << "serving strategies=" << references.size()
              << " mtp=" << use_mtp << " batched_requests=" << batched_requests
              << " C2_and_budgets_exact=1\n"
              << std::flush;
  }
}

int main(int argc, char** argv) {
  const bool execution_only =
      argc == 7 && std::string_view(argv[5]) == "--execution-only";
  std::uint32_t execution_tokens = 4096;
  const bool batch_only =
      argc == 6 && std::string_view(argv[5]) == "--batch-only";
  const bool prefill_only =
      argc == 6 && std::string_view(argv[5]) == "--prefill-only";
  const bool sampling_only =
      argc == 6 && std::string_view(argv[5]) == "--sampling-only";
  const bool cache_only =
      argc == 6 && std::string_view(argv[5]) == "--cache-only";
  const bool eos_only = argc == 6 && std::string_view(argv[5]) == "--eos-only";
  if ((argc != 5 && !batch_only && !prefill_only && !sampling_only &&
       !cache_only && !eos_only && !execution_only) ||
      std::string_view(argv[1]) != "--model" ||
      std::string_view(argv[3]) != "--mtp-model") {
    std::cerr << "Usage: session_test --model FIRST.gguf --mtp-model MTP.gguf "
                 "[--batch-only | --prefill-only | --sampling-only | "
                 "--cache-only | --eos-only | --execution-only TOKENS]\n";
    return 77;
  }
  if (execution_only) {
    const std::string_view value(argv[6]);
    const auto [end, ec] = std::from_chars(
        value.data(), value.data() + value.size(), execution_tokens);
    if (ec != std::errc{} || end != value.data() + value.size() ||
        execution_tokens == 0 || execution_tokens > 262128) {
      std::cerr << "--execution-only requires 1..262128 prompt tokens\n";
      return 2;
    }
  }
  try {
    std::string error;
    auto model = qfn::Model::Load(
        argv[2],
        {.max_context = execution_only ? execution_tokens + 16 : 6145,
         .mtp_model_path = argv[4],
         .max_draft_tokens = 7},
        &error);
    Require(model != nullptr, error);
    if (execution_only) {
      CheckExecutionModes(model, execution_tokens);
      return 0;
    }
    if (eos_only) {
      CheckServingEos(model);
      return 0;
    }
    if (cache_only) {
      CheckMtpCacheReplay(model);
      return 0;
    }
    CheckSnapshotDuringGraphCapture(model, false);
    CheckSnapshotDuringGraphCapture(model, true);
    CheckSnapshotDuringGraphCapture(model, true, true);
    CheckExecutionModes(model);
    if (sampling_only) {
      CheckServingEos(model);
      CheckServingSampling(model);
      return 0;
    }
    if (!batch_only) {
      CheckExecutionModes(model, 2048, true);
      CheckPrefillChunks(model);
    }
    if (prefill_only)
      return 0;
    if (!batch_only)
      CheckMtpCacheReplay(model);
    CheckFailureRecovery(model);
    CheckRollbackReuse(model);
    CheckImageSnapshotAttachment(model);
    CheckBatchFailureIsolation(model);
    CheckBatchedSessions(model);
    if (batch_only)
      return 0;
    CheckServingEos(model);
    const auto pattern = model->Tokenize(
        "The quick brown fox jumps over the lazy dog. "
        "Strix Halo executes this deterministic benchmark sequence. ");
    Require(!pattern.empty(), "empty prompt pattern");
    std::vector<std::int32_t> prompt(4096);
    for (std::size_t i = 0; i < prompt.size(); ++i)
      prompt[i] = pattern[i % pattern.size()];
    const std::array<sampling::SamplingConfig, 4> configs{{
        {.seed = 1},
        {.temperature = 0.7F, .seed = 1},
        {.temperature = 1.0F, .top_p = 0.95F, .seed = 73},
        {.temperature = 0.8F,
         .top_k = 40,
         .top_p = 0.9F,
         .min_p = 0.01F,
         .min_keep = 2,
         .seed = 1,
         .repeat_penalty = 1.1F,
         .repeat_last_n = 16,
         .frequency_penalty = 0.2F,
         .presence_penalty = 0.1F},
    }};
    struct Replay {
      unsigned depth;
      std::size_t config;
      std::vector<float> prefill_logits;
      std::vector<float> final_logits;
      std::vector<std::int32_t> tokens;
      std::uint64_t rng;
    };
    std::vector<Replay> replays;
    for (const auto depth : {16U, 4096U}) {
      const auto prefix = std::span(prompt).first(depth);
      const std::vector<sampling::TokenId> history(prefix.begin(),
                                                   prefix.end());
      for (std::size_t c = 0; c < configs.size(); ++c) {
        const bool sampled = configs[c].uses_random_sampling();
        auto ar = model->CreateSession(gufo::core::SessionMode::kAutoregressive,
                                       6145, &error);
        auto mtp = model->CreateSession(
            model->HasMtp() ? gufo::core::SessionMode::kSpeculative
                            : gufo::core::SessionMode::kAutoregressive,
            6145, &error);
        Require(ar && mtp, error);
        Require(ar->Sync(prefix, &error), error);
        // Prefill the other session before decoding either: retained target
        // hidden rows must remain private to their session.
        Require(mtp->Sync(prefix, &error), error);
        RequireExact(ar->Logits(), mtp->Logits(),
                     "interleaving changed prefill logits");
        Replay saved{depth, c,  {ar->Logits().begin(), ar->Logits().end()},
                     {},    {}, 0};
        sampling::SamplerState a(configs[c], history), b(configs[c], history);
        std::vector<std::int32_t> reference, candidate;
        std::vector<std::size_t> interleaved_widths;
        const std::size_t token_count = c <= 1 ? 32 : 8;
        while (candidate.size() < token_count) {
          qfn::Session::DecodeResult decoded;
          Require(mtp->DecodeStep(token_count - candidate.size(), b, &decoded,
                                  &error, false),
                  error);
          Require(!decoded.tokens.empty(), "empty MTP step");
          interleaved_widths.push_back(decoded.tokens.size());
          candidate.insert(candidate.end(), decoded.tokens.begin(),
                           decoded.tokens.end());
          while (reference.size() < candidate.size()) {
            if (sampled) {
              // Independent target replay checks every committed token's
              // recurrent/cache state, including residual corrections.
              const auto token = candidate[reference.size()];
              Require(ar->Evaluate(token, &error), error);
              reference.push_back(token);
              a.Accept(static_cast<sampling::TokenId>(token));
              continue;
            }
            qfn::Session::DecodeResult single;
            Require(ar->DecodeStep(1, a, &single, &error, false), error);
            reference.insert(reference.end(), single.tokens.begin(),
                             single.tokens.end());
          }
          double squared = 0.0;
          float max_diff = 0.0F;
          for (std::size_t i = 0; i < ar->Logits().size(); ++i) {
            Require(std::isfinite(ar->Logits()[i]) &&
                        std::isfinite(mtp->Logits()[i]),
                    "non-finite logits");
            const float d = ar->Logits()[i] - mtp->Logits()[i];
            squared += static_cast<double>(d) * d;
            max_diff = std::max(max_diff, std::abs(d));
          }
          std::cout << "frontier depth=" << depth << " config=" << c
                    << " tokens=" << candidate.size()
                    << " width=" << decoded.tokens.size()
                    << " max_diff=" << max_diff
                    << " rmse=" << std::sqrt(squared / ar->Logits().size())
                    << " ar_rng=" << a.rng_state()
                    << " mtp_rng=" << b.rng_state() << '\n';
          if (reference != candidate) {
            std::cout << "AR:";
            for (auto t : reference)
              std::cout << ' ' << t;
            std::cout << "\nMTP:";
            for (auto t : candidate)
              std::cout << ' ' << t;
            std::cout << '\n';
          }
          std::cout << std::flush;
          Require(reference == candidate,
                  "AR/MTP token mismatch at depth " + std::to_string(depth) +
                      " config " + std::to_string(c) + " token " +
                      std::to_string(reference.size()));
          Require(max_diff == 0.0F, "AR/MTP frontier logits differ");
          if (!sampled) {
            Require(a.rng_state() == b.rng_state(),
                    "greedy AR/MTP RNG mismatch");
          }
          Require(ar->Position() == mtp->Position(),
                  "AR/MTP position mismatch");
        }
        saved.final_logits.assign(ar->Logits().begin(), ar->Logits().end());
        saved.tokens = reference;
        saved.rng = b.rng_state();
        // Greedy and seeded sampling exercise fresh-load determinism; the
        // remaining configurations cover filters and penalties above.
        if (c <= 1)
          replays.push_back(std::move(saved));
        const auto stats = mtp->Statistics();
        Require(mtp->Sync(prefix, &error), error);
        Require(mtp->Position() == depth, "prefix reset failed");
        sampling::SamplerState replay_sampler(configs[c], history);
        std::vector<std::int32_t> replay;
        std::vector<std::size_t> isolated_widths;
        while (replay.size() < token_count) {
          qfn::Session::DecodeResult decoded;
          Require(mtp->DecodeStep(token_count - replay.size(), replay_sampler,
                                  &decoded, &error, false),
                  error);
          Require(!decoded.tokens.empty(), "empty replay step");
          isolated_widths.push_back(decoded.tokens.size());
          replay.insert(replay.end(), decoded.tokens.begin(),
                        decoded.tokens.end());
        }
        const auto after = mtp->Statistics();
        Require(replay == candidate && isolated_widths == interleaved_widths &&
                    after.drafted - stats.drafted == stats.drafted &&
                    after.accepted - stats.accepted == stats.accepted,
                "interleaving changed MTP proposals or acceptance");
        std::cout << "depth=" << depth << " config=" << c
                  << " tokens=" << token_count
                  << " exact=1 drafted=" << stats.drafted
                  << " accepted=" << stats.accepted << '\n'
                  << std::flush;
      }
    }
    // Recreate device weights, executor and plans. Reverse request order so
    // a previous shape cannot silently determine this request's arithmetic.
    model.reset();
    model = qfn::Model::Load(
        argv[2],
        {.max_context = 6145, .mtp_model_path = argv[4], .max_draft_tokens = 7},
        &error);
    Require(model != nullptr, error);
    std::reverse(replays.begin(), replays.end());
    for (const auto& saved : replays) {
      const auto prefix = std::span(prompt).first(saved.depth);
      const std::vector<sampling::TokenId> history(prefix.begin(),
                                                   prefix.end());
      for (const bool speculative : {false, true}) {
        const bool sampled = configs[saved.config].uses_random_sampling();
        auto session = model->CreateSession(
            model->HasMtp() ? gufo::core::SessionMode::kSpeculative
                            : gufo::core::SessionMode::kAutoregressive,
            6145, &error);
        Require(session && session->Sync(prefix, &error), error);
        RequireExact(saved.prefill_logits, session->Logits(),
                     "fresh load changed prefill logits");
        sampling::SamplerState sampler(configs[saved.config], history);
        std::vector<std::int32_t> tokens;
        constexpr std::array<std::size_t, 4> budgets{2, 8, 1, 4};
        std::size_t step = 0;
        while (tokens.size() < saved.tokens.size()) {
          if (sampled && !speculative) {
            const auto token = saved.tokens[tokens.size()];
            Require(session->Evaluate(token, &error), error);
            tokens.push_back(token);
            continue;
          }
          const auto budget = sampled ? saved.tokens.size() - tokens.size()
                              : speculative ? budgets[step++ % budgets.size()]
                                            : 1;
          qfn::Session::DecodeResult decoded;
          Require(session->DecodeStep(
                      std::min(budget, saved.tokens.size() - tokens.size()),
                      sampler, &decoded, &error, false),
                  error);
          Require(!decoded.tokens.empty(), "empty fresh-load replay step");
          tokens.insert(tokens.end(), decoded.tokens.begin(),
                        decoded.tokens.end());
        }
        Require(tokens == saved.tokens,
                "fresh load or draft width changed tokens");
        RequireExact(saved.final_logits, session->Logits(),
                     "fresh load or draft width changed final logits");
        if (!sampled || speculative) {
          Require(sampler.rng_state() == saved.rng,
                  "fresh load or draft width changed RNG");
        }
        std::cout << "fresh_load depth=" << saved.depth
                  << " config=" << saved.config << " mtp=" << speculative
                  << " tokens=" << tokens.size() << " bitwise_exact=1\n"
                  << std::flush;
      }
    }
    CheckServingSampling(model);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
