#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

#include "gufo/ext/generated/qwen38_flash_next_constants.hpp"
#include "src/models/qwen38_flash_next/runner.hpp"
#include "src/models/qwen38_flash_next/serving_vision.hpp"

namespace qfn = gufo::models::qwen38_flash_next;

static_assert(qfn::DecodeConcurrencyForSessions(0) == 1);
static_assert(qfn::DecodeConcurrencyForSessions(1) == 1);
static_assert(qfn::DecodeConcurrencyForSessions(2) == 2);
static_assert(qfn::DecodeConcurrencyForSessions(4) == 4);
static_assert(qfn::DecodeConcurrencyForSessions(8) == 8);
static_assert(qfn::DecodeConcurrencyForSessions(9) == 8);
static_assert(qfn::DecodeConcurrencyForSessions(
                  std::numeric_limits<std::size_t>::max()) == 8);

namespace {
void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

template<class Action>
void Rejects(Action action, std::string_view diagnostic) {
  bool rejected = false;
  try {
    action();
  } catch (const std::exception& error) {
    rejected = std::string_view(error.what()).find(diagnostic) !=
               std::string_view::npos;
  }
  Expect(rejected, diagnostic);
}

// Injection at the same post-load boundary as the serving factory. Projector
// discovery/loading is supplied, while selection, width configuration, startup
// planning and reservation ownership execute the production implementation.
struct LoadedModel {
  bool projector{};
  bool speculative = true;
  std::uint32_t copy_rows = 64;
  std::uint32_t neural_drafts = 7;
  int estimates = 0;
  int width_changes = 0;
  std::shared_ptr<gufo::models::qwen::vision::AdmissionBudget> budget;

  bool VisionEncoder() const { return projector; }
  bool HasVisionMemoryBudget() const { return bool(budget); }
  void LimitCopyRows(std::uint32_t rows) {
    ++width_changes;
    copy_rows = rows;
  }
  std::size_t TextSessionBytes(gufo::core::SessionMode mode,
                               std::uint32_t context) {
    Expect(mode == (speculative ? gufo::core::SessionMode::kSpeculative
                                : gufo::core::SessionMode::kAutoregressive) &&
               context == 262144,
           "serving preserves execution mode and full context");
    ++estimates;
    return copy_rows == 8 ? 100 : 150;
  }
  std::size_t SessionBytes(gufo::core::SessionMode mode,
                           std::uint32_t context) {
    return TextSessionBytes(mode, context) + 300;
  }
  std::size_t DeferredScratchBytes() const { return copy_rows == 8 ? 20 : 80; }
  void SetVisionMemoryBudget(
      std::shared_ptr<gufo::models::qwen::vision::AdmissionBudget> value) {
    budget = std::move(value);
  }
};

void LoadedProjectorSelection() {
  for (const std::string path : {"projector.gguf", ""}) {
    gufo::ext::TextRunnerOptions options;
    options.vision_model_path = path;
    options.max_context = 262144;
    options.session_count = 4;
    options.speculative.backend = gufo::ext::TextSpeculativeBackend::kMtp;
    options.speculative.max_draft_tokens = 7;
    LoadedModel model{.projector = true};
    const auto selected = qfn::InitializeServingVision(
        model, options.max_context,
        options.speculative.backend == gufo::ext::TextSpeculativeBackend::kMtp,
        options.session_count, 450, [] { return std::size_t{1000}; });
    Expect(options.provider_options.empty() && selected.has_value(),
           "explicit and discovered loaded projectors need no policy argument");
    Expect(
        selected->budget == model.budget &&
            selected->configured_text_bytes == 400 &&
            selected->plan.per_session_bytes == 100 &&
            selected->plan.shared_reservation_bytes == 20 &&
            selected->plan.state_capacity_bytes == 430 &&
            selected->plan.Fits(4),
        "loaded projector installs coordination and text-only startup claim");
    Expect(model.copy_rows == 8 && selected->copy_width.has_value() &&
               selected->copy_width->rows == 8 && model.neural_drafts == 7 &&
               options.speculative.max_draft_tokens == 7,
           "ordinary verification preserves the neural draft limit");
    Expect(selected->budget->ProtectedBytes() == 420 &&
               selected->budget->Outstanding() == 0,
           "execution is pending without an unused startup image pool");
    {
      auto image = selected->budget->Reserve(580);
      Expect(selected->budget->Outstanding() == 580,
             "images use capacity after accepted execution obligations");
      Rejects([&] { (void)selected->budget->ReserveShared(1); },
              "vision memory");
    }
    Expect(selected->budget->Outstanding() == 0,
           "image refusal and release preserve subsequent capacity");
    auto original_budget = model.budget;
    Rejects([&] { qfn::RequireUncoordinatedModel(model); }, "already");
    Rejects(
        [&] {
          (void)qfn::InitializeServingVision(model, options.max_context, true,
                                             options.session_count, 450,
                                             [] { return std::size_t{1000}; });
        },
        "already");
    Expect(model.budget == original_budget && model.width_changes == 1,
           "another runner cannot overwrite the model coordinator");
    selected->budget->ClaimExecutionOwner();
    Rejects([&] { selected->budget->ClaimExecutionOwner(); }, "owner");
  }

  // No loaded projector means no image-related admission or width cap.
  LoadedModel text;
  const auto unselected = qfn::InitializeServingVision(
      text, 262144, true, 4, 0, []() -> std::size_t {
        Expect(false, "text-only selection must not observe image headroom");
        return 0;
      });
  Expect(
      !unselected && !text.budget && text.copy_rows == 64 &&
          text.estimates == 0,
      "no projector leaves existing text admission and optional width alone");
  LoadedModel typed{.projector = true};
  qfn::RequireUncoordinatedModel(typed);
  Expect(!typed.budget && typed.copy_rows == 64,
         "typed factory guard does not silently install a serving coordinator");

  LoadedModel autoregressive{.projector = true, .speculative = false};
  const auto ar = qfn::InitializeServingVision(
      autoregressive, 262144, false, 1, 230, [] { return std::size_t{1000}; });
  Expect(ar && ar->plan.Fits(1) && !ar->copy_width &&
             autoregressive.width_changes == 0 &&
             ar->budget == autoregressive.budget,
         "AR projector installs admission without configuring MTP copy rows");

  LoadedModel insufficient{.projector = true};
  const auto short_plan = qfn::InitializeServingVision(
      insufficient, 262144, true, 4, 419, [] { return std::size_t{1000}; });
  Expect(!short_plan->plan.Fits(4),
         "one-byte text insufficiency remains a startup refusal");
  LoadedModel unknown_sessions{.projector = true};
  Rejects(
      [&] {
        (void)qfn::InitializeServingVision(unknown_sessions, 262144, true, 0,
                                           450,
                                           [] { return std::size_t{1000}; });
      },
      "session");
  Expect(!unknown_sessions.budget && unknown_sessions.width_changes == 0,
         "missing execution ownership cannot guess a session count");
}
}  // namespace

int main() {
  // The metadata is generated from the real provider declaration. Simulate
  // HIP availability without loading the provider or touching a device.
  gufo::ext::Registry registry;
  registry.BeginExtension({"qwen38_flash_next", "test", ""});
  auto provider = gufo::ext::generated::extension_qwen38_flash_next::
      kProvider_flash_next_q4;
  provider.available = true;
  provider.unavailable_reason.clear();
  Expect(std::ranges::none_of(provider.options,
                              [](const auto& option) {
                                return option.name == "--vision-memory-policy";
                              }),
         "provider metadata and its generated help omit the removed option");
  registry.AddProvider(std::move(provider));
  gufo::ext::ProviderInfo unrelated;
  unrelated.id = "unrelated-provider";
  unrelated.options = {{"--unrelated-option",
                        gufo::ext::OptionType::kString,
                        {"serve"},
                        "Independent option"}};
  registry.AddProvider(std::move(unrelated));

  const auto defaults = gufo::ext::ParseOptions(registry, "serve", {});
  gufo::ext::ValidateSelectedOptions(
      registry, "serve", registry.Provider("flash-next-q4"), defaults);
  Expect(defaults.empty(), "normal no-flag serving construction validates");
  for (const auto policy : {"conservative", "on-demand", "automatic", ""}) {
    const std::array<std::string_view, 2> arguments{"--vision-memory-policy",
                                                    policy};
    for (const auto command : {"serve", "prompt", "bench"}) {
      Rejects(
          [&] { (void)gufo::ext::ParseOptions(registry, command, arguments); },
          "unknown option: --vision-memory-policy");
    }
    gufo::ext::ParsedOptions supplied{
        {"--vision-memory-policy", std::string(policy)}};
    Rejects(
        [&] {
          gufo::ext::ValidateSelectedOptions(
              registry, "serve", registry.Provider("flash-next-q4"), supplied);
        },
        "does not support --vision-memory-policy");
  }
  const std::array<std::string_view, 2> other_arguments{"--unrelated-option",
                                                        "unchanged"};
  const auto other =
      gufo::ext::ParseOptions(registry, "serve", other_arguments);
  gufo::ext::ValidateSelectedOptions(
      registry, "serve", registry.Provider("unrelated-provider"), other);
  Rejects(
      [&] {
        gufo::ext::ValidateSelectedOptions(
            registry, "serve", registry.Provider("flash-next-q4"), other);
      },
      "does not support --unrelated-option");
  LoadedProjectorSelection();
  std::cout
      << "Flash-Next automatic serving selection and provider options passed\n";
}
