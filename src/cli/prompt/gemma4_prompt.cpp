#include "src/cli/prompt/gemma4_prompt.hpp"

#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/image.hpp"
#include "src/models/gemma4/chat_template.hpp"
#include "src/models/gemma4/engine.hpp"
#include "src/models/gemma4/vision/encoder.hpp"
#include "src/models/gemma4/vision/prompt.hpp"

namespace gufo::cli {
namespace {

namespace g4 = models::gemma4;

constexpr std::uint32_t kDefaultContext = 4096;

double SecondsSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

std::shared_ptr<g4::Model> LoadModel(
    const PromptOptions& opt, std::chrono::steady_clock::time_point start) {
  if (opt.force_cpu) {
    std::cerr << "Gemma 4 is supported only by the ROCm backend\n";
    return nullptr;
  }
  if (opt.image_tokens != 0 &&
      !g4::vision::IsSoftTokenBudget(opt.image_tokens)) {
    std::cerr << "--image-tokens must be 70, 140, 280, 560 or 1120\n";
    return nullptr;
  }
  const bool mtp = opt.speculative_backend == "mtp";
  if (!opt.speculative_backend.empty() && !mtp) {
    std::cerr << "Gemma 4 supports only --speculative mtp\n";
    return nullptr;
  }
  if (mtp && opt.mtp_model_path.empty()) {
    std::cerr << "--speculative mtp requires --mtp-model\n";
    return nullptr;
  }
  g4::ModelOptions options{.max_context = kDefaultContext,
                           .mtp_model_path = mtp ? opt.mtp_model_path : "",
                           .draft_tokens = opt.draft_tokens_given
                                               ? opt.draft_tokens
                                               : g4::kDefaultDraftTokens,
                           .min_draft_tokens = opt.min_draft_tokens};
  try {
    options.draft_policy = g4::ParseDraftPolicy(opt.draft_policy);
    options.draft_calibration =
        g4::ParseDraftCalibrationScope(opt.draft_calibration);
  } catch (const std::invalid_argument& e) {
    std::cerr << e.what() << '\n';
    return nullptr;
  }
  std::string error;
  auto model = g4::Model::Load(opt.model_path, options, &error);
  if (model == nullptr) {
    std::cerr << "Error loading Gemma 4 model: " << error << '\n';
    std::cerr << "[Model Load]: " << SecondsSince(start) << " s (failed)\n";
    return nullptr;
  }
  std::cout << "[Model Load]: " << SecondsSince(start) << " s\n";
  return model;
}

/// The vision sidecar, when images are attached or --mmproj is given.
std::shared_ptr<g4::vision::Encoder> LoadVision(const PromptOptions& opt,
                                                const g4::Model& model) {
  if (opt.image_paths.empty() && opt.vision_model_path.empty()) {
    return {};
  }
  auto encoder = g4::vision::Encoder::Open(
      opt.model_path, opt.vision_model_path, model.config().hidden_size);
  if (!encoder) {
    throw std::invalid_argument(
        "image input requires a matching --mmproj BF16 sidecar");
  }
  return encoder;
}

void AttachImages(const PromptOptions& opt,
                  tokenization::ChatMessage& message) {
  for (const auto& path : opt.image_paths) {
    message.images.push_back(
        {0, std::make_shared<const std::vector<std::uint8_t>>(
                core::ReadImageFile(path))});
  }
}

/// A rendered conversation with its images, encoded on first use.
struct Prompt {
  std::vector<g4::TokenId> tokens;
  std::vector<g4::ImageSpan> spans;
  std::shared_ptr<const g4::vision::Prompt> vision;
};

Prompt RenderPrompt(const PromptOptions& opt, const g4::Model& model,
                    g4::vision::Encoder* encoder,
                    std::span<const tokenization::ChatMessage> history,
                    const g4::ChatOptions& options) {
  auto vision = std::make_shared<const g4::vision::Prompt>(g4::vision::Prepare(
      model.tokenizer(), history, {}, options,
      encoder != nullptr ? encoder->identity() : "", model.MaxContext(),
      opt.image_tokens != 0 ? opt.image_tokens
                            : g4::vision::kDefaultSoftTokens));
  Prompt prompt{vision->tokens, {}, vision};
  for (const auto& image : vision->images) {
    prompt.spans.push_back(
        {image.span.offset, image.span.rows, image.prefix_identity});
  }
  return prompt;
}

/// Generates one reply after `prompt`; `reply` receives the decoded text.
int Generate(const PromptOptions& opt, g4::Model& model, g4::Session& session,
             g4::vision::Encoder* encoder, const Prompt& input,
             std::string* reply) {
  const std::span<const g4::TokenId> prompt = input.tokens;
  if (prompt.empty()) {
    std::cerr << "Gemma 4 prompt produced no tokens\n";
    return 1;
  }
  if (prompt.size() + opt.max_tokens > session.ContextSize()) {
    std::cerr << "Gemma 4 prompt and output exceed the "
              << session.ContextSize() << "-token CLI context\n";
    return 1;
  }
  std::string error;
  const auto prefill_start = std::chrono::steady_clock::now();
  std::map<std::size_t, std::shared_ptr<const g4::vision::Encoder::Embedding>>
      embeddings;
  const auto embed = [&](std::size_t index) -> const float* {
    auto& embedding = embeddings[index];
    if (!embedding) {
      embedding = encoder->Encode(input.vision->images.at(index).pixels);
    }
    return embedding->data();
  };
  if (!session.Sync(prompt, input.spans, embed, &error)) {
    std::cerr << "Gemma 4 prefill failed: " << error << '\n';
    return 1;
  }
  const double prefill_seconds = SecondsSince(prefill_start);
  if (opt.verbose) {
    std::cout << "[Engine]: Gemma 4 ROCm (gfx1151)\n"
              << "Model: " << model.ModelName() << '\n'
              << "Prompt tokens: " << prompt.size() << '\n'
              << "--- Generation Output ---\n";
  }
  std::vector<sampling::TokenId> history(prompt.begin(), prompt.end());
  sampling::SamplerState sampler(opt.sampling, history);
  const auto decode_start = std::chrono::steady_clock::now();
  std::size_t generated = 0;
  bool stop = false;
  while (generated < opt.max_tokens && !stop) {
    g4::Session::DecodeResult step;
    if (!session.DecodeStep(opt.max_tokens - generated, sampler, &step,
                            &error)) {
      std::cerr << "\nGemma 4 decode failed: " << error << '\n';
      return 1;
    }
    for (const g4::TokenId token : step.tokens) {
      if (model.IsStopToken(token)) {
        stop = true;
        break;
      }
      const std::string piece = model.TokenText(token);
      if (reply != nullptr) {
        reply->append(piece);
      }
      std::cout << piece << std::flush;
      ++generated;
    }
    stop = stop || step.stop || step.tokens.empty();
  }
  std::cout << '\n';
  if (opt.verbose && generated > 0) {
    const double seconds = SecondsSince(decode_start);
    std::cout << "Prefill " << prompt.size() << " tokens in " << prefill_seconds
              << " s (" << static_cast<double>(prompt.size()) / prefill_seconds
              << " tok/s)\nGenerated " << generated << " tokens in " << seconds
              << " s (" << static_cast<double>(generated) / seconds
              << " tok/s)\n";
    if (model.HasMtp()) {
      const auto& stats = session.Statistics();
      std::cout << "[Speculative]: cycles=" << stats.cycles
                << " drafted=" << stats.drafted
                << " accepted=" << stats.accepted << " acceptance="
                << (stats.drafted != 0 ? static_cast<double>(stats.accepted) /
                                             static_cast<double>(stats.drafted)
                                       : 0.0)
                << " siblings=" << stats.siblings
                << " siblings_accepted=" << stats.siblings_accepted << '\n';
    }
  }
  return 0;
}

}  // namespace

bool IsGemma4(const core::GgufReader& reader) {
  return reader.GetMetadataString("general.architecture") == "gemma4";
}

int RunGemma4Prompt(const PromptOptions& opt,
                    std::chrono::steady_clock::time_point load_start) {
  auto model = LoadModel(opt, load_start);
  if (!model) {
    return 1;
  }
  std::string error;
  auto session = model->CreateSession(kDefaultContext, &error);
  if (!session) {
    std::cerr << "Gemma 4 session creation failed: " << error << '\n';
    return 1;
  }
  try {
    const auto encoder = LoadVision(opt, *model);
    Prompt prompt;
    if (opt.use_chat_template) {
      std::vector<tokenization::ChatMessage> messages;
      if (!opt.system_prompt.empty()) {
        messages.emplace_back(tokenization::ChatRole::kSystem,
                              opt.system_prompt);
      }
      messages.emplace_back(tokenization::ChatRole::kUser, opt.prompt_text);
      AttachImages(opt, messages.back());
      prompt =
          RenderPrompt(opt, *model, encoder.get(), messages,
                       g4::ResolveChatOptions(PromptReasoningOptions(opt)));
    } else if (!opt.image_paths.empty()) {
      throw std::invalid_argument("--image requires chat framing");
    } else {
      prompt.tokens = model->tokenizer().Encode(opt.prompt_text, true, true);
    }
    return Generate(opt, *model, *session, encoder.get(), prompt, nullptr);
  } catch (const std::exception& e) {
    std::cerr << "Gemma 4 prompt failed: " << e.what() << '\n';
    return 1;
  }
}

int RunGemma4Chat(const PromptOptions& opt,
                  std::chrono::steady_clock::time_point load_start) {
  if (!opt.use_chat_template) {
    std::cerr << "Gemma 4 interactive chat requires chat framing; use prompt "
                 "--raw for raw text\n";
    return 1;
  }
  auto model = LoadModel(opt, load_start);
  if (!model) {
    return 1;
  }
  std::string error;
  auto session = model->CreateSession(kDefaultContext, &error);
  if (!session) {
    std::cerr << "Gemma 4 session creation failed: " << error << '\n';
    return 1;
  }
  std::shared_ptr<g4::vision::Encoder> encoder;
  try {
    encoder = LoadVision(opt, *model);
  } catch (const std::exception& e) {
    std::cerr << "Gemma 4 vision failed: " << e.what() << '\n';
    return 1;
  }
  const auto chat_options = g4::ResolveChatOptions(PromptReasoningOptions(opt));
  std::vector<tokenization::ChatMessage> history;
  if (!opt.system_prompt.empty()) {
    history.emplace_back(tokenization::ChatRole::kSystem, opt.system_prompt);
  }
  std::cout << "=== Gufo Interactive Chat (Gemma 4) ===\n"
            << "Type 'exit' or Ctrl+D to quit.\n\n";
  for (std::string input;;) {
    std::cout << ">>> User: " << std::flush;
    if (!std::getline(std::cin, input) || input == "exit" || input == "quit") {
      break;
    }
    if (input.empty()) {
      continue;
    }
    history.emplace_back(tokenization::ChatRole::kUser, input);
    if (history.size() == 1 ||
        (history.size() == 2 &&
         history[0].role == tokenization::ChatRole::kSystem)) {
      AttachImages(opt, history.back());
    }
    Prompt prompt;
    try {
      prompt = RenderPrompt(opt, *model, encoder.get(), history, chat_options);
    } catch (const std::exception& e) {
      std::cerr << "Gemma 4 chat template failed: " << e.what() << '\n';
      return 1;
    }
    std::cout << "<<< Assistant: ";
    std::string reply;
    if (Generate(opt, *model, *session, encoder.get(), prompt, &reply) != 0) {
      return 1;
    }
    // The template drops thinking channels from model history itself.
    history.emplace_back(tokenization::ChatRole::kAssistant, std::move(reply));
    std::cout << '\n';
  }
  return 0;
}

}  // namespace gufo::cli
