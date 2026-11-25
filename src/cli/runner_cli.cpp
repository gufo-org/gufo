#include "src/cli/runner_cli.hpp"

#include <algorithm>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/cli/serve/text_generation_backend.hpp"
#include "src/cli/speculative_settings.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include "src/cli/serve/inference_backend.hpp"
#endif

namespace gufo::cli {
namespace {

double Seconds(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

template<typename Options>
bool PrepareFor(const Options& command, const ext::TextProvider& provider,
                ext::TextRunnerOptions* options, std::string* error) {
  try {
    options->provider_options = command.provider_options;
    options->speculative = SpeculativeConfigFor(
        command.speculative_backend, command.mtp_model_path,
        command.dflash_model_path, command.dspark_model_path);
    ApplySpeculativeSettings(
        ResolveSpeculativeSettings(provider, command.speculative_backend, {}),
        &options->speculative);
    return true;
  } catch (const std::invalid_argument& exception) {
    if (error != nullptr)
      *error = exception.what();
    return false;
  }
}

// Splits "<open>reasoning<close>answer" with the request's own delimiters,
// as the server's parsers do.
void SplitReasoning(const std::string& text, const ext::ReasoningMarkup& markup,
                    std::string* thought, std::string* content) {
  const auto close =
      markup.close.empty() ? std::string::npos : text.rfind(markup.close);
  if (close == std::string::npos) {
    *content = text;
    return;
  }
  auto open = markup.open.empty() ? std::string::npos : text.find(markup.open);
  open =
      open == std::string::npos || open > close ? 0 : open + markup.open.size();
  *thought = text.substr(open, close - open);
  *content = text.substr(close + markup.close.size());
  const auto first = content->find_first_not_of(" \t\r\n");
  *content =
      first == std::string::npos ? std::string() : content->substr(first);
}
#if defined(ENGINE_ENABLE_HIP)
// The server's backend with one session, for the provider's runner.
template<typename Options>
bool LoadBackend(server::InferenceBackend* backend, const Options& opt,
                 const std::string& vision_model,
                 std::chrono::steady_clock::time_point start) {
  std::string error;
  const bool loaded = backend->load(
      opt.model_path, &error, 0, 1, {}, {}, {}, vision_model, {}, false,
      [&](const ext::TextProvider& provider, const core::GgufReader*,
          ext::TextRunnerOptions* options, std::string* message) {
        return PrepareRunnerOptions(opt, provider, options, message);
      });
  if (!loaded) {
    std::cerr << "Error loading model '" << opt.model_path << "': " << error
              << "\n";
    return false;
  }
  std::cerr << "Model loaded in " << Seconds(start) << " s\n";
  return true;
}

void PrintStats(const server::TextGenerationBackend::Result& result) {
  const double prefill = result.prefill_ms > 0
                             ? result.prompt_tokens / (result.prefill_ms / 1000)
                             : 0;
  const double decode =
      result.decode_ms > 0 && result.completion_tokens > 1
          ? (result.completion_tokens - 1) / (result.decode_ms / 1000)
          : 0;
  std::cerr << "\n[" << result.prompt_tokens << " prompt tokens, "
            << result.completion_tokens << " generated; prefill " << prefill
            << " tok/s, decode " << decode << " tok/s]\n";
  if (result.draft_rounds > 0) {
    std::cerr << "[Speculative]: cycles=" << result.draft_rounds
              << " drafted=" << result.draft_tokens
              << " accepted=" << result.draft_accepted_tokens
              << " lookup_proposed=" << result.lookup_tokens
              << " lookup_accepted=" << result.lookup_accepted_tokens << '\n';
  }
}

#endif

}  // namespace

bool PrepareRunnerOptions(const ext::TextPromptOptions& command,
                          const ext::TextProvider& provider,
                          ext::TextRunnerOptions* options, std::string* error) {
  return PrepareFor(command, provider, options, error);
}

bool PrepareRunnerOptions(const ext::TextBenchOptions& command,
                          const ext::TextProvider& provider,
                          ext::TextRunnerOptions* options, std::string* error) {
  return PrepareFor(command, provider, options, error);
}

int RunChatSession(server::TextGenerationBackend& backend,
                   const ext::TextPromptOptions& opt, std::istream& in,
                   std::ostream& out, const ChatResultCallback& on_result) {
  server::ChatRequest request;
  request.reasoning.enabled = opt.thinking_mode();
  if (!opt.system_prompt.empty())
    request.messages.emplace_back(ext::ChatRole::kSystem, opt.system_prompt);
  const auto print = [&out](const server::TokenPiece& piece) {
    out << piece.text << std::flush;
    return true;
  };
  for (std::string line;;) {
    out << "\n> " << std::flush;
    if (!std::getline(in, line) || line == "/exit" || line == "/quit")
      break;
    if (line.empty())
      continue;
    request.messages.emplace_back(ext::ChatRole::kUser, line);
    // Streamed: without it the scheduler hands Wait no pieces to print.
    const auto pending =
        backend.start_chat(request, opt.max_tokens, opt.sampling, {},
                           /*stream_output=*/true);
    const auto result = pending->Wait(print);
    std::string thought, content;
    SplitReasoning(result.text, pending->ReasoningDelimiters(), &thought,
                   &content);
    request.messages.emplace_back(ext::ChatRole::kAssistant, content,
                                  std::string(), thought);
    out << "\n";
    if (on_result)
      on_result(result);
  }
  return 0;
}

int RunRunnerPrompt(const ext::TextPromptOptions& opt,
                    std::chrono::steady_clock::time_point start) {
#if defined(ENGINE_ENABLE_HIP)
  try {
    server::InferenceBackend backend;
    if (!LoadBackend(&backend, opt, opt.vision_model_path, start))
      return 1;
    const auto print = [](const server::TokenPiece& piece) {
      std::cout << piece.text << std::flush;
      return true;
    };
    server::TextGenerationBackend::Result result;
    if (opt.use_chat_template) {
      server::ChatRequest request;
      if (!opt.system_prompt.empty())
        request.messages.emplace_back(ext::ChatRole::kSystem,
                                      opt.system_prompt);
      request.messages.emplace_back(ext::ChatRole::kUser, opt.prompt_text);
      request.reasoning.enabled = opt.thinking_mode();
      result = backend.chat(request, opt.max_tokens, opt.sampling, {}, print);
    } else {
      result = backend.complete(opt.prompt_text, opt.max_tokens, opt.sampling,
                                {}, print);
    }
    std::cout << "\n";
    if (opt.verbose)
      PrintStats(result);
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "\nError: " << exception.what() << "\n";
    return 1;
  }
#else
  (void)opt;
  (void)start;
  std::cerr << "Error: this build has no GPU engine\n";
  return 1;
#endif
}

int RunRunnerChat(const ext::TextPromptOptions& opt,
                  std::chrono::steady_clock::time_point start) {
#if defined(ENGINE_ENABLE_HIP)
  try {
    server::InferenceBackend backend;
    if (!LoadBackend(&backend, opt, opt.vision_model_path, start))
      return 1;
    return RunChatSession(backend, opt, std::cin, std::cout,
                          [&](const server::TextGenerationBackend::Result& r) {
                            if (opt.verbose)
                              PrintStats(r);
                          });
  } catch (const std::exception& exception) {
    std::cerr << "\nError: " << exception.what() << "\n";
    return 1;
  }
#else
  (void)opt;
  (void)start;
  std::cerr << "Error: this build has no GPU engine\n";
  return 1;
#endif
}

int RunRunnerBench(const ext::TextBenchOptions& opt,
                   std::chrono::steady_clock::time_point start) {
#if defined(ENGINE_ENABLE_HIP)
  try {
    server::InferenceBackend backend;
    if (!LoadBackend(&backend, opt, {}, start))
      return 1;
    // A prompt of exactly `tokens` tokens, from repeated text.
    const auto prompt_of = [&](std::size_t tokens) {
      std::string text;
      while (backend.count_tokens(text) < tokens)
        text += " the quick brown fox jumps over the lazy dog";
      while (!text.empty() && backend.count_tokens(text) > tokens)
        text.pop_back();
      return text;
    };
    std::cout << "| model | test | t/s |\n|---|---|---|\n";
    for (const auto n_prompt : opt.n_prompts) {
      for (const auto n_gen : opt.n_gens) {
        const std::string prompt = prompt_of(n_prompt);
        std::vector<double> prefill, decode;
        for (std::size_t rep = 0;
             rep < std::max<std::size_t>(opt.repetitions, 1); ++rep) {
          auto request = backend.start_complete(
              prompt, std::max<std::size_t>(n_gen, 1), opt.sampling, {}, false,
              /*ignore_eos=*/true, "bench", {}, false, /*cache_prompt=*/false,
              /*cache_persist=*/false);
          const auto result = request->Wait();
          if (result.prefill_ms > 0)
            prefill.push_back(result.prompt_tokens /
                              (result.prefill_ms / 1000));
          if (result.completion_tokens > 1 && result.decode_ms > 0)
            decode.push_back((result.completion_tokens - 1) /
                             (result.decode_ms / 1000));
        }
        const auto median = [](std::vector<double> values) {
          std::sort(values.begin(), values.end());
          return values.empty() ? 0.0 : values[values.size() / 2];
        };
        std::cout << "| " << backend.model_id() << " | pp" << n_prompt << " | "
                  << median(prefill) << " |\n";
        if (n_gen > 0)
          std::cout << "| " << backend.model_id() << " | tg" << n_gen << " | "
                    << median(decode) << " |\n";
      }
    }
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "\nError: " << exception.what() << "\n";
    return 1;
  }
#else
  (void)opt;
  (void)start;
  std::cerr << "Error: this build has no GPU engine\n";
  return 1;
#endif
}

}  // namespace gufo::cli
