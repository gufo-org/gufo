#include "src/cli/runner_cli.hpp"

#include <algorithm>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include "src/cli/serve/inference_backend.hpp"
#endif

namespace gufo::cli {
namespace {

double Seconds(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

#if defined(ENGINE_ENABLE_HIP)
// The server's backend with one session, for the provider's runner.
bool LoadBackend(server::InferenceBackend* backend, const std::string& model,
                 const std::string& vision_model,
                 const ext::ParsedOptions& provider_options,
                 std::chrono::steady_clock::time_point start) {
  std::string error;
  const bool loaded = backend->load(
      model, &error, 0, 1, {}, {}, {}, vision_model, {}, false,
      [&](const ext::TextProvider&, const core::GgufReader*,
          ext::TextRunnerOptions* options, std::string*) {
        options->provider_options = provider_options;
        return true;
      });
  if (!loaded) {
    std::cerr << "Error loading model '" << model << "': " << error << "\n";
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
}

// Splits "<think>reasoning</think>answer" as the server's parsers do.
void SplitReasoning(const std::string& text, std::string* thought,
                    std::string* content) {
  const auto close = text.rfind("</think>");
  if (close == std::string::npos) {
    *content = text;
    return;
  }
  auto open = text.find("<think>");
  open = open == std::string::npos || open > close ? 0 : open + 7;
  *thought = text.substr(open, close - open);
  *content = text.substr(close + 8);
  const auto first = content->find_first_not_of(" \t\r\n");
  *content = first == std::string::npos ? std::string() : content->substr(first);
}
#endif

}  // namespace

int RunRunnerPrompt(const ext::TextPromptOptions& opt,
                    std::chrono::steady_clock::time_point start) {
#if defined(ENGINE_ENABLE_HIP)
  try {
  server::InferenceBackend backend;
  if (!LoadBackend(&backend, opt.model_path, opt.vision_model_path,
                   opt.provider_options, start))
    return 1;
  const auto print = [](const server::TokenPiece& piece) {
    std::cout << piece.text << std::flush;
    return true;
  };
  server::TextGenerationBackend::Result result;
  if (opt.use_chat_template) {
    server::ChatRequest request;
    if (!opt.system_prompt.empty())
      request.messages.emplace_back(ext::ChatRole::kSystem, opt.system_prompt);
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
  if (!LoadBackend(&backend, opt.model_path, opt.vision_model_path,
                   opt.provider_options, start))
    return 1;
  server::ChatRequest request;
  request.reasoning.enabled = opt.thinking_mode();
  if (!opt.system_prompt.empty())
    request.messages.emplace_back(ext::ChatRole::kSystem, opt.system_prompt);
  const auto print = [](const server::TokenPiece& piece) {
    std::cout << piece.text << std::flush;
    return true;
  };
  for (std::string line;;) {
    std::cout << "\n> " << std::flush;
    if (!std::getline(std::cin, line) || line == "/exit" || line == "/quit")
      break;
    if (line.empty())
      continue;
    request.messages.emplace_back(ext::ChatRole::kUser, line);
    const auto result =
        backend.chat(request, opt.max_tokens, opt.sampling, {}, print);
    std::string thought, content;
    SplitReasoning(result.text, &thought, &content);
    request.messages.emplace_back(ext::ChatRole::kAssistant, content,
                                  std::string(), thought);
    std::cout << "\n";
    if (opt.verbose)
      PrintStats(result);
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

int RunRunnerBench(const ext::TextBenchOptions& opt,
                   std::chrono::steady_clock::time_point start) {
#if defined(ENGINE_ENABLE_HIP)
  try {
  server::InferenceBackend backend;
  if (!LoadBackend(&backend, opt.model_path, {}, opt.provider_options, start))
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
      for (std::size_t rep = 0; rep < std::max<std::size_t>(opt.repetitions, 1);
           ++rep) {
        auto request = backend.start_complete(
            prompt, std::max<std::size_t>(n_gen, 1), opt.sampling, {}, false,
            /*ignore_eos=*/true, "bench", {}, false, /*cache_prompt=*/false,
            /*cache_persist=*/false);
        const auto result = request->Wait();
        prefill.push_back(result.prompt_tokens / (result.prefill_ms / 1000));
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
