#include "src/cli/decide/decide.hpp"

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

#include "gufo/ext/registry.hpp"
#include "src/cli/arg_parser.hpp"
#include "src/cli/serve/decision_service.hpp"
#include "src/core/json.hpp"

namespace gufo::cli {
int RunDecide(std::span<const char* const> args) {
  std::filesystem::path model_path, mmproj, request_path;
  std::size_t context = 8192;
  bool jsonl = false;
  ArgParser parser("gufo decide [OPTIONS]",
                   "Answer typed decision questions about text, images and "
                   "audio with a decision model.");
  parser.AddOption("-m", "--model", "PATH", "Decision model artifact", "Model",
                   &model_path);
  parser.AddOption("", "--mmproj", "GGUF",
                   "Image/audio projector sidecar of models that take one",
                   "Model", &mmproj);
  parser.AddOption("-c", "--context", "N",
                   "Longest complete decision prompt (default: 8192)", "Model",
                   &context);
  parser.AddOption("-f", "--request", "JSON",
                   "Request file (default: standard input)", "Input",
                   &request_path);
  parser.AddFlag(
      "", "--jsonl",
      "Read and answer one JSON request per line, keeping the model loaded",
      "Input", &jsonl);
  std::string error;
  if (!parser.Parse(args, &error)) {
    std::cerr << "Error: " << error << '\n';
    return 2;
  }
  if (parser.IsHelpRequested()) {
    parser.PrintHelp();
    return 0;
  }
  if (model_path.empty() || context == 0) {
    std::cerr << "Error: decide needs --model and a positive --context\n";
    return 2;
  }
  ext::Registry registry;
  ext::RegisterBuiltExtensions(registry);
  auto service =
      server::CreateDecisionService({.model_root = model_path.string(),
                                     .projector_path = mmproj.string(),
                                     .maximum_context_tokens = context},
                                    {}, &error);
  if (!service) {
    std::cerr << "Error: " << error << '\n';
    return 1;
  }
  try {
    std::ifstream file;
    if (!request_path.empty()) {
      file.open(request_path);
      if (!file)
        throw std::invalid_argument("cannot open request file");
    }
    std::istream& input = request_path.empty() ? std::cin : file;
    auto run = [&](const std::string& body) {
      const auto begin = std::chrono::steady_clock::now();
      auto result = service->decide(json::parse(body));
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - begin)
                            .count();
      std::cerr << "Decision: " << ms << " ms\n";
      std::cout << result.dump() << std::endl;
    };
    if (jsonl) {
      std::string line;
      while (std::getline(input, line))
        if (!line.empty())
          run(line);
    } else {
      std::string body(std::istreambuf_iterator<char>(input), {});
      run(body);
    }
    return 0;
  } catch (const std::invalid_argument& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
}
}  // namespace gufo::cli
