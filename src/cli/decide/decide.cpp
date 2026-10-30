#include "src/cli/decide/decide.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

#include "src/cli/arg_parser.hpp"
#include "src/core/json.hpp"
#if defined(ENGINE_ENABLE_HIP)
#include "src/models/d1/model.hpp"
#endif
namespace gufo::cli {
int RunDecide(std::span<const char* const> args) {
  std::filesystem::path model_path, mmproj, request_path;
  unsigned context = 8192;
  bool jsonl = false;
  ArgParser parser("gufo decide [OPTIONS]",
                   "Run d1-Omni GGUF text, image and audio decisions (noul, "
                   "choice, score).");
  parser.AddOption("-m", "--model", "GGUF", "d1-Omni backbone GGUF", "Model",
                   &model_path);
  parser.AddOption("", "--mmproj", "GGUF",
                   "d1-Omni vision/audio projector GGUF", "Model", &mmproj);
  parser.AddOption(
      "-c", "--context", "N",
      "Maximum complete prompt length (default 8192, maximum 16384)", "Model",
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
  if (model_path.empty() || !context || context > 16384) {
    std::cerr << "Error: decide needs --model and --context in 1..16384\n";
    return 2;
  }
#if defined(ENGINE_ENABLE_HIP)
  try {
    auto model =
        models::d1::Model::Load(model_path.string(), mmproj.string(), context);
    std::ifstream file;
    if (!request_path.empty()) {
      file.open(request_path);
      if (!file)
        throw std::invalid_argument("cannot open request file");
    }
    std::istream& input = request_path.empty() ? std::cin : file;
    auto run = [&](const std::string& body) {
      const auto begin = std::chrono::steady_clock::now();
      auto result = model->Decide(json::parse(body));
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
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
#else
  std::cerr << "Error: decide requires a HIP build\n";
  return 1;
#endif
}
}  // namespace gufo::cli
