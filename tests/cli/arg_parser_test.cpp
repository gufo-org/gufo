#include "src/cli/arg_parser.hpp"

#include <array>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

void TestBasicFlagsAndOptions() {
  strix::cli::ArgParser parser("test_app", "A test application");
  bool verbose = false;
  std::string model = "default.gguf";
  int threads = 4;
  float temperature = 0.5f;

  parser.AddFlag("-v", "--verbose", "Enable verbose mode", "General", &verbose);
  parser.AddOption("-m", "--model", "PATH", "Path to model", "Model", &model);
  parser.AddOption("-t", "--threads", "N", "Number of threads", "Compute",
                   &threads);
  parser.AddOption("--temp", "--temperature", "T", "Sampling temperature",
                   "Sampling", &temperature);

  const std::array<const char*, 7> args = {"-v", "--model", "custom.gguf", "-t",
                                           "8",  "--temp",  "0.9"};

  std::string err;
  assert(parser.Parse(args, &err));
  assert(verbose == true);
  assert(model == "custom.gguf");
  assert(threads == 8);
  assert(temperature > 0.89f && temperature < 0.91f);
}

void TestListParsing() {
  strix::cli::ArgParser parser("test_bench");
  std::vector<std::size_t> depths;
  parser.AddOption("", "--n-depth", "LIST", "List of depths", "Bench", &depths);

  const std::array<const char*, 2> args = {"--n-depth", "4096,8192,16384"};
  std::string err;
  assert(parser.Parse(args, &err));
  assert((depths == std::vector<std::size_t>{4096, 8192, 16384}));

  const std::array<const char*, 2> invalid_args = {"--n-depth", "4096,invalid"};
  assert(!parser.Parse(invalid_args, &err));
  assert(!err.empty());
}

void TestPositionals() {
  strix::cli::ArgParser parser("test_prompt");
  std::string prompt;
  std::string model;
  parser.AddOption("-m", "--model", "PATH", "Model path", "Model", &model);
  parser.JoinPositionals(&prompt);

  const std::array<const char*, 4> args = {"-m", "model.gguf", "Hello",
                                           "world"};
  std::string err;
  assert(parser.Parse(args, &err));
  assert(model == "model.gguf");
  assert(prompt == "Hello world");
}

void TestHelpFormatting() {
  strix::cli::ArgParser parser("strix run", "Run inference on Strix Halo");
  std::string model;
  bool interactive = false;
  parser.AddOption("-m", "--model", "PATH", "Model file", "Model", &model);
  parser.AddFlag("-i", "--interactive", "Interactive chat", "Execution",
                 &interactive);

  std::string help = parser.FormatHelp();
  assert(help.find("Usage: strix run [OPTIONS]") != std::string::npos);
  assert(help.find("Model:") != std::string::npos);
  assert(help.find("Execution:") != std::string::npos);
  assert(help.find("--model") != std::string::npos);
  assert(help.find("--interactive") != std::string::npos);
}

int main() {
  TestBasicFlagsAndOptions();
  TestListParsing();
  TestPositionals();
  TestHelpFormatting();
  std::cout << "All ArgParser tests passed.\n";
  return 0;
}
