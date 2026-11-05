#include "src/cli/speak/speak.hpp"

#include <csignal>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "gufo/ext/registry.hpp"
#include "gufo/ext/tts.hpp"
#include "src/cli/arg_parser.hpp"

namespace gufo::cli {
namespace {
volatile std::sig_atomic_t g_speak_cancel = 0;  // NOLINT
void HandleSpeakInterrupt(int) {
  g_speak_cancel = 1;
}
ArgParser MakeParser(ext::SpeakOptions* o,
                     std::vector<std::string>* positionals) {
  ArgParser parser(
      "gufo speak [OPTIONS] [TEXT]",
      "Synthesize speech, or answer a spoken or written message in "
      "interleaved text and speech or in text, with a speech model that "
      "supports it.");
  parser.AddOption("-m", "--model", "PATH", "Speech model artifact", "Model",
                   &o->model);
  parser.AddOption("", "--mmproj", "GGUF",
                   "Audio projector sidecar of models that take one (default: "
                   "the model's own lookup)",
                   "Model", &o->projector_path);
  parser.AddOption("-p", "--text", "TEXT", "Text to speak, or the chat message",
                   "Input", &o->text);
  parser.AddOption("-f", "--audio", "WAV",
                   "Spoken message (chat and text modes)", "Input",
                   &o->audio_path);
  parser.AddOption("", "--mode", "NAME",
                   "tts (speak the text), chat (answer in interleaved text "
                   "and speech) or text (answer in text) (default: tts)",
                   "Input", &o->mode);
  parser.AddOption("", "--system", "TEXT",
                   "System instructions for chat and text modes (default: "
                   "the model's prompt of the mode)",
                   "Input", &o->system);
  parser.AddOption("", "--tools", "JSON",
                   "File with an array of OpenAI function tools; the model "
                   "calls them or answers (chat and text modes)",
                   "Input", &o->tools_path);
  parser.AddFlag("", "--reasoning", "Reason before answering (text mode)",
                 "Input", &o->reasoning);
  parser.AddOption("-l", "--language", "NAME",
                   "Speech language code or name, tts mode (default: en)",
                   "Voice", &o->language);
  parser.AddOption("", "--voice", "NAME",
                   "Voice name (default: the model's first voice)", "Voice",
                   &o->voice);
  parser.AddOption("-o", "--output", "WAV",
                   "Output WAV path (tts and chat modes)", "Output",
                   &o->output_path);
  parser.AddOption("", "--format", "NAME",
                   "Report format: text or json (default: text)", "Output",
                   &o->format);
  parser.AddOption("-n", "--max-frames", "N",
                   "Maximum generated slots (default: 3000)", "Generation",
                   &o->max_frames);
  parser.AddOption("-c", "--context", "N",
                   "Prompt plus generation capacity (default: 4096)",
                   "Generation", &o->context);
  parser.AddOption("", "--temperature", "T",
                   "Speech code temperature; 0 is greedy (default: 1.0)",
                   "Generation", &o->temperature);
  parser.AddOption("", "--top-k", "K", "Speech code top-k (default: 128)",
                   "Generation", &o->top_k);
  parser.AddOption("", "--seed", "N", "Sampling seed (default: 0)",
                   "Generation", &o->seed);
  parser.CollectPositionals(positionals);
  return parser;
}
}  // namespace

void PrintSpeakHelp(std::string_view) {
  ext::SpeakOptions options;
  std::vector<std::string> positionals;
  MakeParser(&options, &positionals).PrintHelp();
}

int RunSpeak(std::span<const char* const> args) {
  ext::SpeakOptions o;
  std::vector<std::string> positionals;
  auto parser = MakeParser(&o, &positionals);
  std::string error;
  if (!parser.Parse(args, &error)) {
    std::cerr << "Error: " << error << '\n';
    return 2;
  }
  if (parser.IsHelpRequested()) {
    parser.PrintHelp();
    return 0;
  }
  if (o.text.empty() && positionals.size() == 1)
    o.text = positionals.front();
  else if (!positionals.empty()) {
    std::cerr << "Error: supply the text once\n";
    return 2;
  }
  const bool text_mode = o.mode == "text";
  if (o.model.empty() || (o.mode != "tts" && o.mode != "chat" && !text_mode) ||
      (o.format != "text" && o.format != "json") ||
      o.output_path.empty() != text_mode || !o.max_frames ||
      o.max_frames >= o.context || (o.text.empty() && o.audio_path.empty())) {
    std::cerr << "Error: speak needs --model, a known --mode and --format, "
                 "text or --audio, --output except in text mode, and "
                 "--max-frames below --context\n";
    return 2;
  }
  ext::Registry registry;
  ext::RegisterBuiltExtensions(registry);
  const auto* provider = ext::ProbeTtsProvider(o.model);
  if (provider == nullptr) {
    std::cerr << "Error: no speech model claims " << o.model << '\n';
    return 2;
  }
  if (provider->speak_driver == nullptr) {
    std::cerr << "Error: " << provider->id
              << (provider->create || provider->create_with_options
                      ? " does not support gufo speak; use gufo serve tts\n"
                      : " is not available in this build\n");
    return 1;
  }
  g_speak_cancel = 0;
  const auto previous = std::signal(SIGINT, HandleSpeakInterrupt);
  o.cancelled = [] { return g_speak_cancel != 0; };
  const int status = provider->speak_driver(o);
  (void)std::signal(SIGINT, previous);
  return status;
}

}  // namespace gufo::cli
