#include "src/cli/speak/speak.hpp"

#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "src/cli/arg_parser.hpp"
#include "src/core/json.hpp"
#if defined(ENGINE_ENABLE_HIP)
#include "src/cli/serve/lfm3_audio_runner.hpp"
#include "src/models/lfm3_audio/generation.hpp"
#endif

namespace gufo::cli {
namespace {
volatile std::sig_atomic_t g_speak_cancel = 0;  // NOLINT
void HandleSpeakInterrupt(int) {
  g_speak_cancel = 1;
}
struct SpeakOptions {
  std::filesystem::path model, mmproj, output, audio, tools;
  std::string text, system, language = "en", voice = "female", mode = "tts",
                            format = "text";
  bool reasoning = false;
  std::size_t max_frames = 3000, context = 4096;
  std::uint64_t seed = 0;
  float temperature = 1.0F;
  int top_k = 128;
};
ArgParser MakeParser(SpeakOptions* o, std::vector<std::string>* positionals) {
  ArgParser parser(
      "gufo speak [OPTIONS] [TEXT]",
      "Synthesize speech, or answer a spoken or written message in "
      "interleaved text and speech or in text, with native HIP "
      "LFM3-Audio-2B.");
  parser.AddOption("-m", "--model", "GGUF", "LFM3-Audio backbone GGUF", "Model",
                   &o->model);
  parser.AddOption("", "--mmproj", "GGUF",
                   "LFM3-Audio audio towers (default: the mmproj*.gguf beside "
                   "--model)",
                   "Model", &o->mmproj);
  parser.AddOption("-p", "--text", "TEXT", "Text to speak, or the chat message",
                   "Input", &o->text);
  parser.AddOption("-f", "--audio", "WAV",
                   "Spoken message (chat and text modes)", "Input", &o->audio);
  parser.AddOption("", "--mode", "NAME",
                   "tts (speak the text), chat (answer in interleaved text "
                   "and speech) or text (answer in text) (default: tts)",
                   "Input", &o->mode);
  parser.AddOption("", "--system", "TEXT",
                   "System instructions for chat and text modes (default: "
                   "the model card prompt of the mode)",
                   "Input", &o->system);
  parser.AddOption("", "--tools", "JSON",
                   "File with an array of OpenAI function tools; the model "
                   "calls them or answers (chat and text modes)",
                   "Input", &o->tools);
  parser.AddFlag("", "--reasoning",
                 "Reason before answering (text mode; prefills <think>)",
                 "Input", &o->reasoning);
  parser.AddOption("-l", "--language", "NAME",
                   "Speech language code or name, tts mode (default: en)",
                   "Voice", &o->language);
  parser.AddOption("", "--voice", "NAME", "female or male (default: female)",
                   "Voice", &o->voice);
  parser.AddOption("-o", "--output", "WAV",
                   "Output 24 kHz WAV path (tts and chat modes)", "Output",
                   &o->output);
  parser.AddOption("", "--format", "NAME",
                   "Report format: text or json (default: text)", "Output",
                   &o->format);
  parser.AddOption("-n", "--max-frames", "N",
                   "Maximum generated slots, 80 ms of speech each "
                   "(default: 3000)",
                   "Generation", &o->max_frames);
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
  SpeakOptions options;
  std::vector<std::string> positionals;
  MakeParser(&options, &positionals).PrintHelp();
}
int RunSpeak(std::span<const char* const> args) {
  SpeakOptions o;
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
  const bool chat = o.mode == "chat", text_mode = o.mode == "text";
  const bool tts = !chat && !text_mode;
  if (o.model.empty() || (o.output.empty() != text_mode) ||
      (o.mode != "tts" && !chat && !text_mode) ||
      (o.format != "text" && o.format != "json") || !o.max_frames ||
      o.max_frames >= o.context || (o.text.empty() && o.audio.empty()) ||
      (tts && (o.text.empty() || !o.audio.empty() || !o.tools.empty() ||
               !o.system.empty())) ||
      (o.reasoning && (!text_mode || !o.tools.empty()))) {
    std::cerr << "Error: speak needs --model and text (tts, with --output), "
                 "or text/--audio (chat with --output, or text); "
                 "--reasoning is for text mode without --tools\n";
    return 2;
  }
#if defined(ENGINE_ENABLE_HIP)
  namespace audio = models::lfm3_audio;
  const auto mmproj = FindLfm3AudioMmproj(o.model, o.mmproj, &error);
  if (mmproj.empty()) {
    std::cerr << "Error: "
              << (error.empty() ? "--model is not an LFM3-Audio GGUF with an "
                                  "mmproj"
                                : error)
              << '\n';
    return 2;
  }
  const auto load_begin = std::chrono::steady_clock::now();
  auto model = audio::Model::Load(o.model.string(), mmproj.string(), &error);
  std::unique_ptr<audio::Generator> generator;
  if (model)
    generator =
        audio::Generator::Create(model, std::uint32_t(o.context), &error);
  if (!generator) {
    std::cerr << "Error loading LFM3-Audio: " << error << '\n';
    return 1;
  }
  const double load_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - load_begin)
                             .count();
  std::vector<audio::ChatMessage> messages(1);
  messages[0].role = "user";
  if (!o.audio.empty()) {
    std::ifstream in(o.audio, std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(in)), {});
    std::vector<float> pcm;
    std::uint32_t rate = 0;
    if (raw.empty() ||
        !audio::DecodeWav(
            {reinterpret_cast<const std::byte*>(raw.data()), raw.size()}, &pcm,
            &rate, &error)) {
      std::cerr << "Error reading " << o.audio << ": " << error << '\n';
      return 1;
    }
    messages[0].parts.push_back(
        {true, {}, audio::Resample(pcm, rate, audio::Model::kSampleRate)});
  }
  if (!o.text.empty())
    messages[0].parts.push_back({false, o.text, {}});
  std::vector<std::string> tools;
  if (!o.tools.empty()) {
    std::ifstream in(o.tools);
    std::string body((std::istreambuf_iterator<char>(in)), {});
    try {
      const auto list = json::parse(body);
      if (!list.is_array() || list.items().empty())
        throw std::invalid_argument("expected a non-empty array");
      for (const auto& tool : list.items())
        tools.push_back(audio::ToolDefinition(tool.dump()));
    } catch (const std::exception& e) {
      std::cerr << "Error reading " << o.tools << ": " << e.what() << '\n';
      return 2;
    }
  }
  // Model card prompts: interleaved replies are prefilled with
  // <|mixed_start|> and text replies with <think> or <think></think>; with
  // tools the model chooses between a call and an answer.
  std::string system = o.system, prefix;
  if (tts) {
    prefix = "<|audio_start|>";
    if (!audio::TtsSystemPrompt(o.language, o.voice, &system, &error)) {
      std::cerr << "Error: " << error << '\n';
      return 2;
    }
  } else if (system.empty() && !tools.empty())
    system = chat ? audio::kToolsInterleavedSystemPrompt
                  : audio::kToolsTextSystemPrompt;
  else if (system.empty() && chat)
    system = audio::kInterleavedSystemPrompt;
  if (tools.empty() && chat)
    prefix = "<|mixed_start|>";
  else if (tools.empty() && text_mode)
    prefix = o.reasoning ? "<think>" : "<think></think>";
  audio::GenerateOptions options;
  if (text_mode)
    options.suppress_mixed = options.suppress_audio_out = true;
  else if (chat && !tools.empty()) {
    // A spoken reply starts with a call or with interleaved speech.
    const auto call = model->tokenizer().Encode("<tool_call>", true);
    if (call.size() == 1)
      options.first_tokens = {call[0], model->tokens().mixed_start};
  }
  options.max_new_tokens = o.max_frames;
  options.seed = o.seed;
  options.audio.temperature = o.temperature;
  options.audio.top_k = o.top_k;
  audio::GenerateCallbacks callbacks;
  g_speak_cancel = 0;
  const auto previous = std::signal(SIGINT, HandleSpeakInterrupt);
  callbacks.cancelled = [] { return g_speak_cancel != 0; };
  // Reasoning streams to stderr, the answer to stdout; tool calls are
  // reported once the reply ends.
  audio::ReplyParser reply(o.reasoning);
  auto show = [&](const audio::ReplyParser::Delta& delta) {
    if (o.format != "text")
      return;
    if (!delta.reasoning.empty())
      std::cerr << delta.reasoning << std::flush;
    if (!delta.content.empty())
      std::cout << delta.content << std::flush;
  };
  if (!tts)
    callbacks.on_text = [&](std::string_view piece) {
      show(reply.Feed(piece));
      return true;
    };
  audio::GenerateResult result;
  const bool ok =
      generator->Generate(audio::BuildPrompt(system, messages, prefix, tools),
                          options, callbacks, &result, &error);
  (void)std::signal(SIGINT, previous);
  if (!ok || result.cancelled) {
    std::cerr << "Error: " << (ok ? "cancelled" : error) << '\n';
    return result.cancelled ? 130 : 1;
  }
  std::vector<audio::ToolCall> calls;
  show(reply.Finish(&calls));
  json::Value tool_calls = json::Value::array();
  for (const auto& call : calls) {
    auto entry = json::Value::object();
    entry["name"] = call.name;
    auto arguments = json::Value::object();
    for (const auto& argument : call.arguments)
      arguments[argument.name] = argument.is_string
                                     ? json::Value(argument.value)
                                     : json::parse(argument.value);
    entry["arguments"] = std::move(arguments);
    tool_calls.push_back(std::move(entry));
  }
  if (!tts && o.format == "text") {
    std::cout << '\n';
    for (const auto& call : tool_calls.items())
      std::cout << "tool call: " << call.dump() << '\n';
  }
  if (!o.output.empty()) {
    const auto wav =
        audio::EncodeWav(result.samples, audio::Model::kOutputRate);
    std::ofstream out(o.output, std::ios::binary);
    out.write(reinterpret_cast<const char*>(wav.data()),
              std::streamsize(wav.size()));
    if (!out) {
      std::cerr << "Error: cannot write " << o.output << '\n';
      return 1;
    }
  }
  const double seconds =
      double(result.samples.size()) / audio::Model::kOutputRate;
  if (o.format == "json") {
    json::Value report = json::Value::object();
    report["schema"] = "gufo.lfm3-audio.speech.v1";
    report["text"] = tts ? result.text : reply.content();
    if (!tts) {
      report["reasoning"] = reply.reasoning();
      report["tool_calls"] = std::move(tool_calls);
    }
    report["frames"] = result.frames;
    report["audio_seconds"] = seconds;
    report["prompt_tokens"] = result.prompt_tokens;
    json::Value t = json::Value::object();
    t["load_ms"] = load_ms;
    t["encode_ms"] = result.timings.encode_ms;
    t["prefill_ms"] = result.timings.prefill_ms;
    t["first_text_ms"] = result.timings.first_text_ms;
    t["first_audio_ms"] = result.timings.first_audio_ms;
    t["backbone_ms"] = result.timings.backbone_ms;
    t["depth_ms"] = result.timings.depth_ms;
    t["total_ms"] = result.timings.total_ms;
    t["real_time_factor"] = result.timings.total_ms > 0
                                ? seconds * 1000 / result.timings.total_ms
                                : 0.0;
    report["timings"] = std::move(t);
    std::cout << report.dump() << '\n';
  } else if (!o.output.empty())
    std::cerr << "wrote " << o.output << ": " << seconds << " s, "
              << result.frames << " frames in " << result.timings.total_ms
              << " ms (first audio " << result.timings.first_audio_ms
              << " ms)\n";
  return 0;
#else
  std::cerr << "Error: speak requires the HIP backend\n";
  return 1;
#endif
}
}  // namespace gufo::cli
