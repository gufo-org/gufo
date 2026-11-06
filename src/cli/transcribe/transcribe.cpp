#include "src/cli/transcribe/transcribe.hpp"

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "gufo/ext/asr.hpp"
#include "gufo/ext/registry.hpp"
#include "src/cli/arg_parser.hpp"
#include "src/cli/serve/audio_buffer.hpp"
#include "src/core/json.hpp"

namespace gufo::cli {
namespace {

namespace json = gufo::json;

constexpr std::uintmax_t kMaximumAudioBytes = 256U << 20U;
volatile std::sig_atomic_t g_cancel_requested = 0;  // NOLINT

void HandleInterrupt(int) {
  g_cancel_requested = 1;
}

bool ReadAudio(const std::filesystem::path& path,
               std::vector<std::byte>* output, std::string* error) {
  std::error_code filesystem_error;
  const std::uintmax_t bytes =
      std::filesystem::file_size(path, filesystem_error);
  if (filesystem_error || bytes == 0U || bytes > kMaximumAudioBytes) {
    if (error != nullptr) {
      *error = "audio file must be between 1 byte and 256 MiB";
    }
    return false;
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    if (error != nullptr) {
      *error = "cannot open audio file: " + path.string();
    }
    return false;
  }
  std::vector<char> chars(static_cast<std::size_t>(bytes));
  input.read(chars.data(), static_cast<std::streamsize>(chars.size()));
  if (!input || static_cast<std::size_t>(input.gcount()) != chars.size()) {
    if (error != nullptr) {
      *error = "cannot read complete audio file: " + path.string();
    }
    return false;
  }
  output->resize(chars.size());
  std::ranges::transform(chars, output->begin(), [](char value) {
    return static_cast<std::byte>(static_cast<unsigned char>(value));
  });
  return true;
}

json::Value TimingsJson(const ext::TranscriptionTimings& timings,
                        double load_ms) {
  json::Value result = json::Value::object();
  result["load_ms"] = load_ms;
  result["decode_audio_ms"] = timings.decode_audio_ms;
  result["feature_extraction_ms"] = timings.feature_extraction_ms;
  result["audio_encoder_ms"] = timings.audio_encoder_ms;
  result["text_decoder_ms"] = timings.text_decoder_ms;
  result["total_ms"] = timings.total_ms;
  return result;
}

// Live transcription of a recording fed as a microphone would deliver it,
// as fast as the model decodes. A piece's time is the delay it adds after
// its audio arrives.
int RunLiveTranscription(ext::AsrRunner& runtime,
                         std::span<const std::byte> wav,
                         const std::string& format) {
  std::string error;
  server::AudioBuffer decoded;
  if (!server::DecodeWav(wav, &decoded, &error)) {
    std::cerr << "Error: " << error << '\n';
    return 1;
  }
  const auto rate = decoded.sample_rate;
  const auto pcm = server::ResampleMono(decoded, rate);
  auto stream = runtime.CreateLiveTranscription(rate, &error);
  if (!stream) {
    std::cerr << "Error: " << error << '\n';
    return 1;
  }
  const bool text = format == "text";
  auto on_text = [&](std::string_view piece) {
    if (text)
      std::cout << piece << std::flush;
    return g_cancel_requested == 0;
  };
  g_cancel_requested = 0;
  const auto previous_interrupt = std::signal(SIGINT, HandleInterrupt);
  using Clock = std::chrono::steady_clock;
  const auto ms = [](Clock::time_point since) {
    return std::chrono::duration<double, std::milli>(Clock::now() - since)
        .count();
  };
  const std::size_t piece = std::max<std::size_t>(1, rate / 10);
  double slowest = 0;
  bool ok = true;
  const auto start = Clock::now();
  for (std::size_t at = 0; ok && at < pcm.size(); at += piece) {
    const auto t = Clock::now();
    ok = stream->Push({pcm.data() + at, std::min(piece, pcm.size() - at)},
                      on_text, &error);
    slowest = std::max(slowest, ms(t));
  }
  const auto t = Clock::now();
  ok = ok && stream->Finish(on_text, &error);
  const double finish = ms(t);
  const double total = ms(start);
  (void)std::signal(SIGINT, previous_interrupt);
  if (!ok) {
    std::cerr << "Error transcribing audio: " << error << '\n';
    return g_cancel_requested != 0 ? 130 : 1;
  }
  if (text) {
    std::cout << '\n';
    return 0;
  }
  json::Value report = json::Value::object();
  report["schema"] = "gufo.live-transcription.v1";
  report["text"] = stream->text();
  report["audio_seconds"] = rate ? double(pcm.size()) / rate : 0.0;
  json::Value timings = json::Value::object();
  timings["slowest_piece_ms"] = slowest;
  timings["finish_ms"] = finish;
  timings["total_ms"] = total;
  report["timings"] = std::move(timings);
  std::cout << report.dump() << '\n';
  return 0;
}

}  // namespace

void PrintTranscribeHelp(std::string_view program_name) {
  std::filesystem::path model;
  std::filesystem::path audio;
  std::string language;
  std::string context;
  std::string format = "text";
  std::size_t maximum_tokens = 256;
  std::size_t capacity = 1024;
  std::size_t repeat = 1;
  std::size_t warmup = 0;
  std::filesystem::path mmproj;
  bool live = false;
  std::vector<std::string> positionals;
  ArgParser parser(
      std::string(program_name) + " transcribe [OPTIONS] [WAV]",
      "Transcribe a PCM or float WAV with native HIP speech recognition.");
  parser.AddOption("-m", "--model", "PATH",
                   "Speech recognition model (Qwen3-ASR-1.7B directory, or "
                   "an extension model's artifact)",
                   "Model", &model);
  parser.AddOption("", "--mmproj", "GGUF",
                   "Audio projector sidecar of models that take one (default: "
                   "the model's own lookup)",
                   "Model", &mmproj);
  parser.AddOption("-f", "--audio", "WAV", "Input WAV path", "Input", &audio);
  parser.AddOption("-l", "--language", "NAME",
                   "Optional language name or ISO code; auto-detect by default",
                   "Input", &language);
  parser.AddOption("", "--prompt", "TEXT", "Optional transcription context",
                   "Input", &context);
  parser.AddOption("-n", "--max-tokens", "N",
                   "Maximum generated tokens per audio chunk (default: 256)",
                   "Generation", &maximum_tokens);
  parser.AddOption("-c", "--context", "N",
                   "Prompt plus generation capacity (default: 1024)",
                   "Generation", &capacity);
  parser.AddOption(
      "", "--repeat", "N",
      "Measured transcriptions with one loaded runtime (default: 1)",
      "Benchmark", &repeat);
  parser.AddOption("", "--warmup", "N",
                   "Discarded warmup transcriptions (default: 0)", "Benchmark",
                   &warmup);
  parser.AddFlag("", "--live",
                 "Live transcription, for models that support it: feed the "
                 "audio in 100 ms pieces and print text as it is decoded",
                 "Input", &live);
  parser.AddOption("", "--format", "NAME",
                   "Output format: text or json (default: text)", "Output",
                   &format);
  parser.CollectPositionals(&positionals);
  parser.PrintHelp();
}

int RunTranscribe(std::span<const char* const> args) {
  std::filesystem::path model;
  std::filesystem::path audio;
  std::string language;
  std::string context;
  std::string format = "text";
  std::size_t maximum_tokens = 256;
  std::size_t capacity = 1024;
  std::size_t repeat = 1;
  std::size_t warmup = 0;
  std::filesystem::path mmproj;
  bool live = false;
  std::vector<std::string> positionals;

  ArgParser parser(
      "gufo transcribe [OPTIONS] [WAV]",
      "Transcribe a PCM or float WAV with native HIP speech recognition.");
  parser.AddOption("-m", "--model", "PATH",
                   "Speech recognition model (Qwen3-ASR-1.7B directory, or "
                   "an extension model's artifact)",
                   "Model", &model);
  parser.AddOption("", "--mmproj", "GGUF",
                   "Audio projector sidecar of models that take one (default: "
                   "the model's own lookup)",
                   "Model", &mmproj);
  parser.AddOption("-f", "--audio", "WAV", "Input WAV path", "Input", &audio);
  parser.AddOption("-l", "--language", "NAME",
                   "Optional language name or ISO code; auto-detect by default",
                   "Input", &language);
  parser.AddOption("", "--prompt", "TEXT", "Optional transcription context",
                   "Input", &context);
  parser.AddOption("-n", "--max-tokens", "N",
                   "Maximum generated tokens per audio chunk (default: 256)",
                   "Generation", &maximum_tokens);
  parser.AddOption("-c", "--context", "N",
                   "Prompt plus generation capacity (default: 1024)",
                   "Generation", &capacity);
  parser.AddOption(
      "", "--repeat", "N",
      "Measured transcriptions with one loaded runtime (default: 1)",
      "Benchmark", &repeat);
  parser.AddOption("", "--warmup", "N",
                   "Discarded warmup transcriptions (default: 0)", "Benchmark",
                   &warmup);
  parser.AddFlag("", "--live",
                 "Live transcription, for models that support it: feed the "
                 "audio in 100 ms pieces and print text as it is decoded",
                 "Input", &live);
  parser.AddOption("", "--format", "NAME",
                   "Output format: text or json (default: text)", "Output",
                   &format);
  parser.CollectPositionals(&positionals);

  std::string error;
  if (!parser.Parse(args, &error)) {
    std::cerr << "Error: " << error << '\n';
    PrintTranscribeHelp("gufo");
    return 2;
  }
  if (parser.IsHelpRequested()) {
    PrintTranscribeHelp("gufo");
    return 0;
  }
  if (model.empty()) {
    std::cerr << "Error: --model <DIR> is required\n";
    return 2;
  }
  if (audio.empty() && positionals.size() == 1U) {
    audio = positionals.front();
  } else if (!positionals.empty()) {
    std::cerr << "Error: supply exactly one WAV path\n";
    return 2;
  }
  if (audio.empty() || maximum_tokens == 0U || capacity < 32U ||
      maximum_tokens >= capacity || repeat == 0U || repeat > 100U ||
      warmup > 100U || (format != "text" && format != "json")) {
    std::cerr << "Error: invalid audio, token capacity, or output format\n";
    return 2;
  }

  std::vector<std::byte> wav;
  if (!ReadAudio(audio, &wav, &error)) {
    std::cerr << "Error: " << error << '\n';
    return 1;
  }

  ext::Registry registry;
  ext::RegisterBuiltExtensions(registry);
  const ext::AsrProvider* provider = ext::ProbeAsrProvider(model.string());
  if (!provider) {
    provider = ext::FindAsrProvider("qwen3-asr");
  }
  if (!provider || (!provider->create && !provider->create_with_options)) {
    std::cerr << "Error loading ASR: provider not available\n";
    return 1;
  }
  if (!mmproj.empty() && !provider->create_with_options) {
    std::cerr << "Error: this model does not take --mmproj\n";
    return 2;
  }

  const auto load_begin = std::chrono::steady_clock::now();
  auto runtime =
      provider->create_with_options
          ? provider->create_with_options({.model_root = model.string(),
                                           .maximum_context_tokens = capacity,
                                           .projector_path = mmproj.string()},
                                          &error)
          : provider->create(model.string(), capacity, &error);
  const auto load_end = std::chrono::steady_clock::now();
  if (runtime == nullptr) {
    std::cerr << "Error loading ASR model: " << error << '\n';
    return 1;
  }
  if (live) {
    if (!runtime->SupportsLiveTranscription()) {
      std::cerr << "Error: --live is not supported by this model\n";
      return 2;
    }
    return RunLiveTranscription(*runtime, wav, format);
  }
  const double load_ms =
      std::chrono::duration<double, std::milli>(load_end - load_begin).count();

  g_cancel_requested = 0;
  const auto previous_interrupt = std::signal(SIGINT, HandleInterrupt);
  const ext::TranscriptionRequest request{
      .wav = wav,
      .context = context,
      .language = language.empty() ? std::nullopt
                                   : std::optional<std::string>(language),
      .max_new_tokens = maximum_tokens,
  };
  ext::TranscriptionResult result;
  bool transcribed = true;
  for (std::size_t run = 0; run < warmup && transcribed; ++run) {
    transcribed = runtime->Transcribe(
        request, [] { return g_cancel_requested != 0; }, &result, &error);
  }
  std::vector<ext::TranscriptionTimings> run_timings;
  run_timings.reserve(repeat);
  std::string expected_text;
  for (std::size_t run = 0; run < repeat && transcribed; ++run) {
    transcribed = runtime->Transcribe(
        request, [] { return g_cancel_requested != 0; }, &result, &error);
    if (!transcribed) {
      break;
    }
    if (run == 0U) {
      expected_text = result.text;
    } else if (result.text != expected_text) {
      transcribed = false;
      error = "repeated Qwen3-ASR transcription changed its transcript";
      break;
    }
    run_timings.push_back(result.timings);
  }
  (void)std::signal(SIGINT, previous_interrupt);
  if (!transcribed) {
    std::cerr << "Error transcribing audio: " << error << '\n';
    return g_cancel_requested != 0 ? 130 : 1;
  }

  if (format == "text") {
    std::cout << result.text << '\n';
    return 0;
  }
  json::Value report = json::Value::object();
  report["schema"] = "gufo.qwen3-asr.transcription.v1";
  report["backend"] = "native-hip";
  report["language"] = result.language;
  report["text"] = result.text;
  report["audio_samples"] = result.audio_samples;
  report["sample_rate"] = static_cast<std::size_t>(result.sample_rate);
  report["mel_frames"] = result.mel_frames;
  report["audio_tokens"] = result.audio_tokens;
  report["prompt_tokens"] = result.prompt_tokens;
  report["generated_tokens"] = result.generated_ids.size();
  report["chunks"] = result.chunks;
  report["timings"] = TimingsJson(result.timings, load_ms);
  json::Value runs = json::Value::array();
  double total_ms = 0.0;
  for (const ext::TranscriptionTimings& timings : run_timings) {
    runs.push_back(TimingsJson(timings, 0.0));
    total_ms += timings.total_ms;
  }
  report["runs"] = std::move(runs);
  json::Value benchmark = json::Value::object();
  benchmark["warmup"] = warmup;
  benchmark["repeat"] = repeat;
  benchmark["mean_total_ms"] =
      total_ms / static_cast<double>(run_timings.size());
  report["benchmark"] = std::move(benchmark);
  std::cout << report.dump() << '\n';
  return 0;
}

}  // namespace gufo::cli
