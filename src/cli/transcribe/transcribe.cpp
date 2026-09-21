#include "src/cli/transcribe/transcribe.hpp"

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "src/cli/arg_parser.hpp"
#include "src/core/json.hpp"
#include "src/models/qwen3_asr/hip/transcription_runtime.hpp"

namespace gufo::cli {
namespace {

namespace qwen3_asr = models::qwen3_asr;
namespace qwen3_asr_hip = models::qwen3_asr::hip;
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

json::Value TimingsJson(const qwen3_asr::TranscriptionTimings& timings,
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
  std::vector<std::string> positionals;
  ArgParser parser(
      std::string(program_name) + " transcribe [OPTIONS] [WAV]",
      "Transcribe a PCM or float WAV with native HIP Qwen3-ASR-1.7B.");
  parser.AddOption("-m", "--model", "DIR", "Qwen3-ASR-1.7B model directory",
                   "Model", &model);
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
  std::vector<std::string> positionals;

  ArgParser parser(
      "gufo transcribe [OPTIONS] [WAV]",
      "Transcribe a PCM or float WAV with native HIP Qwen3-ASR-1.7B.");
  parser.AddOption("-m", "--model", "DIR", "Qwen3-ASR-1.7B model directory",
                   "Model", &model);
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

  const auto load_begin = std::chrono::steady_clock::now();
  auto runtime = qwen3_asr_hip::TranscriptionHipRuntime::Create(
      model.string(), capacity, &error);
  const auto load_end = std::chrono::steady_clock::now();
  if (runtime == nullptr) {
    std::cerr << "Error loading Qwen3-ASR: " << error << '\n';
    return 1;
  }
  const double load_ms =
      std::chrono::duration<double, std::milli>(load_end - load_begin).count();

  g_cancel_requested = 0;
  const auto previous_interrupt = std::signal(SIGINT, HandleInterrupt);
  const qwen3_asr::TranscriptionRequest request{
      .wav = wav,
      .context = context,
      .language = language.empty() ? std::nullopt
                                   : std::optional<std::string>(language),
      .max_new_tokens = maximum_tokens,
  };
  qwen3_asr::TranscriptionResult result;
  bool transcribed = true;
  for (std::size_t run = 0; run < warmup && transcribed; ++run) {
    transcribed = runtime->Transcribe(
        request, [] { return g_cancel_requested != 0; }, &result, &error);
  }
  std::vector<qwen3_asr::TranscriptionTimings> run_timings;
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
  for (const qwen3_asr::TranscriptionTimings& timings : run_timings) {
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
