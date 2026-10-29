#include "src/cli/serve/lfm3_audio_runner.hpp"

#include <algorithm>
#include <cctype>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/lfm3_audio/generation.hpp"
namespace gufo::cli {
namespace audio = models::lfm3_audio;
namespace {
bool IsLfm3AudioMmproj(const std::filesystem::path& path) {
  std::string error;
  auto reader = core::GgufReader::OpenFile(path.string(), &error);
  return reader &&
         reader->GetMetadataString("clip.audio.projector_type") == "lfm3a" &&
         reader->GetMetadataString("clip.gen.audio.projector_type") ==
             "lfm3a_gen";
}
std::string Lower(std::string text) {
  for (auto& c : text)
    c = char(std::tolower(static_cast<unsigned char>(c)));
  return text;
}
// English language names for the ASR system prompt.
std::string AsrLanguage(const std::string& language) {
  static const std::unordered_map<std::string, std::string> names{
      {"ar", "Arabic"},   {"da", "Danish"},     {"de", "German"},
      {"el", "Greek"},    {"en", "English"},    {"fi", "Finnish"},
      {"fr", "French"},   {"he", "Hebrew"},     {"it", "Italian"},
      {"ja", "Japanese"}, {"ko", "Korean"},     {"nl", "Dutch"},
      {"pl", "Polish"},   {"pt", "Portuguese"}, {"es", "Spanish"},
      {"sv", "Swedish"},  {"tr", "Turkish"},    {"zh", "Chinese"}};
  if (auto it = names.find(Lower(language)); it != names.end())
    return it->second;
  return language;
}
// Offline transcription windows: the model card qualifies 30 s contexts.
constexpr std::size_t kWindowSamples = 30 * audio::Model::kSampleRate;
}  // namespace
std::filesystem::path FindLfm3AudioMmproj(const std::filesystem::path& model,
                                          const std::filesystem::path& mmproj,
                                          std::string* error) {
  error->clear();
  std::error_code ec;
  if (model.extension() != ".gguf" ||
      !std::filesystem::is_regular_file(model, ec))
    return {};
  {
    auto reader = core::GgufReader::OpenFile(model.string(), error);
    if (!reader || reader->GetMetadataString("general.architecture") != "lfm3")
      return {};
  }
  if (!mmproj.empty()) {
    if (!IsLfm3AudioMmproj(mmproj)) {
      *error = "--mmproj is not an LFM3-Audio mmproj GGUF: " + mmproj.string();
      return {};
    }
    return mmproj;
  }
  std::vector<std::filesystem::path> found;
  for (const auto& entry : std::filesystem::directory_iterator(
           model.parent_path().empty() ? std::filesystem::path(".")
                                       : model.parent_path(),
           ec)) {
    const auto name = entry.path().filename().string();
    if (entry.is_regular_file() && name.starts_with("mmproj") &&
        entry.path().extension() == ".gguf" && IsLfm3AudioMmproj(entry.path()))
      found.push_back(entry.path());
  }
  if (found.size() > 1)
    *error = "several LFM3-Audio mmproj files beside " + model.string() +
             "; select one with --mmproj";
  return found.size() == 1 ? found.front() : std::filesystem::path{};
}
struct Lfm3AudioRuntime::Impl {
  std::shared_ptr<audio::Model> model;
  std::unique_ptr<audio::Generator> generator;
  std::mutex mutex;
};
Lfm3AudioRuntime::Lfm3AudioRuntime() : impl_(std::make_unique<Impl>()) {}
Lfm3AudioRuntime::~Lfm3AudioRuntime() = default;
std::shared_ptr<Lfm3AudioRuntime> Lfm3AudioRuntime::Create(
    const std::filesystem::path& model, const std::filesystem::path& mmproj,
    std::uint32_t context, std::string* error) {
  auto runtime = std::shared_ptr<Lfm3AudioRuntime>(new Lfm3AudioRuntime);
  runtime->impl_->model =
      audio::Model::Load(model.string(), mmproj.string(), error);
  if (!runtime->impl_->model)
    return nullptr;
  runtime->impl_->generator =
      audio::Generator::Create(runtime->impl_->model, context, error);
  if (!runtime->impl_->generator)
    return nullptr;
  return runtime;
}
bool Lfm3AudioRuntime::Transcribe(
    const models::qwen3_asr::TranscriptionRequest& request,
    const models::qwen3_asr::CancellationCheck& cancelled,
    models::qwen3_asr::TranscriptionResult* result, std::string* error) {
  std::vector<float> pcm;
  if (!request.pcm16k.empty()) {
    pcm.assign(request.pcm16k.begin(), request.pcm16k.end());
  } else {
    std::uint32_t rate = 0;
    if (!audio::DecodeWav(request.wav, &pcm, &rate, error))
      return false;
    pcm = audio::Resample(pcm, rate, audio::Model::kSampleRate);
  }
  if (pcm.empty()) {
    *error = "empty audio";
    return false;
  }
  *result = {};
  result->sample_rate = audio::Model::kSampleRate;
  result->audio_samples = pcm.size();
  const std::string language =
      request.language ? AsrLanguage(*request.language) : std::string{};
  result->language = language;
  const std::lock_guard lock(impl_->mutex);
  for (std::size_t at = 0; at < pcm.size(); at += kWindowSamples) {
    const auto count = std::min(kWindowSamples, pcm.size() - at);
    std::vector<audio::ChatMessage> messages(1);
    messages[0].role = "user";
    messages[0].parts.push_back({true,
                                 {},
                                 {pcm.begin() + std::ptrdiff_t(at),
                                  pcm.begin() + std::ptrdiff_t(at + count)}});
    audio::GenerateOptions options;
    options.max_new_tokens = request.max_new_tokens;
    options.suppress_mixed = options.suppress_audio_out = true;
    const std::string prefix =
        result->text.empty() ? std::string{} : result->text + " ";
    audio::GenerateCallbacks callbacks;
    std::string window_text;
    bool stopped = false;
    callbacks.cancelled = [&] { return cancelled && cancelled(); };
    if (request.on_text)
      callbacks.on_text = [&](std::string_view piece) {
        window_text += piece;
        if (!request.on_text(prefix + window_text))
          stopped = true;
        return !stopped;
      };
    audio::GenerateResult generated;
    if (!impl_->generator->Generate(
            audio::BuildPrompt(audio::AsrSystemPrompt(language), messages),
            options, callbacks, &generated, error))
      return false;
    if (generated.cancelled || stopped) {
      *error = "transcription cancelled";
      return false;
    }
    if (!generated.text.empty())
      result->text = prefix + generated.text;
    result->generated_ids.insert(result->generated_ids.end(),
                                 generated.tokens.begin(),
                                 generated.tokens.end());
    result->audio_tokens += generated.audio_tokens;
    result->prompt_tokens += generated.prompt_tokens;
    result->mel_frames += audio::MelFrames(count);
    ++result->chunks;
    result->timings.audio_encoder_ms += generated.timings.encode_ms;
    result->timings.text_decoder_ms +=
        generated.timings.total_ms - generated.timings.encode_ms;
    result->timings.total_ms += generated.timings.total_ms;
  }
  result->decoded = result->text;
  return true;
}
bool Lfm3AudioRuntime::Synthesize(
    const models::qwen3_tts::SynthesisRequest& request,
    const models::qwen3_tts::CancellationCheck& cancelled,
    models::qwen3_tts::SynthesisResult* result, std::string* error) {
  if (!request.reference_audio.samples.empty()) {
    *error = "LFM3-Audio does not clone reference voices";
    return false;
  }
  if (request.text.empty()) {
    *error = "empty text";
    return false;
  }
  std::string system;
  if (!audio::TtsSystemPrompt(request.language, request.speaker, &system,
                              error))
    return false;
  std::vector<audio::ChatMessage> messages(1);
  messages[0].role = "user";
  messages[0].parts.push_back({false, request.text, {}});
  audio::GenerateOptions options;
  options.max_new_tokens = request.max_new_tokens;
  options.seed = request.sampling.seed;
  if (!request.sampling.sample)
    options.audio.temperature = 0;
  audio::GenerateCallbacks callbacks;
  callbacks.cancelled = [&] { return cancelled && cancelled(); };
  if (request.on_audio)
    callbacks.on_audio = request.on_audio;
  audio::GenerateResult generated;
  const std::lock_guard lock(impl_->mutex);
  if (!impl_->generator->Generate(
          audio::BuildPrompt(system, messages, "<|audio_start|>"), options,
          callbacks, &generated, error))
    return false;
  if (generated.cancelled) {
    *error = "synthesis cancelled";
    return false;
  }
  *result = {};
  result->sample_rate = audio::Model::kOutputRate;
  result->code_groups = audio::AudioWeights::kCodebooks;
  result->codes = std::move(generated.codes);
  result->sample_count = generated.samples.size();
  if (!request.on_audio)
    result->samples = std::move(generated.samples);
  return true;
}
}  // namespace gufo::cli
