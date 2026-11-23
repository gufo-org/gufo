#include <gufo/ext/generated/yue2_constants.hpp>
#include <gufo/ext/music.hpp>
#include <gufo/ext/registry.hpp>

#include "probe.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <gufo/ext/json.hpp>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "config.hpp"
#include "hip/backbone.hpp"
#include "hip/vae.hpp"
#include "noise.hpp"
#include "protocol.hpp"
#include "sampler.hpp"
#include "tokenizer.hpp"
#endif

namespace gufo::models::yue2 {

#if defined(ENGINE_ENABLE_HIP)
namespace {

constexpr double kMaxSeconds = 360.0;

float RoundBfloat16(float value) {
  std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  bits += 0x7FFFU + ((bits >> 16U) & 1U);
  return std::bit_cast<float>(bits & 0xFFFF0000U);
}

double Milliseconds(std::chrono::steady_clock::time_point since) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - since)
      .count();
}

/// Validated `MusicRequest::provider_json`.
struct ProviderRequest {
  Planning planning{Planning::kFull};
  Sampling abc;
  Sampling semantic;
};

bool ApplySampling(const json::Value* value, Sampling* sampling,
                   std::string* error) {
  if (value == nullptr)
    return true;
  if (!value->is_object()) {
    *error = "sampling overrides must be an object";
    return false;
  }
  for (const auto& [key, field] : value->members()) {
    if (!field.is_number()) {
      *error = "sampling option '" + key + "' must be a number";
      return false;
    }
    const double number = field.as_double(0.0);
    const bool integral = std::floor(number) == number && number >= 0.0;
    if (key == "temperature") {
      sampling->temperature = static_cast<float>(number);
    } else if (key == "top_p") {
      sampling->top_p = static_cast<float>(number);
    } else if (key == "repetition_penalty") {
      sampling->repetition_penalty = static_cast<float>(number);
    } else if (key == "top_k" || key == "penalty_window" ||
               key == "min_tokens" || key == "max_tokens") {
      if (!integral || number > 1e6) {
        *error = "sampling option '" + key + "' must be a whole number";
        return false;
      }
      const auto count = static_cast<std::uint32_t>(number);
      (key == "top_k"            ? sampling->top_k
       : key == "penalty_window" ? sampling->penalty_window
       : key == "min_tokens"     ? sampling->min_tokens
                                 : sampling->max_tokens) = count;
    } else {
      *error = "unknown sampling option '" + key + "'";
      return false;
    }
  }
  return ValidateSampling(*sampling, error);
}

bool ParseProviderRequest(std::string_view text,
                          const GenerationDefaults& defaults,
                          ProviderRequest* output, std::string* error) {
  output->abc = defaults.abc;
  output->semantic = defaults.semantic;
  if (text.empty())
    return true;
  json::Value value;
  try {
    value = json::parse(text);
  } catch (const std::exception& exception) {
    *error = std::string("invalid provider options: ") + exception.what();
    return false;
  }
  if (!value.is_object()) {
    *error = "provider options must be an object";
    return false;
  }
  for (const auto& [key, field] : value.members()) {
    if (key == "planning") {
      const auto planning =
          field.is_string() ? ParsePlanning(field.str()) : std::nullopt;
      if (!planning) {
        *error = "planning must be full, melody or off";
        return false;
      }
      output->planning = *planning;
    } else if (key == "abc_sampling") {
      if (!ApplySampling(&field, &output->abc, error))
        return false;
    } else if (key == "semantic_sampling") {
      if (!ApplySampling(&field, &output->semantic, error))
        return false;
    } else {
      *error = "unknown provider option '" + key + "'";
      return false;
    }
  }
  return true;
}

class Yue2Runner final : public ext::MusicRunner {
public:
  Yue2Runner(ModelPaths paths, GenerationDefaults defaults,
             std::unique_ptr<TextTokenizer> tokenizer,
             std::unique_ptr<hip::Backbone> backbone,
             std::unique_ptr<hip::VaeDecoder> vae)
      : paths_(std::move(paths)),
        defaults_(defaults),
        tokenizer_(std::move(tokenizer)),
        backbone_(std::move(backbone)),
        vae_(std::move(vae)) {}

  ext::MusicCapabilities Capabilities() const override {
    ext::MusicCapabilities capabilities;
    capabilities.lyrics = ext::MusicSupport::kOptional;
    capabilities.score = ext::MusicSupport::kOptional;
    capabilities.score_formats = {"abc"};
    capabilities.score_output = true;
    capabilities.plan_only = true;
    capabilities.guidance = true;
    capabilities.steps = true;
    capabilities.duration = ext::MusicDurationControl::kMaximum;
    capabilities.max_seconds = kMaxSeconds;
    return capabilities;
  }

  bool ValidateRequest(const ext::MusicRequest& request,
                       std::string* error) const override {
    std::string message;
    ProviderRequest options;
    if (!ParseProviderRequest(request.provider_json, defaults_, &options,
                              &message)) {
      *error = message;
      return false;
    }
    if (!request.score.empty() && options.planning == Planning::kOff) {
      *error = "a score needs planning full or melody";
      return false;
    }
    // Upstream rejects a score that is only whitespace.
    if (!request.score.empty() &&
        request.score.find_first_not_of(" \t\n\r\f\v") == std::string::npos) {
      *error = "the score is empty";
      return false;
    }
    if (request.plan_only && options.planning == Planning::kOff) {
      *error = "plan_only needs planning full or melody";
      return false;
    }
    if (request.guidance &&
        (*request.guidance < 0.0F || *request.guidance > 20.0F)) {
      *error = "guidance_scale must be in [0, 20]";
      return false;
    }
    if (request.seed && *request.seed >= (1ULL << 63U)) {
      *error = "seed must be below 2^63";
      return false;
    }
    return true;
  }

  std::uint32_t SampleRate() const override {
    return static_cast<std::uint32_t>(vae_->config().sample_rate);
  }
  std::string ModelId() const override { return "YuE2-3B"; }

  bool Generate(const ext::MusicRequest& request,
                const ext::CancellationCheck& is_cancelled,
                const ext::MusicProgress& progress, ext::MusicResult* result,
                std::string* error) override {
    try {
      return Run(request, is_cancelled, progress, result, error);
    } catch (const std::exception& exception) {
      *error = exception.what();
      return false;
    }
  }

private:
  struct Stage {
    std::vector<std::uint32_t> tokens;
    bool truncated{false};
    double ms{0.0};
  };

  // Samples one AR stage; the cache already holds the prefix and `logits`
  // its last row. Generated tokens except the end token are fed back.
  bool Sample(Phase phase, const Sampling& sampling, std::uint64_t seed,
              bool legacy, float guidance, hip::KvCache& cache,
              std::vector<float> logits, hip::KvCache* negative,
              std::vector<float> negative_logits,
              const ext::CancellationCheck& is_cancelled,
              const ext::MusicProgress& progress, Stage* stage,
              std::string* error) {
    const auto start = std::chrono::steady_clock::now();
    const VocabularyWindow window = PhaseWindow(phase);
    const std::uint32_t end = PhaseEndToken(phase);
    const std::string_view name = phase == Phase::kAbc ? "plan" : "semantic";
    TokenSampler sampler(phase, sampling, seed, legacy);
    bool ended = false;
    for (std::uint32_t step = 0; step < sampling.max_tokens; ++step) {
      if (is_cancelled && is_cancelled()) {
        *error = "cancelled";
        return false;
      }
      const std::uint32_t token = sampler.Next(
          logits,
          negative != nullptr ? std::span<const float>(negative_logits)
                              : std::span<const float>(),
          guidance);
      if (token == end) {
        ended = true;
        break;
      }
      stage->tokens.push_back(token);
      if (progress && !progress(name, stage->tokens.size(), 0)) {
        *error = "cancelled";
        return false;
      }
      // Every generated token enters the cache, so the semantic stage and the
      // acoustic prefix can reuse it even when a budget ends the stage.
      backbone_->Append(cache, {&token, 1}, window, &logits);
      if (negative != nullptr)
        backbone_->Append(*negative, {&token, 1}, window, &negative_logits);
    }
    stage->truncated = !ended;
    stage->ms = Milliseconds(start);
    return true;
  }

  bool Run(const ext::MusicRequest& request,
           const ext::CancellationCheck& is_cancelled,
           const ext::MusicProgress& progress, ext::MusicResult* result,
           std::string* error) {
    const auto start = std::chrono::steady_clock::now();
    ProviderRequest options;
    if (!ParseProviderRequest(request.provider_json, defaults_, &options,
                              error))
      return false;
    const SongPrompt prompt{request.prompt, request.lyrics, options.planning};
    const std::uint64_t seed = request.seed.value_or(kDefaultSeed);
    const float guidance =
        request.guidance.value_or(DefaultGuidance(options.planning));
    const int steps = request.steps.value_or(defaults_.ode_steps);
    Sampling semantic = options.semantic;
    if (request.duration_seconds) {
      const auto frames = static_cast<std::uint32_t>(
          std::ceil(*request.duration_seconds * kFrameRate));
      semantic.max_tokens =
          std::clamp<std::uint32_t>(frames, 1, semantic.max_tokens);
      semantic.min_tokens = std::min(semantic.min_tokens, semantic.max_tokens);
    }

    // 1. Plan: a supplied score replaces it; planning off skips it.
    std::vector<std::uint32_t> abc_ids;
    Stage plan;
    std::unique_ptr<hip::KvCache> cache;
    std::vector<float> logits;
    const std::vector<std::uint32_t> plan_prefix =
        PlanPrefix(*tokenizer_, prompt);
    const bool planning =
        request.score.empty() && options.planning != Planning::kOff;
    if (planning) {
      if (plan_prefix.size() + options.abc.max_tokens > kContext) {
        *error = "prompt and planning budget exceed the 24576-token context";
        return false;
      }
      // One cache serves both stages: the score stays in it as the semantic
      // prefix.
      const std::size_t capacity =
          std::min(kContext, plan_prefix.size() + options.abc.max_tokens + 2 +
                                 semantic.max_tokens + 1);
      cache = backbone_->NewCache(capacity);
      backbone_->Append(*cache, plan_prefix, PhaseWindow(Phase::kAbc), &logits);
      if (!Sample(Phase::kAbc, options.abc, seed, false, 1.0F, *cache,
                  std::move(logits), nullptr, {}, is_cancelled, progress, &plan,
                  error))
        return false;
      abc_ids = plan.tokens;
    } else if (!request.score.empty()) {
      abc_ids = tokenizer_->Encode(request.score);
    }
    result->score_format = options.planning == Planning::kOff ? "" : "abc";
    result->score =
        planning ? tokenizer_->Decode(abc_ids) : std::string(request.score);
    if (request.plan_only) {
      result->finish = plan.truncated ? ext::MusicFinish::kTruncated
                                      : ext::MusicFinish::kComplete;
      result->total_ms = Milliseconds(start);
      result->telemetry_json =
          Telemetry(plan, {}, 0, 0, 0, abc_ids.size(), plan_prefix.size(), seed,
                    result->total_ms);
      return true;
    }

    // 2. Semantic codec tokens.
    const std::vector<std::uint32_t> prefix =
        SemanticPrefix(*tokenizer_, prompt, abc_ids);
    if (prefix.size() + semantic.max_tokens > kContext) {
      *error = "prefix and semantic budget exceed the 24576-token context";
      return false;
    }
    const VocabularyWindow window = PhaseWindow(Phase::kSemantic);
    if (cache && cache->length() + 2 == prefix.size() &&
        cache->capacity() >= prefix.size() + semantic.max_tokens + 1) {
      const std::array<std::uint32_t, 2> close{kAbcEnd, kMusicStart};
      backbone_->Append(*cache, close, window, &logits);
    } else {
      cache = backbone_->NewCache(prefix.size() + semantic.max_tokens + 1);
      backbone_->Append(*cache, prefix, window, &logits);
    }
    std::unique_ptr<hip::KvCache> negative;
    std::vector<float> negative_logits;
    if (guidance != 1.0F) {
      const std::vector<std::uint32_t> unconditional =
          NegativePrefix(*tokenizer_, prompt, abc_ids);
      negative =
          backbone_->NewCache(unconditional.size() + semantic.max_tokens);
      backbone_->Append(*negative, unconditional, window, &negative_logits);
    }
    Stage codes;
    if (!Sample(Phase::kSemantic, semantic, seed,
                options.planning == Planning::kOff, guidance, *cache,
                std::move(logits), negative.get(), std::move(negative_logits),
                is_cancelled, progress, &codes, error))
      return false;
    negative.reset();
    if (codes.tokens.empty()) {
      *error = "the model ended the song before any music";
      return false;
    }

    // 3. Acoustic flow matching over every original context chunk.
    const auto acoustic_start = std::chrono::steady_clock::now();
    const std::size_t frames = codes.tokens.size();
    std::vector<float> noise = TorchRandn(frames * kLatentDim, seed);
    // Upstream starts the solver from the BF16 noise.
    for (float& value : noise)
      value = RoundBfloat16(value);
    const auto ranges = ChunkRanges(frames, prefix.size());
    std::vector<float> latents;
    latents.reserve(noise.size());
    for (std::size_t chunk = 0; chunk < ranges.size(); ++chunk) {
      const auto [first, last] = ranges[chunk];
      std::unique_ptr<hip::KvCache> context;
      const std::uint32_t music_end = kMusicEnd;
      if (ranges.size() == 1) {
        // The semantic cache already holds the prefix and every codec token.
        backbone_->Append(*cache, {&music_end, 1}, {}, nullptr);
      } else {
        cache.reset();
        std::vector<std::uint32_t> tokens = prefix;
        tokens.insert(tokens.end(), codes.tokens.begin() + first,
                      codes.tokens.begin() + last);
        tokens.push_back(music_end);
        context = backbone_->NewCache(tokens.size());
        backbone_->Append(*context, tokens, {}, nullptr);
      }
      std::vector<float> part;
      const auto total_steps =
          static_cast<std::uint64_t>(steps) * ranges.size();
      const bool solved = backbone_->Synthesize(
          context ? *context : *cache,
          std::span<const float>(noise).subspan(first * kLatentDim,
                                                (last - first) * kLatentDim),
          steps,
          [&](int completed, int) {
            if (is_cancelled && is_cancelled())
              return false;
            return !progress ||
                   progress("acoustic", chunk * steps + completed, total_steps);
          },
          &part, error);
      if (!solved)
        return false;
      latents.insert(latents.end(), part.begin(), part.end());
    }
    cache.reset();
    const double acoustic_ms = Milliseconds(acoustic_start);

    // 4. Waveform.
    const auto decode_start = std::chrono::steady_clock::now();
    std::vector<float> audio;
    if (!vae_->Decode(
            latents,
            [&](int completed, int total) {
              if (is_cancelled && is_cancelled())
                return false;
              return !progress ||
                     progress("decode", static_cast<std::uint64_t>(completed),
                              static_cast<std::uint64_t>(total));
            },
            &audio, error))
      return false;
    for (float& sample : audio)
      sample = std::clamp(sample, -1.0F, 1.0F);
    const double decode_ms = Milliseconds(decode_start);

    result->audio.sample_rate = SampleRate();
    result->audio.channels = 2;
    result->audio.samples = std::move(audio);
    result->finish = plan.truncated || codes.truncated
                         ? ext::MusicFinish::kTruncated
                         : ext::MusicFinish::kComplete;
    result->total_ms = Milliseconds(start);
    result->telemetry_json =
        Telemetry(plan, codes, acoustic_ms, decode_ms, frames, abc_ids.size(),
                  prefix.size(), seed, result->total_ms);
    return true;
  }

  static std::string Telemetry(const Stage& plan, const Stage& codes,
                               double acoustic_ms, double decode_ms,
                               std::size_t frames, std::size_t score_tokens,
                               std::size_t prefix_tokens, std::uint64_t seed,
                               double total_ms) {
    json::Value value = json::Value::object();
    value["seed"] = static_cast<long long>(seed);
    value["plan_ms"] = plan.ms;
    value["plan_tokens"] = static_cast<long long>(plan.tokens.size());
    value["plan_truncated"] = plan.truncated;
    value["score_tokens"] = static_cast<long long>(score_tokens);
    value["prefix_tokens"] = static_cast<long long>(prefix_tokens);
    value["semantic_ms"] = codes.ms;
    value["semantic_tokens"] = static_cast<long long>(codes.tokens.size());
    value["semantic_truncated"] = codes.truncated;
    value["acoustic_ms"] = acoustic_ms;
    value["decode_ms"] = decode_ms;
    value["audio_seconds"] = static_cast<double>(frames) / kFrameRate;
    value["total_ms"] = total_ms;
    return value.dump();
  }

  ModelPaths paths_;
  GenerationDefaults defaults_;
  std::unique_ptr<TextTokenizer> tokenizer_;
  std::unique_ptr<hip::Backbone> backbone_;
  std::unique_ptr<hip::VaeDecoder> vae_;
};

std::unique_ptr<ext::MusicRunner> CreateYue2Runner(
    const ext::MusicRunnerOptions& options, std::string* error) {
  const std::string* vae =
      ext::FindOption<std::string>(options.options, "--vae-model");
  const auto paths = ResolveModelPaths(
      options.model_root,
      vae != nullptr ? std::filesystem::path(*vae) : std::filesystem::path(),
      error);
  if (!paths)
    return nullptr;
  BackboneConfig config;
  GenerationDefaults defaults;
  if (!LoadBackboneConfig(paths->model, &config, &defaults, error))
    return nullptr;
  auto tokenizer = TextTokenizer::Load(paths->model / "qwen.tiktoken", error);
  if (!tokenizer)
    return nullptr;
  auto vae_decoder = hip::VaeDecoder::Create(paths->vae, error);
  if (!vae_decoder)
    return nullptr;
  auto backbone = hip::Backbone::Create(paths->model, config, error);
  if (!backbone)
    return nullptr;
  return std::make_unique<Yue2Runner>(*paths, defaults, std::move(tokenizer),
                                      std::move(backbone),
                                      std::move(vae_decoder));
}

}  // namespace
#endif

}  // namespace gufo::models::yue2

void yue2_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(gufo::ext::generated::extension_yue2::kProvider_yue2);
#if defined(ENGINE_ENABLE_HIP)
  using namespace gufo::models::yue2;
  registry.AddMusicProvider({
      .id = "yue2",
      .probe = &ProbeYue2,
      .resolve = &ResolveYue2Artifacts,
      .create_with_options = &CreateYue2Runner,
  });
#endif
}
