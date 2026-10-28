#include "src/models/fish_drama3/hip/synthesis_runtime.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <future>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/models/fish_drama3/hip/backbone_runtime.hpp"
#include "src/models/fish_drama3/hip/codec_runtime.hpp"
#include "src/models/fish_drama3/hip/tower_runtime.hpp"
#include "src/models/fish_drama3/loader.hpp"
#include "src/models/fish_drama3/tokenizer.hpp"

namespace gufo::models::fish_drama3::hip {
namespace {

using Clock = std::chrono::steady_clock;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

double ElapsedMs(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

std::vector<float> ResampleLinearMono(std::span<const float> input,
                                      std::uint32_t in_rate,
                                      std::uint32_t out_rate) {
  if (in_rate == out_rate || input.empty()) {
    return std::vector<float>(input.begin(), input.end());
  }
  const double ratio =
      static_cast<double>(in_rate) / static_cast<double>(out_rate);
  const std::size_t out_len = static_cast<std::size_t>(
      std::ceil(static_cast<double>(input.size()) / ratio));
  std::vector<float> out(out_len);
  for (std::size_t i = 0; i < out_len; ++i) {
    const double src_pos = static_cast<double>(i) * ratio;
    const std::size_t idx0 = std::min<std::size_t>(
        static_cast<std::size_t>(src_pos), input.size() - 1U);
    const std::size_t idx1 =
        std::min<std::size_t>(idx0 + 1U, input.size() - 1U);
    const float frac = static_cast<float>(src_pos - static_cast<double>(idx0));
    out[i] = input[idx0] * (1.0F - frac) + input[idx1] * frac;
  }
  return out;
}

struct SamplerScratch {
  std::vector<std::int32_t> order;
  std::vector<double> weights;
  std::vector<double> probabilities;
};

std::int32_t SampleHostTopKTopP(const std::vector<float>& logits,
                                float temperature, int top_k, float top_p,
                                std::mt19937& rng, SamplerScratch& scratch) {
  if (logits.empty()) {
    throw std::runtime_error("Drama 3 sampler received empty logits");
  }
  if (temperature <= 0.0F || top_k <= 1) {
    return static_cast<std::int32_t>(std::distance(
        logits.begin(), std::max_element(logits.begin(), logits.end())));
  }
  auto& order = scratch.order;
  order.resize(logits.size());
  std::iota(order.begin(), order.end(), 0);
  const std::size_t count =
      std::min(logits.size(), static_cast<std::size_t>(std::max(1, top_k)));
  const auto compare = [&](std::int32_t left, std::int32_t right) {
    return logits[left] == logits[right] ? left < right
                                         : logits[left] > logits[right];
  };
  const double maximum =
      *std::max_element(logits.begin(), logits.end()) / temperature;
  double total = 0.0;
  auto& weights = scratch.weights;
  weights.resize(logits.size());
  for (std::size_t i = 0; i < logits.size(); ++i) {
    weights[i] =
        std::exp(static_cast<double>(logits[i]) / temperature - maximum);
    total += weights[i];
  }
  if (!(total > 0.0) || !std::isfinite(total)) {
    return 0;
  }
  std::size_t sorted = std::min<std::size_t>(count, 64U);
  for (;;) {
    std::partial_sort(order.begin(),
                      order.begin() + static_cast<std::ptrdiff_t>(sorted),
                      order.end(), compare);
    double mass = 0.0;
    for (std::size_t r = 0; r < sorted; ++r) {
      mass += weights[static_cast<std::size_t>(order[r])] / total;
    }
    if (sorted == count || mass > top_p) {
      break;
    }
    sorted = std::min<std::size_t>(count, sorted * 2U);
  }
  auto& probabilities = scratch.probabilities;
  probabilities.assign(count, 0.0);
  double cumulative = 0.0;
  for (std::size_t r = 0; r < sorted; ++r) {
    cumulative += weights[static_cast<std::size_t>(order[r])] / total;
    if (r > 0 && cumulative > top_p) {
      break;
    }
    probabilities[r] = weights[static_cast<std::size_t>(order[r])];
  }
  std::discrete_distribution<std::size_t> choice(probabilities.begin(),
                                                 probabilities.end());
  return order[choice(rng)];
}

bool SameTakes(const std::vector<Drama3Take>& left,
               const std::vector<Drama3Take>& right) {
  if (left.size() != right.size()) {
    return false;
  }
  for (std::size_t i = 0; i < left.size(); ++i) {
    if (left[i].text != right[i].text ||
        left[i].codes.codebooks != right[i].codes.codebooks ||
        left[i].codes.frames != right[i].codes.frames ||
        left[i].codes.codes != right[i].codes.codes) {
      return false;
    }
  }
  return true;
}

}  // namespace

struct SynthesisHipRuntime::Impl {
  ModelConfig config;
  std::size_t max_context_frames{4096};
  std::unique_ptr<Drama3Tokenizer> tokenizer;
  std::unique_ptr<BackboneHipRuntime> backbone;
  std::unique_ptr<TowerHipRuntime> tower;
  std::unique_ptr<FishDacCodecHipRuntime> codec;

  // Cached single reference audio to avoid re-encoding identical voice prompts
  std::vector<float> cached_ref_samples;
  std::uint32_t cached_ref_rate{0};
  FishDacCodes cached_ref_codes;

  // Cached prompt prefix state (system + references + history) across requests
  bool has_cached_prefix{false};
  std::vector<Drama3Take> cached_prefix_refs;
  std::vector<Drama3Take> cached_prefix_history;
  std::size_t cached_prefix_len{0};
  Drama3Frame cached_prefix_last_frame{};
  std::vector<float> cached_prefix_last_hidden;

  Impl(const std::string& model_dir, std::size_t max_ctx)
      : max_context_frames(max_ctx) {
    auto parsed = LoadModelConfigFromPath(model_dir);
    if (!parsed.has_value() || !IsSupportedModelConfig(*parsed)) {
      throw std::runtime_error(
          "Unsupported or missing Fish Drama 3 config in " + model_dir);
    }
    config = std::move(*parsed);
    tokenizer = std::make_unique<Drama3Tokenizer>(model_dir);

    std::string err;
    codec = FishDacCodecHipRuntime::Create(model_dir, config.codec, &err);
    if (!codec) {
      throw std::runtime_error("Failed to create Fish DAC codec runtime: " +
                               err);
    }

    // Scope `safetensors` so its mmap pages are released as soon as weights
    // are uploaded to coarse-grained HIP device buffers.
    {
      auto safetensors = LoadSafetensors(model_dir);
      if (!safetensors) {
        throw std::runtime_error(
            "Failed to load Fish Drama 3 model safetensors from " + model_dir);
      }
      backbone = BackboneHipRuntime::Create(*safetensors, config,
                                            max_context_frames, &err);
      if (!backbone) {
        throw std::runtime_error("Failed to create Drama 3 backbone runtime: " +
                                 err);
      }
      tower = TowerHipRuntime::Create(*safetensors, config.tower,
                                      max_context_frames, &err);
      if (!tower) {
        throw std::runtime_error("Failed to create Drama 3 tower runtime: " +
                                 err);
      }
    }
  }
};

SynthesisHipRuntime::SynthesisHipRuntime(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

SynthesisHipRuntime::~SynthesisHipRuntime() = default;

std::unique_ptr<SynthesisHipRuntime> SynthesisHipRuntime::Create(
    const std::string& model_dir, std::size_t max_context_frames,
    std::string* error) {
  try {
    return std::unique_ptr<SynthesisHipRuntime>(new SynthesisHipRuntime(
        std::make_unique<Impl>(model_dir, max_context_frames)));
  } catch (const std::exception& ex) {
    SetError(error, ex.what());
    return nullptr;
  }
}

bool SynthesisHipRuntime::EncodeReference(std::span<const float> mono_samples,
                                          std::uint32_t sample_rate,
                                          FishDacCodes* out_codes,
                                          std::string* error) {
  if (out_codes == nullptr || mono_samples.empty()) {
    SetError(error, "Reference audio is empty");
    return false;
  }
  if (impl_->cached_ref_rate == sample_rate &&
      impl_->cached_ref_samples.size() == mono_samples.size() &&
      std::equal(mono_samples.begin(), mono_samples.end(),
                 impl_->cached_ref_samples.begin())) {
    *out_codes = impl_->cached_ref_codes;
    return true;
  }
  const auto resampled = ResampleLinearMono(mono_samples, sample_rate,
                                            impl_->config.codec.sample_rate);
  FishDacCodes codes;
  if (!impl_->codec->Encode(resampled, &codes, error)) {
    return false;
  }
  if (mono_samples.size() <= 44100U * 30U) {
    impl_->cached_ref_samples.assign(mono_samples.begin(), mono_samples.end());
    impl_->cached_ref_rate = sample_rate;
    impl_->cached_ref_codes = codes;
  }
  *out_codes = std::move(codes);
  return true;
}

bool SynthesisHipRuntime::Generate(const SynthesisRequest& request,
                                   const CancellationCheck& is_cancelled,
                                   SynthesisResult* result,
                                   std::string* error) {
  if (result == nullptr) {
    SetError(error, "SynthesisResult output pointer is null");
    return false;
  }
  *result = {};
  result->sample_rate = impl_->config.codec.sample_rate;
  result->code_groups = impl_->config.codec.total_codebooks;

  const auto total_start = Clock::now();
  try {
    std::vector<Drama3Take> refs = request.references;
    if (!request.reference_audio_mono.empty()) {
      if (request.reference_text.empty()) {
        SetError(error, "Drama 3 reference audio requires reference_text");
        return false;
      }
      FishDacCodes ref_codes;
      if (!EncodeReference(request.reference_audio_mono,
                           request.reference_sample_rate, &ref_codes, error)) {
        return false;
      }
      refs.push_back(Drama3Take{request.reference_text, std::move(ref_codes)});
    }

    const auto prefix_frames = BuildDrama3PromptPrefix(
        impl_->config, *impl_->tokenizer, refs, request.history);
    const auto suffix_frames =
        BuildDrama3PromptSuffix(*impl_->tokenizer, request.text);
    const std::size_t max_new = request.sampling.max_new_frames;
    if (prefix_frames.size() + suffix_frames.size() + max_new >
        impl_->max_context_frames) {
      SetError(error, "Drama 3 prompt + max_new_frames exceeds context limit");
      return false;
    }

    impl_->codec->ResetStream();
    const std::size_t dim = impl_->config.backbone.hidden_size;

    const bool prefix_hit =
        !prefix_frames.empty() && impl_->has_cached_prefix &&
        impl_->cached_prefix_len == prefix_frames.size() &&
        SameTakes(impl_->cached_prefix_refs, refs) &&
        SameTakes(impl_->cached_prefix_history, request.history);

    if (prefix_hit) {
      const auto restore_start = Clock::now();
      if (!impl_->backbone->RestorePrefixState(error) ||
          !impl_->tower->RestorePrefixState(error)) {
        impl_->has_cached_prefix = false;
        return false;
      }
      result->backbone_prefill_ms += ElapsedMs(restore_start);
    } else {
      impl_->has_cached_prefix = false;
      impl_->backbone->Reset();
      impl_->tower->Reset();

      std::vector<std::int32_t> prefix_ids(prefix_frames.size());
      for (std::size_t i = 0; i < prefix_frames.size(); ++i) {
        prefix_ids[i] = prefix_frames[i][0];
      }
      auto prefix_emb = impl_->backbone->TokenEmbedding(prefix_ids);
      impl_->tower->MixBackboneEmbeddings(&prefix_emb, prefix_frames);

      const auto bb_prefix_start = Clock::now();
      std::vector<float> prefix_hidden;
      if (!impl_->backbone->ForwardEmbeddings(prefix_emb, prefix_frames.size(),
                                              &prefix_hidden, error, nullptr)) {
        return false;
      }
      result->backbone_prefill_ms += ElapsedMs(bb_prefix_start);

      if (is_cancelled && is_cancelled()) {
        SetError(error, "Cancelled");
        return false;
      }

      const auto tw_prefix_start = Clock::now();
      if (!impl_->tower->Prefill(prefix_hidden, prefix_frames, error, 0U)) {
        return false;
      }
      result->tower_prefill_ms += ElapsedMs(tw_prefix_start);

      if (!impl_->backbone->SavePrefixState(error) ||
          !impl_->tower->SavePrefixState(prefix_frames.size() - 1U, error)) {
        return false;
      }
      impl_->cached_prefix_refs = refs;
      impl_->cached_prefix_history = request.history;
      impl_->cached_prefix_len = prefix_frames.size();
      impl_->cached_prefix_last_frame = prefix_frames.back();
      impl_->cached_prefix_last_hidden.assign(
          prefix_hidden.end() - static_cast<std::ptrdiff_t>(dim),
          prefix_hidden.end());
      impl_->has_cached_prefix = true;
    }

    // Incremental Suffix Prefill (only user turn + assistant header tokens)
    std::vector<std::int32_t> suffix_ids(suffix_frames.size());
    for (std::size_t i = 0; i < suffix_frames.size(); ++i) {
      suffix_ids[i] = suffix_frames[i][0];
    }
    auto suffix_emb = impl_->backbone->TokenEmbedding(suffix_ids);
    impl_->tower->MixBackboneEmbeddings(&suffix_emb, suffix_frames);

    const auto bb_suffix_start = Clock::now();
    std::vector<float> suffix_hidden;
    std::vector<float> logits;
    if (!impl_->backbone->ForwardEmbeddings(suffix_emb, suffix_frames.size(),
                                            &suffix_hidden, error, &logits)) {
      return false;
    }
    result->backbone_prefill_ms += ElapsedMs(bb_suffix_start);

    if (is_cancelled && is_cancelled()) {
      SetError(error, "Cancelled");
      return false;
    }

    const auto tw_suffix_start = Clock::now();
    std::vector<Drama3Frame> tower_suffix_frames;
    tower_suffix_frames.reserve(1U + suffix_frames.size());
    tower_suffix_frames.push_back(impl_->cached_prefix_last_frame);
    tower_suffix_frames.insert(tower_suffix_frames.end(), suffix_frames.begin(),
                               suffix_frames.end());

    std::vector<float> tower_suffix_hidden;
    tower_suffix_hidden.reserve(tower_suffix_frames.size() * dim);
    tower_suffix_hidden.insert(tower_suffix_hidden.end(),
                               impl_->cached_prefix_last_hidden.begin(),
                               impl_->cached_prefix_last_hidden.end());
    tower_suffix_hidden.insert(tower_suffix_hidden.end(), suffix_hidden.begin(),
                               suffix_hidden.end());

    if (!impl_->tower->Prefill(tower_suffix_hidden, tower_suffix_frames, error,
                               impl_->cached_prefix_len - 1U)) {
      return false;
    }
    result->tower_prefill_ms += ElapsedMs(tw_suffix_start);

    std::vector<float> cur_hidden(
        suffix_hidden.end() - static_cast<std::ptrdiff_t>(dim),
        suffix_hidden.end());
    Drama3Frame cur_frame = suffix_frames.back();

    std::mt19937 rng(request.sampling.seed);
    SamplerScratch sampler_scratch;
    std::deque<std::int32_t> recent_tokens;
    std::vector<Drama3Frame> output_frames;
    output_frames.reserve(max_new);

    const bool streaming = static_cast<bool>(request.on_audio);
    // For non-streaming, decode in 24-frame causal chunks in the background on
    // Codec's independent HIP stream so Codec execution overlaps with
    // Backbone+Tower decode!
    const std::size_t chunk_frames =
        streaming
            ? std::max<std::size_t>(1U, request.sampling.stream_chunk_frames)
            : (max_new > 16U ? max_new - 12U : max_new);
    std::vector<Drama3Frame> pending_stream_frames;
    pending_stream_frames.reserve(chunk_frames);

    struct AsyncCodecResult {
      bool ok{false};
      std::vector<float> pcm;
      std::string err;
    };
    std::future<AsyncCodecResult> pending_codec_future;

    auto wait_pending_codec = [&]() -> bool {
      if (!pending_codec_future.valid()) {
        return true;
      }
      const auto wait_start = Clock::now();
      AsyncCodecResult r = pending_codec_future.get();
      result->codec_decode_ms += ElapsedMs(wait_start);
      if (!r.ok) {
        SetError(error, r.err);
        return false;
      }
      result->sample_count += r.pcm.size();
      if (streaming) {
        if (!request.on_audio(r.pcm)) {
          SetError(error, "Cancelled by audio callback");
          return false;
        }
      } else {
        result->samples.insert(result->samples.end(), r.pcm.begin(),
                               r.pcm.end());
      }
      return true;
    };

    auto launch_codec_chunk_async = [&]() -> bool {
      if (pending_stream_frames.empty()) {
        return true;
      }
      if (!wait_pending_codec()) {
        return false;
      }
      FishDacCodes chunk_codes;
      chunk_codes.codebooks = 10;
      chunk_codes.frames =
          static_cast<std::int64_t>(pending_stream_frames.size());
      chunk_codes.codes.resize(pending_stream_frames.size() * 10U);
      for (std::size_t b = 0; b < 10U; ++b) {
        for (std::size_t f = 0; f < pending_stream_frames.size(); ++f) {
          chunk_codes.codes[b * pending_stream_frames.size() + f] =
              pending_stream_frames[f][b + 1U];
        }
      }
      pending_stream_frames.clear();

      FishDacCodecHipRuntime* codec_ptr = impl_->codec.get();
      pending_codec_future = std::async(
          std::launch::async,
          [codec_ptr, codes = std::move(chunk_codes)]() -> AsyncCodecResult {
            AsyncCodecResult res;
            res.ok = codec_ptr->DecodeStreamChunk(codes, &res.pcm, &res.err);
            return res;
          });
      return true;
    };

    // 4. Autoregressive Decode Loop
    for (std::size_t step = 0; step < max_new; ++step) {
      if (is_cancelled && is_cancelled()) {
        (void)wait_pending_codec();
        SetError(error, "Cancelled");
        return false;
      }

      // Repetition Aware Sampling (RAS) using `logits` already computed by
      // fused ForwardEmbeddings
      const std::int32_t first = SampleHostTopKTopP(
          logits, request.sampling.temperature, request.sampling.top_k,
          request.sampling.top_p, rng, sampler_scratch);
      const std::int32_t second =
          SampleHostTopKTopP(logits, 1.0F, 30, 0.95F, rng, sampler_scratch);
      const std::int32_t first_token =
          first == 0 ? impl_->config.eos_token_id
                     : (impl_->config.tower.semantic_begin + first - 1);
      const bool repeats =
          first != 0 && std::find(recent_tokens.begin(), recent_tokens.end(),
                                  first_token) != recent_tokens.end();
      const std::int32_t selected = repeats ? second : first;
      if (selected == 0) {
        break;  // EOS reached
      }

      const std::int32_t semantic_code = selected - 1;
      const std::size_t position = impl_->backbone->valid_steps() - 1U;
      std::array<float, 9> uniforms{};
      if (request.sampling.temperature > 0.0F && request.sampling.top_k > 1) {
        for (float& u : uniforms) {
          u = static_cast<float>(std::generate_canonical<double, 53>(rng));
        }
      }

      const auto tw_step_start = Clock::now();
      Drama3Frame next_frame{};
      if (!impl_->tower->RunFrame(
              cur_hidden, cur_frame, semantic_code, position,
              request.sampling.temperature, request.sampling.top_k,
              request.sampling.top_p, uniforms, &next_frame, error)) {
        (void)wait_pending_codec();
        return false;
      }
      result->tower_decode_ms += ElapsedMs(tw_step_start);

      output_frames.push_back(next_frame);
      recent_tokens.push_back(next_frame[0]);
      if (recent_tokens.size() > 16U) {
        recent_tokens.pop_front();
      }

      pending_stream_frames.push_back(next_frame);

      if (step + 1U == max_new) {
        result->reached_limit = true;
        break;
      }

      // In streaming mode, launch async Codec decoding right before Backbone
      // ForwardEmbeddings so Codec chunk decoding and Backbone run concurrently
      // on separate HIP streams!
      if (streaming && pending_stream_frames.size() >= chunk_frames) {
        if (!launch_codec_chunk_async()) {
          return false;
        }
      }

      // Advance Backbone by 1 step with the newly sampled frame (fusing LM Head
      // logits!)
      cur_frame = next_frame;
      const auto next_bb_start = Clock::now();
      const std::array<std::int32_t, 1> next_id{cur_frame[0]};
      auto next_emb = impl_->backbone->TokenEmbedding(next_id);
      const std::array<Drama3Frame, 1> next_frame_span{cur_frame};
      impl_->tower->MixBackboneEmbeddings(&next_emb, next_frame_span);
      if (!impl_->backbone->ForwardEmbeddings(next_emb, 1U, &cur_hidden, error,
                                              &logits)) {
        (void)wait_pending_codec();
        return false;
      }
      result->backbone_decode_ms += ElapsedMs(next_bb_start);
    }

    if (output_frames.empty()) {
      (void)wait_pending_codec();
      SetError(error, "Drama 3 ended before producing any audio frames");
      return false;
    }

    if (streaming) {
      if (!pending_stream_frames.empty()) {
        if (!launch_codec_chunk_async()) {
          return false;
        }
      }
      if (!wait_pending_codec()) {
        return false;
      }
    }

    result->codes.codebooks = 10;
    result->codes.frames = static_cast<std::int64_t>(output_frames.size());
    result->codes.codes.resize(output_frames.size() * 10U);
    for (std::size_t b = 0; b < 10U; ++b) {
      for (std::size_t f = 0; f < output_frames.size(); ++f) {
        result->codes.codes[b * output_frames.size() + f] =
            output_frames[f][b + 1U];
      }
    }

    if (!streaming) {
      const auto codec_start = Clock::now();
      if (!impl_->codec->Decode(result->codes, &result->samples, error)) {
        return false;
      }
      result->codec_decode_ms = ElapsedMs(codec_start);
      result->sample_count = result->samples.size();
    }

    result->total_ms = ElapsedMs(total_start);
    return true;
  } catch (const std::exception& ex) {
    SetError(error, ex.what());
    return false;
  }
}

const ModelConfig& SynthesisHipRuntime::config() const noexcept {
  return impl_->config;
}

}  // namespace gufo::models::fish_drama3::hip
