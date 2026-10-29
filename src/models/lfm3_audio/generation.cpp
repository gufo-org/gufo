#include "src/models/lfm3_audio/generation.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <deque>
#include <future>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace gufo::models::lfm3_audio {
namespace {
using Clock = std::chrono::steady_clock;
double Ms(Clock::time_point since) {
  return std::chrono::duration<double, std::milli>(Clock::now() - since)
      .count();
}
std::string Lower(std::string_view text) {
  std::string out(text);
  for (auto& c : out)
    c = char(std::tolower(static_cast<unsigned char>(c)));
  return out;
}
// Bytes of a UTF-8 prefix that ends on a complete character.
std::size_t CompleteUtf8(std::string_view text) {
  std::size_t end = text.size();
  for (std::size_t back = 1; back <= std::min<std::size_t>(3, text.size());
       ++back) {
    const auto c = static_cast<unsigned char>(text[text.size() - back]);
    if ((c & 0xC0) == 0x80)
      continue;
    const std::size_t need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
    if (need > back)
      end = text.size() - back;
    break;
  }
  return end;
}
enum class Mode { kText, kAudio, kInterleaved };
}  // namespace
std::vector<PromptPiece> BuildPrompt(std::string_view system,
                                     std::span<const ChatMessage> messages,
                                     std::string_view assistant_prefix) {
  std::vector<PromptPiece> pieces;
  auto add_template = [&](std::string text) {
    if (!pieces.empty() && pieces.back().kind == PromptPiece::Kind::kTemplate)
      pieces.back().text += text;
    else
      pieces.push_back({PromptPiece::Kind::kTemplate, std::move(text), {}});
  };
  add_template("<|startoftext|>");
  if (!system.empty()) {
    add_template("<|im_start|>system\n");
    pieces.push_back({PromptPiece::Kind::kLiteral, std::string(system), {}});
    add_template("<|im_end|>\n");
  }
  for (const auto& m : messages) {
    add_template("<|im_start|>" + m.role + "\n");
    for (const auto& part : m.parts)
      if (part.audio)
        pieces.push_back({PromptPiece::Kind::kAudio, {}, part.samples});
      else if (!part.text.empty())
        pieces.push_back({PromptPiece::Kind::kLiteral, part.text, {}});
    add_template("<|im_end|>\n");
  }
  add_template("<|im_start|>assistant\n" + std::string(assistant_prefix));
  return pieces;
}
std::string AsrSystemPrompt(std::string_view language) {
  return language.empty() ? "Perform ASR."
                          : "Perform ASR in " + std::string(language) + ".";
}
bool TtsSystemPrompt(std::string_view language, std::string_view voice,
                     std::string* prompt, std::string* error) {
  static const std::unordered_map<std::string, std::string> names{
      {"ar", "Arabic"},
      {"da", "Danish"},
      {"de", "German"},
      {"el", "Greek"},
      {"en", "American English"},
      {"en-us", "American English"},
      {"en-gb", "British English"},
      {"fi", "Finnish"},
      {"fr", "French"},
      {"he", "Hebrew"},
      {"it", "Italian"},
      {"ja", "Japanese"},
      {"ko", "Korean"},
      {"nl", "Dutch"},
      {"pl", "Polish"},
      {"pt", "Portuguese"},
      {"es", "Spanish"},
      {"sv", "Swedish"},
      {"tr", "Turkish"},
      {"zh", "Chinese"}};
  const auto code = Lower(language.empty() ? "en" : language);
  std::string name;
  if (auto it = names.find(code); it != names.end())
    name = it->second;
  else
    for (const auto& [key, value] : names)
      if (Lower(value) == code || (code == "english" && key == "en"))
        name = value;
  if (name.empty()) {
    if (error)
      *error = "unsupported speech language: " + std::string(language);
    return false;
  }
  const auto v = Lower(voice.empty() ? "female" : voice);
  if (v != "female" && v != "male") {
    if (error)
      *error = "voice must be female or male";
    return false;
  }
  *prompt = "Perform TTS in " + name + " using a " + v + " voice.";
  return true;
}

struct Generator::Impl {
  std::shared_ptr<Model> model;
  std::unique_ptr<lfm3::Session> session;
  std::unique_ptr<Voice> voice;
  std::uint32_t context{};
  float* frame_embedding{};  // device row of backbone width
};
Generator::Generator() : impl_(std::make_unique<Impl>()) {}
Generator::~Generator() {
  if (impl_ && impl_->frame_embedding)
    (void)hipFree(impl_->frame_embedding);
}
Model& Generator::model() const {
  return *impl_->model;
}
std::unique_ptr<Generator> Generator::Create(std::shared_ptr<Model> model,
                                             std::uint32_t context,
                                             std::string* error) {
  auto g = std::unique_ptr<Generator>(new Generator);
  g->impl_->model = model;
  g->impl_->context = context;
  g->impl_->session = model->backbone().CreateSession(context, error);
  if (!g->impl_->session)
    return nullptr;
  g->impl_->voice = model->CreateVoice(error);
  if (!g->impl_->voice)
    return nullptr;
  if (hipMalloc(reinterpret_cast<void**>(&g->impl_->frame_embedding),
                std::size_t(model->backbone().config().hidden) * 4) !=
      hipSuccess) {
    if (error)
      *error = "cannot allocate frame embedding";
    return nullptr;
  }
  return g;
}
bool Generator::Generate(std::span<const PromptPiece> prompt,
                         const GenerateOptions& o,
                         const GenerateCallbacks& callbacks,
                         GenerateResult* result, std::string* error) {
  try {
    auto& impl = *impl_;
    auto& model = *impl.model;
    const auto& tok = model.tokenizer();
    const auto& ids = model.tokens();
    const auto& config = model.backbone().config();
    auto& session = *impl.session;
    auto& voice = *impl.voice;
    *result = {};
    const auto start = Clock::now();
    std::string failure;
    auto ok = [&](bool value) {
      if (!value)
        throw std::runtime_error(failure.empty() ? "LFM3-Audio failure"
                                                 : failure);
    };
    ok(session.Reset(&failure));
    voice.Reset();
    std::mt19937_64 rng(o.seed);

    // Prefill: runs of tokens and of audio embeddings, in order.
    std::vector<std::uint32_t> pending;
    std::uint32_t last_token = 0;
    auto flush = [&]() {
      if (pending.empty())
        return;
      if (session.Position() + pending.size() > impl.context)
        throw std::invalid_argument("LFM3-Audio prompt exceeds the context");
      ok(session.Forward(pending, {}, &failure));
      result->prompt_tokens += pending.size();
      pending.clear();
    };
    for (const auto& piece : prompt) {
      if (piece.kind == PromptPiece::Kind::kAudio) {
        const auto t = Clock::now();
        std::vector<float> embeddings;
        ok(model.Encode(piece.audio, &embeddings, &failure));
        result->timings.encode_ms += Ms(t);
        flush();
        const auto rows = embeddings.size() / config.hidden;
        if (session.Position() + rows > impl.context)
          throw std::invalid_argument("LFM3-Audio prompt exceeds the context");
        ok(session.ForwardEmbeddings(embeddings, false, {}, &failure));
        result->prompt_tokens += rows;
        result->audio_tokens += rows;
        last_token = ids.audio;
        continue;
      }
      std::vector<std::uint32_t> encoded;
      if (piece.kind == PromptPiece::Kind::kLiteral) {
        const std::pair<std::size_t, std::size_t> span{0, piece.text.size()};
        encoded = tok.EncodeFramed(piece.text, {&span, 1});
      } else
        encoded = tok.Encode(piece.text, true);
      for (auto id : encoded)
        if (id == ids.audio || id == ids.audio_out)
          throw std::invalid_argument(
              "prompt text cannot hold speech placeholders");
      pending.insert(pending.end(), encoded.begin(), encoded.end());
      if (!encoded.empty())
        last_token = encoded.back();
    }
    flush();
    if (!session.Position())
      throw std::invalid_argument("empty LFM3-Audio prompt");
    result->timings.prefill_ms = Ms(start);

    Mode mode = last_token == ids.audio_start || last_token == ids.audio_out
                    ? Mode::kAudio
                : last_token == ids.mixed_start ? Mode::kInterleaved
                                                : Mode::kText;
    std::vector<std::uint32_t> suppress{ids.audio, ids.audio_out};
    if (o.suppress_mixed)
      suppress.push_back(ids.mixed_start);
    if (o.suppress_audio_out)
      suppress.push_back(ids.audio_start);
    constexpr int kTextSlots = 6, kAudioSlots = 12;
    int slots_left = kTextSlots;
    bool speaking = false, text_done = false, finished = false;
    std::deque<std::uint32_t> text_history;
    std::vector<float> logits(config.vocab);
    std::vector<std::int32_t> unsent;  // frames not yet decoded
    std::string text_bytes;            // decoded but not yet emitted
    std::size_t emitted_text = 0;
    bool cancelled = false;
    auto check_cancel = [&]() {
      if (callbacks.cancelled && callbacks.cancelled())
        cancelled = true;
      return cancelled;
    };
    // Codec work runs on the voice's codec stream in a worker thread, so it
    // overlaps the next backbone step; audio is emitted on this thread.
    std::future<std::vector<float>> pending_audio;
    double decode_started = 0;
    auto join = [&]() {
      if (!pending_audio.valid())
        return;
      auto samples = pending_audio.get();
      result->timings.codec_ms += Ms(start) - decode_started;
      if (samples.empty())
        return;
      if (result->timings.first_audio_ms < 0)
        result->timings.first_audio_ms = Ms(start);
      result->samples.insert(result->samples.end(), samples.begin(),
                             samples.end());
      if (callbacks.on_audio && !callbacks.on_audio(samples))
        cancelled = true;
    };
    auto decode = [&](bool finalize) {
      join();
      decode_started = Ms(start);
      pending_audio = std::async(
          std::launch::async, [&voice, frames = std::move(unsent), finalize]() {
            std::vector<float> samples;
            std::string decode_error;
            if (!voice.Decode(frames, finalize, &samples, &decode_error))
              throw std::runtime_error(decode_error);
            return samples;
          });
      unsent.clear();
    };
    auto emit_text = [&](bool final) {
      const auto end = final ? text_bytes.size() : CompleteUtf8(text_bytes);
      if (end > emitted_text) {
        const std::string_view piece(text_bytes.data() + emitted_text,
                                     end - emitted_text);
        if (result->timings.first_text_ms < 0)
          result->timings.first_text_ms = Ms(start);
        if (callbacks.on_text && !callbacks.on_text(piece))
          cancelled = true;
        emitted_text = end;
      }
    };
    std::array<std::int32_t, AudioWeights::kCodebooks> codes;
    for (std::size_t step = 0; step < o.max_new_tokens && !finished; ++step) {
      if (check_cancel())
        break;
      const bool interleaving = mode == Mode::kInterleaved;
      const bool audio_slot =
          mode == Mode::kAudio || (interleaving && speaking);
      if (audio_slot) {
        auto t = Clock::now();
        ok(voice.SampleFrame(session.DeviceHidden(), o.audio, rng, codes,
                             impl.frame_embedding, false, &failure));
        result->timings.depth_ms += Ms(t);
        const bool eos = std::any_of(codes.begin(), codes.end(), [](auto c) {
          return c == std::int32_t(AudioWeights::kCodes);
        });
        if (eos) {
          // Segments of one reply form a single codec stream: flush, but
          // only finalize at the end of the reply.
          codes.fill(std::int32_t(AudioWeights::kCodes));
          mode = Mode::kText;
          if (!unsent.empty())
            decode(false);
        } else {
          result->codes.insert(result->codes.end(), codes.begin(), codes.end());
          unsent.insert(unsent.end(), codes.begin(), codes.end());
          ++result->frames;
          result->tokens.push_back(ids.audio_out);
          const auto chunk = result->frames <= o.first_chunk_frames
                                 ? o.first_chunk_frames
                                 : o.chunk_frames;
          if (unsent.size() >= chunk * AudioWeights::kCodebooks)
            decode(false);
        }
        if (check_cancel())
          break;
        if (step + 1 == o.max_new_tokens)
          break;
        t = Clock::now();
        if (session.Position() >= impl.context)
          break;
        ok(session.ForwardEmbeddings(
            {impl.frame_embedding, std::size_t(config.hidden)}, true, {},
            &failure));
        result->timings.backbone_ms += Ms(t);
        join();
      } else {
        auto t = Clock::now();
        ok(session.ReadLogits(logits, &failure));
        if (o.repetition_penalty != 1.0F)
          for (auto id : std::unordered_set<std::uint32_t>(text_history.begin(),
                                                           text_history.end()))
            logits[id] = logits[id] < 0 ? logits[id] * o.repetition_penalty
                                        : logits[id] / o.repetition_penalty;
        for (auto id : suppress)
          logits[id] = -INFINITY;
        std::uint32_t token;
        if (o.temperature <= 0) {
          token = std::uint32_t(std::max_element(logits.begin(), logits.end()) -
                                logits.begin());
        } else {
          std::vector<std::pair<float, std::uint32_t>> ranked;
          ranked.reserve(logits.size());
          for (std::uint32_t i = 0; i < logits.size(); ++i)
            if (std::isfinite(logits[i]))
              ranked.push_back({logits[i] / o.temperature, i});
          std::sort(ranked.begin(), ranked.end(), std::greater<>());
          if (o.top_k > 0 && ranked.size() > std::size_t(o.top_k))
            ranked.resize(o.top_k);
          double total = 0;
          std::vector<double> p(ranked.size());
          for (std::size_t i = 0; i < ranked.size(); ++i)
            total += p[i] = std::exp(ranked[i].first - ranked[0].first);
          if (o.top_p < 1.0F) {
            double kept = 0;
            std::size_t n = 0;
            while (n < ranked.size() && (n == 0 || kept / total < o.top_p))
              kept += p[n++];
            ranked.resize(n);
            p.resize(n);
            total = kept;
          }
          double target = std::uniform_real_distribution<double>(0, total)(rng);
          token = ranked.back().second;
          for (std::size_t i = 0; i < ranked.size(); ++i)
            if ((target -= p[i]) <= 0) {
              token = ranked[i].second;
              break;
            }
        }
        result->timings.backbone_ms += Ms(t);
        result->tokens.push_back(token);
        text_history.push_back(token);
        if (text_history.size() > o.repetition_window)
          text_history.pop_front();
        if (token == ids.text_end) {
          if (interleaving) {
            text_done = true;
            speaking = true;
          } else
            finished = true;
        } else if (token == ids.im_end || tok.IsStop(token)) {
          finished = true;
        } else if (!interleaving && token == ids.audio_start) {
          mode = Mode::kAudio;
        } else if (!interleaving && token == ids.mixed_start) {
          mode = Mode::kInterleaved;
          slots_left = kTextSlots;
          speaking = false;
          text_done = false;
        } else if (token != ids.audio_start && token != ids.mixed_start) {
          text_bytes += tok.Piece(token);
          emit_text(false);
        }
        if (finished || check_cancel())
          break;
        if (step + 1 == o.max_new_tokens || session.Position() >= impl.context)
          break;
        t = Clock::now();
        ok(session.Forward({&token, 1}, {}, &failure));
        result->timings.backbone_ms += Ms(t);
        join();
      }
      if (interleaving && !text_done) {
        if (--slots_left == 0) {
          speaking = !speaking;
          slots_left = speaking ? kAudioSlots : kTextSlots;
        }
      }
    }
    if (!cancelled && result->frames)
      decode(true);
    join();
    emit_text(true);
    result->text = text_bytes;
    result->cancelled = cancelled;
    result->timings.total_ms = Ms(start);
    return true;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return false;
  }
}
}  // namespace gufo::models::lfm3_audio
