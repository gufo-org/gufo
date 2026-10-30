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

#include "src/core/json.hpp"
#include "src/models/qwen/chat_template.hpp"

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
class PieceWriter {
public:
  explicit PieceWriter(std::vector<PromptPiece>* pieces) : pieces_(pieces) {}
  void Template(std::string_view text) {
    if (text.empty())
      return;
    if (!pieces_->empty() &&
        pieces_->back().kind == PromptPiece::Kind::kTemplate)
      pieces_->back().text += text;
    else
      pieces_->push_back({PromptPiece::Kind::kTemplate, std::string(text), {}});
  }
  void Literal(std::string_view text) {
    if (!text.empty())
      pieces_->push_back({PromptPiece::Kind::kLiteral, std::string(text), {}});
  }
  void Audio(const std::vector<float>& samples) {
    pieces_->push_back({PromptPiece::Kind::kAudio, {}, samples});
  }

private:
  std::vector<PromptPiece>* pieces_;
};
}  // namespace
std::string ToolDefinition(std::string_view tool_json) {
  auto tool = json::parse(tool_json);
  if (!tool.is_object())
    throw std::invalid_argument("tool definition must be an object");
  const auto* fn = tool.find("function");
  const auto& source = fn ? *fn : tool;
  if (!source.is_object() || source.member_str("name").empty())
    throw std::invalid_argument("tool definition requires a function name");
  auto filtered = json::Value::object();
  for (const auto& [key, value] : source.members())
    if (key != "defer_loading")
      filtered[key] = value;
  return filtered.tojson();
}
std::vector<PromptPiece> SystemPieces(std::string_view system,
                                      std::span<const std::string> tools) {
  std::vector<PromptPiece> pieces;
  PieceWriter out(&pieces);
  out.Template("<|startoftext|>");
  if (system.empty() && tools.empty())
    return pieces;
  out.Template("<|im_start|>system\n");
  out.Literal(system);
  if (!tools.empty()) {
    out.Template(system.empty() ? "<tools>\n" : "\n\n<tools>\n");
    for (std::size_t i = 0; i < tools.size(); ++i) {
      if (i)
        out.Template("\n");
      out.Literal(tools[i]);
    }
    out.Template("\n</tools>");
  }
  out.Template("<|im_end|>\n");
  return pieces;
}
std::vector<PromptPiece> MessagePieces(const ChatMessage& m) {
  std::vector<PromptPiece> pieces;
  PieceWriter out(&pieces);
  out.Template("<|im_start|>" + m.role + "\n");
  const bool tool = m.role == "tool";
  if (tool)
    out.Template("<tool_response>\n");
  for (const auto& part : m.parts)
    if (part.audio)
      out.Audio(part.samples);
    else
      out.Literal(part.text);
  if (tool)
    out.Template("\n</tool_response>");
  out.Template("<|im_end|>\n");
  return pieces;
}
std::vector<PromptPiece> BuildPrompt(std::string_view system,
                                     std::span<const ChatMessage> messages,
                                     std::string_view assistant_prefix,
                                     std::span<const std::string> tools) {
  auto pieces = SystemPieces(system, tools);
  PieceWriter out(&pieces);
  for (const auto& m : messages)
    for (auto& piece : MessagePieces(m))
      if (piece.kind == PromptPiece::Kind::kTemplate)
        out.Template(piece.text);
      else
        pieces.push_back(std::move(piece));
  out.Template("<|im_start|>assistant\n" + std::string(assistant_prefix));
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
  // Inputs of the context in order, so Erase can recompute it: runs of
  // tokens or of embedding rows.
  struct Run {
    std::vector<std::uint32_t> tokens;
    std::vector<float> rows;
  };
  std::vector<Run> log;
  std::uint32_t last_token{};
  std::uint32_t Hidden() const { return model->backbone().config().hidden; }
  void Fail(bool ok, const std::string& failure) const {
    if (!ok)
      throw std::runtime_error(failure.empty() ? "LFM3-Audio failure"
                                               : failure);
  }
  void Record(std::span<const std::uint32_t> tokens) {
    if (log.empty() || !log.back().rows.empty())
      log.emplace_back();
    log.back().tokens.insert(log.back().tokens.end(), tokens.begin(),
                             tokens.end());
  }
  void Record(std::span<const float> rows) {
    if (log.empty() || !log.back().tokens.empty())
      log.emplace_back();
    log.back().rows.insert(log.back().rows.end(), rows.begin(), rows.end());
  }
  std::vector<Run> Prepare(std::span<const PromptPiece> pieces,
                           GenerateResult* result);
  void Feed(const std::vector<Run>& runs);
  void Room(std::size_t rows) const {
    if (session->Position() + rows > context)
      throw std::invalid_argument("LFM3-Audio prompt exceeds the context");
  }
  void Tokens(std::span<const std::uint32_t> tokens) {
    if (tokens.empty())
      return;
    Room(tokens.size());
    std::string failure;
    Fail(session->Forward(tokens, {}, &failure), failure);
    Record(tokens);
    last_token = tokens.back();
  }
  void Rows(std::span<const float> rows, std::uint32_t token) {
    if (rows.empty())
      return;
    Room(rows.size() / Hidden());
    std::string failure;
    Fail(session->ForwardEmbeddings(rows, false, {}, &failure), failure);
    Record(rows);
    last_token = token;
  }
};
Generator::Generator() : impl_(std::make_unique<Impl>()) {}
Generator::~Generator() {
  if (impl_ && impl_->frame_embedding)
    (void)hipFree(impl_->frame_embedding);
}
Model& Generator::model() const {
  return *impl_->model;
}
std::uint32_t Generator::Position() const {
  return impl_->session->Position();
}
std::uint32_t Generator::Context() const {
  return impl_->context;
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
bool Generator::Reset(std::string* error) {
  impl_->log.clear();
  impl_->last_token = 0;
  return impl_->session->Reset(error);
}
std::vector<Generator::Impl::Run> Generator::Impl::Prepare(
    std::span<const PromptPiece> prompt, GenerateResult* result) {
  const auto& tok = model->tokenizer();
  const auto& ids = model->tokens();
  std::vector<Run> runs;
  // Adjacent text pieces encode as one string, client text as literal
  // spans, so piece boundaries do not split BPE merges.
  std::string text;
  std::vector<std::pair<std::size_t, std::size_t>> literals;
  auto flush = [&]() {
    if (text.empty())
      return;
    const auto encoded = tok.EncodeFramed(text, literals);
    for (auto id : encoded)
      if (id == ids.audio || id == ids.audio_out)
        throw std::invalid_argument(
            "prompt text cannot hold speech placeholders");
    if (runs.empty() || !runs.back().rows.empty())
      runs.emplace_back();
    runs.back().tokens.insert(runs.back().tokens.end(), encoded.begin(),
                              encoded.end());
    if (result)
      result->prompt_tokens += encoded.size();
    text.clear();
    literals.clear();
  };
  for (const auto& piece : prompt) {
    if (piece.kind == PromptPiece::Kind::kAudio) {
      const auto t = Clock::now();
      std::vector<float> embeddings;
      std::string failure;
      Fail(model->Encode(piece.audio, &embeddings, &failure), failure);
      if (result)
        result->timings.encode_ms += Ms(t);
      flush();
      if (runs.empty() || !runs.back().tokens.empty())
        runs.emplace_back();
      runs.back().rows.insert(runs.back().rows.end(), embeddings.begin(),
                              embeddings.end());
      if (result) {
        result->prompt_tokens += embeddings.size() / Hidden();
        result->audio_tokens += embeddings.size() / Hidden();
      }
      continue;
    }
    if (piece.kind == PromptPiece::Kind::kLiteral && !piece.text.empty())
      literals.emplace_back(text.size(), piece.text.size());
    text += piece.text;
  }
  flush();
  return runs;
}
void Generator::Impl::Feed(const std::vector<Run>& runs) {
  std::size_t rows = 0;
  for (const auto& run : runs)
    rows += run.tokens.size() + run.rows.size() / Hidden();
  Room(rows);
  for (const auto& run : runs)
    if (run.rows.empty())
      Tokens(run.tokens);
    else
      Rows(run.rows, model->tokens().audio);
}
bool Generator::Append(std::span<const PromptPiece> prompt,
                       GenerateResult* result, std::string* error) {
  try {
    impl_->Feed(impl_->Prepare(prompt, result));
    return true;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return false;
  }
}
bool Generator::AppendEmbeddings(std::span<const float> rows,
                                 std::string* error) {
  try {
    if (rows.size() % impl_->Hidden())
      throw std::invalid_argument("embedding rows must have backbone width");
    impl_->Rows(rows, impl_->model->tokens().audio);
    return true;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return false;
  }
}
bool Generator::Replace(std::uint32_t begin, std::uint32_t end,
                        std::span<const PromptPiece> pieces,
                        std::string* error) {
  try {
    auto& impl = *impl_;
    const auto H = impl.Hidden();
    if (begin > end || end > impl.session->Position())
      throw std::invalid_argument("LFM3-Audio replace outside the context");
    auto inserted = impl.Prepare(pieces, nullptr);
    // Rows [0, begin), the replacement, then rows [end, position).
    std::vector<Impl::Run> kept;
    auto add = [&](const Impl::Run& run, std::size_t i) {
      const bool rows = !run.rows.empty();
      if (kept.empty() || kept.back().rows.empty() == rows)
        kept.emplace_back();
      if (rows)
        kept.back().rows.insert(kept.back().rows.end(),
                                run.rows.begin() + std::ptrdiff_t(i) * H,
                                run.rows.begin() + std::ptrdiff_t(i + 1) * H);
      else
        kept.back().tokens.push_back(run.tokens[i]);
    };
    std::uint32_t at = 0;
    bool spliced = false;
    auto splice = [&]() {
      for (const auto& run : inserted) {
        const auto n =
            run.rows.empty() ? run.tokens.size() : run.rows.size() / H;
        for (std::size_t i = 0; i < n; ++i)
          add(run, i);
      }
      spliced = true;
    };
    for (const auto& run : impl.log) {
      const auto n = std::uint32_t(run.rows.empty() ? run.tokens.size()
                                                    : run.rows.size() / H);
      for (std::uint32_t i = 0; i < n; ++i, ++at) {
        if (at == begin && !spliced)
          splice();
        if (at < begin || at >= end)
          add(run, i);
      }
    }
    if (!spliced)
      splice();
    const auto last = impl.last_token;
    const bool tail = end == impl.session->Position();
    std::string failure;
    impl.Fail(impl.session->Reset(&failure), failure);
    impl.log.clear();
    impl.Feed(kept);
    // The context still ends in the same input unless its end was replaced.
    if (!tail)
      impl.last_token = last;
    return true;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return false;
  }
}
bool Generator::Generate(std::span<const PromptPiece> prompt,
                         const GenerateOptions& o,
                         const GenerateCallbacks& callbacks,
                         GenerateResult* result, std::string* error) {
  *result = {};
  const auto start = Clock::now();
  if (!Reset(error) || !Append(prompt, result, error))
    return false;
  if (!impl_->session->Position()) {
    if (error)
      *error = "empty LFM3-Audio prompt";
    return false;
  }
  const auto prefill_ms = Ms(start);
  GenerateResult reply;
  if (!Continue(o, callbacks, &reply, error))
    return false;
  reply.prompt_tokens = result->prompt_tokens;
  reply.audio_tokens = result->audio_tokens;
  reply.timings.encode_ms = result->timings.encode_ms;
  reply.timings.prefill_ms = prefill_ms;
  // Reply times count from the start of the prompt.
  for (auto* t : {&reply.timings.first_text_ms, &reply.timings.first_audio_ms})
    if (*t >= 0)
      *t += prefill_ms;
  reply.timings.total_ms += prefill_ms;
  *result = std::move(reply);
  return true;
}
bool Generator::Continue(const GenerateOptions& o,
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
    auto ok = [&](bool value) { impl.Fail(value, failure); };
    if (!session.Position())
      throw std::invalid_argument("LFM3-Audio reply needs a prompt");
    voice.Reset();
    std::mt19937_64 rng(o.seed);
    const auto last_token = impl.last_token;
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
    std::vector<float> frame_row(config.hidden);
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
        if (!eos)
          result->frame_positions.push_back(session.Position());
        ok(session.ForwardEmbeddings(
            {impl.frame_embedding, std::size_t(config.hidden)}, true, {},
            &failure));
        // The frame embedding stays recorded for Replace.
        if (hipMemcpyAsync(frame_row.data(), impl.frame_embedding,
                           frame_row.size() * 4, hipMemcpyDeviceToHost,
                           static_cast<hipStream_t>(session.DeviceStream())) !=
                hipSuccess ||
            hipStreamSynchronize(
                static_cast<hipStream_t>(session.DeviceStream())) != hipSuccess)
          throw std::runtime_error("cannot read the frame embedding");
        impl.Record(frame_row);
        impl.last_token = ids.audio_out;
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
        if (result->tokens.empty() && !o.first_tokens.empty()) {
          std::vector<float> allowed(o.first_tokens.size());
          for (std::size_t i = 0; i < allowed.size(); ++i)
            allowed[i] = logits.at(o.first_tokens[i]);
          std::fill(logits.begin(), logits.end(), -INFINITY);
          for (std::size_t i = 0; i < allowed.size(); ++i)
            logits[o.first_tokens[i]] = allowed[i];
        }
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
            finished = result->stopped = true;
        } else if (token == ids.im_end || tok.IsStop(token)) {
          finished = result->stopped = true;
        } else if (!interleaving && token == ids.audio_start) {
          mode = Mode::kAudio;
        } else if (!interleaving && token == ids.mixed_start) {
          mode = Mode::kInterleaved;
          slots_left = kTextSlots;
          speaking = false;
          text_done = false;
        } else if (!tok.IsControl(token)) {
          text_bytes += tok.Piece(token);
          emit_text(false);
        }
        if (finished || check_cancel())
          break;
        if (step + 1 == o.max_new_tokens || session.Position() >= impl.context)
          break;
        t = Clock::now();
        ok(session.Forward({&token, 1}, {}, &failure));
        impl.Record(std::span<const std::uint32_t>(&token, 1));
        impl.last_token = token;
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
struct SpeechBatch::Impl {
  struct Stream {
    std::unique_ptr<Generator> generator;
    bool active{};
    GenerateOptions options;
    GenerateCallbacks callbacks;
    Done done;
    GenerateResult result;
    std::mt19937_64 rng;
    std::vector<std::int32_t> unsent;
    std::future<std::vector<float>> pending;
    Clock::time_point start;
    double decode_started{};
    std::size_t steps{};
    bool cancelled{};
    std::array<std::int32_t, AudioWeights::kCodebooks> codes{};
    Generator::Impl& g() { return *generator->impl_; }
  };
  std::shared_ptr<Model> model;
  std::vector<Stream> streams;
  // Joins the stream's codec work and emits its audio.
  void Join(Stream& st) {
    if (!st.pending.valid())
      return;
    auto samples = st.pending.get();
    st.result.timings.codec_ms += Ms(st.start) - st.decode_started;
    if (samples.empty())
      return;
    if (st.result.timings.first_audio_ms < 0)
      st.result.timings.first_audio_ms = Ms(st.start);
    st.result.samples.insert(st.result.samples.end(), samples.begin(),
                             samples.end());
    if (st.callbacks.on_audio && !st.callbacks.on_audio(samples))
      st.cancelled = true;
  }
  void Decode(Stream& st, bool finalize) {
    Join(st);
    st.decode_started = Ms(st.start);
    st.pending = std::async(
        std::launch::async, [voice = st.g().voice.get(),
                             frames = std::move(st.unsent), finalize]() {
          std::vector<float> samples;
          std::string error;
          if (!voice->Decode(frames, finalize, &samples, &error))
            throw std::runtime_error(error);
          return samples;
        });
    st.unsent.clear();
  }
  void Finish(Stream& st, bool ok, std::string error) {
    try {
      if (ok && !st.cancelled && st.result.frames)
        Decode(st, true);
      Join(st);
    } catch (const std::exception& e) {
      if (ok)
        error = e.what();
      ok = false;
    }
    st.result.cancelled = st.cancelled;
    st.result.timings.total_ms = Ms(st.start);
    st.active = false;
    auto done = std::move(st.done);
    auto result = std::move(st.result);
    st.callbacks = {};
    if (done)
      done(ok, std::move(result), std::move(error));
  }
};
SpeechBatch::SpeechBatch() : impl_(std::make_unique<Impl>()) {}
SpeechBatch::~SpeechBatch() {
  if (!impl_)
    return;
  for (auto& st : impl_->streams)
    if (st.active)
      impl_->Finish(st, false, "speech batch closed");
}
std::size_t SpeechBatch::capacity() const {
  return impl_->streams.size();
}
std::size_t SpeechBatch::active() const {
  return std::size_t(std::count_if(impl_->streams.begin(), impl_->streams.end(),
                                   [](const auto& st) { return st.active; }));
}
std::unique_ptr<SpeechBatch> SpeechBatch::Create(std::shared_ptr<Model> model,
                                                 std::uint32_t context,
                                                 std::size_t streams,
                                                 std::string* error) {
  if (!streams || streams > lfm3::Session::kMaxBatch) {
    if (error)
      *error = "invalid speech batch size";
    return nullptr;
  }
  auto batch = std::unique_ptr<SpeechBatch>(new SpeechBatch);
  batch->impl_->model = model;
  batch->impl_->streams.resize(streams);
  for (auto& st : batch->impl_->streams) {
    st.generator = Generator::Create(model, context, error);
    if (!st.generator)
      return nullptr;
  }
  return batch;
}
bool SpeechBatch::Add(std::span<const PromptPiece> prompt,
                      const GenerateOptions& options,
                      GenerateCallbacks callbacks, Done done,
                      std::string* error) {
  auto& impl = *impl_;
  auto it = std::find_if(impl.streams.begin(), impl.streams.end(),
                         [](const auto& st) { return !st.active; });
  if (it == impl.streams.end()) {
    if (error)
      *error = "speech batch is full";
    return false;
  }
  auto& st = *it;
  st.result = {};
  st.start = Clock::now();
  if (!st.generator->Reset(error) ||
      !st.generator->Append(prompt, &st.result, error))
    return false;
  if (st.g().last_token != impl.model->tokens().audio_start) {
    if (error)
      *error = "batched speech prompts end in <|audio_start|>";
    return false;
  }
  st.result.timings.prefill_ms = Ms(st.start);
  st.g().voice->Reset();
  st.options = options;
  st.callbacks = std::move(callbacks);
  st.done = std::move(done);
  st.rng.seed(options.seed);
  st.unsent.clear();
  st.steps = 0;
  st.cancelled = false;
  st.active = true;
  return true;
}
bool SpeechBatch::Step(std::string* error) {
  auto& impl = *impl_;
  std::vector<Impl::Stream*> live;
  for (auto& st : impl.streams)
    if (st.active) {
      if (st.callbacks.cancelled && st.callbacks.cancelled())
        st.cancelled = true;
      if (st.cancelled)
        impl.Finish(st, true, {});
      else
        live.push_back(&st);
    }
  if (live.empty())
    return true;
  const auto kCodes = std::int32_t(AudioWeights::kCodes);
  const auto ids = impl.model->tokens();
  try {
    // Frames of every stream in one depth pass.
    auto t = Clock::now();
    std::string failure;
    if (live.size() == 1) {
      auto& g = live[0]->g();
      if (!g.voice->SampleFrame(
              g.session->DeviceHidden(), live[0]->options.audio, live[0]->rng,
              live[0]->codes, g.frame_embedding, false, &failure))
        throw std::runtime_error(failure);
    } else {
      std::vector<Voice*> voices;
      std::vector<const float*> hiddens;
      std::vector<Voice::Sampling> sampling;
      std::vector<std::mt19937_64*> rngs;
      std::vector<float*> embeddings;
      std::vector<std::int32_t> codes(live.size() * AudioWeights::kCodebooks);
      for (auto* st : live) {
        voices.push_back(st->g().voice.get());
        hiddens.push_back(st->g().session->DeviceHidden());
        sampling.push_back(st->options.audio);
        rngs.push_back(&st->rng);
        embeddings.push_back(st->g().frame_embedding);
      }
      if (!Voice::SampleFrames(voices, hiddens, sampling, rngs, codes,
                               embeddings, &failure))
        throw std::runtime_error(failure);
      for (std::size_t i = 0; i < live.size(); ++i)
        std::copy_n(
            codes.begin() + std::ptrdiff_t(i) * AudioWeights::kCodebooks,
            AudioWeights::kCodebooks, live[i]->codes.begin());
    }
    const double depth_ms = Ms(t);
    // Streams that continue take one shared backbone step.
    std::vector<Impl::Stream*> next;
    for (auto* st : live) {
      st->result.timings.depth_ms += depth_ms;
      const bool eos = std::any_of(st->codes.begin(), st->codes.end(),
                                   [&](auto c) { return c == kCodes; });
      if (eos) {
        st->result.stopped = true;
        impl.Finish(*st, true, {});
        continue;
      }
      auto& r = st->result;
      r.codes.insert(r.codes.end(), st->codes.begin(), st->codes.end());
      st->unsent.insert(st->unsent.end(), st->codes.begin(), st->codes.end());
      ++r.frames;
      r.tokens.push_back(ids.audio_out);
      const auto chunk = r.frames <= st->options.first_chunk_frames
                             ? st->options.first_chunk_frames
                             : st->options.chunk_frames;
      if (st->unsent.size() >= chunk * AudioWeights::kCodebooks)
        impl.Decode(*st, false);
      if (++st->steps >= st->options.max_new_tokens ||
          st->g().session->Position() >= st->g().context)
        impl.Finish(*st, true, {});
      else
        next.push_back(st);
    }
    if (!next.empty()) {
      t = Clock::now();
      if (next.size() == 1) {
        auto& g = next[0]->g();
        if (!g.session->ForwardEmbeddings(
                {g.frame_embedding, std::size_t(g.Hidden())}, true, {},
                &failure))
          throw std::runtime_error(failure);
      } else {
        std::vector<lfm3::Session*> sessions;
        std::vector<std::uint32_t> tokens(next.size(), 0);
        std::vector<const float*> rows;
        for (auto* st : next) {
          sessions.push_back(st->g().session.get());
          rows.push_back(st->g().frame_embedding);
        }
        if (!lfm3::Session::ForwardBatchInputs(sessions, tokens, rows, false,
                                               &failure))
          throw std::runtime_error(failure);
      }
      const double backbone_ms = Ms(t);
      for (auto* st : next) {
        st->result.timings.backbone_ms += backbone_ms;
        impl.Join(*st);
      }
    }
    return true;
  } catch (const std::exception& e) {
    for (auto* st : live)
      if (st->active)
        impl.Finish(*st, false, e.what());
    if (error)
      *error = e.what();
    return false;
  }
}

struct LiveTranscriber::Impl {
  static constexpr std::uint32_t kChunk = 12;  // audio tokens, 0.96 s
  // Audio tokens per language context: training chains samples of at most
  // 30 seconds.
  static constexpr std::uint32_t kContextAudio = 375;
  static constexpr std::uint32_t kContext = 4096;
  std::shared_ptr<Model> model;
  std::unique_ptr<lfm3::Session> session;
  std::unique_ptr<AudioInput> input;
  std::vector<std::uint32_t> prompt;
  // Fed before the next chunk's audio: the prompt of a new context or the
  // <|text_end|> closing the previous block.
  std::vector<std::uint32_t> pending;
  std::vector<float> rows;  // completed audio tokens not yet in a chunk
  std::vector<float> logits;
  std::string text, bytes;
  std::size_t emitted{};
  std::uint32_t context_audio{}, chunks{};
  bool finished{}, stopped{};
  void Fail(bool ok, const std::string& failure) const {
    if (!ok)
      throw std::runtime_error(failure.empty() ? "LFM3-Audio failure"
                                               : failure);
  }
  // Decodes one block after count audio tokens of rows.
  void Chunk(std::uint32_t count, bool final, const TextCallback& on_text) {
    const auto& ids = model->tokens();
    const auto& tok = model->tokenizer();
    const auto H = model->backbone().config().hidden;
    const std::uint32_t budget = final ? 96 : 48;
    std::string failure;
    if (context_audio + count > kContextAudio ||
        session->Position() + pending.size() + count + budget + 2 > kContext) {
      Fail(session->Reset(&failure), failure);
      pending = prompt;
      context_audio = 0;
    }
    if (!pending.empty())
      Fail(session->Forward(pending, {}, &failure), failure);
    pending.clear();
    if (count) {
      Fail(session->ForwardEmbeddings({rows.data(), std::size_t(count) * H},
                                      false, {}, &failure),
           failure);
      rows.erase(rows.begin(), rows.begin() + std::ptrdiff_t(count) * H);
      context_audio += count;
    }
    ++chunks;
    // Before the final chunk the turn cannot end: speech and the end-of-turn
    // tokens are suppressed; placeholders always are.
    std::vector<std::uint32_t> masked{ids.audio, ids.audio_out};
    if (!final)
      masked.insert(masked.end(), {ids.audio_start, ids.mixed_start, ids.im_end,
                                   ids.mixed_end, ids.eos});
    for (std::uint32_t n = 0; n < budget; ++n) {
      Fail(session->ReadLogits(logits, &failure), failure);
      for (auto id : masked)
        logits[id] = -INFINITY;
      const auto token = std::uint32_t(
          std::max_element(logits.begin(), logits.end()) - logits.begin());
      if (token == ids.text_end ||
          (final && (token == ids.im_end || token == ids.eos ||
                     token == ids.im_start || tok.IsStop(token))))
        break;
      if (!tok.IsControl(token)) {
        bytes += tok.Piece(token);
        const auto end = CompleteUtf8(bytes);
        if (end > emitted) {
          const std::string_view piece(bytes.data() + emitted, end - emitted);
          text += piece;
          emitted = end;
          if (on_text && !on_text(piece)) {
            stopped = true;
            return;
          }
        }
      }
      if (session->Position() >= kContext)
        break;
      Fail(session->Forward({&token, 1}, {}, &failure), failure);
    }
    if (!final)
      pending = {ids.text_end};
  }
};
LiveTranscriber::LiveTranscriber() : impl_(std::make_unique<Impl>()) {}
LiveTranscriber::~LiveTranscriber() = default;
std::unique_ptr<LiveTranscriber> LiveTranscriber::Create(
    std::shared_ptr<Model> model, std::string* error) {
  auto live = std::unique_ptr<LiveTranscriber>(new LiveTranscriber);
  auto& impl = *live->impl_;
  impl.model = model;
  impl.session = model->backbone().CreateSession(Impl::kContext, error);
  if (!impl.session)
    return nullptr;
  impl.logits.resize(model->backbone().config().vocab);
  impl.prompt =
      model->tokenizer().Encode("<|startoftext|><|im_start|>system\n" +
                                    std::string(kLiveAsrSystemPrompt) +
                                    "<|im_end|>\n<|im_start|>assistant\n",
                                true);
  if (!live->Reset(error))
    return nullptr;
  return live;
}
bool LiveTranscriber::Reset(std::string* error) {
  auto& impl = *impl_;
  impl.input = impl.model->CreateAudioInput(error);
  if (!impl.input || !impl.session->Reset(error))
    return false;
  impl.pending = impl.prompt;
  impl.rows.clear();
  impl.text.clear();
  impl.bytes.clear();
  impl.emitted = 0;
  impl.context_audio = impl.chunks = 0;
  impl.finished = impl.stopped = false;
  return true;
}
const std::string& LiveTranscriber::text() const {
  return impl_->text;
}
std::uint64_t LiveTranscriber::samples() const {
  return impl_->input ? impl_->input->samples() : 0;
}
std::uint32_t LiveTranscriber::chunks() const {
  return impl_->chunks;
}
bool LiveTranscriber::Push(std::span<const float> samples,
                           const TextCallback& on_text, std::string* error) {
  try {
    auto& impl = *impl_;
    if (impl.finished || impl.stopped)
      throw std::logic_error("live transcription stream has ended");
    std::vector<float> rows;
    std::string failure;
    impl.Fail(impl.input->Push(samples, false, &rows, &failure), failure);
    impl.rows.insert(impl.rows.end(), rows.begin(), rows.end());
    const auto H = impl.model->backbone().config().hidden;
    while (!impl.stopped && impl.rows.size() >= std::size_t(Impl::kChunk) * H)
      impl.Chunk(Impl::kChunk, false, on_text);
    return true;
  } catch (const std::exception& e) {
    impl_->finished = true;
    if (error)
      *error = e.what();
    return false;
  }
}
bool LiveTranscriber::Finish(const TextCallback& on_text, std::string* error) {
  try {
    auto& impl = *impl_;
    if (impl.finished)
      throw std::logic_error("live transcription stream has ended");
    impl.finished = true;
    if (impl.stopped)
      return true;
    std::vector<float> rows;
    std::string failure;
    impl.Fail(impl.input->Push({}, true, &rows, &failure), failure);
    impl.rows.insert(impl.rows.end(), rows.begin(), rows.end());
    const auto H = impl.model->backbone().config().hidden;
    while (!impl.stopped && impl.rows.size() > std::size_t(Impl::kChunk) * H)
      impl.Chunk(Impl::kChunk, false, on_text);
    if (!impl.stopped)
      impl.Chunk(std::uint32_t(impl.rows.size() / H), true, on_text);
    if (!impl.stopped && impl.bytes.size() > impl.emitted) {
      const std::string_view rest(impl.bytes.data() + impl.emitted,
                                  impl.bytes.size() - impl.emitted);
      impl.text += rest;
      impl.emitted = impl.bytes.size();
      if (on_text)
        (void)on_text(rest);
    }
    return true;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return false;
  }
}

namespace {
constexpr std::string_view kThinkOpen = "<think>", kThinkClose = "</think>",
                           kCallOpen = "<tool_call>";
// Bytes at the end of text that could start tag.
std::size_t PartialTag(std::string_view text, std::string_view tag) {
  for (std::size_t n = std::min(text.size(), tag.size() - 1); n > 0; --n)
    if (text.substr(text.size() - n) == tag.substr(0, n))
      return n;
  return 0;
}
}  // namespace
ReplyParser::Delta ReplyParser::Feed(std::string_view text) {
  pending_ += text;
  return Drain(false);
}
ReplyParser::Delta ReplyParser::Drain(bool final) {
  Delta out;
  // A reply may open its own reasoning.
  if (!thinking_ && reasoning_.empty() && content_.empty() && !calling_) {
    const auto lead = pending_.find_first_not_of(" \n");
    if (lead != std::string::npos &&
        std::string_view(pending_).substr(lead).starts_with(kThinkOpen)) {
      pending_.erase(0, lead + kThinkOpen.size());
      thinking_ = true;
    } else if (!final && (lead == std::string::npos ||
                          kThinkOpen.starts_with(
                              std::string_view(pending_).substr(lead))))
      return out;
  }
  if (thinking_) {
    const auto close = pending_.find(kThinkClose);
    if (close == std::string::npos) {
      const auto keep = final ? 0 : PartialTag(pending_, kThinkClose);
      out.reasoning = pending_.substr(0, pending_.size() - keep);
      pending_.erase(0, out.reasoning.size());
      reasoning_ += out.reasoning;
      return out;
    }
    out.reasoning = pending_.substr(0, close);
    reasoning_ += out.reasoning;
    pending_.erase(0, close + kThinkClose.size());
    thinking_ = false;
  }
  if (!calling_) {
    // Answers follow </think> after whitespace.
    if (content_.empty()) {
      const auto lead = pending_.find_first_not_of(" \n");
      pending_.erase(0, lead == std::string::npos ? pending_.size() : lead);
    }
    const auto call = pending_.find(kCallOpen);
    if (call != std::string::npos) {
      out.content = pending_.substr(0, call);
      calls_ = pending_.substr(call);
      pending_.clear();
      calling_ = true;
    } else {
      const auto keep = final ? 0 : PartialTag(pending_, kCallOpen);
      out.content = pending_.substr(0, pending_.size() - keep);
      pending_.erase(0, out.content.size());
    }
    content_ += out.content;
  } else {
    calls_ += pending_;
    pending_.clear();
  }
  return out;
}
ReplyParser::Delta ReplyParser::Finish(std::vector<ToolCall>* calls) {
  auto out = Drain(true);
  calls->clear();
  std::string_view rest = calls_;
  while (true) {
    const auto lead = rest.find_first_not_of(" \n");
    rest.remove_prefix(lead == std::string_view::npos ? rest.size() : lead);
    if (rest.empty())
      break;
    const auto scan = tokenization::ScanQwenToolCall(rest);
    if (!scan.call || !scan.call->complete || !scan.call->native) {
      // Text that is not a complete call stays visible.
      out.content += rest;
      content_ += rest;
      break;
    }
    ToolCall call{scan.call->name, {}};
    for (const auto& parameter : scan.call->parameters) {
      ToolArgument argument{parameter.name, parameter.value, true};
      try {
        (void)json::parse(parameter.value);
        argument.is_string = false;
      } catch (const std::exception&) {
      }
      call.arguments.push_back(std::move(argument));
    }
    calls->push_back(std::move(call));
    rest.remove_prefix(scan.end);
  }
  calls_.clear();
  return out;
}
}  // namespace gufo::models::lfm3_audio
