#include "src/cli/serve/lfm3_audio_runner.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/json.hpp"
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
  std::uint32_t context{};
  std::mutex mutex;
  std::shared_ptr<std::atomic<int>> realtime =
      std::make_shared<std::atomic<int>>(0);
  // Synthesis requests join one batch on the speaker thread, so concurrent
  // requests share backbone and depth-decoder steps.
  struct Job {
    std::vector<audio::PromptPiece> prompt;
    audio::GenerateOptions options;
    audio::GenerateCallbacks callbacks;
    std::promise<std::pair<std::string, audio::GenerateResult>> done;
  };
  std::mutex speech_mutex;
  std::condition_variable speech_ready;
  std::deque<std::shared_ptr<Job>> jobs;
  bool stopping{false};
  std::thread speaker;
  void Speak();
  ~Impl() {
    {
      const std::lock_guard lock(speech_mutex);
      stopping = true;
    }
    speech_ready.notify_all();
    if (speaker.joinable())
      speaker.join();
  }
};
void Lfm3AudioRuntime::Impl::Speak() {
  std::string error;
  auto batch = audio::SpeechBatch::Create(
      model, context, models::lfm3::Session::kMaxBatch, &error);
  const std::string unavailable = error;
  while (true) {
    std::vector<std::shared_ptr<Job>> admitted;
    {
      std::unique_lock lock(speech_mutex);
      speech_ready.wait(lock, [&] {
        return stopping || !jobs.empty() || (batch && batch->active());
      });
      if (stopping)
        break;
      while (!jobs.empty() && (!batch || admitted.size() + batch->active() <
                                             batch->capacity())) {
        admitted.push_back(std::move(jobs.front()));
        jobs.pop_front();
      }
    }
    for (auto& job : admitted) {
      std::string failure = unavailable;
      if (batch && batch->Add(
                       job->prompt, job->options, job->callbacks,
                       [job](bool ok, audio::GenerateResult result,
                             std::string message) {
                         job->done.set_value(
                             {ok ? std::string() : message, std::move(result)});
                       },
                       &failure))
        continue;
      job->done.set_value({failure.empty() ? "speech batch failed" : failure,
                           audio::GenerateResult{}});
    }
    if (batch && batch->active())
      (void)batch->Step(&error);
    error.clear();
  }
  batch.reset();  // ends active streams with an error
  const std::lock_guard lock(speech_mutex);
  for (auto& job : jobs)
    job->done.set_value({"LFM3-Audio runtime stopped", {}});
  jobs.clear();
}
namespace {
// Holds one of the runtime's realtime slots for the life of a stream.
class Slot {
public:
  explicit Slot(std::shared_ptr<std::atomic<int>> count)
      : count_(std::move(count)) {}
  ~Slot() {
    if (count_)
      --*count_;
  }
  Slot(Slot&& other) noexcept : count_(std::move(other.count_)) {}
  Slot(const Slot&) = delete;
  static std::optional<Slot> Acquire(std::shared_ptr<std::atomic<int>> count,
                                     std::string* error) {
    if (++*count > Lfm3AudioRuntime::kRealtimeSessions) {
      --*count;
      *error = "too many realtime LFM3-Audio sessions";
      return std::nullopt;
    }
    return Slot(std::move(count));
  }

private:
  std::shared_ptr<std::atomic<int>> count_;
};
class LiveStream final : public server::LiveTranscription {
public:
  LiveStream(Slot slot, std::unique_ptr<audio::LiveTranscriber> live,
             std::uint32_t rate)
      : slot_(std::move(slot)),
        live_(std::move(live)),
        resampler_(rate, audio::Model::kSampleRate) {}
  bool Push(std::span<const float> samples, const Text& on_text,
            std::string* error) override {
    return live_->Push(resampler_.Push(samples), on_text, error);
  }
  bool Finish(const Text& on_text, std::string* error) override {
    const auto tail = resampler_.Push({}, true);
    return live_->Push(tail, on_text, error) && live_->Finish(on_text, error);
  }
  const std::string& text() const override { return live_->text(); }

private:
  Slot slot_;
  std::unique_ptr<audio::LiveTranscriber> live_;
  audio::Resampler resampler_;
};
std::string Trimmed(std::string_view text) {
  const auto first = text.find_first_not_of(" \t\r\n");
  if (first == std::string_view::npos)
    return {};
  return std::string(
      text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1));
}
// The official template's spelling of earlier tool calls.
void RenderCalls(std::span<const audio::ToolCall> calls,
                 std::vector<audio::PromptPiece>* pieces) {
  using Kind = audio::PromptPiece::Kind;
  for (std::size_t i = 0; i < calls.size(); ++i) {
    std::string head = i ? "\n\n" : "";
    head += "<tool_call>\n<function=" + calls[i].name + ">\n";
    pieces->push_back({Kind::kTemplate, head, {}});
    for (const auto& argument : calls[i].arguments) {
      pieces->push_back(
          {Kind::kTemplate, "<parameter=" + argument.name + ">\n", {}});
      pieces->push_back({Kind::kLiteral,
                         argument.is_string
                             ? argument.value
                             : json::parse(argument.value).tojson(),
                         {}});
      pieces->push_back({Kind::kTemplate, "\n</parameter>\n", {}});
    }
    pieces->push_back({Kind::kTemplate, "</function>\n</tool_call>", {}});
  }
}
class Conversation final : public server::VoiceConversation {
public:
  // Context of one conversation: turns, speech frames (12.5 per second) and
  // the reply budget. Older turns leave when a new one needs the room.
  static constexpr std::uint32_t kContext = 8192;
  Conversation(Slot slot, std::shared_ptr<audio::Model> model,
               std::unique_ptr<audio::Generator> generator, std::uint32_t rate)
      : slot_(std::move(slot)),
        model_(std::move(model)),
        generator_(std::move(generator)),
        rate_(rate) {}
  bool Configure(const Settings& settings, std::string* error) override {
    try {
      Check(!open_, "commit or clear the audio buffer first");
      std::vector<std::string> tools;
      for (const auto& tool : settings.tools)
        tools.push_back(audio::ToolDefinition(tool));
      const auto system = SystemPrompt(settings, !tools.empty());
      if (installed_ && (system != system_ || tools != tools_)) {
        const auto before = generator_->Position();
        Check(generator_->Replace(0, system_end_,
                                  audio::SystemPieces(system, tools), error));
        const std::int64_t delta =
            std::int64_t(generator_->Position()) - before;
        for (auto& turn : turns_) {
          turn.begin = std::uint32_t(turn.begin + delta);
          turn.end = std::uint32_t(turn.end + delta);
        }
        system_end_ = std::uint32_t(system_end_ + delta);
      }
      settings_ = settings;
      tools_ = std::move(tools);
      system_ = system;
      return true;
    } catch (const std::exception& e) {
      *error = e.what();
      return false;
    }
  }
  bool AppendAudio(std::span<const float> samples,
                   std::string* error) override {
    try {
      if (!open_) {
        BeginTurn(true);
        const auto begin = generator_->Position();
        Template("<|im_start|>user\n");
        input_ = model_->CreateAudioInput(error);
        Check(bool(input_));
        resampler_.emplace(rate_, audio::Model::kSampleRate);
        open_ = true;
        turn_begin_ = begin;
      }
      Rows(resampler_->Push(samples), false, error);
      return true;
    } catch (const std::exception& e) {
      *error = e.what();
      return false;
    }
  }
  bool CommitAudio(std::size_t* samples, std::string* error) override {
    try {
      *samples = 0;
      if (!open_)
        return false;
      Rows(resampler_->Push({}, true), true, error);
      *samples = input_->samples();
      if (!input_->tokens()) {
        Check(ClearAudio(error));
        return false;
      }
      Template("<|im_end|>\n");
      turns_.push_back({turn_begin_, generator_->Position()});
      open_ = false;
      input_.reset();
      return true;
    } catch (const std::exception& e) {
      *error = e.what();
      return false;
    }
  }
  bool ClearAudio(std::string* error) override {
    if (!open_)
      return true;
    open_ = false;
    input_.reset();
    return generator_->Replace(turn_begin_, generator_->Position(), {}, error);
  }
  bool audio_open() const override { return open_; }
  bool AddUserText(std::string_view text, std::string* error) override {
    try {
      Check(!open_, "commit or clear the audio buffer first");
      BeginTurn(true);
      audio::ChatMessage message{"user", {{false, std::string(text), {}}}};
      Message(message, error);
      return true;
    } catch (const std::exception& e) {
      *error = e.what();
      return false;
    }
  }
  bool AddToolOutputs(std::span<const std::string> outputs,
                      std::string* error) override {
    try {
      Check(!open_, "commit or clear the audio buffer first");
      BeginTurn(false);
      std::string content;
      for (const auto& output : outputs)
        content += (content.empty() ? "" : "\n") + output;
      audio::ChatMessage message{"tool", {{false, content, {}}}};
      Message(message, error);
      return true;
    } catch (const std::exception& e) {
      *error = e.what();
      return false;
    }
  }
  bool Respond(const Settings& settings, const Callbacks& callbacks,
               Reply* reply, std::string* error) override {
    try {
      *reply = {};
      Check(!open_, "commit or clear the audio buffer before a response");
      BeginTurn(false);
      Check(!turns_.empty(), "the conversation has no input");
      const auto begin = generator_->Position();
      // Model card prefills: <|mixed_start|> for speech, <think> or
      // <think></think> for text; with tools the model chooses.
      std::string prefix;
      const bool tools = !tools_.empty();
      if (!tools)
        prefix = settings.audio_output ? "<|mixed_start|>"
                 : settings.reasoning  ? "<think>"
                                       : "<think></think>";
      Template("<|im_start|>assistant\n" + prefix);
      reply_begin_ = begin;
      audio::GenerateOptions options;
      const auto room = generator_->Context() - generator_->Position();
      Check(room > 8, "the conversation context is full");
      options.max_new_tokens =
          std::min<std::size_t>(settings.max_output_tokens, room - 4);
      options.seed = settings.seed ? *settings.seed : std::random_device{}();
      if (!settings.audio_output)
        options.suppress_mixed = options.suppress_audio_out = true;
      else if (tools) {
        // A spoken reply starts with a call or with interleaved speech.
        const auto call = model_->tokenizer().Encode("<tool_call>", true);
        Check(call.size() == 1, "tokenizer lacks <tool_call>");
        options.first_tokens = {call[0], model_->tokens().mixed_start};
      }
      audio::ReplyParser parser(prefix == "<think>");
      audio::GenerateCallbacks hooks;
      hooks.cancelled = callbacks.cancelled;
      hooks.on_audio = callbacks.on_audio;
      bool stop = false;
      auto send = [&](const audio::ReplyParser::Delta& delta) {
        if (!delta.reasoning.empty() && callbacks.on_reasoning &&
            !callbacks.on_reasoning(delta.reasoning))
          stop = true;
        if (!delta.content.empty() && callbacks.on_text &&
            !callbacks.on_text(delta.content))
          stop = true;
        return !stop;
      };
      hooks.on_text = [&](std::string_view piece) {
        return send(parser.Feed(piece));
      };
      audio::GenerateResult result;
      Check(generator_->Continue(options, hooks, &result, error));
      std::vector<audio::ToolCall> calls;
      send(parser.Finish(&calls));
      // The reply stays in the context, closed like a template turn.
      Template("<|im_end|>\n");
      turns_.push_back({begin, generator_->Position()});
      // Earlier reasoning leaves the context once the user speaks again, as
      // the template renders it.
      if (!parser.reasoning().empty() || prefix == "<think>") {
        std::vector<audio::PromptPiece> pieces{
            {audio::PromptPiece::Kind::kTemplate,
             "<|im_start|>assistant\n",
             {}},
            {audio::PromptPiece::Kind::kLiteral,
             Trimmed(parser.content()),
             {}}};
        RenderCalls(calls, &pieces);
        pieces.push_back(
            {audio::PromptPiece::Kind::kTemplate, "<|im_end|>\n", {}});
        rewrites_.push_back({turns_.size() - 1, std::move(pieces)});
      }
      reply->reasoning = parser.reasoning();
      reply->text = parser.content();
      for (const auto& call : calls) {
        auto arguments = json::Value::object();
        for (const auto& argument : call.arguments)
          arguments[argument.name] = argument.is_string
                                         ? json::Value(argument.value)
                                         : json::parse(argument.value);
        reply->calls.push_back({call.name, arguments.dump()});
      }
      reply->audio_samples = result.samples.size();
      reply->input_tokens = begin;
      reply->output_tokens = result.tokens.size();
      reply->cancelled = result.cancelled || stop;
      reply->incomplete = !result.stopped && !reply->cancelled;
      reply->first_text_ms = result.timings.first_text_ms;
      reply->first_audio_ms = result.timings.first_audio_ms;
      reply->total_ms = result.timings.total_ms;
      reply_begin_.reset();
      return true;
    } catch (const std::exception& e) {
      if (error->empty())
        *error = e.what();
      // A failed reply leaves no partial turn behind.
      if (reply_begin_ && *reply_begin_ <= generator_->Position()) {
        std::string ignored;
        (void)generator_->Replace(*reply_begin_, generator_->Position(), {},
                                  &ignored);
      }
      reply_begin_.reset();
      return false;
    }
  }

private:
  struct Turn {
    std::uint32_t begin, end;
  };
  struct Rewrite {
    std::size_t turn;
    std::vector<audio::PromptPiece> pieces;
  };
  // The model card's prompt of the session's output without instructions.
  static std::string SystemPrompt(const Settings& settings, bool tools) {
    if (!settings.instructions.empty())
      return settings.instructions;
    if (tools)
      return std::string(settings.audio_output
                             ? audio::kToolsInterleavedSystemPrompt
                             : audio::kToolsTextSystemPrompt);
    return settings.audio_output ? std::string(audio::kInterleavedSystemPrompt)
                                 : std::string();
  }
  static void Check(bool ok, std::string_view what = {}) {
    if (!ok)
      throw std::runtime_error(std::string(what));
  }
  void Template(const std::string& text) {
    const std::vector<audio::PromptPiece> pieces{
        {audio::PromptPiece::Kind::kTemplate, text, {}}};
    std::string failure;
    if (!generator_->Append(pieces, nullptr, &failure))
      throw std::runtime_error(failure);
  }
  void Message(const audio::ChatMessage& message, std::string* error) {
    const auto begin = generator_->Position();
    Check(generator_->Append(audio::MessagePieces(message), nullptr, error));
    turns_.push_back({begin, generator_->Position()});
  }
  void Rows(const std::vector<float>& samples, bool finish,
            std::string* error) {
    std::vector<float> rows;
    Check(input_->Push(samples, finish, &rows, error));
    if (rows.empty())
      return;
    const auto H = model_->backbone().config().hidden;
    Check(
        generator_->Position() + rows.size() / H + 256 <= generator_->Context(),
        "user audio exceeds the conversation context");
    Check(generator_->AppendEmbeddings(rows, error));
  }
  void Replace(std::uint32_t begin, std::uint32_t end,
               std::span<const audio::PromptPiece> pieces) {
    const auto before = generator_->Position();
    std::string failure;
    if (!generator_->Replace(begin, end, pieces, &failure))
      throw std::runtime_error(failure);
    const std::int64_t delta = std::int64_t(generator_->Position()) - before;
    for (auto& turn : turns_)
      if (turn.begin >= end) {
        turn.begin = std::uint32_t(turn.begin + delta);
        turn.end = std::uint32_t(turn.end + delta);
      } else if (turn.begin >= begin && turn.end == end) {
        turn.end = std::uint32_t(turn.end + delta);  // the replaced turn
      }
  }
  // Installs the system turn, rewrites earlier reasoning when the user
  // speaks again and frees room for a reply by dropping the oldest turns.
  void BeginTurn(bool user) {
    if (!installed_) {
      Check(generator_->Reset());
      std::string failure;
      Check(generator_->Append(audio::SystemPieces(system_, tools_), nullptr,
                               &failure),
            failure);
      system_end_ = generator_->Position();
      installed_ = true;
    }
    if (user) {
      for (auto it = rewrites_.rbegin(); it != rewrites_.rend(); ++it)
        Replace(turns_[it->turn].begin, turns_[it->turn].end, it->pieces);
      rewrites_.clear();
    }
    const std::size_t reserve =
        std::min<std::size_t>(settings_.max_output_tokens, kContext / 2) + 512;
    auto freed = [&](std::size_t n) -> std::size_t {
      return n ? turns_[n - 1].end - turns_.front().begin : 0;
    };
    std::size_t drop = 0;
    while (drop < turns_.size() &&
           generator_->Position() - freed(drop) + reserve >
               generator_->Context())
      ++drop;
    if (drop) {
      // Pending rewrites refer to turn indices; dropping turns forgets them.
      rewrites_.clear();
      const auto begin = turns_.front().begin, end = turns_[drop - 1].end;
      turns_.erase(turns_.begin(), turns_.begin() + std::ptrdiff_t(drop));
      Replace(begin, end, {});
    }
  }
  Slot slot_;
  std::shared_ptr<audio::Model> model_;
  std::unique_ptr<audio::Generator> generator_;
  std::uint32_t rate_;
  Settings settings_;
  std::vector<std::string> tools_;  // in the template's spelling
  std::string system_{audio::kInterleavedSystemPrompt};
  bool installed_{false}, open_{false};
  std::uint32_t system_end_{}, turn_begin_{};
  std::optional<std::uint32_t> reply_begin_;
  std::unique_ptr<audio::AudioInput> input_;
  std::optional<audio::Resampler> resampler_;
  std::vector<Turn> turns_;
  std::vector<Rewrite> rewrites_;
};
}  // namespace
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
  runtime->impl_->context = context;
  return runtime;
}
std::unique_ptr<server::LiveTranscription>
Lfm3AudioRuntime::CreateLiveTranscription(std::uint32_t rate,
                                          std::string* error) {
  auto slot = Slot::Acquire(impl_->realtime, error);
  if (!slot)
    return nullptr;
  auto live = audio::LiveTranscriber::Create(impl_->model, error);
  if (!live)
    return nullptr;
  return std::make_unique<LiveStream>(std::move(*slot), std::move(live), rate);
}
std::unique_ptr<server::VoiceConversation>
Lfm3AudioRuntime::CreateVoiceConversation(std::uint32_t rate,
                                          std::string* error) {
  auto slot = Slot::Acquire(impl_->realtime, error);
  if (!slot)
    return nullptr;
  auto generator =
      audio::Generator::Create(impl_->model, Conversation::kContext, error);
  if (!generator)
    return nullptr;
  return std::make_unique<Conversation>(std::move(*slot), impl_->model,
                                        std::move(generator), rate);
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
  auto job = std::make_shared<Impl::Job>();
  job->prompt = audio::BuildPrompt(system, messages, "<|audio_start|>");
  job->options = options;
  job->callbacks = std::move(callbacks);
  auto done = job->done.get_future();
  {
    const std::lock_guard lock(impl_->speech_mutex);
    if (!impl_->speaker.joinable())
      impl_->speaker = std::thread([impl = impl_.get()] { impl->Speak(); });
    impl_->jobs.push_back(std::move(job));
  }
  impl_->speech_ready.notify_all();
  auto [failure, generated] = done.get();
  if (!failure.empty()) {
    *error = failure;
    return false;
  }
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
