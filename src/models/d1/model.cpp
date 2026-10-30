#include "src/models/d1/model.hpp"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdexcept>

#include "src/models/d1/vision.hpp"
#include "src/models/lfm3/engine.hpp"
#include "src/models/lfm3_audio/model.hpp"
#include "src/models/qwen/control_tokens.hpp"
namespace gufo::models::d1 {
namespace {
using json::Value;
std::string Pretty(const Value& v, unsigned depth = 0) {
  if (!v.is_array() && !v.is_object())
    return v.dump();
  if (v.empty())
    return v.dump();
  std::string text = v.is_array() ? "[\n" : "{\n";
  std::size_t i = 0;
  auto append = [&](const std::string& prefix, const Value& item) {
    if (i++)
      text += ",\n";
    text +=
        std::string((depth + 1) * 2, ' ') + prefix + Pretty(item, depth + 1);
  };
  if (v.is_array())
    for (const auto& item : v.items())
      append("", item);
  else
    for (const auto& [name, item] : v.members())
      append(Value(name).dump() + ": ", item);
  return text + '\n' + std::string(depth * 2, ' ') + (v.is_array() ? "]" : "}");
}
struct Media {
  std::uint32_t token{};
  std::string markup;
  std::vector<float> rows;
};
}  // namespace
struct Model::Impl {
  std::shared_ptr<lfm3::Model> text;
  std::shared_ptr<lfm3_audio::Model> audio;
  std::unique_ptr<Vision> vision;
  std::unique_ptr<core::GgufReader> reader;
  lfm3::Weights host;
  std::unique_ptr<lfm3::Session> session;
  std::mutex mutex;
  unsigned context{};
  lfm3::Model& backbone() { return audio ? audio->backbone() : *text; }
  const lfm3::Tokenizer& tokenizer() { return backbone().tokenizer(); }
  std::uint32_t Token(const std::string& piece) {
    auto ids = tokenizer().Encode(piece);
    if (ids.size() != 1)
      throw std::runtime_error("missing d1 token " + piece);
    return ids[0];
  }
  Media ReadMedia(const Value& item, core::ImageReadBudget& budget) {
    if (!audio)
      throw std::invalid_argument("image/audio input needs --mmproj");
    if (!item.is_object())
      throw std::invalid_argument("media must be an object");
    const auto type = item.member_str("type");
    Media out;
    if (type == "image") {
      std::string url = item.member_str("url");
      if (const auto* u = item.find("image_url"))
        url = u->is_string() ? u->str() : u->member_str("url");
      if (url.empty())
        throw std::invalid_argument("image requires url or image_url");
      auto bytes = core::ReadImageUrl(url, budget);
      out.rows = vision->Encode(ProcessImage(core::DecodeImage(bytes)));
      out.token = Token("<|image|>");
    } else if (type == "audio" || type == "input_audio") {
      std::string url = item.member_str("url");
      if (const auto* a = item.find("input_audio")) {
        if (a->member_str("format") != "wav")
          throw std::invalid_argument("input_audio format must be wav");
        url = a->member_str("data");
      }
      if (url.empty())
        throw std::invalid_argument("audio requires url or input_audio data");
      auto bytes = core::ReadAudioUrl(url, budget);
      std::vector<float> samples;
      std::uint32_t rate{};
      std::string error;
      if (!lfm3_audio::DecodeWav(std::as_bytes(std::span(bytes)), &samples,
                                 &rate, &error))
        throw std::invalid_argument(error);
      if (samples.empty() || samples.size() / double(rate) > 300)
        throw std::invalid_argument("audio must contain at most 300 seconds");
      auto pcm = lfm3_audio::Resample(samples, rate, 16000);
      if (!audio->Encode(pcm, &out.rows, &error))
        throw std::runtime_error(error);
      out.token = Token("<|audio|>");
    } else
      throw std::invalid_argument(
          "media type must be image, audio or input_audio");
    const auto count = out.rows.size() / host.config.hidden;
    if (!count || count > context)
      throw std::invalid_argument("media exceeds decision context");
    out.markup = std::string(tokenization::kVisionStart);
    for (std::size_t i = 0; i < count; ++i)
      out.markup += tokenizer().Piece(out.token);
    out.markup += tokenization::kVisionEnd;
    return out;
  }
};
Model::Model() : impl_(std::make_unique<Impl>()) {}
Model::~Model() = default;
std::unique_ptr<Model> Model::Load(const std::string& path,
                                   const std::string& mmproj,
                                   unsigned context) {
  auto model = std::unique_ptr<Model>(new Model);
  auto& m = *model->impl_;
  std::string error;
  m.reader = core::GgufReader::OpenFile(path, &error);
  if (!m.reader)
    throw std::runtime_error(error);
  auto w = lfm3::Weights::Bind(*m.reader, &error);
  if (!w)
    throw std::runtime_error(error);
  m.host = *w;
  if (!w->config.decision)
    throw std::invalid_argument("model is not a d1-Omni decision GGUF");
  if (!context || context > std::min(16384U, w->config.context))
    throw std::invalid_argument("decision context must be 1..16384");
  m.context = context;
  if (mmproj.empty())
    m.text = lfm3::Model::Load(path, &error);
  else {
    m.audio = lfm3_audio::Model::Load(path, mmproj, &error);
    m.vision = Vision::Load(mmproj);
  }
  if (!m.audio && !m.text)
    throw std::runtime_error(error);
  m.session = m.backbone().CreateSession(context, &error);
  if (!m.session)
    throw std::runtime_error(error);
  return model;
}
json::Value Model::Decide(const Value& request) {
  auto& m = *impl_;
  std::lock_guard lock(m.mutex);
  const auto* qs = request.find("questions");
  if (!request.is_object() || !qs || !qs->is_object() || qs->empty() ||
      qs->size() > 128)
    throw std::invalid_argument(
        "questions must be an object with 1..128 named questions");
  if (request.contains("set_size") && request.member_size("set_size") != 1)
    throw std::invalid_argument(
        "this GGUF decision path reads one question per prompt (set_size=1)");
  std::vector<Question> questions;
  for (const auto& [name, value] : qs->members()) {
    auto q = ParseQuestion(m.tokenizer(), name, value);
    q.temperature =
        m.reader->GetMetadataFloat32("lfm3.decision.temperature." + q.type)
            .value_or(1);
    if (!std::isfinite(q.temperature) || q.temperature <= 0)
      throw std::runtime_error("invalid GGUF decision temperature");
    questions.push_back(std::move(q));
  }
  const auto plain = [&](std::string_view s) {
    return Plain(m.tokenizer(), s, m.host.config.vocab);
  };
  std::string prefix(m.tokenizer().Piece(m.tokenizer().Bos()));
  prefix += tokenization::kImStart;
  prefix += "user\n";
  std::vector<Media> media;
  core::ImageReadBudget budget;
  // The official GGUF API puts audio in files; this path accepts WAV files.
  if (const auto* files = request.find("files")) {
    if (!files->is_array() || files->size() + media.size() > 16)
      throw std::invalid_argument("files must list at most 16 WAV URLs");
    for (const auto& file : files->items()) {
      if (!file.is_string())
        throw std::invalid_argument("files must contain WAV URL strings");
      auto item = Value::object();
      item["type"] = "audio";
      item["url"] = file;
      media.push_back(m.ReadMedia(item, budget));
      prefix += media.back().markup;
    }
  }
  if (const auto* images = request.find("images")) {
    if (!images->is_array() || images->size() > 16)
      throw std::invalid_argument(
          "images must list at most 16 data/HTTPS URLs");
    for (const auto& image : images->items()) {
      if (!image.is_string())
        throw std::invalid_argument("images must contain URL strings");
      auto item = Value::object();
      item["type"] = "image";
      item["url"] = image;
      media.push_back(m.ReadMedia(item, budget));
      prefix += media.back().markup;
    }
  }
  if (const auto* items = request.find("media")) {
    if (!items->is_array() || items->size() + media.size() > 16)
      throw std::invalid_argument("media must list at most 16 items");
    for (const auto& item : items->items()) {
      media.push_back(m.ReadMedia(item, budget));
      prefix += media.back().markup;
    }
  }
  const auto* state = request.find("state");
  if (state && !state->is_null())
    prefix += plain(state->is_string() ? state->str() : Pretty(*state)) +
              "\n\n\nQUESTION:\n";
  auto answers = Value::object();
  std::size_t input_tokens = 0;
  std::vector<float> logits(m.host.config.vocab);
  for (const auto& q : questions) {
    std::string body = plain(q.body);
    std::vector<Media> own;
    if (q.instructions.is_object()) {
      own.push_back(m.ReadMedia(q.instructions, budget));
      const auto at = body.find("\xee\x80\x80");
      if (at == std::string::npos)
        throw std::runtime_error("missing spoken-question slot");
      body.replace(at, 3, own.back().markup);
    }
    const std::string prompt =
        prefix + body + std::string(tokenization::kImEnd) + "\n" +
        std::string(tokenization::kImStart) + "assistant\n<think></think>";
    auto ids = m.tokenizer().Encode(prompt);
    if (ids.size() > m.context)
      throw std::invalid_argument(
          "decision prompt exceeds --context; prompts cannot be truncated or "
          "chunked");
    std::vector<float> embeddings;
    if (!media.empty() || !own.empty()) {
      embeddings.resize(ids.size() * m.host.config.hidden);
      for (std::size_t i = 0; i < ids.size(); ++i)
        lfm3::DecodeRow(
            m.host.embedding, ids[i],
            std::span(embeddings)
                .subspan(i * m.host.config.hidden, m.host.config.hidden));
      std::size_t offset = 0;
      const auto place = [&](const Media& item) {
        while (offset < ids.size() && ids[offset] != item.token)
          ++offset;
        const auto rows = item.rows.size() / m.host.config.hidden;
        if (rows > ids.size() - offset ||
            !std::all_of(ids.begin() + offset, ids.begin() + offset + rows,
                         [&](auto id) { return id == item.token; }))
          throw std::runtime_error(
              "media placeholder count differs from embeddings");
        std::copy(item.rows.begin(), item.rows.end(),
                  embeddings.begin() + offset * m.host.config.hidden);
        offset += rows;
      };
      for (const auto& item : media)
        place(item);
      for (const auto& item : own)
        place(item);
    }
    std::string error;
    if (!m.session->ForwardDecision(ids, embeddings, logits, &error))
      throw std::runtime_error(error);
    answers[q.name] = ReadAnswer(q, logits);
    input_tokens += ids.size();
  }
  auto result = Value::object();
  result["answers"] = answers;
  auto usage = Value::object();
  usage["input_tokens"] = input_tokens;
  usage["output_tokens"] = 0;
  result["usage"] = usage;
  return result;
}
}  // namespace gufo::models::d1
