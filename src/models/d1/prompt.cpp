#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>

#include "src/models/d1/model.hpp"
namespace gufo::models::d1 {
std::string Plain(const lfm3::Tokenizer& tok, std::string_view input,
                  unsigned vocab) {
  std::string text(input);
  std::vector<std::string> specials;
  for (unsigned i = 0; i < vocab; ++i)
    if (tok.IsControl(i) && tok.Piece(i).size() > 1)
      specials.emplace_back(tok.Piece(i));
  std::sort(specials.begin(), specials.end(),
            [](const auto& a, const auto& b) { return a.size() > b.size(); });
  bool changed;
  do {
    changed = false;
    for (const auto& piece : specials) {
      std::size_t pos = 0;
      while ((pos = text.find(piece, pos)) != std::string::npos) {
        text.insert(pos + 1, "\xe2\x80\x8b");
        pos += 4;
        changed = true;
      }
    }
  } while (changed);
  return text;
}
Question ParseQuestion(const lfm3::Tokenizer& tok, const std::string& name,
                       const json::Value& value) {
  Question q;
  q.name = name;
  q.type = value.member_str("type");
  if (!value.is_object() ||
      (q.type != "noul" && q.type != "choice" && q.type != "score"))
    throw std::invalid_argument(
        "d1 GGUF questions support type noul, choice or score");
  const auto* ins = value.find("instructions");
  const auto* criteria = value.find("criteria");
  if (!ins || (!ins->is_string() && !ins->is_object()) ||
      (ins->is_string() && ins->str().empty()))
    throw std::invalid_argument(
        "question instructions must be text or input_audio");
  if (value.contains("multiple") &&
      (!value.find("multiple")->is_bool() || value.find("multiple")->as_bool()))
    throw std::invalid_argument(
        "multiple choice is not supported by the d1 GGUF interface");
  q.instructions = *ins;
  q.body = ins->is_string() ? ins->str() : "\xee\x80\x80";
  auto ids = [&](std::initializer_list<std::string> forms) {
    std::vector<std::uint32_t> out;
    for (const auto& form : forms) {
      auto enc = tok.Encode(form, false);
      if (enc.size() == 1 &&
          std::find(out.begin(), out.end(), enc[0]) == out.end())
        out.push_back(enc[0]);
    }
    if (out.empty())
      throw std::invalid_argument(
          "tokenizer has no single-token decision code");
    return out;
  };
  if (q.type == "noul") {
    if (criteria && !criteria->is_null()) {
      if (!criteria->is_object())
        throw std::invalid_argument("noul criteria must be an object");
      if (!criteria->empty()) {
        const auto describe = [&](const char* key) {
          const auto* v = criteria->find(key);
          return !v || v->is_null() ? std::string("None")
                 : v->is_string()   ? v->str()
                                    : v->tojson();
        };
        q.body += "\nYes: " + describe("true") + "\nNo: " + describe("false");
      }
    }
    q.body += "\n\nReply with yes or no only.";
    q.readout = {ids({"yes", "Yes", "YES"}), ids({"no", "No", "NO"})};
  } else if (q.type == "score") {
    if (!criteria || !criteria->is_array() || criteria->size() < 2 ||
        criteria->size() > 10)
      throw std::invalid_argument("score criteria must list 2 to 10 levels");
    q.body += "\n\n";
    for (std::size_t i = 0; i < criteria->size(); ++i) {
      const auto& level = criteria->items()[i];
      if (!level.is_string())
        throw std::invalid_argument("score levels must be strings");
      q.labels.push_back(level.str());
      const auto code = std::to_string(i);
      if (i)
        q.body += '\n';
      q.body += code + " " + level.str();
      q.readout.push_back(ids({code}));
    }
    q.body += "\n\nReply with a single digit 0-" +
              std::to_string(q.labels.size() - 1) + " only.";
  } else {
    if (!criteria || !criteria->is_object() || criteria->empty() ||
        criteria->size() > 255)
      throw std::invalid_argument("choice criteria must name 1 to 255 options");
    bool letters = true;
    for (const auto& [label, description] : criteria->members()) {
      if (!description.is_null() && !description.is_string())
        throw std::invalid_argument(
            "option descriptions must be strings or null");
      letters &= label.size() == 1 && ((label[0] >= 'A' && label[0] <= 'Z') ||
                                       (label[0] >= 'a' && label[0] <= 'z'));
    }
    q.body += "\n\nOptions:\n";
    std::unordered_set<std::uint32_t> used;
    std::vector<std::string> fallback;
    for (char c = 'A'; c <= 'Z'; ++c)
      fallback.emplace_back(1, c);
    for (int i = 0; i < 100; ++i)
      fallback.push_back((i < 10 ? "0" : "") + std::to_string(i));
    for (char c = 'a'; c <= 'z'; ++c)
      fallback.emplace_back(1, c);
    for (int i = 0; i < 200; ++i)
      fallback.push_back("#" + std::to_string(i));
    for (char a = 'A'; a <= 'Z'; ++a)
      for (char b = 'A'; b <= 'Z'; ++b)
        fallback.push_back(std::string(1, a) + b);
    std::size_t i = 0;
    for (const auto& [label, description] : criteria->members()) {
      std::string code = letters  ? label
                         : i < 26 ? std::string(1, char('A' + i))
                                  : std::string(1, char('A' + (i - 26) / 26)) +
                                        char('A' + (i - 26) % 26);
      auto take = [&](const std::string& candidate) {
        auto enc = tok.Encode(candidate, false);
        if (enc.size() != 1 || used.contains(enc[0]))
          return false;
        code = candidate;
        used.insert(enc[0]);
        return true;
      };
      if (!take(code)) {
        bool found = false;
        for (const auto& candidate : fallback)
          if (take(candidate)) {
            found = true;
            break;
          }
        if (!found)
          throw std::invalid_argument("no distinct single-token aliases");
      }
      std::string desc = description.is_string() ? description.str() : "";
      if (desc.empty()) {
        desc = label;
        std::replace(desc.begin(), desc.end(), '_', ' ');
      }
      if (i++)
        q.body += '\n';
      q.body += code + " " + desc;
      q.labels.push_back(label);
      q.readout.push_back(ids({code, " " + code}));
    }
    q.body += "\n\nReply with the option code only.";
  }
  return q;
}
json::Value ReadAnswer(const Question& q, std::span<const float> logits) {
  if (!std::isfinite(q.temperature) || q.temperature <= 0 || q.readout.empty())
    throw std::invalid_argument("invalid decision temperature or readout");
  std::vector<double> scores;
  for (const auto& group : q.readout) {
    float best = -std::numeric_limits<float>::infinity();
    for (auto id : group) {
      if (id >= logits.size() || !std::isfinite(logits[id]))
        throw std::runtime_error("invalid decision logits");
      best = std::max(best, logits[id]);
    }
    scores.push_back(best / double(q.temperature));
  }
  const double maximum = *std::max_element(scores.begin(), scores.end());
  double sum = 0;
  for (auto& score : scores) {
    score = std::exp(score - maximum);
    sum += score;
  }
  for (auto& score : scores)
    score /= sum;
  auto answer = json::Value::object();
  answer["type"] = q.type;
  if (q.type == "noul") {
    answer["noul"] = scores[0];
    return answer;
  }
  auto probs = json::Value::object();
  std::size_t selected =
      std::max_element(scores.begin(), scores.end()) - scores.begin();
  if (q.type == "score") {
    double cumulative = 0;
    for (std::size_t i = 0; i < scores.size(); ++i) {
      cumulative += scores[i];
      if (cumulative >= 0.5) {
        selected = i;
        break;
      }
    }
  }
  for (std::size_t i = 0; i < scores.size(); ++i)
    probs[q.type == "score" ? std::to_string(i) : q.labels[i]] = scores[i];
  answer[q.type] = q.type == "score" ? json::Value(selected)
                                     : json::Value(q.labels[selected]);
  answer["confidence"] = *std::max_element(scores.begin(), scores.end());
  answer["probabilities"] = probs;
  if (q.type == "score") {
    auto legend = json::Value::object();
    for (std::size_t i = 0; i < q.labels.size(); ++i)
      legend[std::to_string(i)] = q.labels[i];
    answer["legend"] = legend;
  }
  return answer;
}
}  // namespace gufo::models::d1
