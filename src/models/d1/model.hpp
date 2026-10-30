#ifndef GUFO_MODELS_D1_MODEL_HPP_
#define GUFO_MODELS_D1_MODEL_HPP_
#include <memory>

#include "src/core/json.hpp"
#include "src/models/lfm3/tokenizer.hpp"
namespace gufo::models::d1 {
struct Question {
  std::string name, type, body;
  json::Value instructions;
  std::vector<std::string> labels;
  std::vector<std::vector<std::uint32_t>> readout;
  float temperature{};
};
// Control-token spellings in caller text are broken with U+200B, as upstream.
std::string Plain(const lfm3::Tokenizer& tokenizer, std::string_view text,
                  unsigned vocab);
Question ParseQuestion(const lfm3::Tokenizer& tokenizer,
                       const std::string& name, const json::Value& question);
json::Value ReadAnswer(const Question& question, std::span<const float> logits);
class Model {
public:
  ~Model();
  static std::unique_ptr<Model> Load(const std::string& path,
                                     const std::string& mmproj,
                                     unsigned context = 8192);
  // Same request/response shape as the official GGUF /v1/systemone interface.
  // Media are data URLs or public HTTPS; filesystem paths are never read here.
  json::Value Decide(const json::Value& request);

private:
  Model();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace gufo::models::d1
#endif
