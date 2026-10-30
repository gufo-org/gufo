#include <fstream>
#include <iostream>
#include <stdexcept>

#include "src/core/json.hpp"
#include "src/models/lfm3/engine.hpp"
#include "src/models/lfm3_vl/prompt.hpp"
namespace vl = gufo::models::lfm3_vl;
namespace l = gufo::models::lfm3;
using gufo::json::Value;
void Require(bool ok, const std::string& error) {
  if (!ok)
    throw std::runtime_error(error);
}
void Write(const std::string& path, std::span<const float> values) {
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(values.data()), values.size_bytes());
  Require(bool(f), "cannot write " + path);
}
int main(int argc, char** argv) {
  try {
    if (argc != 5)
      throw std::invalid_argument(
          "usage: lfm3_vl_probe MODEL MMPROJ REQUEST.json OUTPUT_PREFIX");
    std::ifstream f(argv[3]);
    auto request =
        gufo::json::parse(std::string(std::istreambuf_iterator<char>(f), {}));
    std::string error;
    auto model = l::Model::Load(argv[1], &error);
    Require(bool(model), error);
    auto vision = vl::Vision::Load(argv[2]);
    std::vector<gufo::tokenization::ChatMessage> messages;
    for (const auto& message : request.find("messages")->items()) {
      const auto role = message.member_str("role");
      messages.emplace_back(
          role == "system"      ? gufo::tokenization::ChatRole::kSystem
          : role == "assistant" ? gufo::tokenization::ChatRole::kAssistant
                                : gufo::tokenization::ChatRole::kUser,
          message.member_str("content"));
      if (const auto* images = message.find("images"))
        for (const auto& image : images->items())
          messages.back().images.push_back(
              {std::size_t(image.find("offset")
                               ? image.find("offset")->as_double()
                               : 0),
               std::make_shared<const std::vector<std::uint8_t>>(
                   gufo::core::ReadImageFile(image.member_str("path"))),
               image.member_str("detail") == "low"
                   ? gufo::tokenization::ChatMessage::ImageDetail::kLow
                   : gufo::tokenization::ChatMessage::ImageDetail::kAuto});
    }
    gufo::ReasoningOptions reasoning;
    reasoning.enabled =
        request.find("thinking") && request.find("thinking")->as_bool();
    auto prompt =
        vl::Prepare(model->tokenizer(), messages, {}, reasoning, 131072);
    auto session = model->CreateSession(prompt.tokens.size() + 128, &error);
    Require(bool(session), error);
    std::vector<float> logits(model->config().vocab);
    const std::string output = argv[4];
    std::size_t cursor = 0;
    auto report = Value::object();
    report["tokens"] = Value::array();
    for (auto token : prompt.tokens)
      report["tokens"].push_back(int(token));
    report["images"] = Value::array();
    for (std::size_t i = 0; i < prompt.images.size(); ++i) {
      const auto& image = prompt.images[i];
      Require(
          session->Forward(
              std::span(prompt.tokens).subspan(cursor, image.offset - cursor),
              {}, &error),
          error);
      auto rows = vision->Encode(image.picture);
      Require(session->ForwardEmbeddings(rows, false, {}, &error), error);
      const auto stem = output + ".image" + std::to_string(i);
      Write(stem + ".patches.bin", image.picture.patches);
      Write(stem + ".embeddings.bin", rows);
      auto grid = Value::object();
      grid["offset"] = image.offset;
      grid["height"] = image.picture.height;
      grid["width"] = image.picture.width;
      report["images"].push_back(std::move(grid));
      cursor = image.offset + rows.size() / model->config().hidden;
    }
    Require(session->Forward(std::span(prompt.tokens).subspan(cursor), logits,
                             &error),
            error);
    Write(output + ".logits.bin", logits);
    std::ofstream result(output + ".json");
    result << report.tojson();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
