#include <fstream>
#include <iostream>
#include <stdexcept>

#include "src/models/d1/model.hpp"
#include "src/models/lfm3/engine.hpp"
namespace d = gufo::models::d1;
namespace l = gufo::models::lfm3;
using gufo::json::Value;
Value Read(const char* path) {
  std::ifstream f(path);
  if (!f)
    throw std::runtime_error("cannot open fixture");
  return gufo::json::parse(std::string(std::istreambuf_iterator<char>(f), {}));
}
template<class T>
std::vector<T> Integers(const Value& value) {
  std::vector<T> out;
  for (const auto& item : value.items())
    out.push_back(T(item.as_double()));
  return out;
}
int main(int argc, char** argv) {
  try {
    if (argc != 3)
      throw std::invalid_argument("usage: d1_forward_probe MODEL PLAN.json");
    std::string error;
    auto model = l::Model::Load(argv[1], &error);
    if (!model)
      throw std::runtime_error(error);
    auto session = model->CreateSession(16384, &error);
    if (!session)
      throw std::runtime_error(error);
    const auto plan = Read(argv[2]);
    for (const auto& fixture : plan.items()) {
      std::vector<d::Question> questions;
      std::vector<std::string> suffixes;
      for (const auto& [name, value] : fixture.find("questions")->members()) {
        auto q = d::ParseQuestion(model->tokenizer(), name, value);
        suffixes.push_back(d::Suffix(
            model->tokenizer(), q,
            d::Plain(model->tokenizer(), q.body, model->config().vocab)));
        questions.push_back(std::move(q));
      }
      auto prompt = d::Pack(model->tokenizer(), fixture.member_str("prefix"),
                            fixture.member_str("header"), suffixes, questions);
      const auto tokens = Integers<std::uint32_t>(*fixture.find("tokens"));
      const auto groups = Integers<std::int32_t>(*fixture.find("groups"));
      const auto slots = Integers<std::uint32_t>(*fixture.find("slots"));
      const auto vocabulary =
          Integers<std::uint32_t>(*fixture.find("vocabulary"));
      if (prompt.tokens != tokens || prompt.groups != groups ||
          prompt.slots != slots)
        throw std::runtime_error("prompt/slot golden differs: " +
                                 fixture.member_str("name"));
      std::vector<float> embeddings;
      if (!fixture.member_str("embeddings").empty()) {
        embeddings.resize(tokens.size() * model->config().hidden);
        std::ifstream file(fixture.member_str("embeddings"), std::ios::binary);
        if (!file.read(reinterpret_cast<char*>(embeddings.data()),
                       embeddings.size() * 4))
          throw std::runtime_error("cannot read fixture embeddings");
      }
      std::vector<float> logits(slots.size() * vocabulary.size());
      if (!session->ForwardDecision(tokens, embeddings, groups, slots,
                                    vocabulary, logits, &error))
        throw std::runtime_error(error);
      std::ofstream output(fixture.member_str("output"), std::ios::binary);
      output.write(reinterpret_cast<const char*>(logits.data()),
                   logits.size() * 4);
      if (!output)
        throw std::runtime_error("cannot write logits");
      const auto position = session->Position();
      auto bad = groups;
      bad[0] = 0;
      if (session->ForwardDecision(tokens, embeddings, bad, slots, vocabulary,
                                   logits, &error) ||
          session->Position() != position)
        throw std::runtime_error("invalid group mutated the session");
      auto bad_slots = slots;
      bad_slots.back() = tokens.size();
      if (session->ForwardDecision(tokens, embeddings, groups, bad_slots,
                                   vocabulary, logits, &error) ||
          session->Position() != position)
        throw std::runtime_error("invalid answer slot mutated the session");
      std::cout << "PASS " << fixture.member_str("name")
                << " tokens=" << tokens.size() << " slots=" << slots.size()
                << '\n';
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
