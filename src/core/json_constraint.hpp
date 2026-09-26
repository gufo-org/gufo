#ifndef GUFO_CORE_JSON_CONSTRAINT_HPP_
#define GUFO_CORE_JSON_CONSTRAINT_HPP_

#include <bitset>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include "src/core/json.hpp"
#include "src/core/json_schema_lexeme.hpp"

namespace gufo::sampling {

// Byte grammar: token boundaries may split UTF-8 characters and JSON escapes.
// Programs and vocabulary tries are immutable and shared; stacks belong to a
// request and are copied along with its sampler during speculative
// verification.
class JsonConstraint {
public:
  struct Stack {
    std::vector<std::uint32_t> symbols;
    std::string lexeme;
    auto operator<=>(const Stack&) const = default;
  };
  using State = std::vector<Stack>;
  using Sequence = std::vector<std::uint32_t>;
  using Rule = std::vector<Sequence>;

  static std::shared_ptr<const JsonConstraint> Compile(
      const json::Value& schema, bool strict);
  static std::shared_ptr<const JsonConstraint> Object();
  static std::shared_ptr<const JsonConstraint> WithReasoning(
      std::shared_ptr<const JsonConstraint> answer);
  using Tool = std::pair<std::string, std::shared_ptr<const JsonConstraint>>;
  static std::shared_ptr<const JsonConstraint> WithTools(
      std::shared_ptr<const JsonConstraint> answer, std::vector<Tool> tools,
      bool required);

  [[nodiscard]] State Start() const;
  [[nodiscard]] State Advance(const State& state, unsigned char byte) const;
  [[nodiscard]] bool Complete(const State& state) const;
  [[nodiscard]] const std::string& prompt() const { return prompt_; }

private:
  JsonConstraint() = default;
  friend class JsonConstraintCompiler;
  State Expand(State state) const;
  std::vector<Rule> rules_;
  std::vector<std::bitset<256>> classes_;
  std::vector<std::shared_ptr<const JsonSchemaLexeme>> lexemes_;
  std::uint32_t root_{0};
  std::string prompt_;
};

class ConstraintVocabulary {
public:
  struct Piece {
    std::string text;
    bool stop{false};
  };
  using Reader = std::function<Piece(std::uint32_t)>;
  explicit ConstraintVocabulary(std::uint32_t size, const Reader& reader);

  [[nodiscard]] std::vector<std::uint8_t> Allowed(
      const JsonConstraint& grammar, const JsonConstraint::State& state) const;
  [[nodiscard]] JsonConstraint::State Accept(const JsonConstraint& grammar,
                                             const JsonConstraint::State& state,
                                             std::uint32_t token) const;
  [[nodiscard]] std::size_t size() const { return pieces_.size(); }

private:
  struct Edge {
    std::uint32_t child;
    unsigned char byte;
  };
  struct Node {
    std::vector<Edge> edges;
    std::vector<std::uint32_t> tokens;
  };
  std::vector<Node> trie_{1};
  std::vector<Piece> pieces_;
};

struct TokenConstraint {
  std::shared_ptr<const JsonConstraint> grammar;
  std::shared_ptr<const ConstraintVocabulary> vocabulary;
  [[nodiscard]] std::shared_ptr<const std::vector<std::uint8_t>> Allowed(
      const JsonConstraint::State& state) const;

private:
  mutable std::mutex mutex_;
  mutable std::map<JsonConstraint::State,
                   std::shared_ptr<const std::vector<std::uint8_t>>>
      masks_;
};

}  // namespace gufo::sampling

#endif
