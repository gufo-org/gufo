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
  enum class ToolFormat { kJson, kQwen, kDeepSeek };
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
  // Resolve a local JSON pointer using the same rules as schema compilation.
  // Throws invalid_argument for malformed or missing references.
  static const json::Value* ResolveReference(const json::Value& root,
                                             const json::Value& reference);
  // Returns null when a schema cannot be represented unambiguously by native
  // parameter tags. The caller retains the exact JSON schema in that case.
  // Non-strict tools permit open nested objects. Unsupported keywords retain
  // best-effort native framing without relaxing strict response schemas.
  // Required calls keep compact JSON for extended schemas; ordinary native
  // calls retain their existing prompt and continuation representation.
  static std::shared_ptr<const JsonConstraint> ToolParameters(
      const json::Value& schema, bool strict, ToolFormat format,
      bool required = false);
  // Best-effort native framing for non-strict tools with no declared argument
  // types. It preserves the model's native string/DSML typed-value semantics.
  static std::shared_ptr<const JsonConstraint> OpenToolParameters(
      ToolFormat format);
  static std::shared_ptr<const JsonConstraint> WithReasoning(
      std::shared_ptr<const JsonConstraint> answer);
  using Tool = std::pair<std::string, std::shared_ptr<const JsonConstraint>>;
  static std::shared_ptr<const JsonConstraint> WithTools(
      std::shared_ptr<const JsonConstraint> answer, std::vector<Tool> tools,
      bool required, bool parallel = false,
      ToolFormat format = ToolFormat::kJson);

  [[nodiscard]] State Start() const;
  [[nodiscard]] State Advance(const State& state, unsigned char byte) const;
  [[nodiscard]] bool Complete(const State& state) const;
  [[nodiscard]] const std::string& prompt() const { return prompt_; }

private:
  JsonConstraint() = default;
  friend class JsonConstraintCompiler;
  friend class ConstraintVocabulary;
  friend class SamplerState;
  friend struct TokenConstraint;
  State CanonicalMaskState(const State& state, std::size_t token_bytes) const;
  State Expand(State state) const;
  std::vector<Rule> rules_;
  std::vector<std::bitset<256>> classes_;
  std::vector<std::shared_ptr<const JsonSchemaLexeme>> lexemes_;
  std::uint32_t root_{0};
  std::string prompt_;
  bool stop_only_when_complete_{true};
  bool automatic_tools_{false};
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
  [[nodiscard]] bool Allows(const JsonConstraint& grammar,
                            const JsonConstraint::State& state,
                            std::uint32_t token) const;
  [[nodiscard]] std::size_t size() const { return pieces_.size(); }

private:
  friend struct TokenConstraint;
  struct Node {
    std::uint32_t child{UINT32_MAX};
    std::uint32_t sibling{UINT32_MAX};
    std::uint32_t token{UINT32_MAX};
    unsigned char byte{0};
  };
  std::vector<Node> trie_{1};
  std::vector<Piece> pieces_;
  std::vector<std::uint32_t> next_token_;
  std::size_t max_token_bytes_{0};
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
