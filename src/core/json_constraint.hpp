#ifndef GUFO_CORE_JSON_CONSTRAINT_HPP_
#define GUFO_CORE_JSON_CONSTRAINT_HPP_

#include <bitset>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
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
  // Grammars built into core. kQwen and kDeepSeek are the descriptors that
  // Syntax() returns; a provider whose call syntax is none of these describes
  // it with ToolSyntax instead of adding an entry here.
  enum class ToolFormat { kJson, kQwen, kDeepSeek };
  // How JSON-shaped native arguments spell JSON values. The defaults are JSON;
  // schema semantics (types, bounds, finite values, required and closed
  // properties) are the same in every spelling. Numbers, true, false, null,
  // braces, brackets, ':' and ',' keep their JSON spelling.
  struct ValueSyntax {
    // Both empty: JSON strings with escapes. Both set: a string is raw UTF-8
    // text between these delimiters, without escapes, which therefore cannot
    // contain string_close.
    std::string string_open;
    std::string string_close;
    // Object keys are written unquoted, ending at ':', when they are
    // nonempty, hold none of :{}[], have no surrounding whitespace and do not
    // begin like a string. Other keys keep the quoted string form.
    bool bare_keys{false};
    // JSON's insignificant whitespace between tokens. Without it, values are
    // compact.
    bool whitespace{true};
    // Object members in sorted key order instead of schema order.
    bool sorted_keys{false};
    auto operator<=>(const ValueSyntax&) const = default;
  };
  // Arguments as one tagged parameter per top-level property, in any order:
  //   parameter_open KEY raw_value_open TEXT value_close parameter_separator
  //   parameter_open KEY typed_value_open JSON value_close parameter_separator
  // TEXT is raw and ends at the first value_close. When both openers are
  // equal the tag carries no type (as Qwen's): a parameter admitting strings
  // is raw text and the parser decides types from the schema; undeclared
  // parameters are then not admitted. Otherwise (as DeepSeek's string="true"
  // and string="false") raw and JSON values are both admitted where the
  // schema allows them. A key ends at the first byte of raw_value_open, which
  // typed_value_open must share.
  struct TaggedArguments {
    std::string parameter_open;
    std::string raw_value_open;
    std::string typed_value_open;
    std::string value_close;
    std::string parameter_separator;
    auto operator<=>(const TaggedArguments&) const = default;
  };
  // A native call written as
  //   open name_prefix NAME name_suffix ARGUMENTS close
  // ARGUMENTS is either `tagged` parameters or, without them, the tool's
  // parameter object spelled by `values`.
  //
  // JSON-shaped arguments: each parallel call repeats `open`, ordinary text may
  // precede a call and, without a response format, follow it, and a required
  // call may only be preceded by llama.cpp's bounded SPACE_RULE.
  //
  // Tagged arguments follow llama.cpp's native parsers: calls end the output
  // and a required call may be preceded by whitespace. With an empty
  // block_close each parallel call repeats `open` after optional SPACE_RULE
  // whitespace, which may also follow the last call. Otherwise parallel calls
  // share one block that block_close ends.
  struct ToolSyntax {
    std::string open;
    std::string name_prefix;
    std::string name_suffix;
    std::string close;
    std::string block_close;
    // Bytes that end a function name for the parser; a tool whose name holds
    // one cannot be constrained (std::invalid_argument).
    std::string name_delimiters;
    ValueSyntax values;
    std::optional<TaggedArguments> tagged;
    // A complete call header (`open` + `name_prefix`) written inside the
    // reasoning phase ends it, as for the Qwen and DeepSeek grammars.
    bool reasoning_header{false};
    auto operator<=>(const ToolSyntax&) const = default;
  };
  // The descriptor of a built-in native format; kJson has none
  // (std::invalid_argument). A provider variant can copy one and change its
  // tag text.
  static const ToolSyntax& Syntax(ToolFormat format);
  struct Stack {
    std::vector<std::uint32_t> symbols;
    std::string lexeme;
    // Wide native argument sets retain only the subset chosen by this request.
    // Sampler copies and speculative rollback copy this state with the stack.
    std::vector<std::uint64_t> used_parameters;
    auto operator<=>(const Stack&) const = default;
  };
  using State = std::vector<Stack>;
  using Sequence = std::vector<std::uint32_t>;
  using Rule = std::vector<Sequence>;

  // Response formats pass open_objects: as in llama.cpp, an object then admits
  // extra keys after its declared properties when additionalProperties is
  // true or a schema, or when it declares no properties. Omitting
  // additionalProperties beside properties still closes the object. Tool
  // admission and argument recovery keep closed objects.
  static std::shared_ptr<const JsonConstraint> Compile(
      const json::Value& schema, bool strict, bool open_objects = false);
  static std::shared_ptr<const JsonConstraint> Object();
  // Resolve a local JSON pointer using the same rules as schema compilation.
  // Throws invalid_argument for malformed or missing references.
  static const json::Value* ResolveReference(const json::Value& root,
                                             const json::Value& reference);
  // Native models always retain native parameter tags. Unsupported tool
  // schemas use best-effort arguments without relaxing response-format schemas.
  // Tool choice changes the call envelope, not this cached argument grammar.
  static std::shared_ptr<const JsonConstraint> ToolParameters(
      const json::Value& schema, bool strict, ToolFormat format);
  // Best-effort native framing for non-strict tools with no declared argument
  // types. It preserves the model's native string/DSML typed-value semantics.
  static std::shared_ptr<const JsonConstraint> OpenToolParameters(
      ToolFormat format);
  // The same tool argument contract in a provider-described syntax; the
  // ToolFormat overloads are these for Syntax(format). Tagged arguments keep
  // the native routes above. JSON-shaped arguments compile the parameter
  // object in `syntax.values`: a schema this cannot enforce is
  // std::invalid_argument when strict; otherwise its supported parts are
  // kept, and a schema with no usable object becomes any object in that
  // syntax.
  static std::shared_ptr<const JsonConstraint> ToolParameters(
      const json::Value& schema, bool strict, const ToolSyntax& syntax);
  static std::shared_ptr<const JsonConstraint> OpenToolParameters(
      const ToolSyntax& syntax);
  // Prefixes `answer` with a reasoning phase that ends at `close` (or, for a
  // grammar with `reasoning_header` calls, at a complete call header). With an
  // empty `open` the phase is already open, as when the prompt opened it. A
  // non-empty `open` is the provider's complete opener (it may span tokens and
  // include whitespace): the output either starts with it and reasons, or
  // starts with the answer. An empty `close` is std::invalid_argument.
  static std::shared_ptr<const JsonConstraint> WithReasoning(
      std::shared_ptr<const JsonConstraint> answer,
      std::string_view close = "</think>", std::string_view open = {});
  using Tool = std::pair<std::string, std::shared_ptr<const JsonConstraint>>;
  static std::shared_ptr<const JsonConstraint> WithTools(
      std::shared_ptr<const JsonConstraint> answer, std::vector<Tool> tools,
      bool required, bool parallel = false,
      ToolFormat format = ToolFormat::kJson);
  // Tool choice for a provider-described call syntax. Argument grammars
  // normally come from ToolParameters with the same syntax.
  static std::shared_ptr<const JsonConstraint> WithTools(
      std::shared_ptr<const JsonConstraint> answer, std::vector<Tool> tools,
      bool required, bool parallel, const ToolSyntax& syntax);

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
  static std::shared_ptr<const JsonConstraint> BuildTools(
      std::shared_ptr<const JsonConstraint> answer, std::vector<Tool> tools,
      bool required, bool parallel, const ToolSyntax* syntax);
  State CanonicalMaskState(const State& state, std::size_t token_bytes) const;
  State Expand(State state) const;
  // Whether `state` is still in the WithReasoning phase, or its opener.
  bool InReasoning(const State& state) const;
  std::vector<Rule> rules_;
  std::vector<std::bitset<256>> classes_;
  std::vector<std::shared_ptr<const JsonSchemaLexeme>> lexemes_;
  // Only outer native parameter lists use these sets; their values remain
  // ordinary JSON/string rules, so sets never nest within a request stack.
  struct ArgumentSet {
    Sequence members;
    std::vector<std::uint64_t> required;
  };
  std::vector<ArgumentSet> argument_sets_;
  std::uint32_t root_{0};
  std::string prompt_;
  bool stop_only_when_complete_{true};
  bool automatic_tools_{false};
  std::string reasoning_tool_prefix_;
  std::uint32_t reasoning_tool_root_{0};
  // WithReasoning's rules and lexemes are numbered from these on.
  std::uint32_t reasoning_rules_{UINT32_MAX};
  std::uint32_t reasoning_lexemes_{UINT32_MAX};
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
  std::vector<std::uint32_t> reasoning_boundary_tokens_;
  std::size_t max_token_bytes_{0};
};

struct TokenConstraint {
  std::shared_ptr<const JsonConstraint> grammar;
  std::shared_ptr<const ConstraintVocabulary> vocabulary;
  [[nodiscard]] std::shared_ptr<const std::vector<std::uint8_t>> Allowed(
      const JsonConstraint::State& state) const;

private:
  mutable std::mutex mutex_;
  mutable JsonConstraint::State initial_state_;
  mutable std::map<JsonConstraint::State,
                   std::shared_ptr<const std::vector<std::uint8_t>>>
      masks_;
};

}  // namespace gufo::sampling

#endif
