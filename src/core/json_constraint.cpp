#include "src/core/json_constraint.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <deque>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string_view>
#include <tuple>

namespace gufo::sampling {
namespace {
constexpr std::uint32_t kTerminal = 1U << 31;
constexpr std::uint32_t kLexeme = 1U << 30;
constexpr std::uint32_t kLeaf = kTerminal | kLexeme;
constexpr std::size_t kMaxSchemaBytes = 2 * 1024 * 1024;
constexpr std::size_t kMaxRules = 262144;
constexpr std::size_t kMaxDepth = 16;
constexpr std::size_t kMaxStates = 8192;
constexpr std::size_t kMaxStack = 16384;
constexpr std::size_t kMaxWork = 2000000;

[[noreturn]] void Invalid(std::string_view message) {
  throw std::invalid_argument("JSON Schema: " + std::string(message));
}
}  // namespace

class JsonConstraintCompiler {
public:
  using Sequence = JsonConstraint::Sequence;
  using Rule = JsonConstraint::Rule;
  explicit JsonConstraintCompiler(const json::Value& schema, bool strict)
      : schema_(schema), strict_(strict) {
    // Literal bytes occupy the first 256 terminal classes.
    for (unsigned i = 0; i < 256; ++i) {
      std::bitset<256> bits;
      bits.set(i);
      grammar_->classes_.push_back(bits);
    }
    ws_ = Optional(Lexeme(JsonSchemaLexeme::Whitespace()));
    const auto digits = Class("0123456789");
    const auto hex = Class("0123456789abcdefABCDEF");
    const auto hex_tail = Seq({hex, hex});
    const auto unicode = Alt({
        Seq({Class("0123456789abcefABCEF"), hex, hex, hex}),
        Seq({Class("dD"), Class("01234567"), hex_tail}),
        Seq({Class("dD"), Class("89abAB"), hex_tail, Literal("\\u"),
             Class("dD"), Class("cdefCDEF"), hex_tail}),
    });
    const auto continuation = Range(0x80, 0xbf);
    const auto character = Alt({
        Range(0x20, 0x21),
        Range(0x23, 0x5b),
        Range(0x5d, 0x7f),
        Seq({Range(0xc2, 0xdf), continuation}),
        Seq({Byte(0xe0), Range(0xa0, 0xbf), continuation}),
        Seq({Range(0xe1, 0xec), continuation, continuation}),
        Seq({Byte(0xed), Range(0x80, 0x9f), continuation}),
        Seq({Range(0xee, 0xef), continuation, continuation}),
        Seq({Byte(0xf0), Range(0x90, 0xbf), continuation, continuation}),
        Seq({Range(0xf1, 0xf3), continuation, continuation, continuation}),
        Seq({Byte(0xf4), Range(0x80, 0x8f), continuation, continuation}),
        Seq({Byte('\\'), Class("\"\\/bfnrt")}),
        Seq({Literal("\\u"), unicode}),
    });
    string_ = Seq({Byte('"'), Repeat(character), Byte('"')});
    const auto positive =
        Alt({Byte('0'), Seq({Range('1', '9'), Repeat(digits)})});
    integer_ = Seq({Optional(Byte('-')), positive});
    const auto fraction = Seq({Byte('.'), digits, Repeat(digits)});
    const auto exponent =
        Seq({Class("eE"), Optional(Class("+-")), digits, Repeat(digits)});
    number_ = Seq({integer_, Optional(fraction), Optional(exponent)});
    bool_ = Alt({Literal("true"), Literal("false")});
    null_ = Literal("null");
  }

  std::shared_ptr<const JsonConstraint> Compile(bool object_only) {
    if (object_only) {
      grammar_->root_ = Seq({ws_, GenericObject(kMaxDepth), ws_});
      grammar_->prompt_ = "Respond with a single valid JSON object.";
    } else {
      const auto* root = &schema_;
      std::set<const json::Value*> roots;
      while (const auto* reference = root->find("$ref")) {
        if (!roots.insert(root).second)
          Invalid("root reference cycle");
        root = Reference(*reference);
      }
      if (!root->is_object() || root->member_str("type") != "object" ||
          root->contains("anyOf"))
        Invalid("the root must have type object");
      grammar_->root_ = Seq({ws_, Visit(schema_, 0), ws_});
      grammar_->prompt_ =
          "Respond with a single JSON object matching this JSON Schema:\n" +
          schema_.dump();
    }
    // A recursive schema must have a finite witness. References that recurse
    // without first consuming input are invalid for a predictive grammar.
    std::vector<bool> productive(grammar_->rules_.size()), nullable(productive);
    bool changed = true;
    while (changed) {
      changed = false;
      for (std::size_t id = 0; id < grammar_->rules_.size(); ++id) {
        for (const auto& sequence : grammar_->rules_[id]) {
          const bool p = std::ranges::all_of(sequence, [&](auto symbol) {
            return (symbol & kLeaf) || productive[symbol];
          });
          const bool n = std::ranges::all_of(sequence, [&](auto symbol) {
            return !(symbol & kLeaf) && nullable[symbol];
          });
          changed |= (p && !productive[id]) || (n && !nullable[id]);
          productive[id] = productive[id] || p;
          nullable[id] = nullable[id] || n;
        }
      }
    }
    std::vector<unsigned char> visited(grammar_->rules_.size());
    auto check = [&](auto&& self, std::uint32_t id) -> void {
      if (visited[id] == 1)
        Invalid("reference cycle does not consume input");
      if (visited[id] == 2)
        return;
      visited[id] = 1;
      for (const auto& sequence : grammar_->rules_[id]) {
        for (const auto symbol : sequence) {
          if (symbol & kLeaf)
            break;
          self(self, symbol);
          if (!nullable[symbol])
            break;
        }
      }
      visited[id] = 2;
    };
    for (std::size_t id = 0; id < grammar_->rules_.size(); ++id)
      check(check, id);
    if (!productive[grammar_->root_])
      throw JsonSchemaEmpty("JSON Schema: schema has no finite value");
    // Discard impossible alternatives before they can admit dead prefixes.
    for (auto& rule : grammar_->rules_)
      std::erase_if(rule, [&](const auto& sequence) {
        return std::ranges::any_of(sequence, [&](auto symbol) {
          return !(symbol & kLeaf) && !productive[symbol];
        });
      });
    (void)grammar_->Start();
    return std::move(grammar_);
  }

  std::shared_ptr<const JsonConstraint> ToolParameters(
      JsonConstraint::ToolFormat format) {
    // Native envelope/typed-parameter approach: llama.cpp common/parsers/
    // qwen3-coder.cpp and deepseek.cpp at
    // 6a2743f028f78bfb88a7189607b49bde30df3769. Unlike an injected JSON
    // envelope, this retains the syntax already taught by each model's chat
    // template.
    using Format = JsonConstraint::ToolFormat;
    const auto* root = &schema_;
    while (const auto* ref = root->find("$ref")) {
      if (!Without(*root, {"$ref", "$defs", "title", "description"}).empty())
        return {};
      root = Reference(*ref);
    }
    // JSON remains the exact fallback for object-level unions/finite values
    // and schemas whose raw string/non-string alternatives are ambiguous.
    if (root->member_str("type") != "object" || root->contains("anyOf") ||
        root->contains("const") || root->contains("enum"))
      return {};
    const auto* properties = root->find("properties");
    if (!properties)
      return {};
    std::set<std::string> required;
    if (const auto* fields = root->find("required"))
      for (const auto& field : fields->items())
        required.insert(field.str());
    std::vector<std::pair<std::string, std::uint32_t>> members;
    for (const auto& [name, original] : properties->members()) {
      if (name.empty() || name.find_first_of("<>\"=\r\n") != std::string::npos)
        return {};
      const auto* schema = &original;
      while (const auto* ref = schema->find("$ref")) {
        if (!Without(*schema, {"$ref", "title", "description"}).empty())
          return {};
        schema = Reference(*ref);
      }
      const auto* type = schema->find("type");
      if (!type || !type->is_string() || schema->contains("anyOf"))
        return {};
      const bool string = type->str() == "string";
      const std::string close =
          format == Format::kQwen ? "\n</parameter>" : "</｜DSML｜parameter>";
      std::uint32_t value;
      if (string) {
        Sequence choices;
        if (schema->contains("const") || schema->contains("enum")) {
          auto values = schema->contains("enum")
                            ? schema->find("enum")->items()
                            : json::Value::Array{*schema->find("const")};
          for (const auto& item : values) {
            if (item.str().find(close) != std::string::npos)
              return {};
            if (ValueFor(*schema, item))
              choices.push_back(Literal(item.str()));
          }
          value = Alt(choices);
        } else {
          // A regex may require the raw parameter delimiter itself. Its
          // intersection with native framing could then be empty even though
          // the JSON schema is satisfiable. Preserve the JSON representation.
          if (schema->contains("pattern"))
            return {};
          const auto matcher = JsonSchemaLexeme::RawString(*schema, close);
          value = Lexeme(matcher);
          if (matcher->Check("").complete)
            value = Optional(value);
        }
      } else {
        value = Visit(original, 1);
      }
      const auto open = format == Format::kQwen
                            ? "<parameter=" + name + ">\n"
                            : "<｜DSML｜parameter name=\"" + name +
                                  "\" string=\"" + (string ? "true" : "false") +
                                  "\">";
      members.emplace_back(name,
                           Seq({Literal(open), value, Literal(close + "\n")}));
    }
    // Native Qwen may choose argument order. Share subset suffixes for small
    // objects; bound compilation for large schemas by retaining schema order.
    std::uint32_t body;
    if (members.size() <= 10) {
      std::map<unsigned, std::uint32_t> suffixes;
      auto suffix = [&](auto&& self, unsigned used) -> std::uint32_t {
        if (auto found = suffixes.find(used); found != suffixes.end())
          return found->second;
        Rule alternatives;
        bool complete = true;
        for (unsigned i = 0; i < members.size(); ++i) {
          if (used & (1U << i))
            continue;
          complete &= !required.contains(members[i].first);
          alternatives.push_back(
              {members[i].second, self(self, used | (1U << i))});
        }
        if (complete)
          alternatives.push_back({});
        return suffixes[used] = New(std::move(alternatives));
      };
      body = suffix(suffix, 0);
    } else {
      Sequence ordered;
      for (const auto& [name, rule] : members)
        ordered.push_back(required.contains(name) ? rule : Optional(rule));
      body = Seq(std::move(ordered));
    }
    grammar_->root_ = body;
    (void)grammar_->Start();
    return std::move(grammar_);
  }

private:
  static std::uint32_t Byte(unsigned char byte) { return kTerminal | byte; }
  std::uint32_t New(Rule rule = {}) {
    if (grammar_->rules_.size() >= kMaxRules)
      Invalid("compiled grammar exceeds the rule limit");
    grammar_->rules_.push_back(std::move(rule));
    return static_cast<std::uint32_t>(grammar_->rules_.size() - 1);
  }
  std::uint32_t Seq(Sequence sequence) { return New({std::move(sequence)}); }
  std::uint32_t Alt(const Sequence& alternatives) {
    Rule rule;
    for (auto id : alternatives)
      rule.push_back({id});
    return New(std::move(rule));
  }
  std::uint32_t Class(std::string_view characters) {
    std::bitset<256> bits;
    for (unsigned char byte : characters)
      bits.set(byte);
    grammar_->classes_.push_back(bits);
    return kTerminal | (grammar_->classes_.size() - 1);
  }
  std::uint32_t Range(unsigned first, unsigned last) {
    std::string bytes;
    for (unsigned c = first; c <= last; ++c)
      bytes.push_back(static_cast<char>(c));
    return Class(bytes);
  }
  std::uint32_t Literal(std::string_view value) {
    Sequence chunks;
    for (std::size_t offset = 0; offset < value.size(); offset += 64) {
      Sequence bytes;
      for (unsigned char byte : value.substr(offset, 64))
        bytes.push_back(Byte(byte));
      chunks.push_back(Seq(std::move(bytes)));
    }
    return Seq(std::move(chunks));
  }
  std::uint32_t Optional(std::uint32_t rule) { return New({{}, {rule}}); }
  std::uint32_t Repeat(std::uint32_t rule) {
    const auto id = New();
    grammar_->rules_[id] = {{}, {rule, id}};
    return id;
  }
  std::uint32_t Exact(std::uint32_t item, std::size_t count) {
    Sequence parts;
    while (count) {
      if (count & 1)
        parts.push_back(item);
      count >>= 1;
      if (count)
        item = Seq({item, item});
    }
    return Seq(std::move(parts));
  }
  std::uint32_t AtMost(std::uint32_t item, std::size_t count) {
    if (!count)
      return Seq({});
    if (count == 1)
      return Optional(item);
    // All counts 0..N: pairs cover the even counts; a final item covers the
    // odd counts. For even N the longest pair sequence must have no extra item.
    if (count & 1)
      return Seq({AtMost(Seq({item, item}), count / 2), Optional(item)});
    return Alt({AtMost(item, count - 1), Exact(item, count)});
  }
  std::uint32_t GenericValue(std::size_t depth) {
    if (generic_values_.contains(depth))
      return generic_values_.at(depth);
    Sequence alternatives{string_, number_, bool_, null_};
    if (depth) {
      const auto value = GenericValue(depth - 1);
      const auto tail = Repeat(Seq({ws_, Byte(','), ws_, value}));
      alternatives.push_back(
          Seq({Byte('['), ws_, Optional(Seq({value, tail})), ws_, Byte(']')}));
      alternatives.push_back(GenericObject(depth));
    }
    return generic_values_[depth] = Alt(alternatives);
  }
  std::uint32_t GenericObject(std::size_t depth) {
    const auto value = GenericValue(depth - 1);
    const auto member = Seq({string_, ws_, Byte(':'), ws_, value});
    const auto members =
        Seq({member, Repeat(Seq({ws_, Byte(','), ws_, member}))});
    return Seq({Byte('{'), ws_, Optional(members), ws_, Byte('}')});
  }
  static bool MatchesType(const json::Value& value, std::string_view type) {
    if (type == "object")
      return value.is_object();
    if (type == "array")
      return value.is_array();
    if (type == "string")
      return value.is_string();
    if (type == "boolean")
      return value.is_bool();
    if (type == "null")
      return value.is_null();
    if (type == "number")
      return value.is_number();
    if (type == "integer")
      return value.is_number() &&
             std::floor(value.as_double()) == value.as_double();
    return false;
  }
  static bool EqualValue(const json::Value& a, const json::Value& b) {
    if (a.is_number() && b.is_number())
      return a.as_double() == b.as_double();
    if (a.is_object() && b.is_object()) {
      if (a.size() != b.size())
        return false;
      return std::ranges::all_of(a.members(), [&](const auto& member) {
        const auto* other = b.find(member.first);
        return other && EqualValue(member.second, *other);
      });
    }
    if (a.is_array() && b.is_array())
      return a.size() == b.size() &&
             std::equal(a.items().begin(), a.items().end(), b.items().begin(),
                        EqualValue);
    return a.dump() == b.dump();
  }
  const json::Value& Store(json::Value schema) {
    if (derived_.size() >= kMaxRules)
      Invalid("schema expansion exceeds its resource budget");
    derived_.push_back(std::move(schema));
    return derived_.back();
  }
  static json::Value Without(const json::Value& schema,
                             std::initializer_list<std::string_view> keys) {
    auto result = json::Value::object();
    for (const auto& [key, value] : schema.members())
      if (std::ranges::find(keys, key) == keys.end())
        result.append_member(key, value);
    return result;
  }
  json::Value Conjoin(const json::Value& left, const json::Value& right,
                      unsigned depth = 0) {
    if (depth > 64)
      Invalid("schema intersection exceeds its reference budget");
    Keys(left);
    Keys(right);
    if (const auto* ref = left.find("$ref"))
      return Conjoin(Conjoin(*Reference(*ref), Without(left, {"$ref", "$defs"}),
                             depth + 1),
                     right, depth + 1);
    if (const auto* ref = right.find("$ref"))
      return Conjoin(left,
                     Conjoin(*Reference(*ref),
                             Without(right, {"$ref", "$defs"}), depth + 1),
                     depth + 1);
    if (const auto* any = left.find("anyOf")) {
      auto result = json::Value::object();
      auto branches = json::Value::array();
      const auto siblings = Conjoin(Without(left, {"anyOf"}), right, depth + 1);
      for (const auto& branch : any->items()) {
        try {
          branches.push_back(Conjoin(branch, siblings, depth + 1));
        } catch (const JsonSchemaEmpty&) {
        }
      }
      if (branches.empty())
        throw JsonSchemaEmpty("JSON Schema: schema intersection is empty");
      result["anyOf"] = std::move(branches);
      return result;
    }
    if (right.contains("anyOf"))
      return Conjoin(right, left, depth + 1);
    if (const auto* format = right.find("format");
        format && left.contains("format") &&
        !EqualValue(*left.find("format"), *format)) {
      if (!format->is_string())
        Invalid("format must be a string");
      return Conjoin(
          left,
          Conjoin(Without(right, {"format"}),
                  JsonSchemaLexeme::Format(format->str()), depth + 1),
          depth + 1);
    }
    auto result = left;
    for (const auto& [key, value] : right.members()) {
      const auto* previous = left.find(key);
      if (key == "title" || key == "description" || !previous) {
        result[key] = value;
        continue;
      }
      if (EqualValue(*previous, value))
        continue;
      if (key == "minimum" || key == "exclusiveMinimum" || key == "minLength" ||
          key == "minItems" || key == "maximum" || key == "exclusiveMaximum" ||
          key == "maxLength" || key == "maxItems") {
        if (!previous->is_number() || !value.is_number())
          Invalid("bounds must be numbers");
        const bool lower = key.starts_with("min") || key == "exclusiveMinimum";
        result[key] = lower
                          ? std::max(previous->as_double(), value.as_double())
                          : std::min(previous->as_double(), value.as_double());
      } else if (key == "multipleOf") {
        result[key] = JsonSchemaLexeme::IntersectMultipleOf(*previous, value);
      } else if (key == "pattern") {
        if (!previous->is_string() || !value.is_string())
          Invalid("pattern must be a string");
        result[key] = "(?=[\\s\\S]*(?:" + previous->str() +
                      "))(?=[\\s\\S]*(?:" + value.str() + "))";
      } else if (key == "required") {
        if (!previous->is_array() || !value.is_array())
          Invalid("required must be an array");
        for (const auto& field : value.items())
          if (std::ranges::none_of(previous->items(), [&](const auto& old) {
                return EqualValue(old, field);
              }))
            result[key].push_back(field);
      } else if (key == "enum") {
        if (!previous->is_array() || !value.is_array())
          Invalid("enum must be an array");
        result[key] = json::Value::array();
        for (const auto& item : previous->items())
          if (std::ranges::any_of(value.items(), [&](const auto& other) {
                return EqualValue(item, other);
              }))
            result[key].push_back(item);
        if (result[key].empty())
          throw JsonSchemaEmpty("JSON Schema: enum intersection is empty");
      } else if (key == "type") {
        auto types = [](const json::Value& t) {
          return t.is_array() ? t.items() : json::Value::Array{t};
        };
        auto common = json::Value::array();
        for (const auto& a : types(*previous))
          for (const auto& b : types(value))
            if (EqualValue(a, b))
              common.push_back(a);
            else if ((a.str() == "number" && b.str() == "integer") ||
                     (a.str() == "integer" && b.str() == "number"))
              common.push_back("integer");
        if (common.empty())
          throw JsonSchemaEmpty(
              "JSON Schema: schema constraints have no common type");
        result[key] = common.size() == 1 ? common.items()[0] : common;
      } else if (key == "items") {
        result[key] = Conjoin(*previous, value, depth + 1);
      } else if (key == "properties") {
        if (!previous->is_object() || !value.is_object())
          Invalid("properties must be an object");
        const bool left_closed = left.contains("additionalProperties") &&
                                 !left.find("additionalProperties")->as_bool();
        const bool right_closed =
            right.contains("additionalProperties") &&
            !right.find("additionalProperties")->as_bool();
        auto properties = json::Value::object();
        for (const auto& [name, child] : previous->members()) {
          if (const auto* other = value.find(name))
            properties.append_member(name, Conjoin(child, *other, depth + 1));
          else if (!right_closed)
            properties.append_member(name, child);
        }
        if (!left_closed)
          for (const auto& [name, child] : value.members())
            if (!previous->contains(name))
              properties.append_member(name, child);
        result[key] = std::move(properties);
      } else {
        if (key == "const")
          throw JsonSchemaEmpty("JSON Schema: const intersection is empty");
        Invalid("incompatible schema constraints for " + key);
      }
    }
    if (const auto* properties = result.find("properties")) {
      auto allowed = json::Value::object();
      for (const auto& [name, child] : properties->members()) {
        const auto permits = [&](const json::Value& schema) {
          const auto* closed = schema.find("additionalProperties");
          const auto* fields = schema.find("properties");
          return !closed || closed->as_bool() ||
                 (fields && fields->contains(name));
        };
        if (permits(left) && permits(right))
          allowed.append_member(name, child);
      }
      result["properties"] = std::move(allowed);
    }
    return result;
  }
  // Finite enum/const values still have to satisfy every sibling constraint.
  // Return objects in schema property order, independent of enum key order.
  std::optional<json::Value> ValueFor(const json::Value& schema,
                                      const json::Value& value,
                                      std::size_t depth = 0) {
    if (depth > 256)
      Invalid("enum/const reference expansion exceeds its resource budget");
    auto result = value;
    if (const auto* ref = schema.find("$ref")) {
      auto referenced = ValueFor(*Reference(*ref), value, depth + 1);
      if (!referenced)
        return {};
      result = std::move(*referenced);
    }
    if (const auto* any = schema.find("anyOf")) {
      bool matched = false;
      for (const auto& branch : any->items()) {
        try {
          if (auto candidate = ValueFor(branch, value, depth + 1)) {
            result = std::move(*candidate);
            matched = true;
            break;
          }
        } catch (const JsonSchemaEmpty&) {
        }
      }
      if (!matched)
        return {};
    }
    if (const auto* type = schema.find("type")) {
      if (type->is_string()
              ? !MatchesType(value, type->str())
              : !std::ranges::any_of(type->items(), [&](const auto& t) {
                  return MatchesType(value, t.str());
                }))
        return {};
    }
    if (const auto* enumeration = schema.find("enum");
        enumeration &&
        !std::ranges::any_of(enumeration->items(), [&](const auto& item) {
          return EqualValue(value, item);
        }))
      return {};
    if (const auto* constant = schema.find("const");
        constant && !EqualValue(value, *constant))
      return {};
    if (value.is_string() || value.is_number()) {
      auto& check = value_checks_[&schema];
      if (!check)
        check = value.is_string() ? JsonSchemaLexeme::String(schema)
                                  : JsonSchemaLexeme::Number(schema, false);
      if (!check->AcceptValue(value))
        return {};
    } else if (value.is_array()) {
      if ((schema.contains("minItems") &&
           value.size() < schema.find("minItems")->as_size()) ||
          (schema.contains("maxItems") &&
           value.size() > schema.find("maxItems")->as_size()))
        return {};
      if (const auto* items = schema.find("items")) {
        result = json::Value::array();
        for (const auto& item : value.items()) {
          auto candidate = ValueFor(*items, item, depth + 1);
          if (!candidate)
            return {};
          result.push_back(std::move(*candidate));
        }
      }
    } else if (value.is_object()) {
      if (const auto* required = schema.find("required"))
        for (const auto& key : required->items())
          if (!value.contains(key.str()))
            return {};
      static const auto no_properties = json::Value::object();
      const auto* properties = schema.find("properties");
      if (!properties && schema.contains("additionalProperties") &&
          !schema.find("additionalProperties")->as_bool())
        properties = &no_properties;
      if (properties) {
        result = json::Value::object();
        for (const auto& [name, child] : properties->members()) {
          if (const auto* item = value.find(name)) {
            auto candidate = ValueFor(child, *item, depth + 1);
            if (!candidate)
              return {};
            result.append_member(name, std::move(*candidate));
          }
        }
        for (const auto& [name, item] : value.members())
          if (!properties->contains(name)) {
            if (schema.contains("additionalProperties") &&
                !schema.find("additionalProperties")->as_bool())
              return {};
            result.append_member(name, item);
          }
      }
    }
    return result;
  }
  std::uint32_t ValueLiteral(const json::Value& value) {
    if (!value.is_object() && !value.is_array())
      return Literal(value.dump());
    Sequence parts{Byte(value.is_object() ? '{' : '['), ws_};
    bool comma = false;
    auto separator = [&] {
      if (comma)
        parts.insert(parts.end(), {ws_, Byte(','), ws_});
      comma = true;
    };
    if (value.is_object()) {
      for (const auto& [name, item] : value.members()) {
        separator();
        parts.insert(parts.end(), {Literal(json::Value(name).dump()), ws_,
                                   Byte(':'), ws_, ValueLiteral(item)});
      }
    } else {
      for (const auto& item : value.items()) {
        separator();
        parts.push_back(ValueLiteral(item));
      }
    }
    parts.insert(parts.end(), {ws_, Byte(value.is_object() ? '}' : ']')});
    return Seq(std::move(parts));
  }
  std::uint32_t Primitive(std::string_view type) {
    if (type == "string")
      return string_;
    if (type == "integer")
      return integer_;
    if (type == "number")
      return number_;
    if (type == "boolean")
      return bool_;
    if (type == "null")
      return null_;
    Invalid("unsupported or missing type");
  }
  std::uint32_t Lexeme(std::shared_ptr<const JsonSchemaLexeme> lexeme) {
    const auto symbol = kLexeme | grammar_->lexemes_.size();
    grammar_->lexemes_.push_back(std::move(lexeme));
    return static_cast<std::uint32_t>(symbol);
  }
  // Compile an unsigned interval using shared digit prefixes. Work grows with
  // the number of digits, not the number of integers between the bounds.
  std::uint32_t DigitsBetween(std::string_view low, std::string_view high) {
    if (low.empty())
      return Seq({});
    if (low.find_first_not_of('0') == std::string_view::npos &&
        high.find_first_not_of('9') == std::string_view::npos)
      return Seq(Sequence(low.size(), Range('0', '9')));
    if (low.front() == high.front())
      return Seq(
          {Byte(low.front()), DigitsBetween(low.substr(1), high.substr(1))});
    Sequence choices;
    const std::string zeros(low.size() - 1, '0');
    const std::string nines(low.size() - 1, '9');
    choices.push_back(
        Seq({Byte(low.front()), DigitsBetween(low.substr(1), nines)}));
    if (low.front() + 1 < high.front()) {
      Sequence middle{Range(low.front() + 1, high.front() - 1)};
      middle.insert(middle.end(), low.size() - 1, Range('0', '9'));
      choices.push_back(Seq(std::move(middle)));
    }
    choices.push_back(
        Seq({Byte(high.front()), DigitsBetween(zeros, high.substr(1))}));
    return Alt(choices);
  }
  std::uint32_t UnsignedInterval(const std::string& low,
                                 const std::optional<std::string>& high) {
    Sequence choices;
    const auto last = high ? high->size() : low.size();
    for (std::size_t length = low.size(); length <= last; ++length) {
      const auto first =
          length == low.size() ? low : "1" + std::string(length - 1, '0');
      const auto end =
          high && length == high->size() ? *high : std::string(length, '9');
      if (first <= end)
        choices.push_back(DigitsBetween(first, end));
    }
    if (!high) {
      Sequence longer{Range('1', '9')};
      longer.insert(longer.end(), low.size(), Range('0', '9'));
      longer.push_back(Repeat(Range('0', '9')));
      choices.push_back(Seq(std::move(longer)));
    }
    if (choices.empty())
      throw JsonSchemaEmpty(
          "JSON Schema: numeric bounds describe an empty interval");
    return Alt(choices);
  }
  std::uint32_t Integer(const json::Value& schema) {
    std::optional<double> minimum, maximum;
    for (const auto* key :
         {"minimum", "exclusiveMinimum", "maximum", "exclusiveMaximum"}) {
      const auto* value = schema.find(key);
      if (!value)
        continue;
      if (!value->is_number() || !std::isfinite(value->as_double()))
        Invalid(std::string(key) + " must be finite");
      const double number = value->as_double();
      const bool lower = std::string_view(key).ends_with("Minimum") ||
                         std::string_view(key) == "minimum";
      double rounded = lower ? std::ceil(number) : std::floor(number);
      auto& bound = lower ? minimum : maximum;
      if (!bound || (lower ? rounded > *bound : rounded < *bound))
        bound = rounded;
    }
    if (!minimum && !maximum)
      return integer_;
    // JSON stores finite doubles. Fixed notation gives the exact represented
    // integer even for bounds outside int64, without a narrowing conversion.
    auto decimal = [](double value) {
      std::array<char, 512> buffer;
      auto [end, error] =
          std::to_chars(buffer.data(), buffer.data() + buffer.size(),
                        std::abs(value), std::chars_format::fixed, 0);
      if (error != std::errc{})
        Invalid("numeric bound cannot be represented");
      return std::string(buffer.data(), end);
    };
    auto increment = [](std::string value) {
      for (auto i = value.size(); i > 0; --i) {
        if (value[i - 1] != '9') {
          ++value[i - 1];
          return value;
        }
        value[i - 1] = '0';
      }
      return "1" + value;
    };
    auto decrement = [](std::string value) {
      for (auto i = value.size(); i > 0; --i) {
        if (value[i - 1] != '0') {
          --value[i - 1];
          break;
        }
        value[i - 1] = '9';
      }
      if (value.size() > 1 && value.front() == '0')
        value.erase(0, 1);
      return value;
    };
    struct Bound {
      bool negative;
      std::string digits;
    };
    auto bound = [&](std::optional<double> value, const char* exclusive,
                     bool lower) -> std::optional<Bound> {
      if (!value)
        return {};
      Bound result{*value < 0, decimal(*value)};
      const auto* excluded = schema.find(exclusive);
      if (excluded && excluded->as_double() == *value) {
        if (result.digits == "0") {
          result = {!lower, "1"};
        } else if (lower == result.negative) {
          result.digits = decrement(result.digits);
          if (result.digits == "0")
            result.negative = false;
        } else {
          result.digits = increment(result.digits);
        }
      }
      return result;
    };
    const auto low = bound(minimum, "exclusiveMinimum", true);
    const auto high = bound(maximum, "exclusiveMaximum", false);
    auto compare = [](const std::string& a, const std::string& b) {
      return a.size() != b.size() ? (a.size() < b.size() ? -1 : 1)
                                  : a.compare(b);
    };
    if (low && high &&
        (low->negative != high->negative
             ? !low->negative
             : (low->negative ? compare(low->digits, high->digits) < 0
                              : compare(low->digits, high->digits) > 0)))
      throw JsonSchemaEmpty(
          "JSON Schema: numeric bounds describe an empty interval");
    Sequence choices;
    if (!high || !high->negative) {
      choices.push_back(
          UnsignedInterval(low && !low->negative ? low->digits : "0",
                           high ? std::optional(high->digits) : std::nullopt));
    }
    if (!low || low->negative) {
      choices.push_back(Seq(
          {Byte('-'),
           UnsignedInterval(high && high->negative ? high->digits : "1",
                            low ? std::optional(low->digits) : std::nullopt)}));
    }
    if ((!low || low->negative || low->digits == "0") &&
        (!high || !high->negative))
      choices.push_back(Literal("-0"));
    return Alt(choices);
  }
  void Keys(const json::Value& schema) {
    if (!schema.is_object())
      Invalid("each schema must be an object");
    static const std::set<std::string_view> allowed{"type",
                                                    "properties",
                                                    "required",
                                                    "additionalProperties",
                                                    "items",
                                                    "minItems",
                                                    "maxItems",
                                                    "enum",
                                                    "const",
                                                    "anyOf",
                                                    "$defs",
                                                    "$ref",
                                                    "title",
                                                    "description",
                                                    "minimum",
                                                    "maximum",
                                                    "exclusiveMinimum",
                                                    "exclusiveMaximum",
                                                    "multipleOf",
                                                    "pattern",
                                                    "format",
                                                    "minLength",
                                                    "maxLength"};
    for (const auto& [key, value] : schema.members()) {
      if (!allowed.contains(key))
        Invalid("unsupported keyword: " + key);
      if ((key == "title" || key == "description") && !value.is_string())
        Invalid(key + " must be a string");
      if (key == "anyOf" && (!value.is_array() || value.empty()))
        Invalid("anyOf needs at least one branch");
    }
  }
  std::uint32_t Visit(const json::Value& schema, std::size_t depth) {
    if (const auto found = compiled_.find(&schema); found != compiled_.end())
      return found->second;
    const auto id = New();
    compiled_[&schema] = id;
    try {
      const auto body = VisitBody(schema, depth);
      grammar_->rules_[id] = {{body}};
    } catch (const JsonSchemaEmpty&) {
      grammar_->rules_[id] = {};
    }
    return id;
  }
  void CountStrings(std::string_view value) {
    string_characters_ += std::ranges::count_if(
        value, [](unsigned char c) { return (c & 0xc0) != 0x80; });
    if (string_characters_ > 120000)
      Invalid(
          "property/definition names and enum/const strings exceed 120000 "
          "characters");
  }
  void CountValueStrings(const json::Value& value) {
    if (value.is_string())
      CountStrings(value.str());
    else if (value.is_array())
      for (const auto& item : value.items())
        CountValueStrings(item);
    else if (value.is_object())
      for (const auto& [name, item] : value.members()) {
        CountStrings(name);
        CountValueStrings(item);
      }
  }
  std::uint32_t VisitBody(const json::Value& schema, std::size_t depth) {
    if (!schema.is_object())
      Invalid("each schema must be an object");
    if (depth > kMaxDepth)
      Invalid("maximum schema depth is 16");
    Keys(schema);
    if (const auto* defs = schema.find("$defs")) {
      if (!defs->is_object())
        Invalid("$defs must be an object");
      for (const auto& [name, value] : defs->members()) {
        CountStrings(name);
        Visit(value, depth);
      }
    }
    if (const auto* ref = schema.find("$ref")) {
      const auto siblings =
          Without(schema, {"$ref", "$defs", "title", "description"});
      return siblings.empty()
                 ? Visit(*Reference(*ref), depth)
                 : Visit(Store(Conjoin(*Reference(*ref), siblings)), depth);
    }
    if (const auto* any = schema.find("anyOf")) {
      if (!any->is_array() || any->size() == 0)
        Invalid("anyOf needs at least one branch");
      const auto siblings =
          Without(schema, {"anyOf", "$defs", "title", "description"});
      Sequence branches;
      for (const auto& value : any->items()) {
        try {
          branches.push_back(
              Visit(siblings.empty() ? value : Store(Conjoin(value, siblings)),
                    depth + 1));
        } catch (const JsonSchemaEmpty&) {
        }
      }
      return Alt(branches);
    }
    const auto* type = schema.find("type");
    std::vector<std::string> types;
    if (type && type->is_string())
      types.push_back(type->str());
    else if (type && type->is_array() && type->size() > 0 &&
             type->size() <= 2) {
      for (const auto& value : type->items()) {
        if (!value.is_string())
          Invalid("type members must be strings");
        types.push_back(value.str());
      }
      if (types.size() == 2 &&
          std::count(types.begin(), types.end(), "null") != 1)
        Invalid("type unions must be nullable; use anyOf otherwise");
    } else
      Invalid("type must be a string or nullable type array");
    const bool object = std::ranges::find(types, "object") != types.end();
    const bool array = std::ranges::find(types, "array") != types.end();
    for (const auto& [key, value] : schema.members()) {
      (void)value;
      if ((key == "properties" || key == "required" ||
           key == "additionalProperties") &&
          !object)
        Invalid("object keyword on a non-object schema");
      if ((key == "items" || key == "minItems" || key == "maxItems") && !array)
        Invalid("array keyword on a non-array schema");
      if ((key == "minimum" || key == "maximum" || key == "exclusiveMinimum" ||
           key == "exclusiveMaximum" || key == "multipleOf") &&
          std::ranges::find(types, "integer") == types.end() &&
          std::ranges::find(types, "number") == types.end())
        Invalid("numeric constraint on a non-numeric schema");
      if ((key == "pattern" || key == "format" || key == "minLength" ||
           key == "maxLength") &&
          std::ranges::find(types, "string") == types.end())
        Invalid("string constraint on a non-string schema");
    }
    if (schema.contains("enum") || schema.contains("const")) {
      // Validate the entire underlying schema even when the finite choices
      // would otherwise hide malformed properties, items or predicates.
      const auto& base = Store(Without(schema, {"enum", "const", "$defs"}));
      Visit(base, depth);
      json::Value::Array values;
      if (const auto* enumeration = schema.find("enum")) {
        if (!enumeration->is_array() || enumeration->size() == 0)
          Invalid("enum must be a nonempty array");
        values = enumeration->items();
      } else
        values.push_back(*schema.find("const"));
      enum_values_ += values.size();
      if (enum_values_ > 1000)
        Invalid("maximum enum/const value count is 1000");
      Sequence choices;
      std::size_t enum_characters = 0;
      for (const auto& value : values) {
        CountValueStrings(value);
        if (!std::ranges::any_of(
                types, [&](const auto& t) { return MatchesType(value, t); }))
          Invalid("enum/const value does not match its type");
        if (value.is_string()) {
          enum_characters += std::ranges::count_if(
              value.str(), [](unsigned char c) { return (c & 0xc0) != 0x80; });
          if (values.size() > 250 && enum_characters > 15000)
            Invalid(
                "an enum with more than 250 entries is limited to 15000 string "
                "characters");
        }
        if (const auto* constant = schema.find("const");
            constant && !EqualValue(value, *constant))
          continue;
        try {
          if (auto canonical = ValueFor(base, value))
            choices.push_back(ValueLiteral(*canonical));
        } catch (const JsonSchemaEmpty&) {
        }
      }
      if (choices.empty())
        throw JsonSchemaEmpty(
            "JSON Schema: enum/const has no value satisfying its constraints");
      return Alt(choices);
    }
    Sequence alternatives;
    for (const auto& name : types) {
      if (name == "object")
        alternatives.push_back(Object(schema, depth));
      else if (name == "array")
        alternatives.push_back(Array(schema, depth));
      else if ((name == "integer" && schema.contains("multipleOf")) ||
               (name == "number" &&
                (schema.contains("minimum") || schema.contains("maximum") ||
                 schema.contains("exclusiveMinimum") ||
                 schema.contains("exclusiveMaximum") ||
                 schema.contains("multipleOf"))))
        alternatives.push_back(
            Lexeme(JsonSchemaLexeme::Number(schema, name == "integer")));
      else if (name == "string" &&
               (schema.contains("pattern") || schema.contains("format") ||
                schema.contains("minLength") || schema.contains("maxLength")))
        alternatives.push_back(Lexeme(JsonSchemaLexeme::String(schema)));
      else if (name == "integer")
        alternatives.push_back(Integer(schema));
      else
        alternatives.push_back(Primitive(name));
    }
    return Alt(alternatives);
  }
  const json::Value* Reference(const json::Value& reference) const {
    if (!reference.is_string() ||
        (reference.str() != "#" && !reference.str().starts_with("#/")))
      Invalid("only local JSON pointer references are supported");
    const auto* target = &schema_;
    std::string_view pointer(reference.str());
    pointer.remove_prefix(1);
    while (!pointer.empty()) {
      pointer.remove_prefix(1);
      const auto slash = pointer.find('/');
      const auto segment = pointer.substr(0, slash);
      std::string decoded;
      for (std::size_t i = 0; i < segment.size(); ++i) {
        if (segment[i] == '~') {
          if (++i == segment.size() || (segment[i] != '0' && segment[i] != '1'))
            Invalid("invalid JSON pointer escape");
          decoded += segment[i] == '0' ? '~' : '/';
        } else
          decoded += segment[i];
      }
      if (target->is_array()) {
        std::size_t index = 0;
        const auto result = std::from_chars(
            decoded.data(), decoded.data() + decoded.size(), index);
        target = result.ec == std::errc{} &&
                         result.ptr == decoded.data() + decoded.size() &&
                         index < target->size()
                     ? &target->items()[index]
                     : nullptr;
      } else
        target = target->find(decoded);
      if (!target)
        Invalid("local reference does not exist");
      if (slash == std::string_view::npos)
        break;
      pointer.remove_prefix(slash);
    }
    return target;
  }
  std::uint32_t Object(const json::Value& schema, std::size_t depth) {
    static const auto empty_properties = json::Value::object();
    const auto* properties = schema.find("properties");
    if (!properties)
      properties = &empty_properties;
    const auto* additional = schema.find("additionalProperties");
    if (!properties || !properties->is_object() || !additional ||
        !additional->is_bool() || additional->as_bool())
      Invalid("objects require properties and additionalProperties: false");
    properties_ += properties->size();
    if (properties_ > 5000)
      Invalid("maximum property count is 5000");
    std::set<std::string> required;
    if (const auto* fields = schema.find("required")) {
      if (!fields->is_array())
        Invalid("required must be an array");
      for (const auto& field : fields->items()) {
        if (!field.is_string() || !properties->contains(field.str()) ||
            !required.insert(field.str()).second)
          Invalid("required contains an unknown or duplicate property");
      }
    }
    if (strict_ && required.size() != properties->size())
      Invalid(
          "strict schemas require every property (use null for optional "
          "values)");
    // Two suffix states represent whether a comma is needed. Optional fields
    // remain in schema order without enumerating every property subset.
    std::array<std::uint32_t, 2> suffix{Seq({}), Seq({})};
    for (auto it = properties->members().rbegin();
         it != properties->members().rend(); ++it) {
      const auto& [key, value] = *it;
      CountStrings(key);
      const auto member = Seq({Literal(json::Value(key).dump()), ws_, Byte(':'),
                               ws_, Visit(value, depth + 1)});
      std::array<std::uint32_t, 2> next;
      for (unsigned comma = 0; comma < 2; ++comma) {
        Sequence sequence;
        if (comma)
          sequence = {ws_, Byte(','), ws_};
        sequence.insert(sequence.end(), {member, suffix[1]});
        Rule rule{std::move(sequence)};
        if (!required.contains(key))
          rule.push_back({suffix[comma]});
        next[comma] = New(std::move(rule));
      }
      suffix = next;
    }
    return Seq({Byte('{'), ws_, suffix[0], ws_, Byte('}')});
  }
  std::uint32_t Array(const json::Value& schema, std::size_t depth) {
    const auto* items = schema.find("items");
    if (!items)
      Invalid("arrays require an items schema");
    auto bound = [&](const char* name, std::size_t fallback) {
      const auto* value = schema.find(name);
      if (!value)
        return fallback;
      if (!value->is_number() || value->as_double() < 0 ||
          value->as_double() > std::numeric_limits<std::uint32_t>::max() ||
          std::floor(value->as_double()) != value->as_double())
        Invalid("array bounds must be nonnegative 32-bit integers");
      return value->as_size();
    };
    const auto minimum = bound("minItems", 0);
    const auto maximum =
        bound("maxItems", std::numeric_limits<std::uint32_t>::max());
    if (minimum > maximum)
      throw JsonSchemaEmpty("JSON Schema: minItems exceeds maxItems");
    const auto item = Visit(*items, depth + 1);
    if (maximum == 0)
      return Seq({Byte('['), ws_, Byte(']')});
    const auto additional = Seq({ws_, Byte(','), ws_, item});
    const auto required = minimum ? minimum - 1 : 0;
    const auto tail = schema.contains("maxItems")
                          ? AtMost(additional, maximum - required - 1)
                          : Repeat(additional);
    auto members = Seq({item, Exact(additional, required), tail});
    if (!minimum)
      members = Optional(members);
    return Seq({Byte('['), ws_, members, ws_, Byte(']')});
  }
  const json::Value& schema_;
  bool strict_;
  std::shared_ptr<JsonConstraint> grammar_{new JsonConstraint};
  std::uint32_t ws_, string_, integer_, number_, bool_, null_;
  std::map<std::size_t, std::uint32_t> generic_values_;
  std::map<const json::Value*, std::uint32_t> compiled_;
  std::deque<json::Value> derived_;
  std::map<const json::Value*, std::shared_ptr<const JsonSchemaLexeme>>
      value_checks_;
  std::size_t properties_{0}, enum_values_{0}, string_characters_{0};
};

std::shared_ptr<const JsonConstraint> JsonConstraint::Compile(
    const json::Value& schema, bool strict) {
  const auto key = std::string(strict ? "strict:" : "schema:") + schema.dump();
  if (key.size() > kMaxSchemaBytes)
    Invalid("maximum schema size is 2 MiB");
  // Bounded cache; compile outside the lock so unrelated HTTP requests proceed.
  static std::mutex mutex;
  static std::map<std::string, std::shared_ptr<const JsonConstraint>> cache;
  {
    const std::lock_guard lock(mutex);
    if (const auto found = cache.find(key); found != cache.end())
      return found->second;
  }
  auto grammar = JsonConstraintCompiler(schema, strict).Compile(false);
  const std::lock_guard lock(mutex);
  if (cache.size() >= 16)
    cache.erase(cache.begin());
  return cache.emplace(key, std::move(grammar)).first->second;
}

std::shared_ptr<const JsonConstraint> JsonConstraint::Object() {
  static const auto grammar = [] {
    const json::Value schema;
    return JsonConstraintCompiler(schema, false).Compile(true);
  }();
  return grammar;
}

std::shared_ptr<const JsonConstraint> JsonConstraint::ToolParameters(
    const json::Value& schema, bool strict, ToolFormat format) {
  // Validate with the same compiler as JSON output before selecting a native
  // representation. Unsupported non-strict schemas are handled by the caller.
  const auto validated = Compile(schema, strict);
  if (format == ToolFormat::kJson)
    return validated;
  using Key = std::pair<std::shared_ptr<const JsonConstraint>, ToolFormat>;
  static std::mutex mutex;
  static std::map<Key, std::shared_ptr<const JsonConstraint>> cache;
  const Key key{validated, format};
  {
    const std::lock_guard lock(mutex);
    if (const auto found = cache.find(key); found != cache.end())
      return found->second;
  }
  auto grammar = JsonConstraintCompiler(schema, strict).ToolParameters(format);
  const std::lock_guard lock(mutex);
  if (cache.size() >= 16)
    cache.erase(cache.begin());
  return cache.emplace(key, std::move(grammar)).first->second;
}

std::shared_ptr<const JsonConstraint> JsonConstraint::WithReasoning(
    std::shared_ptr<const JsonConstraint> answer) {
  // Cache separately from schema compilation. No mutable phase belongs to the
  // model or cache: the normal grammar stack carries it through sampler copies,
  // draft rejection and request restart.
  static std::mutex mutex;
  static std::map<std::shared_ptr<const JsonConstraint>,
                  std::shared_ptr<const JsonConstraint>>
      cache;
  {
    const std::lock_guard lock(mutex);
    if (const auto found = cache.find(answer); found != cache.end())
      return found->second;
  }
  auto grammar = std::shared_ptr<JsonConstraint>(new JsonConstraint(*answer));
  constexpr std::string_view end = "</think>";
  const auto base = static_cast<std::uint32_t>(grammar->rules_.size());
  grammar->rules_.resize(base + end.size());
  for (std::size_t prefix = 0; prefix < end.size(); ++prefix) {
    std::array<std::bitset<256>, end.size() + 1> transitions;
    for (unsigned byte = 0; byte < 256; ++byte) {
      std::string candidate(end.substr(0, prefix));
      candidate += static_cast<char>(byte);
      std::size_t matched = std::min(candidate.size(), end.size());
      while (matched && !candidate.ends_with(end.substr(0, matched)))
        --matched;
      transitions[matched].set(byte);
    }
    for (std::size_t matched = 0; matched <= end.size(); ++matched) {
      if (transitions[matched].none())
        continue;
      const auto terminal = kTerminal | grammar->classes_.size();
      grammar->classes_.push_back(transitions[matched]);
      grammar->rules_[base + prefix].push_back(
          {static_cast<std::uint32_t>(terminal),
           matched == end.size() ? answer->root_
                                 : base + static_cast<std::uint32_t>(matched)});
    }
  }
  grammar->root_ = base;
  const std::lock_guard lock(mutex);
  if (const auto found = cache.find(answer); found != cache.end())
    return found->second;
  if (cache.size() >= 16)
    cache.erase(cache.begin());
  return cache.emplace(std::move(answer), std::move(grammar)).first->second;
}

std::shared_ptr<const JsonConstraint> JsonConstraint::WithTools(
    std::shared_ptr<const JsonConstraint> answer, std::vector<Tool> tools,
    bool required, bool parallel, ToolFormat format) {
  if (tools.empty())
    return answer;
  using Key = std::tuple<std::shared_ptr<const JsonConstraint>,
                         std::vector<Tool>, bool, bool, ToolFormat>;
  static std::mutex mutex;
  static std::map<Key, std::shared_ptr<const JsonConstraint>> cache;
  const Key key{answer, tools, required, parallel, format};
  {
    const std::lock_guard lock(mutex);
    if (const auto found = cache.find(key); found != cache.end())
      return found->second;
  }
  const bool plain_answer = !answer;
  auto grammar = std::shared_ptr<JsonConstraint>(
      new JsonConstraint(answer ? *answer : *Object()));
  grammar->stop_only_when_complete_ = !plain_answer && !parallel;
  Rule alternatives;
  if (!required && answer)
    alternatives.push_back({answer->root_});
  auto check_capacity = [&](std::size_t rules, std::size_t classes,
                            std::size_t lexemes) {
    auto fits = [](std::size_t current, std::size_t extra) {
      return current <= kMaxRules && extra <= kMaxRules - current;
    };
    if (!fits(grammar->rules_.size(), rules) ||
        !fits(grammar->classes_.size(), classes) ||
        !fits(grammar->lexemes_.size(), lexemes))
      Invalid("combined tool grammar exceeds its resource budget");
  };
  auto literal = [&](std::string_view text) {
    check_capacity(1, 0, 0);
    Sequence bytes;
    for (unsigned char byte : text)
      bytes.push_back(kTerminal | byte);
    const auto id = static_cast<std::uint32_t>(grammar->rules_.size());
    grammar->rules_.push_back({std::move(bytes)});
    return id;
  };
  const bool deepseek = format == ToolFormat::kDeepSeek;
  const std::string_view marker =
      deepseek ? "<｜DSML｜tool_calls>" : "<tool_call>";
  const auto end = literal(format == ToolFormat::kJson ? "}</tool_call>"
                           : deepseek                  ? "</｜DSML｜invoke>"
                                      : "</function>\n</tool_call>");
  const auto calls = static_cast<std::uint32_t>(grammar->rules_.size());
  grammar->rules_.push_back({});
  // In tool-only mode ordinary text remains unconstrained until a canonical
  // call marker. The small prefix automaton is carried by the request grammar,
  // including across tokens, speculative rollback and sampler copies.
  auto text = [&](std::uint32_t target) {
    check_capacity(marker.size(), marker.size() * 256, 0);
    const auto base = static_cast<std::uint32_t>(grammar->rules_.size());
    grammar->rules_.resize(base + marker.size());
    for (std::size_t prefix = 0; prefix < marker.size(); ++prefix) {
      grammar->rules_[base + prefix].push_back({});
      std::vector<std::bitset<256>> transitions(marker.size() + 1);
      for (unsigned byte = 0; byte < 256; ++byte) {
        std::string candidate(marker.substr(0, prefix));
        candidate += static_cast<char>(byte);
        auto matched = std::min(candidate.size(), marker.size());
        while (matched && !candidate.ends_with(marker.substr(0, matched)))
          --matched;
        transitions[matched].set(byte);
      }
      for (std::size_t matched = 0; matched <= marker.size(); ++matched) {
        if (transitions[matched].none() ||
            (matched == marker.size() && target == UINT32_MAX))
          continue;
        const auto terminal = kTerminal | grammar->classes_.size();
        grammar->classes_.push_back(transitions[matched]);
        grammar->rules_[base + prefix].push_back(
            {static_cast<std::uint32_t>(terminal),
             matched == marker.size()
                 ? target
                 : base + static_cast<std::uint32_t>(matched)});
      }
    }
    return base;
  };
  const auto prose = plain_answer ? text(calls) : UINT32_MAX;
  auto after = plain_answer
                   ? (parallel ? prose : text(UINT32_MAX))
                   : static_cast<std::uint32_t>(grammar->rules_.size());
  if (!plain_answer) {
    grammar->rules_.push_back({{}});
    if (parallel) {
      const auto begin = literal(marker);
      grammar->rules_[after].push_back({begin, calls});
    }
  }
  if (deepseek) {
    const auto finish = literal("\n</｜DSML｜tool_calls>");
    const auto tail = static_cast<std::uint32_t>(grammar->rules_.size());
    grammar->rules_.push_back({{finish, after}});
    if (parallel)
      grammar->rules_[tail].push_back({calls});
    after = tail;
  }
  std::map<const JsonConstraint*, std::uint32_t> imported;
  if (answer)
    imported.emplace(answer.get(), answer->root_);
  for (const auto& [name, arguments] : tools) {
    const auto begin = literal(
        format == ToolFormat::kJson
            ? "{\"name\":" + json::Value(name).dump() + ",\"arguments\":"
        : deepseek ? "\n<｜DSML｜invoke name=\"" + name + "\">\n"
                   : "\n<function=" + name + ">\n");
    if (const auto found = imported.find(arguments.get());
        found != imported.end()) {
      grammar->rules_[calls].push_back({begin, found->second, end, after});
      continue;
    }
    check_capacity(arguments->rules_.size(), arguments->classes_.size(),
                   arguments->lexemes_.size());
    const auto rules = grammar->rules_.size();
    const auto classes = grammar->classes_.size();
    const auto lexemes = grammar->lexemes_.size();
    grammar->classes_.insert(grammar->classes_.end(),
                             arguments->classes_.begin(),
                             arguments->classes_.end());
    grammar->lexemes_.insert(grammar->lexemes_.end(),
                             arguments->lexemes_.begin(),
                             arguments->lexemes_.end());
    for (auto rule : arguments->rules_) {
      for (auto& sequence : rule)
        for (auto& symbol : sequence)
          symbol = (symbol & kTerminal)
                       ? kTerminal | ((symbol & ~kTerminal) + classes)
                   : (symbol & kLexeme)
                       ? kLexeme | ((symbol & ~kLexeme) + lexemes)
                       : symbol + rules;
      grammar->rules_.push_back(std::move(rule));
    }
    const auto root = static_cast<std::uint32_t>(arguments->root_ + rules);
    imported.emplace(arguments.get(), root);
    grammar->rules_[calls].push_back({begin, root, end, after});
  }
  if (plain_answer && !required)
    alternatives.push_back({prose});
  else if (format == ToolFormat::kJson)
    alternatives.push_back({literal(marker), calls});
  else {
    std::bitset<256> whitespace;
    for (const unsigned char byte : std::string_view(" \t\r\n"))
      whitespace.set(byte);
    const auto terminal =
        static_cast<std::uint32_t>(kTerminal | grammar->classes_.size());
    grammar->classes_.push_back(whitespace);
    const auto ws = static_cast<std::uint32_t>(grammar->rules_.size());
    grammar->rules_.push_back({{}, {terminal, ws}});
    alternatives.push_back({ws, literal(marker), calls});
  }
  check_capacity(1, 0, 0);
  grammar->root_ = grammar->rules_.size();
  grammar->rules_.push_back(std::move(alternatives));
  const std::lock_guard lock(mutex);
  if (const auto found = cache.find(key); found != cache.end())
    return found->second;
  if (cache.size() >= 16)
    cache.erase(cache.begin());
  return cache.emplace(key, std::move(grammar)).first->second;
}

JsonConstraint::State JsonConstraint::Expand(State pending) const {
  State output;
  std::size_t work = 0;
  while (!pending.empty()) {
    if (++work > kMaxWork || pending.size() + output.size() > kMaxStates)
      throw std::runtime_error("JSON grammar state limit exceeded");
    auto stack = std::move(pending.back());
    pending.pop_back();
    if (stack.symbols.empty() || (stack.symbols.back() & kLeaf)) {
      output.push_back(std::move(stack));
      continue;
    }
    const auto rule = stack.symbols.back();
    stack.symbols.pop_back();
    for (const auto& sequence : rules_.at(rule)) {
      if (stack.symbols.size() + sequence.size() > kMaxStack)
        throw std::runtime_error("JSON grammar stack limit exceeded");
      auto next = stack;
      next.symbols.insert(next.symbols.end(), sequence.rbegin(),
                          sequence.rend());
      pending.push_back(std::move(next));
    }
  }
  std::ranges::sort(output);
  output.erase(std::unique(output.begin(), output.end()), output.end());
  return output;
}

JsonConstraint::State JsonConstraint::Start() const {
  return Expand({{{root_}, {}}});
}

JsonConstraint::State JsonConstraint::Advance(const State& state,
                                              unsigned char byte) const {
  State next;
  for (const auto& stack : state) {
    if (stack.symbols.empty())
      continue;
    const auto symbol = stack.symbols.back();
    if (symbol & kLexeme) {
      const auto& lexeme = lexemes_.at(symbol & ~kLexeme);
      if (!lexeme->AllowsByte(byte))
        continue;
      auto candidate = stack;
      const auto result = lexeme->Advance(candidate.lexeme, byte);
      if (result.prefix)
        next.push_back(candidate);
      if (result.complete) {
        candidate.symbols.pop_back();
        candidate.lexeme.clear();
        next.push_back(std::move(candidate));
      }
    } else if (classes_.at(symbol & ~kTerminal).test(byte)) {
      next.push_back(stack);
      next.back().symbols.pop_back();
    }
  }
  return Expand(std::move(next));
}

bool JsonConstraint::Complete(const State& state) const {
  return std::ranges::any_of(
      state, [](const auto& stack) { return stack.symbols.empty(); });
}

JsonConstraint::State JsonConstraint::CanonicalMaskState(
    const State& state, std::size_t token_bytes) const {
  auto canonical = state;
  for (auto& stack : canonical)
    if (!stack.symbols.empty() && (stack.symbols.back() & kLexeme))
      lexemes_.at(stack.symbols.back() & ~kLexeme)
          ->CanonicalMaskState(stack.lexeme, token_bytes);
  std::ranges::sort(canonical);
  canonical.erase(std::unique(canonical.begin(), canonical.end()),
                  canonical.end());
  return canonical;
}

ConstraintVocabulary::ConstraintVocabulary(std::uint32_t size,
                                           const Reader& reader) {
  if (size == 0 || size > 1048576)
    throw std::invalid_argument("constraint vocabulary size is unsupported");
  pieces_.reserve(size);
  // Flat links avoid tiny heap allocations. Temporary dense indexes accelerate
  // high-fanout prefixes during construction; retained traversal needs only
  // the compact child lists.
  trie_.reserve(std::min<std::size_t>(size * 4ULL, 4000000));
  next_token_.resize(size, UINT32_MAX);
  std::vector<std::uint32_t> dense_ids{0};
  dense_ids.reserve(trie_.capacity());
  std::vector<std::array<std::uint32_t, 256>> dense(1);
  dense.front().fill(UINT32_MAX);
  std::size_t total_bytes = 0;
  for (std::uint32_t token = 0; token < size; ++token) {
    pieces_.push_back(reader(token));
    const auto& piece = pieces_.back();
    total_bytes += piece.text.size();
    if (total_bytes > 64 * 1024 * 1024)
      throw std::invalid_argument("constraint vocabulary exceeds 64 MiB");
    if (piece.stop || piece.text.empty())
      continue;
    if (piece.text.size() > 4096)
      throw std::invalid_argument(
          "constraint vocabulary token exceeds 4096 bytes");
    max_token_bytes_ = std::max(max_token_bytes_, piece.text.size());
    std::uint32_t node = 0;
    for (unsigned char byte : piece.text) {
      auto index = dense_ids[node];
      auto child = index == UINT32_MAX ? trie_[node].child : dense[index][byte];
      if (index == UINT32_MAX) {
        unsigned searched = 0;
        while (child != UINT32_MAX && trie_[child].byte != byte) {
          child = trie_[child].sibling;
          ++searched;
        }
        if (searched >= 8 && dense.size() < 8192) {
          index = static_cast<std::uint32_t>(dense.size());
          dense_ids[node] = index;
          auto& entries = dense.emplace_back();
          entries.fill(UINT32_MAX);
          for (auto edge = trie_[node].child; edge != UINT32_MAX;
               edge = trie_[edge].sibling)
            entries[trie_[edge].byte] = edge;
        }
      }
      if (child == UINT32_MAX) {
        if (trie_.size() >= 4000000)
          throw std::invalid_argument(
              "constraint token trie exceeds its node limit");
        child = static_cast<std::uint32_t>(trie_.size());
        const auto sibling = trie_[node].child;
        trie_[node].child = child;
        if (index != UINT32_MAX)
          dense[index][byte] = child;
        trie_.push_back({UINT32_MAX, sibling, UINT32_MAX, byte});
        dense_ids.push_back(UINT32_MAX);
      }
      node = child;
    }
    next_token_[token] = trie_[node].token;
    trie_[node].token = token;
  }
}

std::vector<std::uint8_t> ConstraintVocabulary::Allowed(
    const JsonConstraint& grammar, const JsonConstraint::State& state) const {
  std::vector<std::uint8_t> mask(pieces_.size());
  if (grammar.Complete(state)) {
    for (std::size_t i = 0; i < pieces_.size(); ++i)
      mask[i] = pieces_[i].stop;
    if (grammar.stop_only_when_complete_)
      return mask;
  }
  // Intern the grammar states reached while walking the token trie. Long word
  // tokens share both trie prefixes and string-body transitions; expanding a
  // pushdown state once per byte edge would otherwise dominate decode time.
  struct CachedState {
    JsonConstraint::State state;
    std::array<std::uint32_t, 256> next;
    explicit CachedState(JsonConstraint::State s) : state(std::move(s)) {
      next.fill(UINT32_MAX);
    }
  };
  std::deque<CachedState> states;
  states.emplace_back(JsonConstraint::State{});
  states.emplace_back(state);
  std::map<JsonConstraint::State, std::uint32_t> intern{{{}, 0}, {state, 1}};
  std::size_t work = 0;
  // Numeric predicates retain their scalar text; strings use compact lexer/DFA
  // states. Walk numeric branches with depth-bounded scratch rather than
  // allocating a 256-way transition table for every vocabulary prefix.
  auto direct = [&](auto&& self, std::uint32_t node,
                    const JsonConstraint::State& current) -> void {
    if (++work > kMaxWork)
      throw std::runtime_error("JSON token mask work limit exceeded");
    for (auto token = trie_[node].token; token != UINT32_MAX;
         token = next_token_[token])
      mask[token] = 1;
    for (auto child = trie_[node].child; child != UINT32_MAX;
         child = trie_[child].sibling) {
      auto next = grammar.Advance(current, trie_[child].byte);
      if (!next.empty())
        self(self, child, next);
    }
  };
  auto walk = [&](auto&& self, std::uint32_t node,
                  std::uint32_t current) -> void {
    if (std::ranges::any_of(states[current].state, [&](const auto& stack) {
          return !stack.symbols.empty() && (stack.symbols.back() & kLexeme) &&
                 !grammar.lexemes_.at(stack.symbols.back() & ~kLexeme)
                      ->CacheTransitions();
        })) {
      direct(direct, node, states[current].state);
      return;
    }
    if (++work > kMaxWork)
      throw std::runtime_error("JSON token mask work limit exceeded");
    for (auto token = trie_[node].token; token != UINT32_MAX;
         token = next_token_[token])
      mask[token] = 1;
    for (auto child = trie_[node].child; child != UINT32_MAX;
         child = trie_[child].sibling) {
      const auto byte = trie_[child].byte;
      auto& transition = states[current].next[byte];
      if (transition == UINT32_MAX) {
        auto next = grammar.Advance(states[current].state, byte);
        if (states.size() >= 8192 && !intern.contains(next)) {
          // Cache capacity is an optimization limit, not a language limit.
          if (!next.empty())
            direct(direct, child, next);
          continue;
        }
        auto [found, inserted] = intern.emplace(next, states.size());
        transition = found->second;
        if (inserted) {
          states.emplace_back(std::move(next));
        }
      }
      if (transition != 0)
        self(self, child, transition);
    }
  };
  walk(walk, 0, 1);
  if (std::ranges::none_of(mask, [](auto value) { return value != 0; }))
    throw std::runtime_error("JSON constraint has no valid token");
  return mask;
}

JsonConstraint::State ConstraintVocabulary::Accept(
    const JsonConstraint& grammar, const JsonConstraint::State& state,
    std::uint32_t token) const {
  const auto& piece = pieces_.at(token);
  if (piece.stop && grammar.Complete(state))
    return state;
  if (piece.stop || piece.text.empty())
    throw std::runtime_error("invalid token accepted by JSON constraint");
  auto next = state;
  for (unsigned char byte : piece.text)
    next = grammar.Advance(next, byte);
  if (next.empty())
    throw std::runtime_error("invalid token accepted by JSON constraint");
  return next;
}

bool ConstraintVocabulary::Allows(const JsonConstraint& grammar,
                                  const JsonConstraint::State& state,
                                  std::uint32_t token) const {
  const auto& piece = pieces_.at(token);
  if (grammar.stop_only_when_complete_ && grammar.Complete(state))
    return piece.stop;
  if (piece.stop)
    return grammar.Complete(state);
  if (piece.text.empty())
    return false;
  auto next = state;
  for (const unsigned char byte : piece.text) {
    next = grammar.Advance(next, byte);
    if (next.empty())
      return false;
  }
  return true;
}

std::shared_ptr<const std::vector<std::uint8_t>> TokenConstraint::Allowed(
    const JsonConstraint::State& state) const {
  auto canonical =
      grammar->CanonicalMaskState(state, vocabulary->max_token_bytes_);
  {
    const std::lock_guard lock(mutex_);
    if (const auto found = masks_.find(canonical); found != masks_.end())
      return found->second;
  }
  auto mask = std::make_shared<const std::vector<std::uint8_t>>(
      vocabulary->Allowed(*grammar, canonical));
  const std::lock_guard lock(mutex_);
  if (masks_.size() >= 16)
    masks_.erase(masks_.begin());
  return masks_.emplace(std::move(canonical), std::move(mask)).first->second;
}
}  // namespace gufo::sampling
