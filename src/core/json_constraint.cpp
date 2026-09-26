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
    ws_ = Repeat(Class(" \t\r\n"));
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
    for (const auto& [schema, rule] : compiled_) {
      (void)schema;
      if (!productive[rule])
        Invalid("recursive schema has no finite value");
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
      Invalid("numeric bounds describe an empty interval");
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
      Invalid("numeric bounds describe an empty interval");
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
    }
  }
  std::uint32_t Visit(const json::Value& schema, std::size_t depth) {
    if (const auto found = compiled_.find(&schema); found != compiled_.end())
      return found->second;
    const auto id = New();
    compiled_[&schema] = id;
    const auto body = VisitBody(schema, depth);
    grammar_->rules_[id] = {{body}};
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
      for (const auto& [key, value] : schema.members()) {
        (void)value;
        if (key != "$ref" && key != "title" && key != "description" &&
            key != "$defs")
          Invalid("$ref siblings are not supported");
      }
      return Visit(*Reference(*ref), depth);
    }
    if (const auto* any = schema.find("anyOf")) {
      for (const auto& [key, value] : schema.members()) {
        (void)value;
        if (key != "anyOf" && key != "title" && key != "description" &&
            key != "$defs")
          Invalid("anyOf siblings are not supported");
      }
      if (!any->is_array() || any->size() == 0)
        Invalid("anyOf needs at least one branch");
      Sequence branches;
      for (const auto& value : any->items())
        branches.push_back(Visit(value, depth + 1));
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
      if (object || array ||
          (schema.contains("enum") && schema.contains("const")))
        Invalid("enum/const supports one primitive constraint per schema");
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
        if (!std::ranges::any_of(
                types, [&](const auto& t) { return MatchesType(value, t); }))
          Invalid("enum/const value does not match its type");
        if (value.is_number()) {
          auto check = JsonSchemaLexeme::Number(
              schema, std::ranges::find(types, "number") == types.end());
          if (!check->AcceptValue(value))
            continue;
        }
        if (value.is_string()) {
          CountStrings(value.str());
          enum_characters += std::ranges::count_if(
              value.str(), [](unsigned char c) { return (c & 0xc0) != 0x80; });
          if (values.size() > 250 && enum_characters > 15000)
            Invalid(
                "an enum with more than 250 entries is limited to 15000 string "
                "characters");
          auto check = JsonSchemaLexeme::String(schema);
          if (!check->AcceptValue(value))
            continue;
        }
        choices.push_back(Literal(value.dump()));
      }
      if (choices.empty())
        Invalid("enum/const has no value satisfying its constraints");
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
      Invalid("minItems exceeds maxItems");
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

std::shared_ptr<const JsonConstraint> JsonConstraint::WithReasoning(
    std::shared_ptr<const JsonConstraint> answer) {
  // Cache separately from schema compilation. No mutable phase belongs to the
  // model or cache: the normal grammar stack carries it through sampler copies,
  // draft rejection and request restart.
  static std::mutex mutex;
  static std::map<std::shared_ptr<const JsonConstraint>,
                  std::shared_ptr<const JsonConstraint>>
      cache;
  const std::lock_guard lock(mutex);
  if (const auto found = cache.find(answer); found != cache.end())
    return found->second;
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
  if (cache.size() >= 16)
    cache.erase(cache.begin());
  cache.emplace(std::move(answer), grammar);
  return grammar;
}

std::shared_ptr<const JsonConstraint> JsonConstraint::WithTools(
    std::shared_ptr<const JsonConstraint> answer, std::vector<Tool> tools,
    bool required) {
  if (tools.empty())
    return answer;
  using Key = std::tuple<std::shared_ptr<const JsonConstraint>,
                         std::vector<Tool>, bool>;
  static std::mutex mutex;
  static std::map<Key, std::shared_ptr<const JsonConstraint>> cache;
  const std::lock_guard lock(mutex);
  const Key key{answer, tools, required};
  if (const auto found = cache.find(key); found != cache.end())
    return found->second;
  auto grammar = std::shared_ptr<JsonConstraint>(new JsonConstraint(*answer));
  Rule alternatives;
  if (!required)
    alternatives.push_back({answer->root_});
  auto literal = [&](std::string_view text) {
    Sequence bytes;
    for (unsigned char byte : text)
      bytes.push_back(kTerminal | byte);
    const auto id = static_cast<std::uint32_t>(grammar->rules_.size());
    grammar->rules_.push_back({std::move(bytes)});
    return id;
  };
  const auto end = literal("}</tool_call>");
  for (const auto& [name, arguments] : tools) {
    const auto begin = literal(
        "<tool_call>{\"name\":" + json::Value(name).dump() + ",\"arguments\":");
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
    alternatives.push_back(
        {begin, static_cast<std::uint32_t>(arguments->root_ + rules), end});
  }
  grammar->root_ = grammar->rules_.size();
  grammar->rules_.push_back(std::move(alternatives));
  if (cache.size() >= 16)
    cache.erase(cache.begin());
  cache.emplace(key, grammar);
  return grammar;
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
      auto candidate = stack;
      candidate.lexeme += static_cast<char>(byte);
      const auto result =
          lexemes_.at(symbol & ~kLexeme)->Check(candidate.lexeme);
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

ConstraintVocabulary::ConstraintVocabulary(std::uint32_t size,
                                           const Reader& reader) {
  if (size == 0 || size > 1048576)
    throw std::invalid_argument("constraint vocabulary size is unsupported");
  pieces_.reserve(size);
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
    std::uint32_t node = 0;
    for (unsigned char byte : piece.text) {
      auto& edges = trie_[node].edges;
      auto found = std::ranges::find(edges, byte, &Edge::byte);
      if (found != edges.end()) {
        node = found->child;
      } else {
        const auto child = static_cast<std::uint32_t>(trie_.size());
        edges.push_back({child, byte});
        if (trie_.size() >= 4000000)
          throw std::invalid_argument(
              "constraint token trie exceeds its node limit");
        trie_.emplace_back();
        node = child;
      }
    }
    trie_[node].tokens.push_back(token);
  }
}

std::vector<std::uint8_t> ConstraintVocabulary::Allowed(
    const JsonConstraint& grammar, const JsonConstraint::State& state) const {
  std::vector<std::uint8_t> mask(pieces_.size());
  if (grammar.Complete(state)) {
    for (std::size_t i = 0; i < pieces_.size(); ++i)
      mask[i] = pieces_[i].stop;
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
  // Lexical predicates carry the current scalar. Unlike a grammar's string
  // loop, distinct word suffixes cannot be interned into one state. Walk these
  // branches with depth-bounded scratch instead of allocating a 256-way
  // transition table for every vocabulary prefix.
  auto direct = [&](auto&& self, std::uint32_t node,
                    const JsonConstraint::State& current) -> void {
    if (++work > kMaxWork)
      throw std::runtime_error("JSON token mask work limit exceeded");
    for (auto token : trie_[node].tokens)
      mask[token] = 1;
    for (const auto& edge : trie_[node].edges) {
      auto next = grammar.Advance(current, edge.byte);
      if (!next.empty())
        self(self, edge.child, next);
    }
  };
  auto walk = [&](auto&& self, std::uint32_t node,
                  std::uint32_t current) -> void {
    if (std::ranges::any_of(states[current].state, [](const auto& stack) {
          return !stack.symbols.empty() && (stack.symbols.back() & kLexeme);
        })) {
      direct(direct, node, states[current].state);
      return;
    }
    if (++work > kMaxWork)
      throw std::runtime_error("JSON token mask work limit exceeded");
    for (auto token : trie_[node].tokens)
      mask[token] = 1;
    for (const auto& edge : trie_[node].edges) {
      auto& transition = states[current].next[edge.byte];
      if (transition == UINT32_MAX) {
        auto next = grammar.Advance(states[current].state, edge.byte);
        auto [found, inserted] = intern.emplace(next, states.size());
        transition = found->second;
        if (inserted) {
          if (states.size() >= 8192)
            throw std::runtime_error("JSON transition cache limit exceeded");
          states.emplace_back(std::move(next));
        }
      }
      if (transition != 0)
        self(self, edge.child, transition);
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

std::shared_ptr<const std::vector<std::uint8_t>> TokenConstraint::Allowed(
    const JsonConstraint::State& state) const {
  {
    const std::lock_guard lock(mutex_);
    if (const auto found = masks_.find(state); found != masks_.end())
      return found->second;
  }
  auto mask = std::make_shared<const std::vector<std::uint8_t>>(
      vocabulary->Allowed(*grammar, state));
  const std::lock_guard lock(mutex_);
  if (masks_.size() >= 16)
    masks_.erase(masks_.begin());
  return masks_.emplace(state, std::move(mask)).first->second;
}
}  // namespace gufo::sampling
