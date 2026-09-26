#include "src/core/json_constraint.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string_view>

namespace gufo::sampling {
namespace {
constexpr std::uint32_t kTerminal = 1U << 31;
constexpr std::size_t kMaxSchemaBytes = 65536;
constexpr std::size_t kMaxRules = 8192;
constexpr std::size_t kMaxDepth = 16;
constexpr std::size_t kMaxStates = 512;
constexpr std::size_t kMaxStack = 1024;
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
      if (!schema_.is_object() || schema_.member_str("type") != "object" ||
          schema_.contains("anyOf") || schema_.contains("$ref"))
        Invalid("the root must have type object");
      if (const auto* defs = schema_.find("$defs")) {
        if (!defs->is_object() || defs->size() > 128)
          Invalid("$defs must be an object with at most 128 definitions");
        for (const auto& [name, value] : defs->members()) {
          (void)name;
          Visit(value, 1);
        }
      }
      grammar_->root_ = Seq({ws_, Visit(schema_, 0), ws_});
      grammar_->prompt_ =
          "Respond with a single JSON object matching this JSON Schema:\n" +
          schema_.dump();
    }
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
    if (value.size() > 512)
      Invalid("literal values and property names are limited to 512 bytes");
    Sequence sequence;
    for (unsigned char byte : value)
      sequence.push_back(Byte(byte));
    return Seq(std::move(sequence));
  }
  std::uint32_t Optional(std::uint32_t rule) { return New({{}, {rule}}); }
  std::uint32_t Repeat(std::uint32_t rule) {
    const auto id = New();
    grammar_->rules_[id] = {{}, {rule, id}};
    return id;
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
  void Keys(const json::Value& schema) {
    static const std::set<std::string_view> allowed{
        "type",  "properties", "required", "additionalProperties",
        "items", "minItems",   "maxItems", "enum",
        "const", "anyOf",      "$defs",    "$ref",
        "title", "description"};
    for (const auto& [key, value] : schema.members()) {
      if (!allowed.contains(key))
        Invalid("unsupported keyword: " + key);
      if ((key == "title" || key == "description") && !value.is_string())
        Invalid(key + " must be a string");
    }
  }
  std::uint32_t Visit(const json::Value& schema, std::size_t depth) {
    if (!schema.is_object())
      Invalid("each schema must be an object");
    if (depth > kMaxDepth)
      Invalid("maximum schema depth is 16");
    Keys(schema);
    if (const auto* ref = schema.find("$ref")) {
      for (const auto& [key, value] : schema.members()) {
        (void)value;
        if (key != "$ref" && key != "title" && key != "description")
          Invalid("$ref siblings are not supported");
      }
      if (!ref->is_string() || !ref->str().starts_with("#/$defs/"))
        Invalid("only local #/$defs/ references are supported");
      std::string name = ref->str().substr(8);
      std::string decoded;
      for (std::size_t i = 0; i < name.size(); ++i) {
        if (name[i] == '~') {
          if (++i == name.size() || (name[i] != '0' && name[i] != '1'))
            Invalid("invalid JSON pointer escape");
          decoded += name[i] == '0' ? '~' : '/';
        } else {
          if (name[i] == '/')
            Invalid("nested definition pointers are not supported");
          decoded += name[i];
        }
      }
      const auto* defs = schema_.find("$defs");
      const auto* target = defs ? defs->find(decoded) : nullptr;
      if (!target || !active_refs_.insert(decoded).second)
        Invalid("missing or recursive local reference");
      const auto result = Visit(*target, depth + 1);
      active_refs_.erase(decoded);
      return result;
    }
    if (const auto* any = schema.find("anyOf")) {
      for (const auto& [key, value] : schema.members()) {
        (void)value;
        if (key != "anyOf" && key != "title" && key != "description")
          Invalid("anyOf siblings are not supported");
      }
      if (!any->is_array() || any->size() == 0 || any->size() > 8)
        Invalid("anyOf needs between 1 and 8 branches");
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
      if (key == "$defs" && &schema != &schema_)
        Invalid("$defs is only supported at the root");
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
      if (enum_values_ > 256)
        Invalid("maximum enum/const value count is 256");
      Sequence choices;
      for (const auto& value : values) {
        if (!std::ranges::any_of(
                types, [&](const auto& t) { return MatchesType(value, t); }))
          Invalid("enum/const value does not match its type");
        choices.push_back(Literal(value.dump()));
      }
      return Alt(choices);
    }
    Sequence alternatives;
    for (const auto& name : types) {
      if (name == "object")
        alternatives.push_back(Object(schema, depth));
      else if (name == "array")
        alternatives.push_back(Array(schema, depth));
      else
        alternatives.push_back(Primitive(name));
    }
    return Alt(alternatives);
  }
  std::uint32_t Object(const json::Value& schema, std::size_t depth) {
    const auto* properties = schema.find("properties");
    const auto* additional = schema.find("additionalProperties");
    if (!properties || !properties->is_object() || !additional ||
        !additional->is_bool() || additional->as_bool())
      Invalid("objects require properties and additionalProperties: false");
    properties_ += properties->size();
    if (properties_ > 128)
      Invalid("maximum property count is 128");
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
          value->as_double() > 256 ||
          std::floor(value->as_double()) != value->as_double())
        Invalid("array bounds must be integers from 0 to 256");
      return value->as_size();
    };
    const auto minimum = bound("minItems", 0);
    const auto maximum = bound("maxItems", 256);
    if (minimum > maximum)
      Invalid("minItems exceeds maxItems");
    const auto item = Visit(*items, depth + 1);
    if (!schema.contains("maxItems")) {
      const auto additional = Seq({ws_, Byte(','), ws_, item});
      Sequence required{item};
      for (std::size_t count = 1; count < minimum; ++count)
        required.push_back(additional);
      required.push_back(Repeat(additional));
      auto members = Seq(std::move(required));
      if (minimum == 0)
        members = Optional(members);
      return Seq({Byte('['), ws_, members, ws_, Byte(']')});
    }
    auto suffix = Seq({});
    for (std::size_t count = maximum; count > 0; --count) {
      Sequence sequence;
      if (count > 1)
        sequence = {ws_, Byte(','), ws_};
      sequence.insert(sequence.end(), {item, suffix});
      Rule rule{std::move(sequence)};
      if (count > minimum)
        rule.push_back({});
      suffix = New(std::move(rule));
    }
    return Seq({Byte('['), ws_, suffix, ws_, Byte(']')});
  }
  const json::Value& schema_;
  bool strict_;
  std::shared_ptr<JsonConstraint> grammar_{new JsonConstraint};
  std::uint32_t ws_, string_, integer_, number_, bool_, null_;
  std::map<std::size_t, std::uint32_t> generic_values_;
  std::set<std::string> active_refs_;
  std::size_t properties_{0}, enum_values_{0};
};

std::shared_ptr<const JsonConstraint> JsonConstraint::Compile(
    const json::Value& schema, bool strict) {
  const auto key = std::string(strict ? "strict:" : "schema:") + schema.dump();
  if (key.size() > kMaxSchemaBytes)
    Invalid("maximum schema size is 64 KiB");
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

JsonConstraint::State JsonConstraint::Expand(State pending) const {
  State output;
  std::size_t work = 0;
  while (!pending.empty()) {
    if (++work > kMaxWork || pending.size() + output.size() > kMaxStates)
      throw std::runtime_error("JSON grammar state limit exceeded");
    auto stack = std::move(pending.back());
    pending.pop_back();
    if (stack.empty() || (stack.back() & kTerminal)) {
      output.push_back(std::move(stack));
      continue;
    }
    const auto rule = stack.back();
    stack.pop_back();
    for (const auto& sequence : rules_.at(rule)) {
      if (stack.size() + sequence.size() > kMaxStack)
        throw std::runtime_error("JSON grammar stack limit exceeded");
      auto next = stack;
      next.insert(next.end(), sequence.rbegin(), sequence.rend());
      pending.push_back(std::move(next));
    }
  }
  std::ranges::sort(output);
  output.erase(std::unique(output.begin(), output.end()), output.end());
  return output;
}

JsonConstraint::State JsonConstraint::Start() const {
  return Expand({{root_}});
}

JsonConstraint::State JsonConstraint::Advance(const State& state,
                                              unsigned char byte) const {
  State next;
  for (const auto& stack : state) {
    if (!stack.empty() && classes_.at(stack.back() & ~kTerminal).test(byte)) {
      next.push_back(stack);
      next.back().pop_back();
    }
  }
  return Expand(std::move(next));
}

bool JsonConstraint::Complete(const State& state) const {
  return std::ranges::any_of(state,
                             [](const auto& stack) { return stack.empty(); });
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
  auto walk = [&](auto&& self, std::uint32_t node,
                  std::uint32_t current) -> void {
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
