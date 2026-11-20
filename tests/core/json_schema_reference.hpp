#ifndef GUFO_TESTS_CORE_JSON_SCHEMA_REFERENCE_HPP_
#define GUFO_TESTS_CORE_JSON_SCHEMA_REFERENCE_HPP_

// A frozen copy of the C++ JSON Schema compiler of Gufo 0.35.0
// (src/core/json_schema_regex.cpp and the string and number lexemes of
// src/core/json_schema_lexeme.cpp), which json_schema_test compares with the
// Rust port. Only names and visibility changed. Do not fix it: a difference
// from the port must be explained by the port.

#include <unicode/uniset.h>
#include <unicode/unistr.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "src/core/json.hpp"
#include "src/core/json_number_limits.hpp"
#include "src/core/json_schema_lexeme.hpp"

// NOLINTBEGIN
namespace gufo::sampling::reference {

// The compiler of Gufo 0.35.0, with its tables public for comparison.
class Regex {
public:
  static std::shared_ptr<const Regex> Compile(
      std::span<const std::string> patterns, std::uint32_t maximum);
  [[nodiscard]] std::uint32_t Start() const { return 0; }
  [[nodiscard]] bool Accepting(std::uint32_t state) const;
  [[nodiscard]] std::uint32_t MaximumSuffix() const;
  [[nodiscard]] std::uint32_t Advance(std::uint32_t state,
                                      std::uint32_t codepoint) const;
  [[nodiscard]] bool CanFinish(std::uint32_t state, std::uint32_t minimum,
                               std::uint32_t maximum) const;
  [[nodiscard]] bool CanAdvance(std::uint32_t state, std::uint32_t first,
                                std::uint32_t last, std::uint32_t minimum,
                                std::uint32_t maximum) const;
  static constexpr std::uint32_t kDead = UINT32_MAX;
  struct Impl;
  explicit Regex(std::shared_ptr<Impl> impl);
  std::shared_ptr<const Impl> impl_;
};

namespace {
using Id = std::uint32_t;
constexpr Id kDead = Regex::kDead;
constexpr Id kInfinite = UINT32_MAX;
constexpr std::size_t kMaxExpressions = 32768;
constexpr std::size_t kMaxStates = 4096;
constexpr std::size_t kMaxTransitions = 262144;

[[noreturn]] void Invalid(std::string_view reason) {
  throw std::invalid_argument("JSON Schema: " + std::string(reason));
}

icu::UnicodeSet Scalars() {
  icu::UnicodeSet set(0, 0x10ffff);
  return set.remove(0xd800, 0xdfff);
}

bool Word(UChar32 cp) {
  return (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z') ||
         (cp >= '0' && cp <= '9') || cp == '_';
}

enum class Kind {
  Empty,
  Epsilon,
  Chars,
  Start,
  Boundary,
  Or,
  And,
  Not,
  Concat,
  Repeat
};
struct Expression {
  Kind kind;
  std::vector<Id> children{};
  Id low{0}, high{0};
  auto operator<=>(const Expression&) const = default;
};

// Regex derivatives preserve intersections and lookaheads, unlike testing
// predicates separately against the already-generated prefix.
class Expressions {
public:
  explicit Expressions(Id maximum) : maximum_(maximum) {
    empty = Add({Kind::Empty});
    epsilon = Add({Kind::Epsilon});
    start = Add({Kind::Start});
    any = Chars(Scalars());
    all = Repeat(any, 0, kInfinite);
  }
  Id empty, epsilon, start, any, all{kDead};
  bool word_boundaries{false};
  std::vector<icu::UnicodeSet> classes;

  Id Boundary(bool positive) {
    word_boundaries = true;
    icu::UnicodeSet words('a', 'z');
    words.add('A', 'Z').add('0', '9').add('_');
    Chars(std::move(words));  // Include word membership in the DFA alphabet.
    return Add({Kind::Boundary, {}, positive ? 1U : 0U});
  }
  Id Chars(icu::UnicodeSet set) {
    set.remove(0xd800, 0xdfff);
    if (set.isEmpty())
      return empty;
    auto found = std::ranges::find(classes, set);
    Id index = found - classes.begin();
    if (found == classes.end())
      classes.push_back(std::move(set));
    return Add({Kind::Chars, {}, index});
  }
  Id Not(Id child) {
    if (child == empty)
      return all;
    if (child == all)
      return empty;
    if (nodes_[child].kind == Kind::Not)
      return nodes_[child].children[0];
    return Add({Kind::Not, {child}});
  }
  Id Combine(Kind kind, std::vector<Id> parts) {
    std::vector<Id> flat;
    for (auto part : parts) {
      if (kind == Kind::Concat) {
        if (part == empty)
          return empty;
        if (part == epsilon)
          continue;
      } else {
        if (part == (kind == Kind::Or ? all : empty))
          return part;
        if (part == (kind == Kind::Or ? empty : all))
          continue;
      }
      if (nodes_[part].kind == kind)
        flat.insert(flat.end(), nodes_[part].children.begin(),
                    nodes_[part].children.end());
      else if (kind != Kind::Concat || part != all || flat.empty() ||
               flat.back() != all)
        flat.push_back(part);
    }
    if (kind != Kind::Concat) {
      std::ranges::sort(flat);
      flat.erase(std::unique(flat.begin(), flat.end()), flat.end());
      for (auto part : flat)
        if (nodes_[part].kind == Kind::Not &&
            std::ranges::binary_search(flat, nodes_[part].children[0]))
          return kind == Kind::Or ? all : empty;
    }
    if (flat.empty())
      return kind == Kind::Or ? empty : kind == Kind::And ? all : epsilon;
    if (flat.size() == 1)
      return flat[0];
    return Add({kind, std::move(flat)});
  }
  Id Repeat(Id child, Id low, Id high) {
    if (high < low)
      Invalid("invalid regex repetition");
    if (!high || child == epsilon)
      return epsilon;
    if (child == empty)
      return low ? empty : epsilon;
    if (low == 1 && high == 1)
      return child;
    // Repetitions of a nullable language need only its nonempty members.
    // Missing mandatory copies can be filled with epsilon.
    if (Nullable(child, false)) {
      child = Combine(Kind::And, {child, Not(epsilon)});
      low = 0;
    }
    const auto width = minimums_[child];
    if (width && high != kInfinite)
      high = std::min<std::uint64_t>(high, maximum_ / width);
    if (high < low)
      return empty;
    return Add({Kind::Repeat, {child}, low, high});
  }
  bool Nullable(Id id, bool at_start, bool previous_word = false,
                bool next_word = false) const {
    const auto& node = nodes_[id];
    switch (node.kind) {
      case Kind::Empty:
      case Kind::Chars:
        return false;
      case Kind::Epsilon:
        return true;
      case Kind::Start:
        return at_start;
      case Kind::Boundary:
        return (previous_word != next_word) == (node.low != 0);
      case Kind::Not:
        return !Nullable(node.children[0], at_start, previous_word, next_word);
      case Kind::Or:
        return std::ranges::any_of(node.children, [&](Id c) {
          return Nullable(c, at_start, previous_word, next_word);
        });
      case Kind::And:
      case Kind::Concat:
        return std::ranges::all_of(node.children, [&](Id c) {
          return Nullable(c, at_start, previous_word, next_word);
        });
      case Kind::Repeat:
        return node.low == 0;
    }
    return false;
  }
  Id Derive(Id id, UChar32 cp, bool at_start, bool previous_word) {
    const auto key = std::tuple{id, cp, at_start, previous_word};
    if (const auto found = derivatives_.find(key); found != derivatives_.end())
      return found->second;
    // Add() may move the node vector while deriving children.
    const auto node = nodes_[id];
    Id result = empty;
    switch (node.kind) {
      case Kind::Empty:
      case Kind::Epsilon:
      case Kind::Start:
      case Kind::Boundary:
        break;
      case Kind::Chars:
        result = classes[node.low].contains(cp) ? epsilon : empty;
        break;
      case Kind::Not:
        result = Not(Derive(node.children[0], cp, at_start, previous_word));
        break;
      case Kind::Or:
      case Kind::And: {
        std::vector<Id> parts;
        for (auto child : node.children)
          parts.push_back(Derive(child, cp, at_start, previous_word));
        result = Combine(node.kind, std::move(parts));
        break;
      }
      case Kind::Concat: {
        std::vector<Id> choices;
        for (std::size_t i = 0; i < node.children.size(); ++i) {
          std::vector<Id> suffix{
              Derive(node.children[i], cp, at_start, previous_word)};
          suffix.insert(suffix.end(), node.children.begin() + i + 1,
                        node.children.end());
          choices.push_back(Combine(Kind::Concat, std::move(suffix)));
          if (!Nullable(node.children[i], at_start, previous_word, Word(cp)))
            break;
        }
        result = Combine(Kind::Or, std::move(choices));
        break;
      }
      case Kind::Repeat:
        result = Combine(
            Kind::Concat,
            {Derive(node.children[0], cp, at_start, previous_word),
             Repeat(node.children[0], node.low ? node.low - 1 : 0,
                    node.high == kInfinite ? kInfinite : node.high - 1)});
        break;
    }
    if (derivatives_.size() >= 1048576)
      Invalid("regex derivative budget exceeded");
    derivatives_.emplace(key, result);
    return result;
  }

private:
  Id Add(Expression node) {
    if (const auto found = ids_.find(node); found != ids_.end())
      return found->second;
    if (nodes_.size() >= kMaxExpressions)
      Invalid("regex expression budget exceeded");
    std::uint64_t minimum = node.kind == Kind::Empty   ? kInfinite
                            : node.kind == Kind::Chars ? 1
                                                       : 0;
    if (node.kind == Kind::Or) {
      minimum = kInfinite;
      for (auto child : node.children)
        minimum = std::min(minimum, minimums_[child]);
    } else if (node.kind == Kind::Concat) {
      for (auto child : node.children)
        minimum =
            std::min<std::uint64_t>(kInfinite, minimum + minimums_[child]);
    } else if (node.kind == Kind::And) {
      for (auto child : node.children)
        minimum = std::max(minimum, minimums_[child]);
    } else if (node.kind == Kind::Repeat) {
      minimum = std::min<std::uint64_t>(kInfinite,
                                        minimums_[node.children[0]] * node.low);
    }
    if (!nodes_.empty() && minimum > maximum_)
      return empty;
    const Id id = nodes_.size();
    ids_.emplace(node, id);
    nodes_.push_back(std::move(node));
    minimums_.push_back(minimum);
    return id;
  }
  Id maximum_;
  std::vector<Expression> nodes_;
  std::vector<std::uint64_t> minimums_;
  std::map<Expression, Id> ids_;
  std::map<std::tuple<Id, UChar32, bool, bool>, Id> derivatives_;
};

struct Syntax {
  enum Kind {
    Atom,
    Sequence,
    Alternative,
    Repeat,
    Positive,
    Negative,
    End
  } kind{Atom};
  Id atom{0}, low{0}, high{0};
  bool assertion{false};
  std::vector<Syntax> children{};
};

class Parser {
public:
  Parser(Expressions& expressions, std::string_view pattern)
      : e_(expressions), text_(icu::UnicodeString::fromUTF8(pattern)) {
    if (pattern.size() > 16384)
      Invalid("regex exceeds 16384 bytes");
  }
  Id Parse() {
    auto syntax = Alternatives(0);
    if (position_ != text_.length())
      Invalid("unbalanced regex parentheses");
    return e_.Combine(Kind::Concat, {e_.all, Expand(syntax, e_.all)});
  }

private:
  UChar32 Peek() const { return text_.char32At(position_); }
  UChar32 Take() {
    if (position_ >= text_.length())
      Invalid("truncated regex");
    const auto cp = Peek();
    position_ += U16_LENGTH(cp);
    return cp;
  }
  Id Count() {
    if (Peek() < '0' || Peek() > '9')
      Invalid("invalid regex repetition");
    std::uint64_t value = 0;
    while (Peek() >= '0' && Peek() <= '9') {
      value = value * 10 + Take() - '0';
      if (value >= kInfinite)
        Invalid("regex repetition is too large");
    }
    return value;
  }
  UChar32 Hex(unsigned count) {
    UChar32 value = 0;
    for (unsigned i = 0; i < count; ++i) {
      const auto cp = Take();
      const auto digit = cp >= '0' && cp <= '9'   ? cp - '0'
                         : cp >= 'a' && cp <= 'f' ? cp - 'a' + 10
                         : cp >= 'A' && cp <= 'F' ? cp - 'A' + 10
                                                  : -1;
      if (digit < 0)
        Invalid("invalid regex Unicode escape");
      value = (value << 4) | digit;
    }
    return value;
  }
  icu::UnicodeSet Escape(bool in_class) {
    auto cp = Take();
    std::string property;
    switch (cp) {
      case 'd':
      case 'D':
        property = "[0-9]";
        break;
      case 's':
      case 'S':
        // ECMA-262 WhiteSpace + LineTerminator, not ICU White_Space.
        property = "[\\p{Zs}\\u0009-\\u000d\\u2028\\u2029\\ufeff]";
        break;
      case 'w':
      case 'W':
        property = "[a-zA-Z0-9_]";
        break;
      case 'p':
      case 'P': {
        if (Take() != '{')
          Invalid("Unicode property escape requires braces");
        std::string name;
        while (Peek() != '}') {
          const auto c = Take();
          if (c > 127 || name.size() >= 128)
            Invalid("invalid Unicode property name");
          name += static_cast<char>(c);
        }
        ++position_;
        property =
            "[\\" + std::string(1, static_cast<char>(cp)) + "{" + name + "}]";
        break;
      }
      case 'n':
        cp = '\n';
        break;
      case 'r':
        cp = '\r';
        break;
      case 't':
        cp = '\t';
        break;
      case 'f':
        cp = '\f';
        break;
      case 'v':
        cp = '\v';
        break;
      case 'b':
        if (!in_class)
          Invalid("word-boundary regex assertions are unsupported");
        cp = '\b';
        break;
      case '0':
        if (Peek() >= '0' && Peek() <= '9')
          Invalid("legacy octal regex escapes are unsupported");
        cp = 0;
        break;
      case 'c':
        cp = Take();
        if (!((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z')))
          Invalid("regex control escape requires an ASCII letter");
        cp %= 32;
        break;
      case 'u':
        if (Peek() == '{') {
          ++position_;
          cp = 0;
          unsigned digits = 0;
          while (Peek() != '}') {
            ++digits;
            cp = (cp << 4) | Hex(1);
            if (cp > 0x10ffff)
              Invalid("invalid regex Unicode code point");
          }
          ++position_;
          if (!digits)
            Invalid("empty regex Unicode code point");
          break;
        }
        cp = Hex(4);
        if (cp >= 0xd800 && cp <= 0xdbff) {
          if (Take() != '\\' || Take() != 'u')
            Invalid("unpaired surrogate in regex");
          const auto low = Hex(4);
          if (low < 0xdc00 || low > 0xdfff)
            Invalid("unpaired surrogate in regex");
          cp = 0x10000 + ((cp - 0xd800) << 10) + low - 0xdc00;
        }
        break;
      case 'x':
        cp = Hex(2);
        break;
      default:
        if (cp > 127 || (std::string_view("^$\\.*+?()[]{}|/").find(cp) ==
                             std::string_view::npos &&
                         !(in_class && cp == '-')))
          Invalid("unsupported regex escape or backreference");
    }
    if (property.empty())
      return icu::UnicodeSet(cp, cp);
    UErrorCode status = U_ZERO_ERROR;
    icu::UnicodeSet result(icu::UnicodeString::fromUTF8(property), 0, nullptr,
                           status);
    if (U_FAILURE(status))
      Invalid("invalid regex Unicode property");
    if (cp == 'D' || cp == 'S' || cp == 'W')
      result.complement();
    return result;
  }
  icu::UnicodeSet Class() {
    // Parse range endpoints explicitly: whitespace is a literal here, and
    // regex shorthands differ from ICU UnicodeSet shorthand syntax.
    icu::UnicodeSet result;
    const bool negate = Peek() == '^';
    if (negate)
      ++position_;
    while (Peek() != ']') {
      auto cp = Take();
      auto chars = cp == '\\' ? Escape(true) : icu::UnicodeSet(cp, cp);
      if (Peek() == '-' && text_.charAt(position_ + 1) != ']') {
        ++position_;
        cp = Take();
        auto end = cp == '\\' ? Escape(true) : icu::UnicodeSet(cp, cp);
        if (chars.size() != 1 || end.size() != 1 ||
            chars.charAt(0) > end.charAt(0))
          Invalid("invalid regex character range");
        chars.add(chars.charAt(0), end.charAt(0));
      }
      result.addAll(chars);
    }
    ++position_;
    if (negate)
      result.complement();
    return result;
  }
  Syntax Alternatives(unsigned depth) {
    if (depth > 32)
      Invalid("regex nesting exceeds 32 levels");
    Syntax alternatives{Syntax::Alternative};
    do {
      Syntax sequence{Syntax::Sequence};
      while (position_ < text_.length() && Peek() != ')' && Peek() != '|') {
        auto cp = Take();
        Syntax atom;
        if (cp == '(') {
          bool positive = false, negative = false;
          if (Peek() == '?') {
            ++position_;
            const auto modifier = Take();
            positive = modifier == '=';
            negative = modifier == '!';
            if (!positive && !negative && modifier != ':')
              Invalid("unsupported regex group, lookbehind or inline flag");
          }
          atom = Alternatives(depth + 1);
          if (Take() != ')')
            Invalid("unbalanced regex parentheses");
          if (positive || negative) {
            Syntax assertion{positive ? Syntax::Positive : Syntax::Negative};
            assertion.assertion = true;
            assertion.children.push_back(std::move(atom));
            atom = std::move(assertion);
          }
        } else if (cp == '[') {
          atom.atom = e_.Chars(Class());
        } else if (cp == '^') {
          atom.atom = e_.start;
          atom.assertion = true;
        } else if (cp == '$') {
          atom.kind = Syntax::End;
          atom.assertion = true;
          atom.atom = e_.epsilon;
        } else if (cp == '\\') {
          if (Peek() == 'b' || Peek() == 'B') {
            atom.atom = e_.Boundary(Take() == 'b');
            atom.assertion = true;
          } else {
            atom.atom = e_.Chars(Escape(false));
          }
        } else if (cp == '.') {
          auto chars = Scalars();
          for (auto c : {'\n', '\r'})
            chars.remove(c);
          chars.remove(0x2028).remove(0x2029);
          atom.atom = e_.Chars(chars);
        } else {
          if (cp < 128 &&
              std::string_view("*+?{}").find(cp) != std::string_view::npos)
            Invalid("regex repetition has no preceding atom");
          atom.atom = e_.Chars(icu::UnicodeSet(cp, cp));
        }
        auto quantifier = Peek();
        if (quantifier == '*' || quantifier == '+' || quantifier == '?' ||
            quantifier == '{') {
          ++position_;
          Syntax repeated{Syntax::Repeat};
          repeated.low = quantifier == '+' ? 1 : 0;
          repeated.high = quantifier == '?' ? 1 : kInfinite;
          if (quantifier == '{') {
            repeated.low = repeated.high = Count();
            if (Peek() == ',') {
              ++position_;
              repeated.high = Peek() == '}' ? kInfinite : Count();
            }
            if (Take() != '}' || repeated.high < repeated.low)
              Invalid("invalid regex repetition");
          }
          if (atom.assertion &&
              (repeated.high == kInfinite || repeated.high > 32))
            Invalid(
                "regex assertions in unbounded or large repetitions are "
                "unsupported");
          repeated.assertion = atom.assertion;
          repeated.children.push_back(std::move(atom));
          atom = std::move(repeated);
          if (Peek() == '?')
            ++position_;
        }
        sequence.assertion |= atom.assertion;
        sequence.children.push_back(std::move(atom));
      }
      alternatives.assertion |= sequence.assertion;
      alternatives.children.push_back(std::move(sequence));
      if (Peek() != '|')
        break;
      ++position_;
    } while (true);
    return alternatives;
  }
  Id Expand(const Syntax& node, Id tail) {
    switch (node.kind) {
      case Syntax::Atom:
        return e_.Combine(Kind::Concat, {node.atom, tail});
      case Syntax::Sequence:
        for (auto it = node.children.rbegin(); it != node.children.rend(); ++it)
          tail = Expand(*it, tail);
        return tail;
      case Syntax::Alternative: {
        std::vector<Id> choices;
        for (const auto& child : node.children)
          choices.push_back(Expand(child, tail));
        return e_.Combine(Kind::Or, std::move(choices));
      }
      case Syntax::Repeat:
        if (node.assertion) {
          std::vector<Id> choices;
          for (Id count = 0; count <= node.high; ++count) {
            if (count >= node.low)
              choices.push_back(tail);
            if (count < node.high)
              tail = Expand(node.children[0], tail);
          }
          return e_.Combine(Kind::Or, std::move(choices));
        }
        return e_.Combine(Kind::Concat,
                          {e_.Repeat(Expand(node.children[0], e_.epsilon),
                                     node.low, node.high),
                           tail});
      case Syntax::Positive:
      case Syntax::Negative: {
        auto condition = Expand(node.children[0], e_.all);
        if (node.kind == Syntax::Negative)
          condition = e_.Not(condition);
        return e_.Combine(Kind::And, {condition, tail});
      }
      case Syntax::End:
        return e_.Combine(Kind::And, {node.atom, tail});
    }
    return e_.empty;
  }
  Expressions& e_;
  icu::UnicodeString text_;
  int32_t position_{0};
};

}  // namespace

struct Regex::Impl {
  struct State {
    bool accepting{false};
    std::vector<Id> next;
    std::vector<Id> successors;
    Id distance{kDead};
  };
  std::vector<icu::UnicodeSet> alphabet;
  std::vector<State> states;
  Id maximum_suffix{0};
};

Regex::Regex(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}

std::shared_ptr<const Regex> Regex::Compile(
    std::span<const std::string> patterns, Id maximum) {
  Expressions expressions(maximum);
  std::vector<Id> conditions;
  for (const auto& pattern : patterns)
    conditions.push_back(Parser(expressions, pattern).Parse());
  const auto root = expressions.Combine(Kind::And, std::move(conditions));
  auto impl = std::make_shared<Impl>();

  // Partition Unicode once by membership in all character classes. Every
  // codepoint in a partition has the same derivative for every expression.
  std::vector<UChar32> endpoints{0, 0xd800, 0xe000, 0x110000};
  for (const auto& chars : expressions.classes)
    for (int32_t i = 0; i < chars.getRangeCount(); ++i) {
      endpoints.push_back(chars.getRangeStart(i));
      endpoints.push_back(chars.getRangeEnd(i) + 1);
    }
  std::ranges::sort(endpoints);
  endpoints.erase(std::unique(endpoints.begin(), endpoints.end()),
                  endpoints.end());
  std::map<std::vector<Id>, Id> partitions;
  for (std::size_t i = 0; i + 1 < endpoints.size(); ++i) {
    const auto first = endpoints[i], last = endpoints[i + 1] - 1;
    if (first >= 0xd800 && first <= 0xdfff)
      continue;
    std::vector<Id> membership;
    for (Id c = 0; c < expressions.classes.size(); ++c)
      if (expressions.classes[c].contains(first))
        membership.push_back(c);
    const auto [found, inserted] =
        partitions.emplace(membership, impl->alphabet.size());
    if (inserted)
      impl->alphabet.emplace_back();
    impl->alphabet[found->second].add(first, last);
  }

  using Key = std::tuple<Id, bool, bool>;
  std::map<Key, Id> ids;
  std::vector<Key> pending;
  std::vector<Id> depths;
  auto intern = [&](Key key, Id depth) {
    const auto [found, inserted] = ids.emplace(key, ids.size());
    if (inserted) {
      if (ids.size() > kMaxStates ||
          ids.size() * impl->alphabet.size() > kMaxTransitions)
        Invalid("compiled regex exceeds the state budget");
      pending.push_back(key);
      depths.push_back(depth);
      impl->states.push_back({});
    }
    return found->second;
  };
  intern({root, true, false}, 0);
  for (Id i = 0; i < pending.size(); ++i) {
    const auto [expression, at_start, previous_word] = pending[i];
    impl->states[i].accepting =
        expressions.Nullable(expression, at_start, previous_word);
    // BFS discovers each state at its minimum reachable length. Beyond the
    // schema's maxLength no transition can contribute to an accepted value.
    if (depths[i] >= maximum) {
      impl->states[i].next.assign(impl->alphabet.size(), kDead);
      continue;
    }
    for (const auto& chars : impl->alphabet) {
      auto derivative = expressions.Derive(expression, chars.charAt(0),
                                           at_start, previous_word);
      const auto next =
          derivative == expressions.empty
              ? kDead
              : intern({derivative, false,
                        expressions.word_boundaries && Word(chars.charAt(0))},
                       depths[i] + 1);
      impl->states[i].next.push_back(next);
    }
  }
  std::vector<std::vector<Id>> predecessors(impl->states.size());
  std::deque<Id> queue;
  for (Id i = 0; i < impl->states.size(); ++i) {
    auto& state = impl->states[i];
    if (state.accepting) {
      state.distance = 0;
      queue.push_back(i);
    }
    std::set<Id> unique;
    for (auto next : state.next)
      if (next != kDead)
        unique.insert(next);
    state.successors.assign(unique.begin(), unique.end());
    for (auto next : unique)
      predecessors[next].push_back(i);
  }
  while (!queue.empty()) {
    const auto state = queue.front();
    queue.pop_front();
    for (auto previous : predecessors[state])
      if (impl->states[previous].distance == kDead) {
        impl->states[previous].distance = impl->states[state].distance + 1;
        queue.push_back(previous);
      }
  }
  for (auto& state : impl->states)
    for (auto& next : state.next)
      if (next != kDead && impl->states[next].distance == kDead)
        next = kDead;
  for (const auto& state : impl->states)
    if (state.distance != kDead)
      impl->maximum_suffix = std::max(impl->maximum_suffix, state.distance);
  return std::shared_ptr<const Regex>(new Regex(std::move(impl)));
}

bool Regex::Accepting(Id state) const {
  return state != kDead && impl_->states.at(state).accepting;
}
Id Regex::MaximumSuffix() const {
  return impl_->maximum_suffix;
}
Id Regex::Advance(Id state, Id cp) const {
  if (state == kDead)
    return kDead;
  for (Id i = 0; i < impl_->alphabet.size(); ++i)
    if (impl_->alphabet[i].contains(static_cast<UChar32>(cp)))
      return impl_->states.at(state).next[i];
  return kDead;
}
bool Regex::CanFinish(Id state, Id minimum, Id maximum) const {
  if (state == kDead || minimum > maximum)
    return false;
  const auto distance = impl_->states.at(state).distance;
  if (distance == kDead || distance > maximum)
    return false;
  if (distance >= minimum)
    return true;
  std::vector<Id> active{state};
  auto anchor = active;
  std::vector<bool> reachable(impl_->states.size());
  Id steps = 0, block = 1, period = 0;
  bool cycle = false;
  while (steps < minimum) {
    std::fill(reachable.begin(), reachable.end(), false);
    for (auto current : active)
      for (auto successor : impl_->states[current].successors)
        if (impl_->states[successor].distance != kDead)
          reachable[successor] = true;
    active.clear();
    for (Id next = 0; next < reachable.size(); ++next)
      if (reachable[next])
        active.push_back(next);
    if (active.empty())
      return false;
    ++steps;
    if (!cycle) {
      ++period;
      if (active == anchor) {
        // Brent cycle detection bounds scratch memory even for very large
        // minLength. Whole periods preserve the set of possible DFA states.
        steps += ((minimum - steps) / period) * period;
        cycle = true;
      } else if (period == block) {
        anchor = active;
        block *= 2;
        period = 0;
      }
    }
  }
  return std::ranges::any_of(active, [&](Id current) {
    return impl_->states[current].distance <= maximum - minimum;
  });
}
bool Regex::CanAdvance(Id state, Id first, Id last, Id minimum,
                       Id maximum) const {
  if (state == kDead || first > last)
    return false;
  const auto& row = impl_->states.at(state);
  for (Id i = 0; i < impl_->alphabet.size(); ++i)
    if (row.next[i] != kDead &&
        impl_->alphabet[i].containsSome(static_cast<UChar32>(first),
                                        static_cast<UChar32>(last)) &&
        CanFinish(row.next[i], minimum, maximum))
      return true;
  return false;
}

namespace {

// Exact finite-decimal arithmetic. JSON Schema multipleOf must not rely on a
// floating-point remainder (e.g. 0.3 is a multiple of 0.1).
struct Decimal {
  std::string digits{"0"};
  int scale{0};
  bool negative{false};

  static Decimal Parse(std::string_view text) {
    Decimal value;
    value.digits.clear();
    if (text.starts_with('-')) {
      value.negative = true;
      text.remove_prefix(1);
    }
    bool fraction = false;
    std::size_t i = 0;
    for (; i < text.size() && text[i] != 'e' && text[i] != 'E'; ++i) {
      if (text[i] == '.') {
        fraction = true;
      } else {
        value.digits += text[i];
        value.scale += fraction;
      }
    }
    if (i < text.size()) {
      int exponent = 0;
      ++i;
      if (i < text.size() && text[i] == '+')
        ++i;
      const auto result =
          std::from_chars(text.data() + i, text.data() + text.size(), exponent);
      if (result.ec != std::errc{})
        Invalid("numeric exponent is outside the supported range");
      value.scale -= exponent;
    }
    value.Normalize();
    return value;
  }
  void Normalize() {
    const auto first = digits.find_first_not_of('0');
    if (first == std::string::npos) {
      digits = "0";
      scale = 0;
      negative = false;
      return;
    }
    digits.erase(0, first);
    while (digits.size() > 1 && digits.back() == '0') {
      digits.pop_back();
      --scale;
    }
  }
};

int Magnitude(const Decimal& a, const Decimal& b) {
  if (a.digits == "0" || b.digits == "0")
    return (a.digits != "0") - (b.digits != "0");
  const auto ae = static_cast<int>(a.digits.size()) - a.scale;
  const auto be = static_cast<int>(b.digits.size()) - b.scale;
  if (ae != be)
    return ae < be ? -1 : 1;
  for (std::size_t i = 0; i < std::max(a.digits.size(), b.digits.size()); ++i) {
    const char x = i < a.digits.size() ? a.digits[i] : '0';
    const char y = i < b.digits.size() ? b.digits[i] : '0';
    if (x != y)
      return x < y ? -1 : 1;
  }
  return 0;
}

int Compare(const Decimal& a, const Decimal& b) {
  if (a.negative != b.negative)
    return a.negative ? -1 : 1;
  return (a.negative ? -1 : 1) * Magnitude(a, b);
}

int NaturalCompare(std::string_view a, std::string_view b) {
  if (a.size() != b.size())
    return a.size() < b.size() ? -1 : 1;
  return a.compare(b);
}

void Subtract(std::string& a, std::string_view b) {
  int borrow = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    int digit = a[a.size() - 1 - i] - '0' - borrow;
    if (i < b.size())
      digit -= b[b.size() - 1 - i] - '0';
    borrow = digit < 0;
    a[a.size() - 1 - i] = '0' + digit + 10 * borrow;
  }
  const auto first = a.find_first_not_of('0');
  a = first == std::string::npos ? "0" : a.substr(first);
}

std::pair<std::string, std::string> Divide(const Decimal& value,
                                           const Decimal& divisor) {
  const int scale = std::max(value.scale, divisor.scale);
  std::string numerator = value.digits + std::string(scale - value.scale, '0');
  std::string denominator =
      divisor.digits + std::string(scale - divisor.scale, '0');
  std::string remainder{"0"};
  std::string quotient;
  for (const char digit : numerator) {
    if (remainder == "0")
      remainder.clear();
    remainder += digit;
    unsigned count = 0;
    while (NaturalCompare(remainder, denominator) >= 0)
      Subtract(remainder, denominator), ++count;
    if (count || !quotient.empty())
      quotient += static_cast<char>('0' + count);
  }
  return {quotient.empty() ? "0" : quotient, remainder};
}

bool Multiple(const Decimal& value, const Decimal& divisor) {
  return value.digits == "0" || Divide(value, divisor).second == "0";
}

std::string Increment(std::string value) {
  for (std::size_t i = value.size(); i > 0; --i) {
    if (value[i - 1] != '9') {
      ++value[i - 1];
      return value;
    }
    value[i - 1] = '0';
  }
  return "1" + value;
}

Decimal Product(const Decimal& value, std::string_view factor) {
  std::vector<unsigned> digits(value.digits.size() + factor.size());
  for (std::size_t i = 0; i < value.digits.size(); ++i)
    for (std::size_t j = 0; j < factor.size(); ++j)
      digits[i + j + 1] += (value.digits[i] - '0') * (factor[j] - '0');
  for (std::size_t i = digits.size() - 1; i > 0; --i) {
    digits[i - 1] += digits[i] / 10;
    digits[i] %= 10;
  }
  Decimal result{"", value.scale, false};
  for (unsigned digit : digits)
    result.digits += static_cast<char>('0' + digit);
  result.Normalize();
  return result;
}

Decimal AddUnit(const Decimal& value, int scale) {
  Decimal next = value;
  next.negative = false;
  if (next.digits == "0")
    next = Decimal{"1", scale, false};
  else {
    next.digits.append(scale - next.scale, '0');
    next.scale = scale;
    bool carry = true;
    for (std::size_t i = next.digits.size(); i && carry; --i) {
      carry = next.digits[i - 1] == '9';
      next.digits[i - 1] = carry ? '0' : next.digits[i - 1] + 1;
    }
    if (carry)
      next.digits.insert(0, 1, '1');
    next.Normalize();
  }
  return next;
}

class NumberLexeme final : public JsonSchemaLexeme {
public:
  NumberLexeme(const json::Value& schema, bool integer)
      : JsonSchemaLexeme("0123456789-."), integer_(integer) {
    for (const auto* key : {"minimum", "maximum", "exclusiveMinimum",
                            "exclusiveMaximum", "multipleOf"}) {
      const auto* entry = schema.find(key);
      if (!entry)
        continue;
      if (!entry->is_number() || !std::isfinite(entry->as_double()))
        Invalid(std::string(key) + " must be a finite number");
      const auto value = Decimal::Parse(entry->dump());
      if (std::string_view(key) == "multipleOf") {
        if (value.negative || value.digits == "0")
          Invalid("multipleOf must be positive");
        multiple_ = value;
        continue;
      }
      const bool lower = std::string_view(key) == "minimum" ||
                         std::string_view(key) == "exclusiveMinimum";
      auto& bound = lower ? lower_ : upper_;
      const bool exclusive = std::string_view(key).starts_with("exclusive");
      const int comparison = bound ? Compare(value, bound->value) : 0;
      if (!bound || (lower ? comparison > 0 : comparison < 0) ||
          (comparison == 0 && exclusive))
        bound = Bound{value, exclusive};
    }
    if (lower_ && upper_) {
      const int comparison = Compare(lower_->value, upper_->value);
      if (comparison > 0 ||
          (comparison == 0 &&
           (lower_->exclusive || upper_->exclusive || !Valid(lower_->value))))
        throw JsonSchemaEmpty(
            "JSON Schema: numeric constraints describe an empty interval");
    }
    if (integer_) {
      if (!multiple_)
        multiple_ = Decimal{"1", 0, false};
      if (multiple_->scale > 0) {
        const int scale = multiple_->scale;
        multiple_->scale = 0;
        for (const int factor : {2, 5}) {
          for (int count = 0; count < scale; ++count) {
            const auto division =
                Divide(*multiple_, Decimal{std::to_string(factor), 0, false});
            if (division.second != "0")
              break;
            multiple_->digits = division.first;
          }
        }
        multiple_->Normalize();
      }
    }
    if (lower_ && upper_ && multiple_ && !GridPoint(*lower_, *upper_))
      throw JsonSchemaEmpty(
          "JSON Schema: numeric constraints contain no multipleOf value");
    // Widen the digit caps only for schema numbers that need longer values.
    for (const auto* value :
         {lower_ ? &lower_->value : nullptr, upper_ ? &upper_->value : nullptr,
          multiple_ ? &*multiple_ : nullptr}) {
      if (!value)
        continue;
      const int integral =
          static_cast<int>(value->digits.size()) - value->scale;
      if (integral >= integral_digits_)
        integral_digits_ = integral + 1;
      if (value->scale >= fraction_digits_)
        fraction_digits_ = value->scale + 1;
    }
  }
  bool AcceptValue(const json::Value& value) const override {
    return value.is_number() && Valid(Decimal::Parse(value.dump()));
  }

  Match Check(std::string_view bytes) const override {
    if (bytes.empty())
      return {true, false};
    if (bytes.size() > 4096)
      return {};
    const bool negative = bytes.starts_with('-');
    auto text = bytes;
    if (negative)
      text.remove_prefix(1);
    if (text.empty())
      return {!lower_ || lower_->value.negative ||
                  (lower_->value.digits == "0" && !lower_->exclusive),
              false};
    const auto dot = text.find('.');
    if (dot == 0 || (integer_ && dot != std::string_view::npos))
      return {};
    if (text.front() == '0' && text.size() > 1 && text[1] != '.')
      return {};
    for (std::size_t i = 0; i < text.size(); ++i)
      if ((text[i] < '0' || text[i] > '9') && (text[i] != '.' || i != dot))
        return {};
    const int integral =
        static_cast<int>(dot == std::string_view::npos ? text.size() : dot);
    const int places =
        dot == std::string_view::npos ? 0 : text.size() - dot - 1;
    if (integral > integral_digits_ || places > fraction_digits_)
      return {};
    const auto value = Decimal::Parse(bytes);
    const bool complete = text.back() != '.' && Valid(value);
    // At the scalar budget or the decimal cap there is no room for another
    // digit. Do not admit an unfinished prefix which would fail next step.
    if (bytes.size() == 4096 || places == fraction_digits_)
      return {complete, complete};
    // Decimal prefixes cover an interval. Before the decimal point, appending
    // integer digits also shifts that interval by powers of ten.
    Decimal low = value;
    low.negative = false;
    Decimal high = AddUnit(low, places);
    const bool extend_integer =
        dot == std::string_view::npos && text.front() != '0';
    for (unsigned shift = 0; shift <= 1024; ++shift) {
      Decimal first = negative ? high : low;
      Decimal last = negative ? low : high;
      first.negative = negative && first.digits != "0";
      last.negative = negative && last.digits != "0";
      const bool intersects_lower =
          !lower_ || Compare(last, lower_->value) > 0 ||
          (negative && !lower_->exclusive && Compare(last, lower_->value) == 0);
      const bool intersects_upper = !upper_ ||
                                    Compare(first, upper_->value) < 0 ||
                                    (!negative && !upper_->exclusive &&
                                     Compare(first, upper_->value) == 0);
      if (intersects_lower && intersects_upper) {
        Bound lower{first, negative}, upper{last, !negative};
        if (lower_ &&
            (Compare(lower_->value, lower.value) > 0 ||
             (Compare(lower_->value, lower.value) == 0 && lower_->exclusive)))
          lower = *lower_;
        if (upper_ &&
            (Compare(upper_->value, upper.value) < 0 ||
             (Compare(upper_->value, upper.value) == 0 && upper_->exclusive)))
          upper = *upper_;
        if (!multiple_ || GridPoint(lower, upper))
          return {true, complete};
      }
      if (!extend_integer ||
          integral + static_cast<int>(shift) + 1 > integral_digits_ ||
          (negative && !intersects_lower) || (!negative && !intersects_upper))
        break;
      --low.scale;
      --high.scale;
    }
    return {complete, complete};
  }

private:
  struct Bound {
    Decimal value;
    bool exclusive;
  };
  bool GridPoint(Bound low, Bound high) const {
    if (low.value.negative && !high.value.negative && high.value.digits != "0")
      return true;  // Zero is inside.
    if (low.value.negative) {
      std::swap(low, high);
      low.value.negative = false;
      high.value.negative = false;
    }
    auto [quotient, remainder] = Divide(low.value, *multiple_);
    if (remainder != "0" || low.exclusive)
      quotient = Increment(std::move(quotient));
    const auto candidate = Product(*multiple_, quotient);
    const auto comparison = Compare(candidate, high.value);
    return comparison < 0 || (comparison == 0 && !high.exclusive);
  }
  bool Valid(const Decimal& value) const {
    if (integer_ && value.scale > 0)
      return false;
    if (lower_ &&
        Compare(value, lower_->value) < static_cast<int>(lower_->exclusive))
      return false;
    if (upper_ &&
        Compare(value, upper_->value) > -static_cast<int>(upper_->exclusive))
      return false;
    return !multiple_ || Multiple(value, *multiple_);
  }
  bool integer_;
  std::optional<Bound> lower_, upper_;
  std::optional<Decimal> multiple_;
  int integral_digits_{kMaxJsonNumberDigits},
      fraction_digits_{kMaxJsonNumberDigits};
};

// Formats follow JSON Schema's RFC-backed string formats. The grammar validates
// decoded strings, so an escaped character has exactly the same meaning as its
// UTF-8 spelling.
std::string FormatPattern(std::string_view format) {
  const std::string leap =
      R"((?:[0-9]{2}(?:0[48]|[2468][048]|[13579][26])|(?:[02468][048]|[13579][26])00))";
  const std::string date =
      R"((?:[0-9]{4}-(?:(?:0[13578]|1[02])-(?:0[1-9]|[12][0-9]|3[01])|(?:0[469]|11)-(?:0[1-9]|[12][0-9]|30)|02-(?:0[1-9]|1[0-9]|2[0-8]))|)" +
      leap + R"(-02-29))";
  const std::string time =
      R"((?:[01][0-9]|2[0-3]):[0-5][0-9]:(?:[0-5][0-9]|60)(?:\.[0-9]+)?(?:[zZ]|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9]))";
  if (format == "date")
    return "^" + date + "$";
  if (format == "time")
    return "^" + time + "$";
  if (format == "date-time")
    return "^" + date + "[tT]" + time + "$";
  if (format == "uuid")
    return R"(^[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$)";
  if (format == "ipv4")
    return R"(^(?:(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])$)";
  if (format == "hostname")
    return R"(^[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*\.?$)";
  if (format == "email")
    return R"(^[^@\s]+@[^@\s]+$)";
  if (format == "duration")
    return R"(^P(?:[0-9]+W|(?=[0-9]|T[0-9])(?:[0-9]+Y)?(?:[0-9]+M)?(?:[0-9]+D)?(?:T(?=[0-9])(?:[0-9]+H)?(?:[0-9]+M)?(?:[0-9]+(?:\.[0-9]+)?S)?)?)$)";
  if (format == "ipv6") {
    const std::string group = "[0-9a-fA-F]{1,4}";
    auto groups = [&](int count) {
      std::string result;
      for (int i = 0; i < count; ++i)
        result += (i ? ":" : "") + group;
      return result;
    };
    std::string result = "^(?:" + groups(8);
    for (int left = 0; left < 8; ++left)
      for (int right = 0; left + right < 8; ++right)
        result += "|" + groups(left) + "::" + groups(right);
    auto ipv4 = FormatPattern("ipv4");
    ipv4 = ipv4.substr(1, ipv4.size() - 2);
    result += "|" + groups(6) + ":" + ipv4;
    for (int left = 0; left < 6; ++left)
      for (int right = 0; left + right < 6; ++right)
        result += "|" + groups(left) + "::" + groups(right) +
                  (right ? ":" : "") + ipv4;
    return result + ")$";
  }
  Invalid("unsupported string format");
}

class StringLexeme final : public JsonSchemaLexeme {
  enum Phase : std::uint32_t {
    Body,
    Escape,
    Hex,
    LowSlash,
    LowU,
    LowHex,
    Utf8
  };
  // Canonical request-local state. Completed characters do not retain their
  // spelling, so literal/escaped Unicode shares cached vocabulary transitions.
  enum Field { Dfa, Count, Value, Extra, Mode, Fields };
  using State = std::array<std::uint32_t, Fields>;

public:
  explicit StringLexeme(const json::Value& schema) {
    for (const auto* key : {"minLength", "maxLength"}) {
      if (const auto* value = schema.find(key)) {
        if (!value->is_number() || value->as_double() < 0 ||
            value->as_double() > 1048576 ||
            std::floor(value->as_double()) != value->as_double())
          Invalid("string lengths must be nonnegative integers up to 1048576");
        (std::string_view(key) == "minLength" ? minimum_ : maximum_) =
            value->as_size();
      }
    }
    if (minimum_ > maximum_)
      throw JsonSchemaEmpty("JSON Schema: minLength exceeds maxLength");
    std::vector<std::string> patterns;
    for (const auto* key : {"pattern", "format"}) {
      const auto* value = schema.find(key);
      if (!value)
        continue;
      if (!value->is_string())
        Invalid(std::string(key) + " must be a string");
      auto expression = std::string_view(key) == "format"
                            ? FormatPattern(value->str())
                            : value->str();
      if (std::string_view(key) == "format") {
        if (value->str() == "hostname")
          maximum_ = std::min(maximum_, 253U);
      }
      patterns.push_back(std::move(expression));
    }
    if (patterns.empty()) {
      scalar_only_ = true;
      static const auto unconstrained = Regex::Compile({}, UINT32_MAX);
      regex_ = unconstrained;
    } else {
      regex_ = Regex::Compile(patterns, maximum_);
    }
    if (!regex_->CanFinish(regex_->Start(), minimum_, maximum_))
      throw JsonSchemaEmpty(
          "JSON Schema: string predicates and lengths have no matching value");
  }

  bool CacheTransitions() const override { return true; }
  void CanonicalMaskState(std::string& encoded,
                          std::size_t token_bytes) const override {
    if (encoded.empty())
      return;
    State state{};
    if (encoded.size() != sizeof(state))
      throw std::logic_error("invalid JSON string matcher state");
    std::memcpy(state.data(), encoded.data(), sizeof(state));
    // A token can complete at most one Unicode character per byte. If even
    // its longest accepting suffix fits, the upper bound cannot affect its
    // mask. Only canonicalize the cache key; the request keeps its true count.
    const auto window = token_bytes + regex_->MaximumSuffix();
    if (scalar_only_ && state[Count] < minimum_ &&
        minimum_ - state[Count] > token_bytes) {
      // No vocabulary token can reach the minimum or close this string yet.
      // With no pattern, every valid Unicode continuation can still finish.
      state[Count] = 0;
    } else if (state[Count] >= minimum_ && maximum_ - state[Count] >= window) {
      state[Count] = minimum_;
    }
    encoded.assign(reinterpret_cast<const char*>(state.data()), sizeof(state));
  }
  Match Check(std::string_view bytes) const override {
    std::string state;
    Match result{true, false};
    for (unsigned char byte : bytes) {
      if (!result.prefix)
        return {};
      result = Advance(state, byte);
    }
    return result;
  }
  Match Advance(std::string& encoded, unsigned char byte) const override {
    State state{};
    if (encoded.empty()) {
      if (byte != '"')
        return {};
      state[Dfa] = regex_->Start();
    } else {
      if (encoded.size() != sizeof(state))
        throw std::logic_error("invalid JSON string matcher state");
      std::memcpy(state.data(), encoded.data(), sizeof(state));
      switch (state[Mode]) {
        case Body:
          if (byte == '"')
            return {false,
                    state[Count] >= minimum_ && regex_->Accepting(state[Dfa])};
          if (byte == '\\') {
            state[Mode] = Escape;
          } else if (byte < 0x20) {
            return {};
          } else if (byte < 0x80) {
            if (!Character(state, byte))
              return {};
          } else {
            const unsigned width = byte >= 0xc2 && byte <= 0xdf   ? 2
                                   : byte >= 0xe0 && byte <= 0xef ? 3
                                   : byte >= 0xf0 && byte <= 0xf4 ? 4
                                                                  : 0;
            if (!width)
              return {};
            state[Mode] = Utf8;
            if (scalar_only_) {
              const auto low = byte == 0xe0 ? 0xa0 : byte == 0xf0 ? 0x90 : 0x80;
              const auto high = byte == 0xed   ? 0x9f
                                : byte == 0xf4 ? 0x8f
                                               : 0xbf;
              state[Value] = (low << 8) | high;
              state[Extra] = width - 1;
            } else {
              state[Value] = byte & ((1U << (7 - width)) - 1);
              state[Extra] = (width << 8) | (width - 1);
            }
          }
          break;
        case Escape:
          if (byte == 'u') {
            state[Mode] = Hex;
            state[Extra] = 4;
          } else {
            const auto index = std::string_view("\"\\/bfnrt").find(byte);
            if (index == std::string_view::npos)
              return {};
            constexpr std::array<unsigned char, 8> values{
                '"', '\\', '/', '\b', '\f', '\n', '\r', '\t'};
            if (!Character(state, values[index]))
              return {};
          }
          break;
        case LowSlash:
          if (byte != '\\')
            return {};
          state[Mode] = LowU;
          break;
        case LowU:
          if (byte != 'u')
            return {};
          state[Mode] = LowHex;
          state[Extra] = (state[Value] << 8) | 4;
          state[Value] = 0;
          break;
        case Hex:
        case LowHex: {
          const int digit = byte >= '0' && byte <= '9'   ? byte - '0'
                            : byte >= 'a' && byte <= 'f' ? byte - 'a' + 10
                            : byte >= 'A' && byte <= 'F' ? byte - 'A' + 10
                                                         : -1;
          if (digit < 0)
            return {};
          if (scalar_only_) {
            const auto remaining = state[Extra] & 0xff;
            if (state[Mode] == LowHex) {
              if ((remaining == 4 && digit != 13) ||
                  (remaining == 3 && digit < 12))
                return {};
            } else if (remaining == 4) {
              state[Value] = digit == 13 ? 1 : 0;
            } else if (remaining == 3 && state[Value] == 1) {
              if (digit >= 12)
                return {};
              state[Value] = digit >= 8 ? 2 : 0;
            }
            --state[Extra];
            if (!(state[Extra] & 0xff)) {
              if (state[Mode] == Hex && state[Value] == 2) {
                state[Value] = 0xd800;
                state[Mode] = LowSlash;
              } else if (!Character(state, 0)) {
                return {};
              }
            }
            break;
          }
          state[Value] = (state[Value] << 4) | digit;
          --state[Extra];
          if (!(state[Extra] & 0xff)) {
            auto cp = state[Value];
            if (state[Mode] == LowHex) {
              if (cp < 0xdc00 || cp > 0xdfff)
                return {};
              cp =
                  0x10000 + ((state[Extra] >> 8) - 0xd800) * 1024 + cp - 0xdc00;
            } else if (cp >= 0xd800 && cp <= 0xdbff) {
              state[Mode] = LowSlash;
              state[Extra] = 0;
              break;
            }
            if (!Character(state, cp))
              return {};
          }
          break;
        }
        case Utf8:
          if (scalar_only_) {
            if (byte < (state[Value] >> 8) || byte > (state[Value] & 0xff))
              return {};
            if (!--state[Extra]) {
              if (!Character(state, 0))
                return {};
            } else {
              state[Value] = (0x80 << 8) | 0xbf;
            }
            break;
          }
          if (byte < 0x80 || byte > 0xbf)
            return {};
          state[Value] = (state[Value] << 6) | (byte & 0x3f);
          --state[Extra];
          if (!(state[Extra] & 0xff)) {
            const auto width = state[Extra] >> 8;
            const auto minimum = width == 2   ? 0x80U
                                 : width == 3 ? 0x800U
                                              : 0x10000U;
            if (state[Value] < minimum || !Character(state, state[Value]))
              return {};
          }
          break;
        default:
          throw std::logic_error("invalid JSON string matcher phase");
      }
    }
    if (state[Mode] != Body && !Pending(state))
      return {};
    encoded.assign(reinterpret_cast<const char*>(state.data()), sizeof(state));
    return {true, false};
  }

private:
  std::uint32_t RemainingMinimum(std::uint32_t count) const {
    return count >= minimum_ ? 0 : minimum_ - count;
  }
  bool Character(State& state, std::uint32_t cp) const {
    if (state[Count] == maximum_ || cp > 0x10ffff ||
        (cp >= 0xd800 && cp <= 0xdfff))
      return false;
    const auto count = state[Count] + 1;
    if (!scalar_only_) {
      const auto next = regex_->Advance(state[Dfa], cp);
      if (!regex_->CanFinish(next, RemainingMinimum(count), maximum_ - count))
        return false;
      state[Dfa] = next;
    }
    state[Count] = maximum_ == UINT32_MAX ? std::min(count, minimum_) : count;
    state[Value] = state[Extra] = 0;
    state[Mode] = Body;
    return true;
  }
  bool Range(const State& state, std::uint32_t first,
             std::uint32_t last) const {
    if (first > last || state[Count] == maximum_)
      return false;
    const auto count = state[Count] + 1;
    return regex_->CanAdvance(state[Dfa], first, last, RemainingMinimum(count),
                              maximum_ - count);
  }
  bool Pending(const State& state) const {
    if (scalar_only_)
      return state[Count] < maximum_;
    if (state[Mode] == Escape)
      return Range(state, 0, 0x10ffff);
    if (state[Mode] == Utf8) {
      const auto shift = 6 * (state[Extra] & 0xff);
      const auto width = state[Extra] >> 8;
      const auto minimum = width == 2 ? 0x80U : width == 3 ? 0x800U : 0x10000U;
      return Range(state, std::max(minimum, state[Value] << shift),
                   std::min(0x10ffffU, ((state[Value] + 1) << shift) - 1));
    }
    if (state[Mode] == LowSlash || state[Mode] == LowU) {
      const auto first = 0x10000 + (state[Value] - 0xd800) * 1024;
      return Range(state, first, first + 1023);
    }
    const auto shift = 4 * (state[Extra] & 0xff);
    const auto first = state[Value] << shift;
    const auto last = ((state[Value] + 1) << shift) - 1;
    if (state[Mode] == LowHex) {
      const auto low = std::max(first, 0xdc00U), high = std::min(last, 0xdfffU);
      if (low > high)
        return false;
      const auto base = 0x10000 + ((state[Extra] >> 8) - 0xd800) * 1024;
      return Range(state, base + low - 0xdc00, base + high - 0xdc00);
    }
    if (first <= 0xd7ff && Range(state, first, std::min(last, 0xd7ffU)))
      return true;
    if (last >= 0xe000 && Range(state, std::max(first, 0xe000U), last))
      return true;
    if (first <= 0xdbff && last >= 0xd800)
      return Range(state, 0x10000 + (std::max(first, 0xd800U) - 0xd800) * 1024,
                   0x10000 + (std::min(last, 0xdbffU) - 0xd800 + 1) * 1024 - 1);
    return false;
  }
  std::uint32_t minimum_{0}, maximum_{UINT32_MAX};
  bool scalar_only_{false};
  std::shared_ptr<const Regex> regex_;
};

}  // namespace

std::shared_ptr<const JsonSchemaLexeme> String(const json::Value& schema) {
  return std::make_shared<StringLexeme>(schema);
}

json::Value IntersectMultipleOf(const json::Value& a, const json::Value& b) {
  for (const auto* value : {&a, &b})
    if (!value->is_number() || !std::isfinite(value->as_double()) ||
        value->as_double() <= 0)
      Invalid("multipleOf must be a positive finite number");
  auto left = Decimal::Parse(a.dump()), right = Decimal::Parse(b.dump());
  // Put both steps on the same integer grid, then compute the exact LCM.
  const int scale = std::max(left.scale, right.scale);
  left.digits.append(scale - left.scale, '0');
  right.digits.append(scale - right.scale, '0');
  left.scale = right.scale = 0;
  auto x = left, y = right;
  while (y.digits != "0") {
    auto remainder = Divide(x, y).second;
    x = std::move(y);
    y = Decimal{std::move(remainder), 0, false};
  }
  auto common = Product(Decimal{Divide(left, x).first, 0, false}, right.digits);
  common.scale += scale;
  common.Normalize();
  const auto result =
      json::parse(common.digits + "e" + std::to_string(-common.scale));
  if (!result.is_number() || !std::isfinite(result.as_double()) ||
      Compare(Decimal::Parse(result.dump()), common) != 0)
    Invalid("combined multipleOf exceeds the exact schema-number range");
  return result;
}

json::Value Format(std::string_view format) {
  auto schema = json::Value::object();
  schema["pattern"] = FormatPattern(format);
  if (format == "hostname")
    schema["maxLength"] = 253;
  return schema;
}
std::shared_ptr<const JsonSchemaLexeme> Number(const json::Value& schema,
                                               bool integer) {
  return std::make_shared<NumberLexeme>(schema, integer);
}

}  // namespace gufo::sampling::reference
// NOLINTEND
#endif
