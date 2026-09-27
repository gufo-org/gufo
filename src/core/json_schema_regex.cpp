#include "src/core/json_schema_regex.hpp"

#include <unicode/uniset.h>
#include <unicode/unistr.h>

#include <algorithm>
#include <deque>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace gufo::sampling {
namespace {
using Id = std::uint32_t;
constexpr Id kDead = JsonSchemaRegex::kDead;
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

struct JsonSchemaRegex::Impl {
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

JsonSchemaRegex::JsonSchemaRegex(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

std::shared_ptr<const JsonSchemaRegex> JsonSchemaRegex::Compile(
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
  return std::shared_ptr<const JsonSchemaRegex>(
      new JsonSchemaRegex(std::move(impl)));
}

bool JsonSchemaRegex::Accepting(Id state) const {
  return state != kDead && impl_->states.at(state).accepting;
}
Id JsonSchemaRegex::MaximumSuffix() const {
  return impl_->maximum_suffix;
}
Id JsonSchemaRegex::Advance(Id state, Id cp) const {
  if (state == kDead)
    return kDead;
  for (Id i = 0; i < impl_->alphabet.size(); ++i)
    if (impl_->alphabet[i].contains(static_cast<UChar32>(cp)))
      return impl_->states.at(state).next[i];
  return kDead;
}
bool JsonSchemaRegex::CanFinish(Id state, Id minimum, Id maximum) const {
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
bool JsonSchemaRegex::CanAdvance(Id state, Id first, Id last, Id minimum,
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

}  // namespace gufo::sampling
