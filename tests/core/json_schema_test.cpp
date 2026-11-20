// Compares the Rust JSON Schema compiler, through JsonSchemaRegex and the
// string and number lexemes, with the frozen C++ compiler of Gufo 0.35.0 in
// json_schema_reference.hpp. Generated patterns and schemas, deterministic
// mutations and every format must compile to the same DFA tables, step the
// same way, or fail with the same exception and message.

#include <gufo/json_schema.h>
#include <unicode/uniset.h>
#include <unicode/unistr.h>
#include <unicode/usetiter.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <typeinfo>
#include <vector>

#include "src/core/json.hpp"
#include "src/core/json_schema_lexeme.hpp"
#include "src/core/json_schema_regex.hpp"
#include "tests/core/json_schema_reference.hpp"

namespace {

using gufo::json::Value;
using gufo::sampling::JsonSchemaEmpty;
using gufo::sampling::JsonSchemaLexeme;
using gufo::sampling::JsonSchemaRegex;
namespace reference = gufo::sampling::reference;

int failures = 0;
std::size_t compared = 0;
// Cases both compilers accepted; the rest were rejected by both.
std::size_t accepted = 0;

void Fail(const std::string& what) {
  if (++failures <= 40)
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
}

// Deterministic generator: the same cases on every run and platform.
struct Random {
  std::uint64_t state;
  std::uint32_t Next() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<std::uint32_t>(state >> 33);
  }
  std::size_t Below(std::size_t n) { return Next() % n; }
};

// What a call did: its result, or which exception it threw and why.
struct Outcome {
  bool ok{false};
  std::string error;
  bool operator==(const Outcome&) const = default;
};

template<class F>
Outcome Run(F&& call) {
  try {
    call();
    return {true, ""};
  } catch (const JsonSchemaEmpty& e) {
    return {false, std::string("empty: ") + e.what()};
  } catch (const std::invalid_argument& e) {
    return {false, std::string("invalid: ") + e.what()};
  } catch (const std::exception& e) {
    return {false, std::string("other: ") + e.what()};
  }
}

std::string Show(std::string_view text) {
  std::string out;
  for (unsigned char c : text) {
    if (c >= 0x20 && c < 0x7f && c != '\\') {
      out += static_cast<char>(c);
    } else {
      char buffer[8];
      std::snprintf(buffer, sizeof(buffer), "\\x%02x", c);
      out += buffer;
    }
  }
  return out;
}

// ICU stands behind the Rust compiler's sets here as in production.
struct Sets {
  bool failed{false};
};

int Resolve(void* context, const std::uint8_t* pattern, std::size_t len,
            GufoJsonSchemaRangeFn add, GufoJsonSchemaStringFn add_string,
            void* sink) noexcept {
  try {
    UErrorCode status = U_ZERO_ERROR;
    const icu::UnicodeSet set(icu::UnicodeString::fromUTF8(std::string_view(
                                  reinterpret_cast<const char*>(pattern), len)),
                              0, nullptr, status);
    if (U_FAILURE(status))
      return 1;
    for (int32_t i = 0; i < set.getRangeCount(); ++i)
      add(sink, set.getRangeStart(i), set.getRangeEnd(i));
    icu::UnicodeSetIterator strings(set);
    strings.skipToStrings();
    std::vector<std::uint32_t> codepoints;
    while (strings.next()) {
      const auto& string = strings.getString();
      codepoints.clear();
      for (int32_t i = 0; i < string.length(); i = string.moveIndex32(i, 1))
        codepoints.push_back(string.char32At(i));
      add_string(sink, codepoints.data(), codepoints.size());
    }
    return 0;
  } catch (...) {
    static_cast<Sets*>(context)->failed = true;
    return 2;
  }
}

// The Rust tables must equal the reference compiler's.
void CompareTables(const std::vector<std::string>& patterns,
                   std::uint32_t maximum, const reference::Regex& expected,
                   const std::string& name) {
  std::vector<GufoJsonSchemaBytes> entries;
  for (const auto& pattern : patterns)
    entries.push_back({reinterpret_cast<const std::uint8_t*>(pattern.data()),
                       pattern.size()});
  Sets sets;
  GufoJsonSchemaRegex* handle = nullptr;
  const auto status =
      gufo_json_schema_regex_compile(entries.data(), entries.size(), maximum,
                                     Resolve, nullptr, &sets, &handle);
  if (status != GUFO_JSON_SCHEMA_COMPLETE) {
    Fail(name + ": tables did not compile");
    return;
  }
  GufoJsonSchemaRegexTables t{};
  gufo_json_schema_regex_tables(handle, &t);
  const auto& impl = *expected.impl_;
  bool same = t.states == impl.states.size() &&
              t.classes == impl.alphabet.size() &&
              t.maximum_suffix == impl.maximum_suffix;
  for (std::uint32_t c = 0; same && c < t.classes; ++c) {
    const auto& chars = impl.alphabet[c];
    const auto count = t.class_offsets[c + 1] - t.class_offsets[c];
    same = count == static_cast<std::uint32_t>(chars.getRangeCount());
    for (std::uint32_t r = 0; same && r < count; ++r) {
      const auto* range = t.class_ranges + 2 * (t.class_offsets[c] + r);
      same = range[0] == static_cast<std::uint32_t>(chars.getRangeStart(r)) &&
             range[1] == static_cast<std::uint32_t>(chars.getRangeEnd(r));
    }
  }
  for (std::uint32_t s = 0; same && s < t.states; ++s) {
    const auto& state = impl.states[s];
    same = (t.accepting[s] != 0) == state.accepting &&
           t.distance[s] == state.distance &&
           t.successor_offsets[s + 1] - t.successor_offsets[s] ==
               state.successors.size();
    for (std::uint32_t c = 0; same && c < t.classes; ++c)
      same = t.next[s * t.classes + c] == state.next[c];
    for (std::size_t i = 0; same && i < state.successors.size(); ++i)
      same = t.successors[t.successor_offsets[s] + i] == state.successors[i];
  }
  // The lookup covers every codepoint, from 0 to 0x110000, with its class.
  same = same && t.lookup_len > 0 && t.lookup_starts[0] == 0 &&
         t.lookup_starts[t.lookup_len] == 0x110000;
  for (std::size_t i = 0; same && i < t.lookup_len; ++i) {
    const auto first = t.lookup_starts[i], last = t.lookup_starts[i + 1] - 1,
               cls = t.lookup_classes[i];
    same = first <= last;
    if (same && cls == UINT32_MAX)
      same = first == 0xd800 && last == 0xdfff;
    else if (same)
      same = cls < t.classes &&
             impl.alphabet[cls].contains(static_cast<UChar32>(first),
                                         static_cast<UChar32>(last));
  }
  if (!same)
    Fail(name + ": tables differ");
  gufo_json_schema_regex_free(handle);
}

// Codepoints that exercise the alphabet: each class's ends, their
// neighbours, and fixed probes.
std::vector<std::uint32_t> Probes(const reference::Regex& regex) {
  std::vector<std::uint32_t> probes{0,      9,       'a',      'Z',     '_',
                                    ' ',    0x7f,    0xa0,     0xe9,    0x2028,
                                    0xd7ff, 0xd800,  0xdfff,   0xe000,  0xfeff,
                                    0xffff, 0x1f600, 0x10ffff, 0x110000};
  for (const auto& chars : regex.impl_->alphabet)
    for (int32_t i = 0; i < chars.getRangeCount() && i < 4; ++i)
      for (auto cp : {chars.getRangeStart(i), chars.getRangeEnd(i)})
        for (int delta : {-1, 0, 1})
          if (cp + delta >= 0)
            probes.push_back(static_cast<std::uint32_t>(cp + delta));
  return probes;
}

void CompareRegex(const std::vector<std::string>& patterns,
                  std::uint32_t maximum) {
  std::string name = "regex";
  for (const auto& pattern : patterns)
    name += " /" + Show(pattern) + "/";
  name += " max " + std::to_string(maximum);
  std::shared_ptr<const reference::Regex> expected;
  std::shared_ptr<const JsonSchemaRegex> actual;
  const auto want =
      Run([&] { expected = reference::Regex::Compile(patterns, maximum); });
  const auto got =
      Run([&] { actual = JsonSchemaRegex::Compile(patterns, maximum); });
  ++compared;
  if (want != got) {
    Fail(name + ": reference " + (want.ok ? "compiles" : want.error) +
         ", port " + (got.ok ? "compiles" : got.error));
    return;
  }
  if (!want.ok)
    return;
  ++accepted;
  CompareTables(patterns, maximum, *expected, name);
  const auto probes = Probes(*expected);
  // The tables are compared in full; stepping samples up to 48 states.
  const auto states =
      static_cast<std::uint32_t>(expected->impl_->states.size());
  const auto stride = std::max<std::uint32_t>(1, states / 48);
  for (std::uint32_t s = 0; s < states; s += stride) {
    if (expected->Accepting(s) != actual->Accepting(s))
      Fail(name + ": Accepting differs");
    for (auto cp : probes)
      if (expected->Advance(s, cp) != actual->Advance(s, cp))
        Fail(name + ": Advance differs at " + std::to_string(cp));
    for (auto [low, high] :
         std::initializer_list<std::pair<std::uint32_t, std::uint32_t>>{
             {0, 0}, {0, 3}, {2, 2}, {1, 40}, {7, 1000}, {64, UINT32_MAX}}) {
      if (expected->CanFinish(s, low, high) != actual->CanFinish(s, low, high))
        Fail(name + ": CanFinish differs");
      for (auto [first, last] :
           std::initializer_list<std::pair<std::uint32_t, std::uint32_t>>{
               {0, 0x10ffff},
               {'a', 'a'},
               {'0', '9'},
               {0x80, 0x7ff},
               {0x10000, 0x103ff},
               {0xd800, 0xdfff},
               {5, 4}})
        if (expected->CanAdvance(s, first, last, low, high) !=
            actual->CanAdvance(s, first, last, low, high))
          Fail(name + ": CanAdvance differs");
    }
  }
  // Brent's cycle skip, with minLength far beyond the states, from the
  // first sampled states.
  for (std::uint32_t s = 0; s < std::min(states, 8 * stride); s += stride)
    for (auto [low, high] :
         std::initializer_list<std::pair<std::uint32_t, std::uint32_t>>{
             {1000003, UINT32_MAX},
             {4294967290U, UINT32_MAX},
             {999999937, 999999942}}) {
      if (expected->CanFinish(s, low, high) != actual->CanFinish(s, low, high))
        Fail(name + ": CanFinish differs for large bounds");
      if (expected->CanAdvance(s, 0, 0x10ffff, low, high) !=
          actual->CanAdvance(s, 0, 0x10ffff, low, high))
        Fail(name + ": CanAdvance differs for large bounds");
    }
  if (expected->MaximumSuffix() != actual->MaximumSuffix())
    Fail(name + ": MaximumSuffix differs");
}

const std::vector<std::string> kAtoms{"a",         "b",
                                      "[a-c]",     "[^x]",
                                      ".",         "\\d",
                                      "\\D",       "\\w",
                                      "\\W",       "\\s",
                                      "\\S",       "\\p{L}",
                                      "\\p{Lu}",   "\\P{N}",
                                      "\\u00e9",   "\\u{1F600}",
                                      "\\x41",     "\\n",
                                      "\\.",       "\\/",
                                      "[\\b]",     "[\\d-]",
                                      "[a\\-z]",   "\\b",
                                      "\\B",       "^",
                                      "$",         "(?=a)",
                                      "(?!b)",     "(a|b)",
                                      "(?:ab)",    "[]",
                                      "[^]",       "\\cJ",
                                      "\\0",       "é",
                                      "😀",        "\\ud83d\\ude00",
                                      "[\\s\\S]",  "\\p{Script=Greek}",
                                      "\\p{Nope}", "(?<=a)",
                                      "\\k",       "[z-a]",
                                      "a{,2}",     "\\u{}"};
const std::vector<std::string> kQuantifiers{
    "", "*", "+", "?", "{2}", "{1,3}", "{2,}", "*?", "{0}", "{33}"};

void RegexGrid() {
  for (const auto& atom : kAtoms)
    for (const auto& quantifier : kQuantifiers)
      for (std::uint32_t maximum : {0U, 2U, UINT32_MAX})
        CompareRegex({atom + quantifier}, maximum);
  for (std::size_t i = 0; i < kAtoms.size(); ++i)
    for (std::size_t j = 0; j < kAtoms.size(); j += 3) {
      CompareRegex({"^" + kAtoms[i] + kAtoms[j] + "$"}, 6);
      CompareRegex({kAtoms[i] + "|" + kAtoms[j]}, UINT32_MAX);
      CompareRegex({kAtoms[i], kAtoms[j]}, 4);
    }
}

std::vector<std::string> FormatPatterns() {
  std::vector<std::string> patterns;
  for (const char* format : {"date", "time", "date-time", "uuid", "ipv4",
                             "ipv6", "hostname", "email", "duration"})
    patterns.push_back(reference::Format(format)["pattern"].str());
  return patterns;
}

const std::vector<std::string> kFixtures{
    "^[a-z]+(?:-[a-z]+)*$",
    "^(?!.*--)[a-z0-9-]{3,12}$",
    "^\\p{Lu}\\p{Ll}*(?: \\p{Lu}\\p{Ll}*)*$",
    "^(?=.*\\d)(?=.*[A-Z]).{8,}$",
    "\\bword\\b",
    "^\\w+@\\w+\\.\\w{2,3}$",
    "^(?:(?:25[0-5]|2[0-4]\\d|1?\\d?\\d)\\.){3}\\d{1,3}$",
    "^[\\u0400-\\u04ff\\s]+$",
    "^\\S+(?:\\s\\S+){0,3}$",
    "(?:a|ab)(?:c|bcd)d",
    "^(?:x(?=y)|y(?!x))+$",
    "^[ab]*a[ab]{12}$",
    "^(?:(?:a{0,3}b){0,4}c)+$",
    "^\\$\\d+(?:\\.\\d\\d)?$",
};

void Mutations() {
  auto sources = kFixtures;
  for (auto& pattern : FormatPatterns())
    sources.push_back(std::move(pattern));
  for (const auto& pattern : sources)
    for (std::uint32_t maximum : {1U, 12U, 253U, UINT32_MAX})
      CompareRegex({pattern}, maximum);
  // Deterministic mutations reach the parser's rejections.
  const std::string alphabet = "ab09()[]{}|^$\\.*+?-,:=!<pPuUxdswbB";
  Random random{1278};
  for (int round = 0; round < 1500; ++round) {
    auto pattern = sources[random.Below(sources.size())];
    if (pattern.size() > 120)
      pattern = pattern.substr(random.Below(pattern.size() - 100), 100);
    for (int edits = 1 + static_cast<int>(random.Below(3)); edits; --edits) {
      const auto at = random.Below(pattern.size() + 1);
      switch (random.Below(3)) {
        case 0:
          pattern.insert(at, 1, alphabet[random.Below(alphabet.size())]);
          break;
        case 1:
          if (at < pattern.size())
            pattern.erase(at, 1);
          break;
        default:
          if (at < pattern.size())
            pattern[at] = alphabet[random.Below(alphabet.size())];
      }
    }
    CompareRegex({pattern}, round % 2 ? 16U : UINT32_MAX);
  }
  // Properties of strings keep their sets nonempty and distinct, as in ICU.
  for (const char* pattern :
       {"\\p{RGI_Emoji_Flag_Sequence}", "^\\p{Emoji_Keycap_Sequence}$",
        "[^\\p{RGI_Emoji_Flag_Sequence}]*a[^\\p{RGI_Emoji_Flag_Sequence}]*",
        "^[\\p{RGI_Emoji}a]+$", "[\\p{RGI_Emoji_Flag_Sequence}-a]",
        "[a-\\p{Emoji_Keycap_Sequence}]", "\\P{RGI_Emoji}",
        "^(?:[\\p{Basic_Emoji}]|x)*$"})
    for (std::uint32_t maximum : {3U, UINT32_MAX})
      CompareRegex({pattern}, maximum);
  // Ill-formed UTF-8 decodes to U+FFFD, as in ICU.
  for (const char* bytes :
       {"\xff", "a\xc3", "\xe2\x82", "\xed\xa0\x80", "[\xf0\x9f]", "\xc0\xaf"})
    CompareRegex({bytes}, UINT32_MAX);
  // Limits.
  std::string limit;
  for (int i = 0; i < 4096; ++i)
    limit += "(?:)";
  CompareRegex({limit}, UINT32_MAX);
  CompareRegex({limit + "a"}, UINT32_MAX);
  CompareRegex({std::string(33, '(') + std::string(33, ')')}, UINT32_MAX);
  CompareRegex({std::string(32, '(') + "a" + std::string(32, ')')}, UINT32_MAX);
  CompareRegex({"a{4294967294}", "a{4294967295}"}, UINT32_MAX);
  CompareRegex({"(?:(?:(?:a|b|c|d){1,20}x){1,20}y){1,20}"}, UINT32_MAX);
}

// The string lexeme: keywords, then matching over generated JSON strings.
void CompareString(const Value& schema, Random& random) {
  const auto name = "string " + schema.dump();
  std::shared_ptr<const JsonSchemaLexeme> expected, actual;
  const auto want = Run([&] { expected = reference::String(schema); });
  const auto got = Run([&] { actual = JsonSchemaLexeme::String(schema); });
  ++compared;
  if (want != got) {
    Fail(name + ": reference " + (want.ok ? "compiles" : want.error) +
         ", port " + (got.ok ? "compiles" : got.error));
    return;
  }
  if (!want.ok)
    return;
  ++accepted;
  static const std::vector<std::string> pieces{"a",
                                               "b",
                                               "Z",
                                               "1",
                                               "-",
                                               " ",
                                               ".",
                                               "é",
                                               "😀",
                                               "\\u00e9",
                                               "\\ud83d\\ude00",
                                               "\\n",
                                               "\\\"",
                                               "\\/",
                                               "\\",
                                               "\\u",
                                               "\\ud83d",
                                               "\xff",
                                               "\x01",
                                               "@",
                                               ":"};
  for (int i = 0; i < 40; ++i) {
    std::string text = "\"";
    for (auto n = random.Below(8); n; --n)
      text += pieces[random.Below(pieces.size())];
    if (random.Below(3))
      text += '"';
    std::string a, b;
    for (std::size_t k = 0; k < text.size(); ++k) {
      const auto byte = static_cast<unsigned char>(text[k]);
      const auto x = expected->Advance(a, byte);
      const auto y = actual->Advance(b, byte);
      if (x.prefix != y.prefix || x.complete != y.complete || a != b) {
        Fail(name + ": Advance differs on " + Show(text.substr(0, k + 1)));
        break;
      }
      if (!x.prefix)
        break;
      auto ca = a, cb = b;
      expected->CanonicalMaskState(ca, 1 + k % 5);
      actual->CanonicalMaskState(cb, 1 + k % 5);
      if (ca != cb)
        Fail(name + ": CanonicalMaskState differs");
    }
    const auto x = expected->Check(text), y = actual->Check(text);
    if (x.prefix != y.prefix || x.complete != y.complete)
      Fail(name + ": Check differs on " + Show(text));
  }
}

void StringGrid() {
  Random random{1};
  const std::vector<std::string> lengths{"",
                                         R"("minLength":2,)",
                                         R"("maxLength":3,)",
                                         R"("minLength":1,"maxLength":1,)",
                                         R"("minLength":4,"maxLength":2,)",
                                         R"("minLength":-1,)",
                                         R"("maxLength":1.5,)",
                                         R"("maxLength":1048577,)",
                                         R"("minLength":"2",)",
                                         R"("minLength":300,)",
                                         R"("maxLength":1048576,)"};
  const std::vector<std::string> predicates{"",
                                            R"("pattern":"^[a-z]+$",)",
                                            R"("pattern":"\\d",)",
                                            R"("pattern":"^\\p{L}{2,}$",)",
                                            R"("pattern":"(",)",
                                            R"("pattern":5,)",
                                            R"("format":"date",)",
                                            R"("format":"email",)",
                                            R"("format":"hostname",)",
                                            R"("format":"ipv4",)",
                                            R"("format":"color",)",
                                            R"("format":null,)",
                                            R"("pattern":"^a",)"
                                            R"("format":"uuid",)",
                                            R"("pattern":"^[0-9]{3}$",)"
                                            R"("format":"ipv4",)"};
  for (const auto& length : lengths)
    for (const auto& predicate : predicates) {
      auto text = "{" + length + predicate;
      if (text.back() == ',')
        text.pop_back();
      CompareString(gufo::json::parse(text + "}"), random);
    }
}

// The number lexeme: keywords over bound grids, then every short prefix and
// generated longer ones.
void CompareNumber(const Value& schema, bool integer,
                   const std::vector<std::string>& inputs) {
  const auto name =
      std::string(integer ? "integer " : "number ") + schema.dump();
  std::shared_ptr<const JsonSchemaLexeme> expected, actual;
  const auto want = Run([&] { expected = reference::Number(schema, integer); });
  const auto got =
      Run([&] { actual = JsonSchemaLexeme::Number(schema, integer); });
  ++compared;
  if (want != got) {
    Fail(name + ": reference " + (want.ok ? "compiles" : want.error) +
         ", port " + (got.ok ? "compiles" : got.error));
    return;
  }
  if (!want.ok)
    return;
  ++accepted;
  for (const auto& input : inputs) {
    const auto x = expected->Check(input), y = actual->Check(input);
    if (x.prefix != y.prefix || x.complete != y.complete)
      Fail(name + ": Check differs on " + input);
  }
  for (const char* text : {"0", "-0", "1", "0.5", "-2.5", "10", "1e3", "1e-7",
                           "1e300", "123.456", "-1e22", "\"1\""}) {
    const auto value = gufo::json::parse(text);
    if (expected->AcceptValue(value) != actual->AcceptValue(value))
      Fail(name + ": AcceptValue differs on " + text);
  }
}

void NumberGrid() {
  std::vector<std::string> inputs{""};
  const std::string alphabet = "0123456789-.";
  for (std::size_t length = 1; length <= 3; ++length) {
    std::vector<std::string> next;
    for (const auto& prefix : inputs)
      if (prefix.size() == length - 1)
        for (char c : alphabet)
          next.push_back(prefix + c);
    inputs.insert(inputs.end(), next.begin(), next.end());
  }
  Random random{7};
  for (int i = 0; i < 300; ++i) {
    std::string text = random.Below(3) ? "" : "-";
    for (auto n = 1 + random.Below(18); n; --n)
      text += alphabet[random.Below(random.Below(4) ? 10 : 12)];
    inputs.push_back(text);
  }
  inputs.push_back(std::string(4096, '1'));
  inputs.push_back(std::string(4097, '1'));
  inputs.push_back("0." + std::string(20, '0') + "1");
  const std::vector<std::string> bounds{"-10", "-1.5", "-1",  "0",      "0.1",
                                        "1",   "2.5",  "1e3", "123.456"};
  const std::vector<std::string> steps{"",     "0.1",  "0.5", "1", "3",
                                       "0.25", "1e-3", "7.5", "0", "-2"};
  auto make = [](const std::string& lower_key, const std::string& lower,
                 const std::string& upper_key, const std::string& upper,
                 const std::string& step) {
    std::string text = "{";
    auto add = [&](const std::string& key, const std::string& value) {
      if (value.empty())
        return;
      if (text.size() > 1)
        text += ',';
      text += "\"" + key + "\":" + value;
    };
    add(lower_key, lower);
    add(upper_key, upper);
    add("multipleOf", step);
    return gufo::json::parse(text + "}");
  };
  for (bool integer : {false, true})
    for (const auto& step : steps) {
      for (std::size_t i = 0; i < bounds.size(); ++i)
        for (std::size_t j = i; j < bounds.size(); j += 2)
          for (const char* lower_key : {"minimum", "exclusiveMinimum"})
            for (const char* upper_key : {"maximum", "exclusiveMaximum"})
              CompareNumber(
                  make(lower_key, bounds[i], upper_key, bounds[j], step),
                  integer, inputs);
      for (const auto& bound : bounds) {
        CompareNumber(make("minimum", bound, "maximum", "", step), integer,
                      inputs);
        CompareNumber(make("minimum", "", "exclusiveMaximum", bound, step),
                      integer, inputs);
      }
    }
  // Reversed, coinciding and repeated bounds; non-numbers; large exponents.
  for (const char* text :
       {R"({"minimum":2,"maximum":1})", R"({"minimum":1,"maximum":1})",
        R"({"minimum":1,"exclusiveMaximum":1})",
        R"({"minimum":1,"exclusiveMinimum":1})",
        R"({"minimum":0.5,"maximum":0.5})", R"({"minimum":"1"})",
        R"({"maximum":null})", R"({"multipleOf":true})",
        R"({"minimum":1e300,"multipleOf":1e-300})",
        R"({"minimum":-1e-300,"maximum":1e-300,"multipleOf":1e-301})",
        R"({"minimum":0.30000000000000004,"maximum":0.4,"multipleOf":0.1})",
        R"({"minimum":1.1,"maximum":1.9})", "{}"})
    for (bool integer : {false, true})
      CompareNumber(gufo::json::parse(text), integer, inputs);
}

void MultipleOfAndFormat() {
  const std::vector<std::string> values{
      "0.1", "0.25",  "3",      "1e-3",  "7.5",           "0",
      "-2",  "1e300", "1e-300", "\"1\"", "123456789.125", "0.3"};
  // The parser rejects 1e400, so no schema holds an infinite bound.
  if (Run([] { gufo::json::parse("1e400"); }).ok)
    Fail("1e400 parsed");
  for (const auto& a : values)
    for (const auto& b : values) {
      const auto x = gufo::json::parse(a), y = gufo::json::parse(b);
      Value expected, actual;
      const auto want =
          Run([&] { expected = reference::IntersectMultipleOf(x, y); });
      const auto got =
          Run([&] { actual = JsonSchemaLexeme::IntersectMultipleOf(x, y); });
      ++compared;
      if (want != got || (want.ok && expected.dump() != actual.dump()))
        Fail("multipleOf " + a + " " + b + ": " +
             (want.ok ? expected.dump() : want.error) + " vs " +
             (got.ok ? actual.dump() : got.error));
    }
  for (const char* format :
       {"date", "time", "date-time", "uuid", "ipv4", "ipv6", "hostname",
        "email", "duration", "uri", ""}) {
    Value expected, actual;
    const auto want = Run([&] { expected = reference::Format(format); });
    const auto got = Run([&] { actual = JsonSchemaLexeme::Format(format); });
    ++compared;
    if (want != got || (want.ok && expected.dump() != actual.dump()))
      Fail(std::string("format ") + format);
  }
}

}  // namespace

int main() {
  for (auto [name, phase] :
       std::initializer_list<std::pair<const char*, void (*)()>>{
           {"regex grid", RegexGrid},
           {"regex mutations", Mutations},
           {"string lexemes", StringGrid},
           {"number lexemes", NumberGrid},
           {"multipleOf and formats", MultipleOfAndFormat}}) {
    const auto start = std::chrono::steady_clock::now();
    phase();
    const std::chrono::duration<double> elapsed =
        std::chrono::steady_clock::now() - start;
    std::fprintf(stderr, "%s: %.2f s\n", name, elapsed.count());
  }
  std::fprintf(stderr, "compared %zu cases (%zu accepted), %d failures\n",
               compared, accepted, failures);
  return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
