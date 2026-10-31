#include <cstdio>
#include <string>
#include <vector>

#include "src/models/gemma4/tokenizer.hpp"
#include "tests/models/gemma4/check.hpp"

namespace g4 = gufo::models::gemma4;
using g4::TokenId;
using g4::TokenType;
using gemma4_test::Require;

namespace {

struct Fixture {
  std::unique_ptr<g4::Tokenizer> tokenizer;
  TokenId Id(const std::string& text) const {
    const auto id = tokenizer->FindToken(text);
    Require(id.has_value(), "fixture token missing: " + text);
    return *id;
  }
};

/// A synthetic vocabulary shaped like Gemma 4's: control/user-defined
/// specials first, 256 byte-fallback tokens, then SentencePiece pieces.
Fixture Build(bool add_bos = true) {
  g4::Tokenizer::Vocabulary v;
  const auto add = [&](std::string text, TokenType type) {
    v.tokens.push_back(std::move(text));
    v.types.push_back(type);
  };
  add("<pad>", TokenType::kControl);             // 0
  add("<eos>", TokenType::kNormal);              // 1, promoted to control (EOG)
  add("<bos>", TokenType::kControl);             // 2
  add("<unk>", TokenType::kUnknown);             // 3
  add("<|turn>", TokenType::kControl);           // 4
  add("<turn|>", TokenType::kControl);           // 5
  add("<|tool_call>", TokenType::kUserDefined);  // 6
  add("<|tool_response>", TokenType::kUserDefined);  // 7, promoted (EOG)
  for (int b = 0; b < 256; ++b) {
    char text[8];
    std::snprintf(text, sizeof(text), "<0x%02X>", b);
    add(text, TokenType::kByte);
  }
  for (const char* piece :
       {"\xE2\x96\x81", "h", "e", "l", "o", "u", "s", "r", "\n", "he", "ll",
        "llo", "hello", "\xE2\x96\x81hello", "\n\n", "\xE2\x96\x81\xE2\x96\x81",
        "us", "er", "user"}) {
    add(piece, TokenType::kNormal);
  }
  v.merges = {"h e",
              "l l",
              "ll o",
              "he llo",
              "\xE2\x96\x81 hello",
              "\n \n",
              "\xE2\x96\x81 \xE2\x96\x81",
              "u s",
              "e r",
              "us er"};
  v.bos = 2;
  v.eos = 5;
  v.pad = 0;
  v.unknown = 3;
  v.add_bos = add_bos;
  std::string error;
  Fixture f{g4::Tokenizer::Create(std::move(v), &error)};
  Require(f.tokenizer != nullptr, "synthetic vocabulary rejected: " + error);
  return f;
}

void Expect(const Fixture& f, const std::string& text, bool parse_special,
            const std::vector<TokenId>& expected, const char* what) {
  const auto got = f.tokenizer->Encode(text, false, parse_special);
  if (got != expected) {
    std::string message = std::string(what) + ": got [";
    for (auto t : got) {
      message += std::to_string(t) + " ";
    }
    throw std::runtime_error(message + "]");
  }
}

void CheckMerges(const Fixture& f) {
  Expect(f, "hello", false, {f.Id("hello")}, "full merge chain");
  Expect(f, " hello", false, {f.Id("\xE2\x96\x81hello")},
         "space escapes to U+2581");
  Expect(f, "  ", false, {f.Id("\xE2\x96\x81\xE2\x96\x81")}, "space pair");
  // Equal ranks resolve leftmost first: lll -> ll l, never l ll.
  Expect(f, "lll", false, {f.Id("ll"), f.Id("l")}, "leftmost tie break");
  Expect(f, "user", false, {f.Id("user")}, "multi-step merges");
}

void CheckNewlines(const Fixture& f) {
  Expect(f, "hello\n\nhello", false,
         {f.Id("hello"), f.Id("\n\n"), f.Id("hello")},
         "newline runs split words");
  // A run that is not itself a token is merged like any other word.
  Expect(f, "\n\n\n", false, {f.Id("\n\n"), f.Id("\n")},
         "newline run outside the vocabulary");
  Expect(f, "\n", false, {f.Id("\n")}, "single newline");
}

/// Long lines are cached across calls: a line encodes the same whether it is
/// new, repeated alone, or repeated inside other text.
void CheckWordCache(const Fixture& f) {
  std::string line;
  std::vector<TokenId> expected;
  for (int i = 0; i < 20; ++i) {
    line += i % 3 == 0 ? " user" : " hello";
    expected.push_back(f.Id(i % 3 == 0 ? "\xE2\x96\x81" : "\xE2\x96\x81hello"));
    if (i % 3 == 0) {
      expected.push_back(f.Id("user"));
    }
  }
  Expect(f, line, false, expected, "long line, first encode");
  Expect(f, line, false, expected, "long line, cached");
  std::vector<TokenId> twice = expected;
  twice.push_back(f.Id("\n\n"));
  twice.insert(twice.end(), expected.begin(), expected.end());
  Expect(f, line + "\n\n" + line, false, twice, "cached line inside text");
  // A line differing only at its end is its own entry.
  std::vector<TokenId> longer = expected;
  longer.push_back(f.Id("\xE2\x96\x81hello"));
  Expect(f, line + " hello", false, longer, "extended long line");
}

void CheckByteFallback(const Fixture& f) {
  Expect(f, "\xC3\xA9", false, {f.Id("<0xC3>"), f.Id("<0xA9>")},
         "UTF-8 character outside the vocabulary");
  // Invalid UTF-8 is replaced by U+FFFD before merging.
  Expect(f, "\xFF", false, {f.Id("<0xEF>"), f.Id("<0xBF>"), f.Id("<0xBD>")},
         "invalid byte becomes U+FFFD");
  Expect(f, "h\xE2\x82", false,
         {f.Id("h"), f.Id("<0xEF>"), f.Id("<0xBF>"), f.Id("<0xBD>"),
          f.Id("<0xEF>"), f.Id("<0xBF>"), f.Id("<0xBD>")},
         "truncated sequence");
}

void CheckSpecials(const Fixture& f) {
  const TokenId turn = f.Id("<|turn>");
  Expect(f, "<|turn>user", true, {turn, f.Id("user")},
         "control token parsed as special");
  const auto plain = f.tokenizer->Encode("<|turn>", false, false);
  Require(plain.size() > 1 && plain[0] != turn,
          "control token matched without parse_special");
  Expect(f, "<|tool_call>hello", false, {f.Id("<|tool_call>"), f.Id("hello")},
         "user-defined token always matched");
  Expect(f, "hello<eos>", true, {f.Id("hello"), 1},
         "EOG text promoted to a control token");
  // Fragments around specials are tokenized independently.
  Expect(f, "he<turn|>llo", true, {f.Id("he"), f.Id("<turn|>"), f.Id("llo")},
         "special splits words");
  const auto with_bos = f.tokenizer->Encode("hello", true, true);
  Require(with_bos.size() == 2 && with_bos[0] == 2, "BOS added on request");
  const Fixture no_bos = Build(false);
  Require(no_bos.tokenizer->Encode("hello", true, true).size() == 1,
          "BOS follows the vocabulary flag");
}

void CheckDecode(const Fixture& f) {
  const std::vector<TokenId> tokens = {f.Id("\xE2\x96\x81hello"),
                                       f.Id("<turn|>"), f.Id("<|tool_call>"),
                                       f.Id("<0xC3>"), f.Id("<0xA9>")};
  Require(f.tokenizer->Decode(tokens, false) == " hello<|tool_call>\xC3\xA9",
          "control hidden, user-defined printed, bytes raw");
  Require(
      f.tokenizer->Decode(tokens, true) == " hello<turn|><|tool_call>\xC3\xA9",
      "control printed with special");
}

void CheckEndOfGeneration(const Fixture& f) {
  Require(f.tokenizer->IsEndOfGeneration(1), "<eos> ends generation");
  Require(f.tokenizer->IsEndOfGeneration(5), "<turn|> ends generation");
  Require(f.tokenizer->IsEndOfGeneration(7),
          "<|tool_response> ends generation");
  Require(!f.tokenizer->IsEndOfGeneration(4), "<|turn> does not");
  Require(f.tokenizer->Type(7) == TokenType::kControl,
          "EOG user-defined token promoted to control");
}

void CheckMalformed() {
  g4::Tokenizer::Vocabulary v;
  v.tokens = {"a", "b"};
  v.types = {TokenType::kNormal, TokenType::kNormal};
  v.merges = {"a b"};  // "ab" is not a token
  v.bos = 0;
  v.eos = 1;
  std::string error;
  Require(!g4::Tokenizer::Create(v, &error) && !error.empty(),
          "merge outside the vocabulary accepted");
  v.merges = {"ab"};
  Require(!g4::Tokenizer::Create(v, &error), "merge without a space accepted");
  v.merges = {};
  v.types.pop_back();
  Require(!g4::Tokenizer::Create(v, &error), "type count mismatch accepted");
}

}  // namespace

int main() {
  return gemma4_test::Run([] {
    const Fixture f = Build();
    CheckMerges(f);
    CheckNewlines(f);
    CheckWordCache(f);
    CheckByteFallback(f);
    CheckSpecials(f);
    CheckDecode(f);
    CheckEndOfGeneration(f);
    CheckMalformed();
  });
}
