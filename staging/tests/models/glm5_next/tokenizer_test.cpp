#include "staging/models/glm5_next/tokenizer.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace glm = gufo::models::glm5_next;

namespace {

// gpt2 byte_to_unicode: every raw byte has a visible single-token name.
std::string VisibleByte(std::uint8_t b) {
  const int i = b;
  const bool direct = (i >= '!' && i <= '~') || (i >= 161 && i <= 172) ||
                      (i >= 174 && i <= 255);
  if (direct) {
    return std::string(1, static_cast<char>(b));
  }
  int n = 0;
  for (int c = 0; c < 256; ++c) {
    const bool c_direct = (c >= '!' && c <= '~') || (c >= 161 && c <= 172) ||
                          (c >= 174 && c <= 255);
    if (!c_direct) {
      if (c == i) {
        break;
      }
      ++n;
    }
  }
  const int cp = 256 + n;
  // All such code points fit in two UTF-8 bytes.
  return std::string({static_cast<char>(0xC0 | (cp >> 6)),
                      static_cast<char>(0x80 | (cp & 0x3F))});
}

// Control-token spellings are assembled from fragments so that no full
// delimiter sequence ever appears as a literal in this source file.
std::string SpecialName(std::string_view inner) {
  return std::string("\x3C\x7C") + std::string(inner) + "\x7C\x3E";
}

constexpr const char* kSpaceName = "\xC4\xA0";  // U+0120: gpt2 byte name

struct Fixture {
  std::vector<std::string> tokens;
  std::vector<std::string> merges;
  std::vector<glm::TokenId> special_ids;
  glm::TokenId bos{};
  glm::TokenId eos{};
  glm::TokenId pad{};
  std::vector<glm::TokenId> eog_ids;
};

Fixture MakeFixture() {
  Fixture fx;
  // Special CONTROL/USER_DEFINED entries first: ids 0..3.
  fx.tokens = {SpecialName("endoftext"), SpecialName("system"),
               SpecialName("user"), SpecialName("assistant")};
  fx.special_ids = {0, 1, 2, 3};
  fx.eos = 0;
  fx.bos = 1;
  fx.pad = 0;
  fx.eog_ids = {0};

  // Words referenced through the merges below (merge-rank order in fx.merges).
  fx.tokens.insert(fx.tokens.end(),
                   {
                       "a",
                       std::string(kSpaceName) + "k",
                       "no",
                       "now",
                       std::string(kSpaceName) + "know",
                       "'t",
                       "1",
                       "2",
                       "5",
                       "7",
                       "8",
                       "9",
                       "89",
                       "15",
                       "157",
                       "1578",
                       std::string(VisibleByte(0xA0)) + "2",
                       std::string(kSpaceName) + "x",
                   });
  fx.merges = {
      std::string(kSpaceName) + " k",   // space + k -> space-k
      "no w",                           // no + w -> now (rank after space-k)
      std::string(kSpaceName) + "k now",  // space-k + now -> space-know
      "' t",                            // clitic pieces merge to 't
      "n o",                            // no (needed by the know chain)
      "1 5",
      "15 7",
      "8 9",
      "157 8",  // fires only if digit grouping ran past three digits
      // The run-interruption probe: only merges when a digit joined a
      // preceding CJK letter run across the piece boundary.
      VisibleByte(0xA0) + " 2",
      std::string(kSpaceName) + " x",  // space + x -> space-x
  };
  // Byte-level entries round out the vocabulary without duplicating names
  // the word list already provides, mirroring a real unique-token GGUF.
  for (int b = 0; b < 256; ++b) {
    const std::string name = VisibleByte(static_cast<std::uint8_t>(b));
    if (std::find(fx.tokens.begin(), fx.tokens.end(), name) ==
        fx.tokens.end()) {
      fx.tokens.push_back(name);
    }
  }
  return fx;
}

class Tok {
public:
  explicit Tok(const Fixture& fx)
      : tokenizer_(glm::GlmTokenizer::CreateFromVocabulary(
            fx.tokens, fx.merges, fx.special_ids, fx.bos, fx.eos, fx.pad,
            fx.eog_ids)),
        tokens_(fx.tokens) {
    if (tokenizer_ == nullptr) {
      throw std::runtime_error("CreateFromVocabulary failed");
    }
  }

  glm::TokenId Id(std::string_view name) const {
    for (std::size_t i = 0; i < tokens_.size(); ++i) {
      if (tokens_[i] == name) {
        return static_cast<glm::TokenId>(i);
      }
    }
    throw std::runtime_error("unknown token name in expectation: " +
                             std::string(name));
  }

  glm::TokenId Byte(std::uint8_t b) const { return Id(VisibleByte(b)); }

  void Expect(const std::string_view text,
              const std::vector<std::string>& token_names) const {
    std::vector<glm::TokenId> expected;
    expected.reserve(token_names.size());
    for (const auto& name : token_names) {
      expected.push_back(Id(name));
    }
    const auto actual = tokenizer_->Encode(text);
    if (actual != expected) {
      const auto dump = [this](const std::vector<glm::TokenId>& ids) {
        std::string out;
        for (const auto id : ids) {
          out += (out.empty() ? "" : " | ");
          out += id < tokens_.size() ? tokens_[id] : std::string("?");
          out += "#" + std::to_string(id);
        }
        return out;
      };
      throw std::runtime_error("Encode mismatch for \"" + std::string(text) +
                               "\": expected [" + dump(expected) +
                               "] got [" + dump(actual) + "]");
    }
  }

  const glm::GlmTokenizer& operator*() const { return *tokenizer_; }
  const glm::GlmTokenizer* operator->() const { return tokenizer_.get(); }

private:
  std::unique_ptr<glm::GlmTokenizer> tokenizer_;
  std::vector<std::string> tokens_;
};

// The chatglm4 regex splits clitics into their own pieces; unlike flint the
// clitic is NOT appended to the preceding word.
void TestCliticPiece(const Tok& tok) {
  tok.Expect("don't know", {"d", "o", "n", "'t",
                            std::string(kSpaceName) + "know"});
}

// \p{N}{1,3} caps numeric pieces at three digits; the run boundary before
// the trailing pair prevents the "157 8" merge from ever being offered.
void TestDigitGrouping(const Tok& tok) {
  tok.Expect("a15789", {"a", "157", "89"});
}

// Letter runs end at digits (the glm4 regex has no CJK-plus-digit branch),
// so the cross-boundary V(A0)+2 merge stays unoffered.
void TestDigitInterruptsLetterRun(const Tok& tok) {
  tok.Expect("\xE4\xBD\xA0" "2",
             {std::string(VisibleByte(0xE4)), std::string(VisibleByte(0xBD)),
              std::string(VisibleByte(0xA0)), "2"});
}

// CJK codepoints are \p{L} members: one run, plain byte-level encoding
// without any cross-character merge in this fixture.
void TestCjkSingleRun(const Tok& tok) {
  tok.Expect("\xE4\xBD\xA0\xE5\xA5\xBD",
             {std::string(VisibleByte(0xE4)), std::string(VisibleByte(0xBD)),
              std::string(VisibleByte(0xA0)), std::string(VisibleByte(0xE5)),
              std::string(VisibleByte(0xA5)), std::string(VisibleByte(0xBD))});
}

// \s+(?!\S) backtracks the last whitespace before non-whitespace: the first
// space stands alone and the second joins the following letter.
void TestWhitespaceBacktrack(const Tok& tok) {
  tok.Expect("  x", {std::string(kSpaceName),
                     std::string(kSpaceName) + "x"});
}

void TestSpecials(const Tok& tok) {
  const auto user = tok.Id(SpecialName("user"));
  const auto eos = tok.Id(SpecialName("endoftext"));
  if (tok->GetEosTokenId() != eos) {
    throw std::runtime_error("eos id mismatch");
  }
  if (!tok->IsStopToken(eos) || tok->IsStopToken(user)) {
    throw std::runtime_error("stop token set mismatch");
  }
  if (!tok->IsSpecialToken(user) || tok->IsSpecialToken(tok.Byte('a'))) {
    throw std::runtime_error("special classification mismatch");
  }
  const auto found = tok->FindSpecialToken(SpecialName("endoftext"));
  if (!found.has_value() || *found != eos) {
    throw std::runtime_error("FindSpecialToken mismatch");
  }
  if (tok->Encode(SpecialName("user")) != std::vector<glm::TokenId>{user}) {
    throw std::runtime_error("standalone special did not encode singly");
  }
  std::vector<glm::TokenId> framed{tok.Id("a"), user, tok.Id("a")};
  if (tok->Encode(std::string("a") + SpecialName("user") + "a") != framed) {
    throw std::runtime_error("embedded special split mismatch");
  }
  // parse_special off keeps the spelling inside the BPE stream.
  glm::TokenizerOptions options;
  options.parse_special_tokens = false;
  if (tok->Encode(SpecialName("user"), options).size() <= 1) {
    throw std::runtime_error("parse_special off must not yield one token");
  }
  const auto decoded = tok->Decode(framed);
  if (decoded != std::string("a") + SpecialName("user") + "a") {
    throw std::runtime_error("Decode roundtrip mismatch");
  }
}

void Run(const char* name, const Tok& tok, void (*test)(const Tok&)) {
  try {
    test(tok);
  } catch (const std::exception& error) {
    throw std::runtime_error(std::string(name) + ": " + error.what());
  }
}

}  // namespace

int main() {
  try {
    const Fixture fx = MakeFixture();
    const Tok tok(fx);
    Run("TestCliticPiece", tok, TestCliticPiece);
    Run("TestDigitGrouping", tok, TestDigitGrouping);
    Run("TestDigitInterruptsLetterRun", tok, TestDigitInterruptsLetterRun);
    Run("TestCjkSingleRun", tok, TestCjkSingleRun);
    Run("TestWhitespaceBacktrack", tok, TestWhitespaceBacktrack);
    Run("TestSpecials", tok, TestSpecials);
    std::cout << "glm5_next tokenizer: OK" << std::endl;
  } catch (const std::exception& error) {
    std::cerr << "glm5_next tokenizer FAILED: " << error.what() << std::endl;
    return 1;
  }
  return 0;
}
