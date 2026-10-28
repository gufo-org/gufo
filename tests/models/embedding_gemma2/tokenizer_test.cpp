#include "src/models/embedding_gemma2/tokenizer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/json.hpp"
#include "src/core/platform/os_info.hpp"

namespace embedding_gemma2 = gufo::models::embedding_gemma2;

#ifndef GUFO_EMBEDDING_GEMMA2_TOKENIZER_GOLDEN
#define GUFO_EMBEDDING_GEMMA2_TOKENIZER_GOLDEN \
  "tests/fixtures/embedding_gemma2/tokenizer_golden.json"
#endif

namespace {

// The compiled-in fixture path exists only where the tree was built. A packaged
// test image points GUFO_FIXTURE_DIR at a directory holding the fixtures.
std::string FixturePath(std::string_view compiled) {
  if (!std::filesystem::exists(std::string(compiled))) {
    if (const char* directory = std::getenv("GUFO_FIXTURE_DIR")) {
      return (std::filesystem::path(directory) /
              std::filesystem::path(compiled).filename())
          .string();
    }
  }
  return std::string(compiled);
}

constexpr double kMaximumLongEncodeMilliseconds = 200.0;
constexpr std::size_t kLongRepeats = 6;
// The golden document ends in "! ", so a repeat merges only its trailing U+2581
// into the next copy's first word. A different tail can change a few more
// tokens; 2 leaves one token of slack.
constexpr std::size_t kLongBoundaryTokens = 2;

void Check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL embedding_gemma2_tokenizer_test: " << message << '\n';
    std::exit(1);
  }
}

std::string ReadAll(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  Check(static_cast<bool>(input), "cannot open " + path.string());
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

std::string Format(const std::vector<std::uint32_t>& ids) {
  std::string out = "[";
  for (std::size_t i = 0; i < ids.size(); ++i) {
    out += (i == 0 ? "" : ",") + std::to_string(ids[i]);
  }
  return out + "]";
}

// The loader must report malformed files instead of crashing or accepting
// them. These checks need no model weights.
void CheckMalformedInputs() {
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() /
      ("embedding_gemma2_tokenizer_" +
       std::to_string(gufo::platform::ProcessId()));
  std::filesystem::create_directories(dir);
  std::string error;
  Check(!embedding_gemma2::Tokenizer::Load((dir / "missing").string(), &error)
                .has_value() &&
            !error.empty(),
        "a missing directory must fail with a message");
  for (const std::string_view contents :
       {std::string_view("{"), std::string_view("[]"),
        std::string_view("{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0},"
                         "\"merges\":[[\"a\"]]}}"),
        std::string_view("{\"model\":{\"type\":\"Unigram\"}} trailing")}) {
    {
      std::ofstream output(dir / "tokenizer.json", std::ios::binary);
      output << contents;
    }
    error.clear();
    Check(
        !embedding_gemma2::Tokenizer::Load(dir.string(), &error).has_value() &&
            !error.empty(),
        "malformed tokenizer.json must fail with a message");
  }
  // Shapes that parse as JSON but break the schema name the bad section.
  const std::string sections =
      R"("normalizer":{"type":"Replace","pattern":{"String":" "},)"
      R"("content":"\u2581"},"pre_tokenizer":{"type":"Split",)"
      R"("pattern":{"String":" "},"behavior":"MergedWithPrevious",)"
      R"("invert":false},)";
  for (const auto& [model, expected] :
       {std::pair<std::string_view, std::string_view>{
            R"("model":[])", "invalid model: expected an object"},
        {R"("model":{"type":"BPE","vocab":{"a":0.5},"merges":[]})",
         "is not an integer"}}) {
    {
      std::ofstream output(dir / "tokenizer.json", std::ios::binary);
      output << "{" << sections << model << "}";
    }
    error.clear();
    Check(
        !embedding_gemma2::Tokenizer::Load(dir.string(), &error).has_value() &&
            error.find(expected) != std::string::npos,
        "expected '" + std::string(expected) + "', got '" + error + "'");
  }
  std::filesystem::remove_all(dir);
}

}  // namespace

int main(int argc, char** argv) {
  CheckMalformedInputs();

  const char* configured = std::getenv("EMBEDDING_GEMMA2_MODEL_ROOT");
  const std::filesystem::path model_root =
      argc > 1
          ? argv[1]
          : (configured != nullptr ? configured : "models/embeddinggemma-2");
  if (!std::filesystem::exists(model_root / "tokenizer.json")) {
    std::cerr << "SKIP embedding_gemma2_tokenizer_test: external model "
                 "unavailable\n";
    return 77;
  }

  std::string error;
  const auto load_start = std::chrono::steady_clock::now();
  const auto loaded =
      embedding_gemma2::Tokenizer::Load(model_root.string(), &error);
  const double load_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - load_start)
                             .count();
  Check(loaded.has_value(), "tokenizer.json must load: " + error);
  const embedding_gemma2::Tokenizer& tokenizer = *loaded;
  Check(tokenizer.vocab_size() == 262144, "the vocabulary has 262144 entries");
  Check(tokenizer.TokenId("<pad>") == 0u && tokenizer.TokenId("<eos>") == 1u &&
            tokenizer.TokenId("<bos>") == 2u &&
            tokenizer.TokenId("<unk>") == 3u &&
            tokenizer.TokenId("<mask>") == 4u,
        "special token ids");
  Check(tokenizer.TokenId("<0x0A>").has_value(), "byte tokens must exist");
  Check(!tokenizer.TokenId("definitely not a token \x01").has_value(),
        "unknown tokens have no id");

  const gufo::json::Value golden = gufo::json::parse(
      ReadAll(FixturePath(GUFO_EMBEDDING_GEMMA2_TOKENIZER_GOLDEN)));
  const gufo::json::Value* cases = golden.find("cases");
  Check(cases != nullptr && cases->is_array() && cases->size() >= 80,
        "the golden file needs at least 80 cases");

  std::size_t passed = 0;
  std::string long_document;
  std::vector<std::uint32_t> long_document_ids;
  for (std::size_t index = 0; index < cases->size(); ++index) {
    const gufo::json::Value& entry = cases->items()[index];
    const std::string name = entry.member_str("name");
    const std::string text = entry.member_str("text");
    const gufo::json::Value* add = entry.find("add_special_tokens");
    const bool add_special_tokens = add == nullptr || add->as_bool(true);
    std::vector<std::uint32_t> expected;
    for (const gufo::json::Value& id : entry.find("ids")->items()) {
      expected.push_back(static_cast<std::uint32_t>(id.as_double()));
    }
    const std::vector<std::uint32_t> actual =
        tokenizer.Encode(text, add_special_tokens);
    if (name == "long_document") {
      long_document = text;
      long_document_ids = expected;
    }
    if (actual != expected) {
      std::size_t first = 0;
      while (first < actual.size() && first < expected.size() &&
             actual[first] == expected[first]) {
        ++first;
      }
      std::cerr << "FAIL embedding_gemma2_tokenizer_test: case " << index
                << " '" << name << "' differs at token " << first
                << "\n  expected " << Format(expected) << "\n  actual   "
                << Format(actual) << '\n';
      return 1;
    }
    ++passed;
  }

  // The whole text merges as one BPE word, so a quadratic merge loop shows
  // up on a long input.
  Check(!long_document.empty(), "the golden file needs a long_document case");
  std::string long_text;
  for (std::size_t i = 0; i < kLongRepeats; ++i) {
    long_text += long_document;
  }
  Check(long_text.size() >= 40000, "the long text must exceed 40000 bytes");
  const auto long_start = std::chrono::steady_clock::now();
  const std::vector<std::uint32_t> long_ids = tokenizer.Encode(long_text);
  const double long_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - long_start)
                             .count();
  // Without <bos> and <eos>, the long text holds kLongRepeats copies of the
  // golden document, give or take a few tokens at each boundary.
  Check(long_document_ids.size() > kLongBoundaryTokens + 2,
        "the golden long document is too short");
  const std::size_t document_tokens = long_document_ids.size() - 2;
  const std::size_t expected_tokens = kLongRepeats * document_tokens;
  Check(
      long_ids.size() >= expected_tokens - kLongRepeats * kLongBoundaryTokens &&
          long_ids.size() <=
              expected_tokens + 2 + kLongRepeats * kLongBoundaryTokens,
      "the long text has " + std::to_string(long_ids.size()) +
          " tokens, expected about " + std::to_string(expected_tokens));
  const std::size_t stable_tokens = document_tokens - kLongBoundaryTokens;
  Check(std::equal(long_document_ids.begin(),
                   long_document_ids.begin() + 1 + stable_tokens,
                   long_ids.begin()),
        "the long text must start with the golden long document tokens");
  // Copies between two others encode the same way, so the tokens repeat with
  // the period one more copy adds. The first and last copies differ at their
  // outer edges and are left out.
  const std::size_t period =
      long_ids.size() - tokenizer
                            .Encode(std::string_view(long_text).substr(
                                0, long_text.size() - long_document.size()))
                            .size();
  Check(long_ids.size() > 2 + 3 * period,
        "the long text must hold at least four copies");
  Check(
      std::equal(long_ids.begin() + 1 + period, long_ids.end() - 1 - 2 * period,
                 long_ids.begin() + 1 + 2 * period),
      "interior copies of the long text must encode identically");
  Check(long_ms < kMaximumLongEncodeMilliseconds,
        "the long text took " + std::to_string(long_ms) + " ms");

  // Invalid UTF-8 and stray bytes must neither crash nor hang.
  const std::string invalid("a\xff\xc3\xe2\x82 b\x80\xf0\x9f\x98", 10);
  const std::vector<std::uint32_t> invalid_ids = tokenizer.Encode(invalid);
  Check(invalid_ids.size() > 2 && invalid_ids.front() == 2u &&
            invalid_ids.back() == 1u,
        "invalid UTF-8 must still be wrapped by bos and eos");
  Check(tokenizer.Encode("", false).empty(), "empty text without specials");

  // #726: truncation keeps <bos>, the first content tokens and <eos>.
  std::string over_limit;
  while (over_limit.size() < 64 * 1024) {
    over_limit += "The quick brown fox jumps over the lazy dog. ";
  }
  const std::vector<std::uint32_t> full_ids = tokenizer.Encode(over_limit);
  Check(full_ids.size() > 8192, "the over-limit text exceeds 8192 tokens");
  const std::vector<std::uint32_t> cut_ids =
      tokenizer.EncodeTruncated(over_limit, 8192);
  Check(cut_ids.size() == 8192 && cut_ids.front() == 2u && cut_ids.back() == 1u,
        "a truncated encoding fills the limit between bos and eos");
  Check(std::equal(cut_ids.begin(), cut_ids.end() - 1, full_ids.begin()),
        "a truncated encoding keeps the first tokens of the full encoding");
  Check(tokenizer.EncodeTruncated("short text", 8192) ==
            tokenizer.Encode("short text"),
        "a text within the limit is not changed");
  Check(tokenizer.EncodeTruncated(over_limit, 1) == full_ids,
        "a limit below the two markers leaves the encoding whole");
  // A small limit encodes only a short leading slice of the text; the kept
  // tokens must still match the full encoding.
  for (const std::size_t limit : {3, 16, 64, 257}) {
    const std::vector<std::uint32_t> small =
        tokenizer.EncodeTruncated(over_limit, limit);
    Check(small.size() == limit && small.front() == 2u && small.back() == 1u &&
              std::equal(small.begin(), small.end() - 1, full_ids.begin()),
          "a short truncation keeps the full encoding's first tokens (limit " +
              std::to_string(limit) + ")");
  }

  std::cout << "PASS embedding_gemma2_tokenizer_test: " << passed
            << " golden cases, load " << load_ms << " ms, " << long_text.size()
            << " byte encode " << long_ms << " ms\n";
  return 0;
}
