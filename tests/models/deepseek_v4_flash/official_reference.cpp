#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/json.hpp"
#include "src/core/crypto/sha256.hpp"
#include "src/models/deepseek_v4_flash/engine.hpp"

namespace gufo::testing::ds4 {
namespace {
namespace json = gufo::server::json;

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

const json::Value& Field(const json::Value& object, std::string_view key) {
  const auto* value = object.find(std::string(key));
  Require(value != nullptr,
          "missing official fixture field: " + std::string(key));
  return *value;
}

std::string Hex(std::string_view bytes) {
  constexpr char digits[] = "0123456789abcdef";
  std::string encoded;
  encoded.reserve(bytes.size() * 2);
  for (const unsigned char byte : bytes) {
    encoded.push_back(digits[byte >> 4]);
    encoded.push_back(digits[byte & 15]);
  }
  return encoded;
}

struct Totals {
  std::size_t cases{}, tokens{}, first_matches{}, matching_prefix{}, matches{};
  double nll{};
};

// The shared HTTP serializer uses six significant digits. Keep every scoring
// digit without changing serialization of ordinary server responses.
void WriteScoreJson(const json::Value& value, std::ostream& output) {
  if (value.is_number()) {
    Require(std::isfinite(value.as_double()), "non-finite score output");
    output << std::setprecision(17) << value.as_double();
  } else if (value.is_array()) {
    output << '[';
    bool first = true;
    for (const auto& item : value.items()) {
      if (!first)
        output << ',';
      first = false;
      WriteScoreJson(item, output);
    }
    output << ']';
  } else if (value.is_object()) {
    output << '{';
    bool first = true;
    for (const auto& [key, item] : value.members()) {
      if (!first)
        output << ',';
      first = false;
      output << json::Value(key).dump() << ':';
      WriteScoreJson(item, output);
    }
    output << '}';
  } else {
    output << value.dump();
  }
}
}  // namespace

// Uses external continuations from the matching hosted checkpoint. This is
// deliberately separate from comparing two routes through our own decoder.
void ScoreOfficialReference(
    const std::shared_ptr<models::deepseek_v4_flash::Model>& model,
    const std::string& output_path) {
  Require(!model->HasDspark(), "official scoring requires ordinary decoding");
  std::ifstream input(GUFO_DS4_OFFICIAL_FIXTURE, std::ios::binary);
  Require(input.good(), "cannot open official continuation fixture");
  const std::string bytes{std::istreambuf_iterator<char>(input), {}};
  const auto fixture = json::parse(bytes);
  Require(fixture.member_str("schema") == "gufo.ds4-official-continuations.v1",
          "unexpected official fixture schema");
  const auto& cases = Field(fixture, "cases").items();
  Require(cases.size() == 105, "incomplete official fixture");

  json::Value report = json::Value::object();
  report["schema"] = "gufo.ds4-official-score.v1";
  report["source_revision"] = fixture.member_str("source_revision");
  report["fixture_sha256"] = crypto::Sha256Hex(std::span(
      reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()));
  report["prefill_policy"] = "native automatic";
  report["cases"] = json::Value::array();
  std::map<std::string, Totals> groups;
  for (const auto& item : cases) {
    const auto id = item.member_str("id");
    const auto group = item.member_str("group");
    Require(!id.empty() && !group.empty(), "missing official case identity");
    const auto prompt = model->EncodeChat("", item.member_str("prompt"));
    const auto target = model->Tokenize(item.member_str("continuation"));
    const auto& official_tokens = Field(item, "token_bytes_hex").items();
    Require(!prompt.empty() && !target.empty() &&
                target.size() == official_tokens.size(),
            id + ": tokenization differs from official continuation");
    for (std::size_t index = 0; index < target.size(); ++index) {
      Require(Hex(model->DecodeToken(target[index])) ==
                  official_tokens[index].str(),
              id + ": official token boundaries differ");
    }

    const std::uint32_t context = group == "continuation-100" ? 4096 : 16384;
    Require(prompt.size() + target.size() + 1 < context,
            id + ": official case exceeds context");
    std::string error;
    auto session = model->CreateSession(context, &error);
    Require(session != nullptr, error);
    Require(session->Sync(prompt, &error), error);
    Totals result{.cases = 1, .tokens = target.size()};
    bool prefix_matches = true;
    json::Value steps = json::Value::array();
    for (std::size_t index = 0; index < target.size(); ++index) {
      const auto logits = session->CopyLogits(&error);
      Require(logits.size() == static_cast<std::size_t>(model->VocabSize()) &&
                  std::all_of(logits.begin(), logits.end(),
                              [](float value) { return std::isfinite(value); }),
              id + ": invalid target logits");
      const int greedy = session->SelectNext(0.0F, nullptr);
      const auto maximum_logit = std::max_element(logits.begin(), logits.end());
      Require(greedy == static_cast<int>(maximum_logit - logits.begin()),
              id + ": decoder argmax differs from its full logits");
      const bool matches = greedy == target[index];
      result.first_matches += index == 0 && matches;
      result.matches += matches;
      prefix_matches = prefix_matches && matches;
      result.matching_prefix += prefix_matches;
      const double maximum = *maximum_logit;
      double normalizer = 0;
      for (const float logit : logits)
        normalizer += std::exp(static_cast<double>(logit) - maximum);
      const double logprob = static_cast<double>(logits[target[index]]) -
                             (maximum + std::log(normalizer));
      Require(std::isfinite(logprob), id + ": invalid target log probability");
      result.nll -= logprob;
      json::Value step = json::Value::object();
      step["target"] = target[index];
      step["greedy"] = greedy;
      step["logprob"] = logprob;
      steps.push_back(std::move(step));
      Require(session->Evaluate(target[index], &error), error);
    }
    auto& total = groups[group];
    total.cases += result.cases;
    total.tokens += result.tokens;
    total.first_matches += result.first_matches;
    total.matching_prefix += result.matching_prefix;
    total.matches += result.matches;
    total.nll += result.nll;
    json::Value record = json::Value::object();
    record["id"] = id;
    record["group"] = group;
    record["context_tokens"] = static_cast<std::size_t>(context);
    record["prefill_capacity"] =
        static_cast<std::size_t>(session->PrefillCapacity());
    record["prompt_tokens"] = prompt.size();
    record["target_tokens"] = target.size();
    record["nll"] = result.nll;
    record["first_match"] = result.first_matches;
    record["greedy_lcp"] = result.matching_prefix;
    record["steps"] = std::move(steps);
    report["cases"].push_back(std::move(record));
    std::cerr << "Official " << id << ": prompt=" << prompt.size()
              << " tokens=" << target.size()
              << " NLL=" << result.nll / static_cast<double>(result.tokens)
              << " prefix=" << result.matching_prefix << '\n';
  }
  Require(groups["continuation-100"].cases == 100 &&
              groups["continuation-100"].tokens == 2313 &&
              groups["smoke-5"].cases == 5 && groups["smoke-5"].tokens == 14,
          "official scoring coverage changed");
  report["summary"] = json::Value::object();
  for (const auto& [name, total] : groups) {
    auto summary = json::Value::object();
    summary["cases"] = total.cases;
    summary["tokens"] = total.tokens;
    summary["nll"] = total.nll / static_cast<double>(total.tokens);
    summary["first_matches"] = total.first_matches;
    summary["mean_greedy_prefix"] = static_cast<double>(total.matching_prefix) /
                                    static_cast<double>(total.cases);
    summary["greedy_matches"] = total.matches;
    report["summary"][name] = std::move(summary);
  }
  std::ofstream output(output_path);
  Require(output.good(), "cannot open official score output");
  WriteScoreJson(report, output);
  output << '\n';
  Require(output.good(), "cannot write official score output");
  WriteScoreJson(Field(report, "summary"), std::cout);
  std::cout << '\n';
}

void DumpReferenceFrontiers(
    const std::shared_ptr<models::deepseek_v4_flash::Model>& model,
    const std::string& output_directory) {
  namespace fs = std::filesystem;
  using models::deepseek_v4_flash::ChatMessage;
  using models::deepseek_v4_flash::RenderChat;
  Require(!model->HasDspark(),
          "frontier comparison requires ordinary decoding");
  const fs::path directory(output_directory);
  Require(!fs::exists(directory), "frontier output directory already exists");
  fs::create_directories(directory);

  std::string continuation;
  std::vector<int> targets;
  while (targets.size() < 128) {
    continuation +=
        " Virtual memory maps pages to physical memory. "
        "The operating system tracks permissions and page faults.";
    targets = model->Tokenize(continuation);
  }
  targets.resize(128);
  std::ofstream target_file(directory / "targets.txt");
  for (const int token : targets)
    target_file << token << '\n';
  Require(target_file.good(), "cannot write frontier target tokens");
  target_file.close();

  auto messages = [](std::string body) {
    return std::vector<ChatMessage>{{.role = "user",
                                     .content = std::move(body),
                                     .reasoning_content = {},
                                     .tool_calls = {}}};
  };
  Require(model->Tokenize("x").size() == 1 && model->Tokenize(" x").size() == 1,
          "frontier prefix requires single-token words");
  const auto overhead = model->EncodeChat(messages("x")).size() - 1;
  json::Value report = json::Value::object();
  report["vocab"] = model->VocabSize();
  report["dtype"] = "float32-le";
  report["prefill_policy"] = "explicit 2048/4096-token calls";
  report["context_capacity"] = 20480;
  report["prefix"] =
      "repeated x words inside the native no-thinking chat template";
  report["decode_tokens"] = targets.size();
  report["points"] = json::Value::array();
  std::ofstream manifest(directory / "manifest.tsv");
  for (const std::size_t depth : {0u, 4096u, 8192u, 12288u, 16384u}) {
    const std::size_t prefix = depth == 0 ? 16 : depth;
    Require(prefix > overhead, "frontier prefix is shorter than chat framing");
    std::string body = "x";
    for (std::size_t word = 1; word < prefix - overhead; ++word)
      body += " x";
    const auto chat = messages(std::move(body));
    const auto prompt = model->EncodeChat(chat);
    Require(prompt.size() == prefix, "frontier prefix depth changed");
    const std::string stem = std::to_string(depth);
    std::ofstream prompt_tokens_file(directory / (stem + ".tokens.txt"));
    for (const int token : prompt)
      prompt_tokens_file << token << '\n';
    Require(prompt_tokens_file.good(), "cannot write frontier prompt tokens");
    prompt_tokens_file.close();
    const auto prompt_path = directory / (stem + ".prompt.txt");
    std::ofstream prompt_file(prompt_path, std::ios::binary);
    prompt_file << RenderChat(chat);
    Require(prompt_file.good(), "cannot write frontier prompt");
    prompt_file.close();
    for (const std::size_t prefill_step : {2048u, 4096u}) {
      std::string error;
      auto session = model->CreateSession(20480, &error);
      Require(session != nullptr, error);
      manifest << prefill_step << '\t' << depth << '\t' << prefix << '\t'
               << session->PrefillCapacity() << '\t'
               << fs::absolute(prompt_path).string() << '\n';
      std::size_t end = 0;
      while (end < prompt.size()) {
        end = std::min(prompt.size(), end + prefill_step);
        Require(session->Sync(std::span(prompt).first(end), &error), error);
      }
      json::Value point = json::Value::object();
      point["depth"] = depth;
      point["prefill_step"] = prefill_step;
      point["prefill_capacity"] =
          static_cast<std::size_t>(session->PrefillCapacity());
      point["prompt_tokens"] = prefix;
      point["greedy"] = json::Value::array();
      point["target_logprobs"] = json::Value::array();
      auto dump = [&](std::string_view suffix,
                      const std::vector<float>& logits) {
        const auto path = directory / (std::to_string(prefill_step) + "-" +
                                       stem + std::string(suffix) + ".f32");
        const auto view = std::as_bytes(std::span(logits));
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(view.data()),
                   static_cast<std::streamsize>(view.size()));
        Require(file.good(), "cannot write frontier logits");
        return crypto::Sha256Hex(std::span(
            reinterpret_cast<const std::uint8_t*>(view.data()), view.size()));
      };
      for (std::size_t step = 0; step <= targets.size(); ++step) {
        const auto logits = session->CopyLogits(&error);
        Require(
            logits.size() == static_cast<std::size_t>(model->VocabSize()) &&
                std::all_of(logits.begin(), logits.end(),
                            [](float value) { return std::isfinite(value); }),
            "invalid frontier logits");
        const auto maximum_logit =
            std::max_element(logits.begin(), logits.end());
        const int greedy = session->SelectNext(0.0F, nullptr);
        Require(greedy == static_cast<int>(maximum_logit - logits.begin()),
                "frontier argmax differs from its full logits");
        if (step == 0)
          point["before_sha256"] = dump("-before", logits);
        if (step == targets.size()) {
          point["after_sha256"] = dump("-after", logits);
          break;
        }
        const double maximum = *maximum_logit;
        double sum = 0;
        for (const float logit : logits)
          sum += std::exp(static_cast<double>(logit) - maximum);
        const double logprob = static_cast<double>(logits[targets[step]]) -
                               (maximum + std::log(sum));
        Require(std::isfinite(logprob), "invalid frontier target likelihood");
        point["greedy"].push_back(greedy);
        point["target_logprobs"].push_back(logprob);
        Require(session->Evaluate(targets[step], &error), error);
      }
      report["points"].push_back(std::move(point));
      std::cerr << "AR frontier prefill=" << prefill_step << " depth=" << depth
                << " tokens=128 complete\n";
    }
  }
  Require(manifest.good(), "cannot write frontier manifest");
  std::ofstream output(directory / "summary.json");
  WriteScoreJson(report, output);
  output << '\n';
  Require(output.good(), "cannot write frontier summary");
}
}  // namespace gufo::testing::ds4
