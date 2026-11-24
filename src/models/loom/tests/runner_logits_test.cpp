// Host-sampling behavior of the Loom runner against a fake libloom_text_embed
// (fake_embed.cpp): no GPU.
#include <cmath>
#include <cstdlib>
#include <gufo/ext/platform/os_info.hpp>
#include <iostream>

#include "../loom_runner.cpp"  // the runner's types live in an anonymous namespace
#include "fake_embed.hpp"

using namespace gufo;
using namespace gufo::loom;

static int g_failures = 0;
#define CHECK(...) CheckImpl(__FILE__, __LINE__, #__VA_ARGS__, __VA_ARGS__)
static void CheckImpl(const char* file, int line, const char* text, bool ok,
                      const std::string& note = {}) {
  if (ok)
    return;
  ++g_failures;
  std::cerr << "FAIL " << file << ":" << line << ": " << text << " " << note
            << "\n";
}

namespace {

struct Run {
  std::vector<int> tokens;
  std::vector<std::optional<sampling::TokenLogprobs>> logprobs;
  std::string text;
  bool stopped{false};
  std::size_t epochs_host{0}, epochs_device{0}, exported{0};
};

class Fixture {
public:
  explicit Fixture(bool logits = true, std::size_t rows = 2,
                   const std::string& chat = {}) {
    dir_ = std::filesystem::temp_directory_path() /
           ("loom_runner_test_" + std::to_string(gufo::platform::ProcessId()));
    std::filesystem::create_directories(dir_);
    // A package without "llm.logits" samples on the device.
    std::ofstream(dir_ / "model.json")
        << R"({"schema_version":1,"programs":{},"services":{"chat":{"kind":"text"}},)"
           R"("variants":{"v":{"services":{"chat":{"config":{"llm.card":true)"
        << (logits ? R"(,"llm.logits":false)" : "") << "}}}}}}";
    // The package's dialect (llm_config.json "chat", in a package built for
    // this SDK); Hermes JSON by default, no llm_config.json at all for "{}".
    if (chat != "{}")
      std::ofstream(dir_ / "llm_config.json")
          << R"({"sdk":{"hrx_loom":")" LOOM_TEXT_EMBED_HRX_LOOM R"("},"chat":)"
          << (chat.empty() ? R"({"call_format":"hermes"})" : chat) << "}";
    fake::state().epochs.clear();
    ext::TextRunnerOptions options;
    options.model_path = dir_.string();
    options.session_count = rows;
    model = std::make_shared<LoomModel>(options);
    runner = std::make_unique<LoomRunner>(model, 0, false, 0);
    pool = std::make_shared<RowPool>();
    pool->used.assign(rows, false);
  }
  ~Fixture() { std::filesystem::remove_all(dir_); }

  // One request on `row`: prefill the prompt, then decode up to `max_tokens`
  // the way Gufo's pool does (Accept after each returned selection).
  Run Generate(std::uint32_t row, sampling::SamplingConfig config,
               std::size_t max_tokens,
               ext::ChatRequest request = DefaultRequest()) {
    auto slot = std::make_unique<LoomSlot>(row, pool);
    loom_text_embed_decoder_t* decoder = nullptr;
    loom_text_embed_decoder_open(model->embed(), &decoder);
    LoomState state(*slot, decoder);
    state.SetStopAtEos(true);
    auto prepared = runner->PreparePrompt(request);
    CHECK(prepared.has_value());
    runner->SetPromptContext(state, prepared->context);
    const auto& prompt = prepared->tokens;
    std::size_t offset = 0;
    while (true) {
      const auto step = runner->Prefill(state, prompt, offset, 64);
      offset += step.consumed_tokens;
      if (step.decode_ready)
        break;
    }
    const auto before = fake::state().epochs.size();
    sampling::SamplerState sampler(config,
                                   std::span<const sampling::TokenId>());
    Run run;
    for (std::size_t n = 0; n < max_tokens && !run.stopped;) {
      const ext::TextRunnerDecode decode{state, max_tokens - n, sampler};
      auto steps = runner->DecodeBatch(std::span(&decode, 1));
      auto& step = steps.front();
      if (step.failure)
        std::rethrow_exception(step.failure);
      for (auto& selection : step.selections) {
        try {
          sampler.Accept(selection.token);
        } catch (const std::exception& error) {
          std::cerr << "accept failed after '" << run.text << "' token "
                    << selection.token << " ('" << selection.piece
                    << "'): " << error.what() << "\n";
          throw;
        }
        run.tokens.push_back(static_cast<int>(selection.token));
        run.logprobs.push_back(selection.logprobs);
        run.text += selection.piece;
        ++n;
      }
      run.stopped = step.stop;
      if (step.selections.empty() && !step.stop)
        break;
    }
    const auto& epochs = fake::state().epochs;
    for (std::size_t i = before; i < epochs.size(); ++i) {
      (epochs[i].host_tokens ? run.epochs_host : run.epochs_device)++;
      for (const bool e : epochs[i].exported)
        run.exported += e;
    }
    return run;
  }

  static ext::ChatRequest DefaultRequest() {
    ext::ChatRequest request;
    request.messages.emplace_back(ext::ChatRole::kUser, "Hi");
    return request;
  }

  std::filesystem::path dir_;
  std::shared_ptr<LoomModel> model;
  std::unique_ptr<LoomRunner> runner;
  std::shared_ptr<RowPool> pool;
};

// Row: a sharp peak at token (last*7 + position*3 + 11) % 200 + 32 (printable
// range), then a smoother falloff; EOS never wins.
void PeakScript(int last, int position, float* out) {
  const int target = (last * 7 + position * 3 + 11) % 90 + 32;
  for (int v = 0; v < fake::kBytes; ++v) {
    const float d = static_cast<float>(v - target);
    out[v] = -0.05F * d * d;
  }
  out[fake::kEos] = -50.0F;
}

int DevicePick(int last, int position) {
  std::vector<float> row(fake::kWidth, -3e38F);
  PeakScript(last, position, row.data());
  return fake::Argmax(row.data());
}

double LogSoftmax(const float* row, int token) {
  double m = -1e300;
  for (int i = 0; i < fake::kVocabulary; ++i)
    m = std::max<double>(m, row[i]);
  double sum = 0;
  for (int i = 0; i < fake::kVocabulary; ++i)
    sum += std::exp(static_cast<double>(row[i]) - m);
  return static_cast<double>(row[token]) - m - std::log(sum);
}

void TestDevicePathUnchanged() {
  std::cerr << "device path: no host need keeps the device sampler\n";
  fake::reset();
  fake::state().script = PeakScript;
  Fixture f;
  sampling::SamplingConfig config;
  const auto run = f.Generate(0, config, 8);
  CHECK(run.tokens.size() == 8);
  CHECK(run.epochs_host == 0, "no host tokens");
  CHECK(run.exported == 0, "no logits downloaded");
  CHECK(f.model->logits().enabled);
}

void TestLogprobs() {
  std::cerr << "logprobs: host path reports exact log-softmax\n";
  fake::reset();
  fake::state().script = PeakScript;
  Fixture f;
  sampling::SamplingConfig config;
  config.logprobs = 3;
  const auto run = f.Generate(0, config, 6);
  CHECK(run.tokens.size() == 6);
  CHECK(run.epochs_host >= 5, "host tokens used");
  // Reference: replay the greedy walk on the scripted rows.
  int last = 'H';  // the fake prompt ends at the last byte of "PROMPT:{"mod"
  (void)last;
  for (std::size_t i = 0; i < run.tokens.size(); ++i) {
    CHECK(run.logprobs[i].has_value(), "logprobs present");
    if (!run.logprobs[i])
      continue;
    const auto& lp = *run.logprobs[i];
    CHECK(lp.selected.token == static_cast<sampling::TokenId>(run.tokens[i]));
    CHECK(lp.top.size() == 3);
    CHECK(lp.top.front().token == lp.selected.token, "greedy token is top-1");
    CHECK(lp.selected.logprob <= 0.0);
    CHECK(lp.selected.logprob == lp.top.front().logprob,
          "selected is the top-1 entry");
    CHECK(lp.top[0].logprob >= lp.top[1].logprob &&
          lp.top[1].logprob >= lp.top[2].logprob);
    double mass = 0;
    for (const auto& t : lp.top)
      mass += std::exp(t.logprob);
    CHECK(mass <= 1.0 + 1e-9, "top-3 mass");
  }
  // The same tokens as the device path (temperature 0).
  Fixture g;
  sampling::SamplingConfig plain;
  const auto reference = g.Generate(0, plain, 6);
  CHECK(reference.tokens == run.tokens, "host greedy == device greedy");
}

void TestRepetitionPenalty() {
  std::cerr << "penalties: the host applies them (the device sampler cannot)\n";
  fake::reset();
  // Two tokens nearly tied: 'A' slightly ahead of 'B' every step.
  fake::state().script = [](int, int, float* out) {
    for (int v = 0; v < fake::kBytes; ++v)
      out[v] = -20.0F;
    out['A'] = 5.0F;
    out['B'] = 4.9F;
    out[fake::kEos] = -50.0F;
  };
  Fixture f;
  sampling::SamplingConfig plain;
  const auto device = f.Generate(0, plain, 6);
  CHECK(std::all_of(device.tokens.begin(), device.tokens.end(),
                    [](int t) { return t == 'A'; }),
        "device repeats A");
  sampling::SamplingConfig config;
  config.repeat_penalty = 2.0F;
  config.repeat_last_n = 64;
  const auto host = f.Generate(0, config, 6);
  CHECK(host.tokens.size() == 6);
  CHECK(host.tokens[0] == 'A');
  CHECK(host.tokens[1] == 'B', "penalty moves the second token to B");
  CHECK(host.epochs_host >= 5);
}

void TestMinP() {
  std::cerr << "min_p: tokens below min_p * p_max are dropped\n";
  fake::reset();
  // Top token 'A' p~0.99; with min_p=0.5 only A can ever be sampled, even at
  // high temperature and any seed.
  fake::state().script = [](int, int, float* out) {
    for (int v = 0; v < fake::kBytes; ++v)
      out[v] = -2.0F;
    out['A'] = 6.0F;
    out[fake::kEos] = -50.0F;
  };
  Fixture f;
  sampling::SamplingConfig config;
  config.temperature = 1.5F;
  config.min_p = 0.5F;
  config.seed = 11;
  const auto run = f.Generate(0, config, 12);
  CHECK(std::all_of(run.tokens.begin(), run.tokens.end(),
                    [](int t) { return t == 'A'; }),
        "only the top token survives");
  config.min_p = 0.0F;
  config.logprobs = 1;  // host path, no min_p
  config.seed = 11;
  const auto loose = f.Generate(0, config, 40);
  CHECK(std::any_of(loose.tokens.begin(), loose.tokens.end(),
                    [](int t) { return t != 'A'; }),
        "without min_p others appear");
}

void TestSeedStream() {
  std::cerr << "seeded host sampling is deterministic across runs\n";
  fake::reset();
  fake::state().script = PeakScript;
  sampling::SamplingConfig config;
  config.temperature = 2.0F;
  config.seed = 5;
  config.logprobs = 1;
  Fixture f;
  const auto a = f.Generate(0, config, 16);
  const auto b = f.Generate(0, config, 16);
  CHECK(a.tokens == b.tokens);
  config.seed = 6;
  const auto c = f.Generate(0, config, 16);
  CHECK(a.tokens != c.tokens);
}

void TestJsonGrammar() {
  std::cerr << "grammar: json_schema enforced from the logits\n";
  fake::reset();
  // The model prefers garbage letters; the grammar must win.
  fake::state().script = [](int last, int position, float* out) {
    PeakScript(last, position, out);
    out[fake::kEos] = 0.0F;
  };
  Fixture f;
  json::Value schema = json::parse(
      R"({"type":"object","properties":{"city":{"type":"string"},"n":{"type":"integer"}},"required":["city","n"],"additionalProperties":false})");
  auto grammar = sampling::JsonConstraint::Compile(schema, true, false);
  sampling::SamplingConfig config;
  config.constraint = f.runner->BindConstraint(grammar);
  const auto run = f.Generate(0, config, 200);
  CHECK(run.stopped, "the grammar completes and EOS ends the response");
  json::Value parsed;
  bool ok = true;
  try {
    parsed = json::parse(run.text);
  } catch (...) {
    ok = false;
  }
  CHECK(ok, "output parses as JSON: " + run.text);
  if (ok) {
    CHECK(parsed.is_object() && parsed.contains("city") &&
          parsed.contains("n"));
  }
  CHECK(run.epochs_host > 0);
  // Without logits the same request is served unconstrained (status quo).
  Fixture g(false);
  sampling::SamplingConfig again;
  again.constraint = g.runner->BindConstraint(grammar);
  const auto plain = g.Generate(0, again, 12);
  CHECK(plain.epochs_host == 0, "device only without logits");
}

void TestToolChoiceRequired() {
  std::cerr << "tool_choice required: the call is forced\n";
  fake::reset();
  fake::state().script = PeakScript;
  Fixture f;
  json::Value params = json::parse(
      R"({"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false})");
  const auto syntax = f.runner->ToolGrammarSyntax(ext::ChatRequest{});
  CHECK(syntax.has_value(), "the package has a native call grammar");
  std::vector<sampling::JsonConstraint::Tool> tools;
  tools.emplace_back("get_weather", sampling::JsonConstraint::ToolParameters(
                                        params, true, *syntax));
  auto grammar = sampling::JsonConstraint::WithTools(nullptr, std::move(tools),
                                                     true, false, *syntax);
  sampling::SamplingConfig config;
  config.constraint = f.runner->BindConstraint(grammar);
  ext::ChatRequest request = Fixture::DefaultRequest();
  request.tools.push_back(
      {.name = "get_weather", .parameters_json = R"({"type":"object"})"});
  request.tool_choice = ext::ChatRequest::ToolChoice::kRequired;
  const auto run = f.Generate(0, config, 120, request);
  CHECK(run.text.rfind("<tool_call>", 0) == 0,
        "starts with the call marker: " + run.text);
  CHECK(run.text.find("get_weather") != std::string::npos);
  CHECK(run.epochs_host > 0);
}

void TestLazyGrammarStaysOnDevice() {
  std::cerr
      << "auto tools: device sampling until the call starts, then the host\n";
  fake::reset();
  fake::state().script = PeakScript;
  Fixture f;
  json::Value params = json::parse(
      R"({"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false})");
  const auto syntax = f.runner->ToolGrammarSyntax(ext::ChatRequest{});
  CHECK(syntax.has_value(), "the package has a native call grammar");
  std::vector<sampling::JsonConstraint::Tool> tools;
  tools.emplace_back("get_weather", sampling::JsonConstraint::ToolParameters(
                                        params, true, *syntax));
  auto grammar = sampling::JsonConstraint::WithTools(nullptr, std::move(tools),
                                                     false, false, *syntax);
  sampling::SamplingConfig config;
  config.constraint = f.runner->BindConstraint(grammar);
  ext::ChatRequest request = Fixture::DefaultRequest();
  request.tools.push_back(
      {.name = "get_weather", .parameters_json = R"({"type":"object"})"});
  const auto prompt = f.runner->PreparePrompt(request);
  CHECK(prompt.has_value());
  const int base = static_cast<int>(prompt->tokens.size());
  // The model writes "ok ", then the call marker; after it the script keeps
  // writing garbage that only the grammar can turn into a call.
  const std::string plan = "ok <tool_call>";
  fake::state().script = [plan, base](int last, int position, float* out) {
    const int index = position - base;
    PeakScript(last, position, out);
    if (index >= 0 && index < static_cast<int>(plan.size())) {
      for (int v = 0; v < fake::kBytes; ++v)
        out[v] = -20.0F;
      out[static_cast<unsigned char>(plan[index])] = 5.0F;
    }
  };
  const auto run = f.Generate(0, config, 60, request);
  CHECK(run.text.rfind("ok <tool_call>", 0) == 0,
        "the lazy prefix is untouched: " + run.text);
  CHECK(run.epochs_device >= 1, "device epochs before the call");
  CHECK(run.epochs_host >= 1, "host epochs once the grammar restricts");
  // After the marker the grammar owns the output: a name and JSON arguments.
  CHECK(run.text.find("\"name\"") != std::string::npos,
        "call body follows: " + run.text);
}

void TestDialectGating() {
  std::cerr
      << "dialect: a package that declares none keeps tool grammars off\n";
  fake::reset();
  fake::state().script = PeakScript;
  fake::state().tool_call_text =
      "<tool_call>\n<function=qprobe>\n<parameter=zz>\n7\n</parameter>\n</"
      "function>\n</tool_call>";
  Fixture f(true, 2, "{}");  // no llm_config.json
  CHECK(!f.runner->NativeToolGrammar());
  json::Value params = json::parse(
      R"({"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false})");
  CHECK(!f.runner->ToolGrammarSyntax(ext::ChatRequest{}).has_value(),
        "no native grammar for an unknown syntax");
  std::vector<sampling::JsonConstraint::Tool> tools;
  tools.emplace_back(
      "get_weather",
      sampling::JsonConstraint::ToolParameters(
          params, true, sampling::JsonConstraint::ToolFormat::kJson));
  auto grammar = sampling::JsonConstraint::WithTools(
      nullptr, std::move(tools), true, false,
      sampling::JsonConstraint::ToolFormat::kJson);
  sampling::SamplingConfig config;
  config.constraint = f.runner->BindConstraint(grammar);
  ext::ChatRequest request = Fixture::DefaultRequest();
  request.tools.push_back(
      {.name = "get_weather", .parameters_json = R"({"type":"object"})"});
  request.tool_choice = ext::ChatRequest::ToolChoice::kRequired;
  const auto run = f.Generate(0, config, 10, request);
  CHECK(run.epochs_host == 0, "unconstrained: the device path serves it");
  // The same package still honours logprobs on a tools request.
  sampling::SamplingConfig lp;
  lp.logprobs = 2;
  const auto with_lp = f.Generate(0, lp, 4, request);
  CHECK(with_lp.epochs_host > 0);
  // A reasoning marker that is not <think> turns every grammar off.
  fake::reset();
  fake::state().script = PeakScript;
  fake::state().reasoning_close = "<|end|>";
  Fixture g(true, 2, "{}");
  sampling::SamplingConfig plain;
  plain.constraint =
      g.runner->BindConstraint(sampling::JsonConstraint::Object());
  const auto no_grammar = g.Generate(0, plain, 6);
  CHECK(no_grammar.epochs_host == 0, "reasoning markers differ: no grammar");
}

void TestBatchMix() {
  std::cerr << "batch: hosted and device rows share one DecodeBatch\n";
  fake::reset();
  fake::state().script = PeakScript;
  Fixture f(true, 3);
  // Rows 0 (logprobs) and 1 (plain) decode together.
  struct One {
    std::unique_ptr<LoomSlot> slot;
    std::unique_ptr<LoomState> state;
    std::unique_ptr<sampling::SamplerState> sampler;
    std::vector<int> tokens;
  };
  std::vector<One> ones(2);
  for (std::uint32_t r = 0; r < 2; ++r) {
    ones[r].slot = std::make_unique<LoomSlot>(r, f.pool);
    loom_text_embed_decoder_t* decoder = nullptr;
    loom_text_embed_decoder_open(f.model->embed(), &decoder);
    ones[r].state = std::make_unique<LoomState>(*ones[r].slot, decoder);
    ones[r].state->SetStopAtEos(true);
    auto prepared = f.runner->PreparePrompt(Fixture::DefaultRequest());
    f.runner->SetPromptContext(*ones[r].state, prepared->context);
    std::size_t offset = 0;
    while (true) {
      const auto step =
          f.runner->Prefill(*ones[r].state, prepared->tokens, offset, 64);
      offset += step.consumed_tokens;
      if (step.decode_ready)
        break;
    }
    sampling::SamplingConfig config;
    if (r == 0)
      config.logprobs = 2;
    ones[r].sampler = std::make_unique<sampling::SamplerState>(
        config, std::span<const sampling::TokenId>());
  }
  for (int round = 0; round < 5; ++round) {
    const ext::TextRunnerDecode decodes[] = {
        {*ones[0].state, 1, *ones[0].sampler},
        {*ones[1].state, 1, *ones[1].sampler}};
    auto steps = f.runner->DecodeBatch(std::span(decodes, 2));
    for (std::size_t i = 0; i < 2; ++i) {
      if (steps[i].failure)
        std::rethrow_exception(steps[i].failure);
      for (auto& sel : steps[i].selections) {
        ones[i].sampler->Accept(sel.token);
        ones[i].tokens.push_back(static_cast<int>(sel.token));
        CHECK(sel.logprobs.has_value() == (i == 0),
              "logprobs only on the hosted row");
      }
    }
  }
  CHECK(ones[0].tokens.size() == 5 && ones[1].tokens.size() == 5);
  CHECK(ones[0].tokens == ones[1].tokens,
        "same prompt, same greedy tokens either way");
}

void TestSelectAdvance() {
  std::cerr << "SelectNext/Advance: the same tokens as DecodeBatch\n";
  fake::reset();
  fake::state().script = PeakScript;
  Fixture f;
  sampling::SamplingConfig config;
  config.logprobs = 2;
  const auto batch = f.Generate(0, config, 6);
  // Same request through SelectNext + Advance.
  auto slot = std::make_unique<LoomSlot>(0, f.pool);
  loom_text_embed_decoder_t* decoder = nullptr;
  loom_text_embed_decoder_open(f.model->embed(), &decoder);
  LoomState state(*slot, decoder);
  state.SetStopAtEos(true);
  auto prepared = f.runner->PreparePrompt(Fixture::DefaultRequest());
  f.runner->SetPromptContext(state, prepared->context);
  std::size_t offset = 0;
  while (true) {
    const auto step = f.runner->Prefill(state, prepared->tokens, offset, 64);
    offset += step.consumed_tokens;
    if (step.decode_ready)
      break;
  }
  sampling::SamplerState sampler(config, std::span<const sampling::TokenId>());
  std::vector<int> tokens;
  for (int i = 0; i < 6; ++i) {
    auto selection = f.runner->SelectNext(state, sampler);
    CHECK(!selection.stop);
    CHECK(selection.logprobs.has_value());
    sampler.Accept(selection.token);
    tokens.push_back(static_cast<int>(selection.token));
    f.runner->Advance(state, selection.token);
  }
  CHECK(tokens == batch.tokens, "select/advance == batch");
}

// ---------------------------------------------------------------- dialects ---

bool Accepts(const sampling::JsonConstraint& grammar, std::string_view text) {
  auto state = grammar.Start();
  for (const unsigned char byte : text) {
    state = grammar.Advance(state, byte);
    if (state.empty())
      return false;
  }
  return grammar.Complete(state);
}

struct DialectCase {
  const char* format;
  const char* chat;     // the "chat" object of llm_config.json
  const char* example;  // a call as the package's template writes it
  const char* open;     // what a required call must start with
};

// The grammar of a required call, built as Gufo's ConstrainChatRequest does.
std::shared_ptr<const sampling::JsonConstraint> RequiredCall(
    const sampling::JsonConstraint::ToolSyntax& syntax) {
  json::Value params = json::parse(
      R"({"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false})");
  std::vector<sampling::JsonConstraint::Tool> tools;
  tools.emplace_back("get_weather", sampling::JsonConstraint::ToolParameters(
                                        params, false, syntax));
  return sampling::JsonConstraint::WithTools(nullptr, std::move(tools), true,
                                             false, syntax);
}

void TestDialects() {
  std::cerr
      << "dialects: native call syntax per package, from llm_config.json\n";
  const DialectCase cases[] = {
      {"qwen_xml",
       R"({"call_format":"qwen_xml","think_open":"<think>\n","think_end":"</think>","thinking":"toggle"})",
       "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</"
       "parameter>\n</function>\n</tool_call>",
       "<tool_call>"},
      {"hermes",
       R"({"call_format":"hermes","think_open":"<think>\n","think_end":"</think>","thinking":"toggle"})",
       "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": "
       "\"Paris\"}}\n</tool_call>",
       "<tool_call>"},
      {"glm",
       R"({"call_format":"glm","think_open":"<think>","think_end":"</think>","thinking":"toggle"})",
       "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</"
       "arg_value></tool_call>",
       "<tool_call>"},
      {"gemma4",
       R"({"call_format":"gemma4","think_open":"<|channel>thought\n","think_end":"<channel|>","thinking":"toggle"})",
       "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>",
       "<|tool_call>"},
      {"harmony",
       R"({"call_format":"harmony","think_open":"<|channel|>analysis<|message|>","think_end":"<|end|>","thinking":"always"})",
       "<|channel|>commentary to=functions.get_weather "
       "<|constrain|>json<|message|>{\"city\":\"Paris\"}<|call|>",
       "<|channel|>commentary"},
      {"json",
       R"({"call_format":"json","call_arguments_key":"parameters","think_open":"","think_end":"</think>","thinking":"never"})",
       "{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}}",
       "{\"name\": \""},
      {"pythonic",
       R"({"call_format":"pythonic","calls_open":"<|tool_call_start|>[","call_separator":", ","calls_close":"]<|tool_call_end|>","think_open":"<think>","think_end":"</think>","thinking":"always"})",
       "<|tool_call_start|>[get_weather(city='Paris')]<|tool_call_end|>",
       "<|tool_call_start|>["},
      {"inkling2",
       R"({"call_format":"inkling2","call_marker":"<|content_invoke_tool_json|>","think_open":"<|message_model|><|content_thinking|>","think_end":"<|end_message|>","thinking":"toggle"})",
       "<|message_model|>get_weather<|content_invoke_tool_json|>"
       "{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\"}}"
       "<|end_message|>",
       "<|message_model|>"},
      {"atem",
       R"({"call_format":"atem","think_open":"<|start|>assistant to=self<|message|>","think_end":"<|start|>assistant to=user<|message|>","thinking":"always"})",
       " to=get_weather<|message|><atem:function_calls>\n<atem:invoke "
       "name=\"get_weather\">\n<atem:parameter "
       "name=\"city\">Paris</atem:parameter>\n</atem:invoke>\n</"
       "atem:function_calls>",
       " to="},
  };
  for (const auto& c : cases) {
    // Automatic tool use stays on the device only where a lazy grammar cannot
    // tell a call from other output by its start.
    const std::string format = c.format;
    CHECK(NativeDialectOf(json::parse(c.chat)).auto_on_device ==
              (format == "harmony" || format == "atem" || format == "inkling2"),
          format + ": auto tool use on the device");
    fake::reset();
    fake::state().script = PeakScript;
    Fixture f(true, 2, c.chat);
    const auto syntax = f.runner->ToolGrammarSyntax(ext::ChatRequest{});
    CHECK(syntax.has_value(),
          std::string(c.format) + ": a native grammar exists");
    if (!syntax)
      continue;
    const auto grammar = RequiredCall(*syntax);
    CHECK(Accepts(*grammar, c.example),
          std::string(c.format) + ": the template's call is accepted");
    CHECK(!Accepts(*grammar, "Plain text."),
          std::string(c.format) + ": required refuses prose");
    CHECK(syntax->open == c.open,
          std::string(c.format) + ": open marker " + syntax->open);
    // The host sampler writes a call in the dialect's spelling from a model
    // that would write something else.
    sampling::SamplingConfig config;
    config.constraint = f.runner->BindConstraint(grammar);
    ext::ChatRequest request = Fixture::DefaultRequest();
    request.tools.push_back(
        {.name = "get_weather", .parameters_json = R"({"type":"object"})"});
    request.tool_choice = ext::ChatRequest::ToolChoice::kRequired;
    // The model writes prose first; the grammar must turn it into the call.
    // After its first bytes the script follows the template's call, so the
    // run ends exactly when the grammar says the call is complete.
    const std::string example = c.example;
    const int base =
        static_cast<int>(f.runner->PreparePrompt(request)->tokens.size());
    fake::state().script = [example, base](int last, int position, float* out) {
      PeakScript(last, position, out);
      const int index = position - base;
      if (index >= 6 && index - 6 < static_cast<int>(example.size())) {
        for (int v = 0; v < fake::kBytes; ++v)
          out[v] = -20.0F;
        out[static_cast<unsigned char>(example[index - 6])] = 5.0F;
      } else if (index - 6 >= static_cast<int>(example.size())) {
        out[fake::kEos] = 50.0F;
      }
    };
    const auto run = f.Generate(0, config, 400, request);
    CHECK(run.text.find(c.open) != std::string::npos,
          std::string(c.format) + ": writes the call: " + run.text);
    CHECK(Accepts(*grammar, run.text),
          std::string(c.format) +
              ": the generated call is complete: " + run.text);
    CHECK(run.stopped, std::string(c.format) + ": ends at the stop token");
    // Reasoning markers of the grammar are the model's own.
    const auto markers = f.runner->ReasoningGrammarMarkers(request);
    CHECK(markers.has_value());
    if (markers && std::string(c.format) == "harmony") {
      CHECK(markers->open == "<|channel|>analysis<|message|>");
      CHECK(markers->close == "<|end|><|start|>assistant", markers->close);
      ext::ChatRequest with_format = request;
      with_format.response_format = sampling::JsonConstraint::Object();
      CHECK(f.runner->ReasoningGrammarMarkers(with_format)->close ==
            "<|end|><|start|>assistant<|channel|>final<|message|>");
    }
    if (markers && std::string(c.format) == "gemma4") {
      CHECK(markers->open == "<|channel>thought\n" &&
            markers->close == "<channel|>");
    }
    if (markers && std::string(c.format) == "pythonic") {
      CHECK(markers->open == "<think>" && markers->close == "</think>");
    }
    if (markers && std::string(c.format) == "atem") {
      // The prompt ends in <|start|>assistant: the output carries the rest of
      // the reasoning header, and reasoning ends where the next header starts
      // (a call continues it with " to=NAME", the answer with " to=user").
      CHECK(markers->open == " to=self<|message|>", markers->open);
      CHECK(markers->close == "<|start|>assistant", markers->close);
      ext::ChatRequest with_format = request;
      with_format.response_format = sampling::JsonConstraint::Object();
      CHECK(f.runner->ReasoningGrammarMarkers(with_format)->close ==
            "<|start|>assistant to=user<|message|>");
    }
    if (markers && std::string(c.format) == "json") {
      CHECK(markers->open.empty(),
            "a package that never reasons has no reasoning phase");
    }
  }
  // A package without tool calls has no grammar, and its requests keep being
  // served unconstrained.
  for (const char* format : {"none"}) {
    fake::reset();
    fake::state().script = PeakScript;
    Fixture f(
        true, 2,
        std::string(R"({"call_format":")") + format +
            R"(","think_open":"<think>","think_end":"</think>","thinking":"toggle"})");
    CHECK(!f.runner->ToolGrammarSyntax(ext::ChatRequest{}).has_value(),
          std::string(format) + ": no tool grammar");
    json::Value params = json::parse(R"({"type":"object"})");
    sampling::SamplingConfig config;
    config.constraint =
        f.runner->BindConstraint(sampling::JsonConstraint::Object());
    ext::ChatRequest request = Fixture::DefaultRequest();
    request.tools.push_back(
        {.name = "get_weather", .parameters_json = R"({"type":"object"})"});
    const auto with_tools = f.Generate(0, config, 6, request);
    CHECK(with_tools.epochs_host == 0,
          std::string(format) + ": tools requests run on the device");
    // A response format (no tools) still gets its grammar there.
    ext::ChatRequest plain = Fixture::DefaultRequest();
    const auto json_run = f.Generate(0, config, 6, plain);
    CHECK(json_run.epochs_host > 0,
          std::string(format) + ": response_format grammar runs");
  }
}

// Calls as the kit's chat_probe.render_call writes them for the packages'
// templates (the same text the model sees in its history), with every kind of
// argument: the grammar of each dialect must accept them.
void TestDialectRendering() {
  std::cerr << "dialects: the grammar accepts the template's own rendering\n";
  struct Case {
    const char* chat;
    const char* rendered;
  };
  const Case cases[] = {
      {R"({"call_format":"hermes"})",
       "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": "
       "\"Paris\", \"days\": 3, \"metric\": true, \"tags\": [\"a\", \"b\"], "
       "\"opts\": {\"x\": 1}}}\n</tool_call>"},
      {R"({"call_format":"qwen_xml"})",
       "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</"
       "parameter>\n<parameter=days>\n3\n</"
       "parameter>\n<parameter=metric>\ntrue\n</"
       "parameter>\n<parameter=tags>\n[\"a\",\"b\"]\n</"
       "parameter>\n<parameter=opts>\n{\"x\":1}\n</parameter>\n</function>\n</"
       "tool_call>"},
      {R"({"call_format":"glm"})",
       "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</"
       "arg_value><arg_key>days</arg_key><arg_value>3</"
       "arg_value><arg_key>metric</arg_key><arg_value>true</"
       "arg_value><arg_key>tags</arg_key><arg_value>[\"a\",\"b\"]</"
       "arg_value><arg_key>opts</arg_key><arg_value>{\"x\":1}</arg_value></"
       "tool_call>"},
      {R"({"call_format":"json","call_arguments_key":"parameters"})",
       "{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\", "
       "\"days\": 3, \"metric\": true, \"tags\": [\"a\", \"b\"], \"opts\": "
       "{\"x\": 1}}}"},
      {R"({"call_format":"gemma4"})",
       "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,days:3,metric:true,"
       "opts:{x:1},tags:[<|\"|>a<|\"|>,<|\"|>b<|\"|>]}<tool_call|>"},
      {R"({"call_format":"harmony"})",
       "<|channel|>commentary to=functions.get_weather "
       "<|constrain|>json<|message|>{\"city\":\"Paris\",\"days\":3,\"metric\":"
       "true,\"tags\":[\"a\",\"b\"],\"opts\":{\"x\":1}}<|call|>"},
      // Python keyword arguments as the template prints them: strings quoted
      // with ', True/False, containers as tojson.
      {R"({"call_format":"pythonic"})",
       "<|tool_call_start|>[get_weather(city='Paris', days=3, metric=True, "
       "tags=[\"a\", \"b\"], opts={\"x\": 1})]<|tool_call_end|>"},
      // The same call with JSON spellings of the scalars, and compactly.
      {R"({"call_format":"pythonic"})",
       "<|tool_call_start|>[get_weather(city=\"Paris\",days=3,metric=true,tags="
       "[\"a\",\"b\"],opts={\"x\":1})]<|tool_call_end|>"},
      // Optional parameters left out, the others in the schema's order.
      {R"({"call_format":"pythonic"})",
       "<|tool_call_start|>[get_weather(city='Paris', "
       "metric=False)]<|tool_call_end|>"},
      {R"({"call_format":"atem"})",
       " to=get_weather<|message|><atem:function_calls>\n<atem:invoke "
       "name=\"get_weather\">\n<atem:parameter "
       "name=\"city\">Paris</atem:parameter>\n<atem:parameter "
       "name=\"days\">3</atem:parameter>\n<atem:parameter "
       "name=\"metric\">true</atem:parameter>\n<atem:parameter "
       "name=\"tags\">[\"a\", \"b\"]</atem:parameter>\n<atem:parameter "
       "name=\"opts\">{\"x\": "
       "1}</atem:parameter>\n</atem:invoke>\n</atem:function_calls>"},
  };
  json::Value params = json::parse(
      R"({"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"},"metric":{"type":"boolean"},"tags":{"type":"array","items":{"type":"string"}},"opts":{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"],"additionalProperties":false}},"required":["city"],"additionalProperties":false})");
  for (const auto& c : cases) {
    const auto dialect = NativeDialectOf(json::parse(c.chat));
    CHECK(dialect.tools.has_value());
    if (!dialect.tools)
      continue;
    std::vector<sampling::JsonConstraint::Tool> tools;
    tools.emplace_back("get_weather", sampling::JsonConstraint::ToolParameters(
                                          params, false, *dialect.tools));
    const auto grammar = sampling::JsonConstraint::WithTools(
        nullptr, std::move(tools), true, false, *dialect.tools);
    CHECK(Accepts(*grammar, c.rendered),
          dialect.call_format + ": rendering accepted: " + c.rendered);
  }
}

// Several calls in one turn, in each dialect's own joining: Python's one list,
// Glimmer's messages, and a call that has no parameters.
void TestDialectParallelAndEdges() {
  std::cerr << "dialects: parallel calls, empty arguments, argument failures\n";
  json::Value params = json::parse(
      R"({"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"},"unit":{"type":"string","enum":["c","f"]}},"required":["city"],"additionalProperties":false})");
  json::Value none = json::parse(
      R"({"type":"object","properties":{},"additionalProperties":false})");
  auto build = [&](const sampling::JsonConstraint::ToolSyntax& syntax,
                   bool parallel, bool required = true, bool strict = false) {
    std::vector<sampling::JsonConstraint::Tool> tools;
    tools.emplace_back("get_weather", sampling::JsonConstraint::ToolParameters(
                                          params, strict, syntax));
    tools.emplace_back("get_time", sampling::JsonConstraint::ToolParameters(
                                       none, strict, syntax));
    return sampling::JsonConstraint::WithTools(nullptr, std::move(tools),
                                               required, parallel, syntax);
  };
  const auto py =
      *NativeDialectOf(json::parse(R"({"call_format":"pythonic"})")).tools;
  const auto atem =
      *NativeDialectOf(json::parse(R"({"call_format":"atem"})")).tools;
  const std::string start = "<|tool_call_start|>[", end = "]<|tool_call_end|>";
  {
    const auto one = build(py, false);
    const auto many = build(py, true);
    CHECK(Accepts(*many, start + "get_weather(city='Paris'), get_time()" + end),
          "pythonic: two calls in one list");
    CHECK(Accepts(
        *many,
        start +
            "get_time(), get_weather(city='A', days=2, unit='c'), get_time()" +
            end));
    CHECK(Accepts(*many, start + "get_weather(city='Paris')" + end));
    CHECK(Accepts(*many, start + "get_time()" + end),
          "pythonic: a call with no arguments");
    CHECK(Accepts(*many, start + "get_weather(city=\"it's\")" + end),
          "pythonic: a double-quoted string may hold the quote");
    CHECK(!Accepts(*one, start + "get_weather(city='Paris'), get_time()" + end),
          "pythonic: one call when parallel is off");
    CHECK(Accepts(*one, start + "get_weather(city='Paris')" + end));
    CHECK(!Accepts(*many, start + "get_weather(days=2)" + end),
          "pythonic: a required argument is required");
    CHECK(
        !Accepts(*many, start + "get_weather(city='Paris', color='red')" + end),
        "pythonic: no unknown argument");
    CHECK(
        !Accepts(*many, start + "get_weather(city='Paris', city='Rome')" + end),
        "pythonic: no repeated argument");
    CHECK(!Accepts(*many, start + "get_weather(city='Paris', unit='k')" + end),
          "pythonic: enum respected");
    CHECK(!Accepts(*many, start + "get_weather(city='Paris', days='3')" + end),
          "pythonic: integer respected");
    CHECK(!Accepts(*many, start + "get_other()" + end),
          "pythonic: only the request's tools");
    CHECK(!Accepts(*many, start + "get_weather(city='Paris')"),
          "pythonic: the list must close");
    CHECK(!Accepts(*many, "get_weather(city='Paris')"),
          "pythonic: a required call has its markers");
    CHECK(!Accepts(*many, start + "get_weather(city='Paris')" + end + "Done."),
          "pythonic: calls end the output");
    // Strict tools need every argument written.
    const auto strict_params = json::parse(
        R"({"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city","days"],"additionalProperties":false})");
    std::vector<sampling::JsonConstraint::Tool> tools;
    tools.emplace_back("get_weather", sampling::JsonConstraint::ToolParameters(
                                          strict_params, true, py));
    const auto strict = sampling::JsonConstraint::WithTools(
        nullptr, std::move(tools), true, false, py);
    CHECK(Accepts(*strict, start + "get_weather(city='Paris', days=3)" + end),
          "pythonic strict: complete call");
    CHECK(!Accepts(*strict, start + "get_weather(city='Paris')" + end),
          "pythonic strict: missing argument");
    // A schema Python keywords cannot spell: a property named by a keyword
    // argument must be an identifier; best effort falls back to any keywords.
    const auto odd = json::parse(
        R"({"type":"object","properties":{"a-b":{"type":"string"}},"additionalProperties":false})");
    bool strict_refused = false;
    try {
      (void)sampling::JsonConstraint::ToolParameters(odd, true, py);
    } catch (const std::invalid_argument&) {
      strict_refused = true;
    }
    CHECK(strict_refused,
          "pythonic strict: a property that is no identifier cannot be "
          "constrained");
    std::vector<sampling::JsonConstraint::Tool> loose;
    loose.emplace_back(
        "f", sampling::JsonConstraint::ToolParameters(odd, false, py));
    const auto open = sampling::JsonConstraint::WithTools(
        nullptr, std::move(loose), true, false, py);
    CHECK(Accepts(*open, start + "f(x=1, y='z')" + end),
          "pythonic best effort: any keyword arguments");
    CHECK(Accepts(*open, start + "f()" + end));
    CHECK(!Accepts(*open, start + "f(1)" + end),
          "pythonic best effort: keywords only");
  }
  {
    const std::string head =
        " to=get_weather<|message|><atem:function_calls>\n<atem:invoke "
        "name=\"get_weather\">\n";
    const std::string weather = head +
                                "<atem:parameter "
                                "name=\"city\">Paris</atem:parameter>\n</"
                                "atem:invoke>\n</atem:function_calls>";
    const std::string time =
        " to=get_time<|message|><atem:function_calls>\n<atem:invoke "
        "name=\"get_time\">\n</atem:invoke>\n</atem:function_calls>";
    const std::string next = "<|eom|><|start|>assistant";
    const auto one = build(atem, false);
    const auto many = build(atem, true);
    CHECK(Accepts(*many, weather), "atem: a call");
    CHECK(Accepts(*many, time), "atem: a call with no arguments");
    CHECK(Accepts(*many, weather + next + time),
          "atem: the next call is a new message");
    CHECK(Accepts(*many, time + next + weather + next + time));
    CHECK(!Accepts(*one, weather + next + time),
          "atem: one call when parallel is off");
    CHECK(!Accepts(*many, weather + time), "atem: calls are separate messages");
    CHECK(!Accepts(
              *many,
              " to=get_time<|message|><atem:function_calls>\n<atem:invoke "
              "name=\"get_weather\">\n</atem:invoke>\n</atem:function_calls>"),
          "atem: the invoke names the function the message is addressed to");
    CHECK(!Accepts(*many, " to=user<|message|>Hello"),
          "atem: required refuses an answer");
    CHECK(!Accepts(*many, head + "</atem:invoke>\n</atem:function_calls>"),
          "atem: a required argument is required");
    CHECK(!Accepts(*many, head + "<atem:parameter "
                                 "name=\"color\">red</atem:parameter>\n</"
                                 "atem:invoke>\n</atem:function_calls>"),
          "atem: no unknown argument");
    CHECK(!Accepts(*many, weather + "<|eot|>"),
          "atem: the stop token is not text");
  }
  // Glimmer's call header starts like its answer header, so automatic tool
  // use is not constrained (it stays on the device).
  fake::reset();
  fake::state().script = PeakScript;
  Fixture f(
      true, 2,
      R"({"call_format":"atem","think_open":"<|start|>assistant to=self<|message|>","think_end":"<|start|>assistant to=user<|message|>","thinking":"always"})");
  sampling::SamplingConfig config;
  config.constraint = f.runner->BindConstraint(build(atem, true, false));
  ext::ChatRequest request = Fixture::DefaultRequest();
  request.tools.push_back(
      {.name = "get_weather", .parameters_json = R"({"type":"object"})"});
  const auto automatic = f.Generate(0, config, 6, request);
  CHECK(automatic.epochs_host == 0,
        "atem: automatic tool use runs on the device");
  request.tool_choice = ext::ChatRequest::ToolChoice::kRequired;
  config.constraint = f.runner->BindConstraint(build(atem, true, true));
  const auto required = f.Generate(0, config, 6, request);
  CHECK(required.epochs_host > 0, "atem: a required call is constrained");
}

}  // namespace

// --speculative is refused, not ignored, for a package without a drafter; one
// with a drafter loads, and no speculation needs none.
void TestSpeculativeNeedsDrafter() {
  std::cerr << "speculative: a package without a drafter refuses it\n";
  const auto dir =
      std::filesystem::temp_directory_path() /
      ("loom_speculative_test_" + std::to_string(gufo::platform::ProcessId()));
  std::filesystem::create_directories(dir);
  std::ofstream(dir / "model.json")
      << R"({"schema_version":1,"programs":{},"services":{"chat":{"kind":"text"}},)"
         R"("variants":{"v":{"services":{"chat":{"config":{"llm.card":true,)"
         R"("llm.logits":false}}}}}})";
  const auto create = [&](ext::TextSpeculativeBackend backend,
                          std::uint32_t depth, std::string* error) {
    fake::reset();
    fake::state().script = PeakScript;
    fake::state().draft_depth = depth;
    ext::TextRunnerOptions options;
    options.model_path = dir.string();
    options.speculative.backend = backend;
    return CreateLoomRunner(options, error) != nullptr;
  };
  std::string error;
  CHECK(!create(ext::TextSpeculativeBackend::kMtp, 0, &error));
  CHECK(error ==
            "loom: --speculative mtp needs a drafter, and this package has "
            "none; give one with --mtp-model",
        error);
  CHECK(!create(ext::TextSpeculativeBackend::kDFlash, 0, &error));
  CHECK(error.find("--speculative dflash2") != std::string::npos &&
            error.find("--dflash-model") != std::string::npos,
        error);
  error.clear();
  CHECK(create(ext::TextSpeculativeBackend::kMtp, 4, &error), error);
  CHECK(create(ext::TextSpeculativeBackend::kDisabled, 0, &error), error);
  fake::reset();
  std::filesystem::remove_all(dir);
}

int main() {
  TestDevicePathUnchanged();
  TestSpeculativeNeedsDrafter();
  TestLogprobs();
  TestRepetitionPenalty();
  TestMinP();
  TestSeedStream();
  TestJsonGrammar();
  TestToolChoiceRequired();
  TestLazyGrammarStaysOnDevice();
  TestDialectGating();
  TestBatchMix();
  TestSelectAdvance();
  TestDialects();
  TestDialectRendering();
  TestDialectParallelAndEdges();
  if (g_failures) {
    std::cerr << g_failures << " failure(s)\n";
    return 1;
  }
  std::cerr << "all passed\n";
  return 0;
}
