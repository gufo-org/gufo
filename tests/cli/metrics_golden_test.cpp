#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "tests/cli/http_server_test_support.hpp"

namespace {

using gufo::server::TextGenerationBackend;
using Result = TextGenerationBackend::Result;

void SeedSchedulerCounters(std::uint64_t device_lost, std::int64_t processing,
                           std::int64_t deferred) {
  auto& metrics = gufo::server::Metrics();
  metrics.device_lost.Set(device_lost);
  metrics.requests_processing.Set(processing);
  metrics.requests_deferred.Set(deferred);
}

Result Finished(std::size_t prompt, std::size_t cached, std::size_t prefill,
                double prefill_ms, std::size_t completion, double decode_ms) {
  Result result;
  result.prompt_tokens = prompt;
  result.cached_prompt_tokens = cached;
  result.prefill_tokens = prefill;
  result.prefill_ms = prefill_ms;
  result.completion_tokens = completion;
  result.decode_ms = decode_ms;
  return result;
}

std::string Scrape(RunningServer& server) {
  const auto response = server.Send("GET /metrics HTTP/1.1\r\n\r\n");
  ExpectStatus(response, 200);
  const auto split = response.find("\r\n\r\n");
  assert(split != std::string::npos);
  assert(response.find("Content-Type: text/plain; version=0.0.4\r\n") < split);
  return response.substr(split + 4);
}

std::string Section(const std::string& name, const std::string& body) {
  return "=== " + name + " ===\n" + body;
}

}  // namespace

int main() {
  RunningServer server;
  std::string actual;
  const auto scrape = [&](const std::string& name) {
    actual += Section(name, Scrape(server));
  };

  scrape("fresh process, nothing recorded");

  gufo::server::RecordServerMetrics(
      Finished(1000, 400, 600, 100.0, 50, 2000.0));
  scrape("one request, 0.1 s of prefill");

  auto drafted = Finished(20, 0, 20, 4.0, 8, 4.0);
  drafted.draft_rounds = 3;
  drafted.draft_tokens = 12;
  drafted.draft_accepted_tokens = 9;
  drafted.lookup_tokens = 5;
  drafted.lookup_accepted_tokens = 2;
  gufo::server::RecordServerMetrics(drafted);
  scrape("speculative and lookup counters");

  gufo::server::RecordServerMetrics(Finished(1, 0, 1, 1e12, 1, 1e12));
  scrape("speeds of 1e-9 tokens per second");

  gufo::server::RecordServerMetrics(
      Finished(12345678, 0, 12345678, 1e7, 12345678, 1e7));
  scrape("speeds of 1234.5678 tokens per second");

  gufo::server::RecordServerMetrics(Finished(1, 0, 1, 1e4, 1, 1e4));
  scrape("speeds of 0.1 tokens per second");

  gufo::server::RecordServerMetrics(
      Finished(123456789012, 0, 123456789012, 1.0, 123456789012, 1.0));
  scrape("speeds of 1.23456789012e14 tokens per second");

  gufo::server::RecordServerMetrics(
      Finished(100000000000000, 0, 100000000000000, 1.0, 100000000000000, 1.0));
  scrape("speeds of 1e17 tokens per second");

  gufo::server::RecordServerMetrics(Finished(
      100000000000000000, 0, 100000000000000000, 1.0, 100000000000000000, 1.0));
  scrape("speeds of 1e20 tokens per second");

  gufo::server::RecordServerMetrics(Finished(0, 0, 0, 0.0, 0, 0.0));
  scrape("an empty request keeps the last speeds");

  SeedSchedulerCounters(7, 3, 2);
  scrape("scheduler counters");
  SeedSchedulerCounters(0, -3, 0);
  scrape("a negative request gauge");
  SeedSchedulerCounters(0, 0, 0);

  server.backend->sessions.resize(2);
  server.backend->sessions[1].prompt_tokens = 10;
  server.backend->sessions[1].generated_tokens = 3;
  scrape("kv ratio 13 over two sessions of 65536");
  server.backend->sessions.resize(3);
  server.backend->sessions[0].prompt_tokens = 1;
  server.backend->sessions[1] = {};
  server.backend->sessions[2] = {};
  scrape("kv ratio 1 over three sessions of 65536");
  server.backend->sessions.assign(
      1, TextGenerationBackend::SessionState{.prompt_tokens = 65536});
  scrape("kv ratio exactly 1");

  server.backend->shared = TextGenerationBackend::SharedState{
      .max_prompt_tokens = 131072,
      .shared_bytes = 8589934592,
      .charged_bytes = 4294967296,
      .reclaimable_bytes = 1048576,
      .waits = 4,
      .pauses = 2,
      .preemptions = 1,
  };
  scrape("shared state only");
  server.backend->prompt_budget = TextGenerationBackend::PromptAdmissionBudget{
      .budget_bytes = 4096,
      .used_bytes = 1024,
      .available_bytes = 3072,
  };
  scrape("shared state and prompt budget");
  server.backend->shared.reset();
  scrape("prompt budget only");
  server.backend->prompt_budget.reset();
  scrape("both absent again");

  const std::string path =
      std::string(GUFO_METRICS_FIXTURES) + "/text_server.golden";
  if (std::getenv("GUFO_METRICS_GOLDEN_WRITE") != nullptr) {
    std::ofstream(path, std::ios::binary) << actual;
    std::cout << "Wrote " << path << '\n';
    return 0;
  }
  std::ifstream in(path, std::ios::binary);
  assert(in);
  const std::string golden((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());
  if (actual != golden) {
    std::cerr << "/metrics differs from " << path << "\n--- actual ---\n"
              << actual;
    return 1;
  }
  std::cout << "Text server /metrics matches the golden.\n";
  return 0;
}
