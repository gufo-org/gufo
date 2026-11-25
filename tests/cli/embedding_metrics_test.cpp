#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "src/cli/serve/embedding_service.hpp"
#include "tests/cli/http_server_test_support.hpp"

namespace {

using gufo::server::EmbeddingBatch;
using gufo::server::EmbeddingInput;
using gufo::server::EmbeddingPart;
using gufo::server::EmbeddingRequestOptions;
using gufo::server::EmbeddingService;
using gufo::server::EmbeddingServiceOptions;
using gufo::server::EmbeddingStatus;
using Type = EmbeddingPart::Type;

EmbeddingInput Input(std::vector<EmbeddingPart> parts) {
  return {.content = std::move(parts)};
}

enum class Shape { kGood, kEmptyVector, kNonFinite, kMissingVector };

class EmbeddingServer {
public:
  EmbeddingServer()
      : service(std::make_shared<EmbeddingService>(EmbeddingServiceOptions{
            .validate_model = false,
            .model_id = "embedding-test",
            .runner =
                [this](const std::vector<EmbeddingInput>& inputs,
                       const EmbeddingRequestOptions&, EmbeddingBatch* batch,
                       std::string* error) {
                  std::this_thread::sleep_for(delay.load());
                  if (throws.load())
                    throw std::runtime_error("injected throw");
                  if (status.load() != EmbeddingStatus::kOk) {
                    *error = "injected";
                    return status.load();
                  }
                  const Shape wanted = shape.load();
                  for (std::size_t index = 0; index < inputs.size(); ++index) {
                    if (wanted == Shape::kMissingVector && index != 0)
                      break;
                    batch->embeddings.push_back(
                        wanted == Shape::kEmptyVector
                            ? std::vector<float>{}
                            : std::vector<float>{
                                  wanted == Shape::kNonFinite
                                      ? std::numeric_limits<float>::quiet_NaN()
                                      : 1.0F});
                    batch->token_counts.push_back(10 + index);
                  }
                  late_cancel = cancel_after_run.load();
                  return EmbeddingStatus::kOk;
                },
        })),
        server("127.0.0.1", 0, nullptr, nullptr, nullptr, nullptr, {}, nullptr,
               service) {
    std::string error;
    const bool started = server.start(&error);
    assert(started);
    (void)started;
    worker = std::jthread([this] { server.run(false); });
  }
  ~EmbeddingServer() {
    server.stop();
    worker.join();
  }

  EmbeddingStatus Embed(std::vector<EmbeddingInput> inputs,
                        bool cancelled = false) {
    EmbeddingBatch batch;
    std::string error;
    return service->EmbedInputs(
        inputs, [this, cancelled] { return cancelled || late_cancel.load(); },
        &batch, &error);
  }

  std::string Get(std::string_view path) {
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(server.port());
    assert(::inet_pton(AF_INET, "127.0.0.1", &address.sin_addr) == 1);
    const net::Socket fd = net::CreateStreamSocket(AF_INET);
    assert(fd != net::kInvalidSocket);
    assert(net::SetSocketTimeouts(fd, 3'000));
    assert(::connect(fd, reinterpret_cast<const sockaddr*>(&address),
                     sizeof(address)) == 0);
    const std::string request =
        "GET " + std::string(path) + " HTTP/1.1\r\n\r\n";
    assert(net::SendNoSignal(fd, request.data(), request.size()) ==
           static_cast<std::intptr_t>(request.size()));
    std::string response;
    char buffer[4096];
    for (;;) {
      const auto count = net::Receive(fd, buffer, sizeof(buffer));
      assert(count >= 0);
      if (count == 0)
        break;
      response.append(buffer, static_cast<std::size_t>(count));
    }
    net::CloseSocket(fd);
    return response;
  }

  std::string Scrape(std::string_view path = "/metrics") {
    const auto response = Get(path);
    ExpectStatus(response, 200);
    const auto split = response.find("\r\n\r\n");
    assert(response.find("Content-Type: text/plain; version=0.0.4\r\n") <
           split);
    return response.substr(split + 4);
  }

  std::atomic<EmbeddingStatus> status{EmbeddingStatus::kOk};
  std::atomic<Shape> shape{Shape::kGood};
  std::atomic<bool> throws{false};
  std::atomic<bool> cancel_after_run{false};
  std::atomic<bool> late_cancel{false};
  std::atomic<std::chrono::milliseconds> delay{std::chrono::milliseconds(0)};
  std::shared_ptr<EmbeddingService> service;
  HttpServer server;
  std::jthread worker;
};

std::string Sample(const std::string& body, std::string_view series) {
  const std::string text = "\n" + body;
  const std::string prefix = "\n" + std::string(series) + " ";
  const auto start = text.find(prefix);
  if (start == std::string::npos) {
    std::cerr << "no sample " << series << " in\n" << body;
    std::abort();
  }
  const auto value = start + prefix.size();
  return text.substr(value, text.find('\n', value) - value);
}

void Expect(const std::string& body, std::string_view series,
            std::string_view value) {
  if (Sample(body, series) != value) {
    std::cerr << series << " is " << Sample(body, series) << ", want " << value
              << '\n';
    std::abort();
  }
}

std::uint64_t Bucket(const std::string& body, std::string_view bound) {
  return std::stoull(
      Sample(body, "gufo_embedding_request_seconds_bucket{le=\"" +
                       std::string(bound) + "\"}"));
}

}  // namespace

int main() {
  EmbeddingServer server;

  const auto fresh = server.Scrape();
  Expect(fresh, "gufo_embedding_requests_total{status=\"ok\"}", "0");
  Expect(fresh, "gufo_embedding_requests_in_flight", "0");
  Expect(fresh, "gufo_embedding_request_seconds_count", "0");
  Expect(fresh, "gufo_device_lost_total", "0");
  assert(fresh.find("llamacpp:") == std::string::npos);
  assert(server.Scrape("/v1/metrics") == fresh);

  assert(server.Embed({Input({{Type::kText, "a"}}),
                       Input({{Type::kText, "b"}})}) == EmbeddingStatus::kOk);
  assert(server.Embed({Input({{Type::kText, "caption"},
                              {Type::kImage, "x"}})}) == EmbeddingStatus::kOk);
  auto body = server.Scrape();
  Expect(body, "gufo_embedding_requests_total{status=\"ok\"}", "2");
  Expect(body, "gufo_embedding_inputs_total{modality=\"text\"}", "3");
  Expect(body, "gufo_embedding_inputs_total{modality=\"image\"}", "1");
  Expect(body, "gufo_embedding_inputs_total{modality=\"audio\"}", "0");
  Expect(body, "gufo_embedding_inputs_total{modality=\"video\"}", "0");
  Expect(body, "gufo_embedding_input_tokens_total", "31");
  Expect(body, "gufo_embedding_request_seconds_count", "2");
  Expect(body, "gufo_embedding_requests_in_flight", "0");

  assert(server.Embed({Input({{Type::kText, "a"}})}, true) ==
         EmbeddingStatus::kCancelled);
  server.status = EmbeddingStatus::kFailed;
  assert(server.Embed({Input({{Type::kText, "a"}})}) ==
         EmbeddingStatus::kFailed);
  body = server.Scrape();
  Expect(body, "gufo_embedding_requests_total{status=\"cancelled\"}", "1");
  Expect(body, "gufo_embedding_requests_total{status=\"error\"}", "1");
  Expect(body, "gufo_embedding_requests_total{status=\"ok\"}", "2");
  Expect(body, "gufo_embedding_input_tokens_total", "31");
  Expect(body, "gufo_embedding_request_seconds_count", "4");

  for (const auto status :
       {EmbeddingStatus::kInputTooLong, EmbeddingStatus::kInvalidInput,
        EmbeddingStatus::kInvalidAudioInput,
        EmbeddingStatus::kInvalidVideoInput}) {
    server.status = status;
    assert(server.Embed({Input({{Type::kText, "a"}})}) == status);
  }
  server.status = EmbeddingStatus::kOk;
  body = server.Scrape();
  Expect(body, "gufo_embedding_requests_total{status=\"invalid\"}", "4");
  Expect(body, "gufo_embedding_requests_total{status=\"error\"}", "1");
  Expect(body, "gufo_embedding_input_tokens_total", "31");

  for (const auto shape :
       {Shape::kEmptyVector, Shape::kNonFinite, Shape::kMissingVector}) {
    server.shape = shape;
    assert(server.Embed(
               {Input({{Type::kText, "a"}}), Input({{Type::kText, "b"}})}) ==
           EmbeddingStatus::kFailed);
  }
  server.shape = Shape::kGood;
  body = server.Scrape();
  Expect(body, "gufo_embedding_requests_total{status=\"error\"}", "4");
  Expect(body, "gufo_embedding_requests_total{status=\"ok\"}", "2");
  Expect(body, "gufo_embedding_input_tokens_total", "31");

  server.cancel_after_run = true;
  assert(server.Embed({Input({{Type::kText, "a"}})}) ==
         EmbeddingStatus::kCancelled);
  server.cancel_after_run = false;
  server.late_cancel = false;
  body = server.Scrape();
  Expect(body, "gufo_embedding_requests_total{status=\"cancelled\"}", "2");
  Expect(body, "gufo_embedding_requests_total{status=\"ok\"}", "2");
  Expect(body, "gufo_embedding_input_tokens_total", "31");

  server.throws = true;
  bool threw = false;
  try {
    (void)server.Embed({Input({{Type::kText, "a"}})});
  } catch (const std::runtime_error&) {
    threw = true;
  }
  server.throws = false;
  assert(threw);
  body = server.Scrape();
  Expect(body, "gufo_embedding_requests_total{status=\"error\"}", "5");
  Expect(body, "gufo_embedding_requests_in_flight", "0");

  const auto fast_before = Bucket(body, "0.01");
  const auto all_before = Bucket(body, "+Inf");
  server.delay = std::chrono::milliseconds(20);
  assert(server.Embed({Input({{Type::kText, "a"}})}) == EmbeddingStatus::kOk);
  server.delay = std::chrono::milliseconds(0);
  body = server.Scrape();
  assert(Bucket(body, "0.01") == fast_before);
  assert(Bucket(body, "30") == Bucket(body, "+Inf"));
  assert(Bucket(body, "+Inf") == all_before + 1);

  server.status = EmbeddingStatus::kDeviceLost;
  assert(server.Embed({Input({{Type::kText, "a"}})}) ==
         EmbeddingStatus::kDeviceLost);
  assert(server.Embed({Input({{Type::kText, "a"}})}) ==
         EmbeddingStatus::kDeviceLost);
  body = server.Scrape();
  Expect(body, "gufo_embedding_requests_total{status=\"device_lost\"}", "2");
  Expect(body, "gufo_device_lost_total", "1");
  Expect(body, "gufo_embedding_requests_in_flight", "0");
  assert(body.find("llamacpp:") == std::string::npos);
}
