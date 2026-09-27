// Measures the cost of the per-token instruction rank 0 sends rank 1.
//
// This is a MEASUREMENT, not a correctness test: nothing here asserts a
// threshold, and a slow result is a result, not a failure. The correctness
// properties of the same code live in `tp_control_test`.
//
// SCOPE, because the number is easy to over-read: the pair is a loopback TCP
// socket on ONE host. It therefore contains the framing cost, the syscall cost
// and the loopback path, but NOT the two hosts' InfiniBand round trip, and
// not a second host's scheduling delay. Treat the result as a LOWER BOUND on
// what the two-host path costs. The cross-host number has to come from a
// two-host run.
//
// Every decoded token costs one `kAdvance` instruction on the control channel,
// so the metric that matters is sustained instructions per second; the latency
// percentiles say where that ceiling comes from.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "src/cli/serve/tp_control.hpp"

namespace {

using gufo::server::TpControlChannel;
using gufo::server::TpControlCommand;
using gufo::server::TpControlCommandKind;
using gufo::server::TpControlConfig;
using gufo::server::TpInstructionOp;

std::uint16_t FreePort() {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = 0;
  ::bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address));
  socklen_t length = sizeof(address);
  ::getsockname(fd, reinterpret_cast<sockaddr*>(&address), &length);
  const auto port = ntohs(address.sin_port);
  ::close(fd);
  return port;
}

struct Samples {
  std::vector<double> micros;

  void Add(std::chrono::steady_clock::time_point start) {
    micros.push_back(std::chrono::duration<double, std::micro>(
                         std::chrono::steady_clock::now() - start)
                         .count());
  }
  /// Nearest-rank percentile, so a reported p99 is a measurement that occurred
  /// rather than an interpolation between two that did not.
  [[nodiscard]] double Percentile(double fraction) const {
    if (micros.empty()) {
      return 0.0;
    }
    auto sorted = micros;
    std::sort(sorted.begin(), sorted.end());
    const auto rank = static_cast<std::size_t>(
        std::ceil(fraction * static_cast<double>(sorted.size())));
    return sorted[std::min(sorted.size(), std::max<std::size_t>(rank, 1)) - 1];
  }
  [[nodiscard]] double Mean() const {
    if (micros.empty()) {
      return 0.0;
    }
    double total = 0.0;
    for (const double value : micros) {
      total += value;
    }
    return total / static_cast<double>(micros.size());
  }
};

/// One token's instruction, as rank 0 sends it.
bool SendAdvance(TpControlChannel& channel, std::uint64_t index,
                 std::int32_t token, std::string* error) {
  const TpControlCommand command{
      .sequence = 1,
      .kind = TpControlCommandKind::kInstruction,
      .instruction = {
          .op = TpInstructionOp::kAdvance, .index = index, .token = token}};
  return channel.SendCommand(command, error);
}

/// Receives one instruction and checks it is the advance that was sent.
bool ReceiveAdvance(TpControlChannel& channel, std::uint64_t index,
                    std::string* error) {
  TpControlCommand command;
  if (!channel.ReceiveCommand(&command, error)) {
    return false;
  }
  if (command.kind != TpControlCommandKind::kInstruction ||
      command.instruction.op != TpInstructionOp::kAdvance ||
      command.instruction.index != index) {
    *error = "unexpected instruction";
    return false;
  }
  return true;
}

void Report(const char* label, const Samples& samples) {
  std::printf(
      "  %-22s n=%zu  mean=%8.1f  p50=%8.1f  p99=%8.1f  max=%8.1f  (us)\n",
      label, samples.micros.size(), samples.Mean(), samples.Percentile(0.50),
      samples.Percentile(0.99),
      samples.micros.empty()
          ? 0.0
          : *std::max_element(samples.micros.begin(), samples.micros.end()));
}

}  // namespace

int main(int argc, char** argv) {
  // Enough steps that the tail is measured rather than guessed, few enough
  // that this stays a test-length run.
  const std::size_t steps =
      argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 20000;
  if (steps == 0) {
    std::fprintf(stderr, "step count must be positive\n");
    return 2;
  }

  const auto port = FreePort();
  std::string server_error;
  std::string client_error;
  std::shared_ptr<TpControlChannel> client;
  std::thread connector([&] {
    client = TpControlChannel::Connect("127.0.0.1", port, &client_error);
  });
  auto server = TpControlChannel::Listen(port, &server_error);
  connector.join();
  if (server == nullptr || client == nullptr) {
    std::fprintf(stderr, "pair failed: %s / %s\n", server_error.c_str(),
                 client_error.c_str());
    return 1;
  }

  const TpControlConfig rank0{.rank = 0,
                              .world_size = 2,
                              .max_context = 4096,
                              .max_draft_tokens = 7,
                              .use_mtp = false,
                              .auth_token = "latency",
                              .prefill_chunk_tokens = 512};
  TpControlConfig rank1 = rank0;
  rank1.rank = 1;
  bool server_ok = false;
  bool client_ok = false;
  std::thread server_thread(
      [&] { server_ok = server->Handshake(rank0, &server_error); });
  std::thread client_thread(
      [&] { client_ok = client->Handshake(rank1, &client_error); });
  server_thread.join();
  client_thread.join();
  if (!server_ok || !client_ok) {
    std::fprintf(stderr, "handshake failed: %s / %s\n", server_error.c_str(),
                 client_error.c_str());
    return 1;
  }

  // Warm the path so the first steps do not pay for first-touch page faults and
  // scheduler placement, which are not what the per-token ceiling is made of.
  std::uint64_t index = 0;
  for (std::size_t i = 0; i < 256; ++i, ++index) {
    std::string error;
    std::thread send([&] {
      (void)SendAdvance(*server, index, static_cast<std::int32_t>(i), &error);
    });
    (void)ReceiveAdvance(*client, index, &error);
    send.join();
  }

  Samples publish_samples;
  Samples consume_samples;
  std::atomic<bool> failed{false};
  std::string failure;

  const auto wall_start = std::chrono::steady_clock::now();
  const std::uint64_t first = index;
  std::thread publish_thread([&] {
    for (std::size_t i = 0; i < steps; ++i) {
      const auto start = std::chrono::steady_clock::now();
      std::string error;
      if (!SendAdvance(*server, first + i, static_cast<std::int32_t>(i),
                       &error)) {
        failure = "send: " + error;
        failed.store(true);
        return;
      }
      publish_samples.Add(start);
    }
  });
  std::thread consume_thread([&] {
    for (std::size_t i = 0; i < steps; ++i) {
      const auto start = std::chrono::steady_clock::now();
      std::string error;
      if (!ReceiveAdvance(*client, first + i, &error)) {
        failure = "receive: " + error;
        failed.store(true);
        return;
      }
      consume_samples.Add(start);
    }
  });
  publish_thread.join();
  consume_thread.join();
  const auto wall = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - wall_start);

  if (failed.load()) {
    std::fprintf(stderr, "FAIL: %s\n", failure.c_str());
    return 1;
  }

  index = first + steps;
  std::printf(
      "TP instruction exchange, %zu instructions, loopback TCP on one "
      "host\n",
      steps);
  std::printf("  LOWER BOUND: excludes the cross-host InfiniBand round trip\n");
  Report("send", publish_samples);
  Report("receive (and check)", consume_samples);
  std::printf("  %-22s %.0f instructions/s (%.3f s wall)\n", "sustained",
              static_cast<double>(steps) / wall.count(), wall.count());

  // The pipelined phase above measures the THROUGHPUT ceiling, not latency:
  // the sender runs ahead and the receiver only drains the socket buffer, so
  // its per-call cost hides how long a message actually takes to arrive. That
  // latency is what rank 1 waits on once per token in the decode loop, so
  // measure it properly by serializing: send, wait for the receiver to
  // confirm, repeat. Half the round trip approximates the one-way delay.
  Samples round_trip;
  std::atomic<std::size_t> consumed_count{0};
  const std::size_t rt_steps = std::min<std::size_t>(steps, 5000);
  const std::uint64_t rt_first = index;
  std::thread rt_consumer_thread([&] {
    for (std::size_t i = 0; i < rt_steps; ++i) {
      std::string error;
      if (!ReceiveAdvance(*client, rt_first + i, &error)) {
        failure = "round-trip receive: " + error;
        failed.store(true);
        return;
      }
      consumed_count.store(i + 1, std::memory_order_release);
    }
  });
  for (std::size_t i = 0; i < rt_steps && !failed.load(); ++i) {
    const auto start = std::chrono::steady_clock::now();
    std::string error;
    if (!SendAdvance(*server, rt_first + i, static_cast<std::int32_t>(i),
                     &error)) {
      failure = "round-trip send: " + error;
      failed.store(true);
      break;
    }
    while (consumed_count.load(std::memory_order_acquire) < i + 1) {
      std::this_thread::yield();
    }
    round_trip.Add(start);
  }
  rt_consumer_thread.join();
  if (failed.load()) {
    std::fprintf(stderr, "FAIL: %s\n", failure.c_str());
    return 1;
  }
  Report("serialized round trip", round_trip);
  std::printf("  %-22s %.1f us\n", "one-way (rtt/2)",
              round_trip.Percentile(0.50) / 2.0);

  std::puts("PASS: measurement completed");
  return 0;
}
