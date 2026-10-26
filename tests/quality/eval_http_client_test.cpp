
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <semaphore>
#include <string>
#include <string_view>
#include <thread>

#include "src/core/platform/net.hpp"
#include "src/eval/http_client.hpp"

namespace {

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

void CheckTransfer(bool post, std::string_view response, bool stalls) {
  const int listener = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
  Expect(listener >= 0, "create fixture socket");
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  Expect(bind(listener, reinterpret_cast<sockaddr*>(&address),
              sizeof(address)) == 0 &&
             listen(listener, 1) == 0,
         "listen on loopback");
  socklen_t size = sizeof(address);
  Expect(
      getsockname(listener, reinterpret_cast<sockaddr*>(&address), &size) == 0,
      "read fixture port");
  std::binary_semaphore release(0);
  std::jthread server([&] {
    pollfd pending{listener, POLLIN, 0};
    Expect(poll(&pending, 1, 2000) == 1, "client connects");
    const int peer = accept4(listener, nullptr, nullptr, SOCK_CLOEXEC);
    Expect(peer >= 0, "accept client");
    char request[4096];
    pending = {peer, POLLIN, 0};
    Expect(poll(&pending, 1, 2000) == 1 &&
               recv(peer, request, sizeof(request), 0) > 0,
           "client sends request");
    if (!response.empty())
      Expect(send(peer, response.data(), response.size(), MSG_NOSIGNAL) ==
                 static_cast<ssize_t>(response.size()),
             "send fixture response");
    // Also bounds this test if the timeout regresses.
    if (stalls)
      (void)release.try_acquire_for(std::chrono::seconds(2));
    close(peer);
  });
  const gufo::eval::HttpClient client(
      "http://127.0.0.1:" + std::to_string(ntohs(address.sin_port)), "",
      std::chrono::milliseconds(100));
  const auto result =
      post ? client.PostJson("/completion", "{}") : client.Get("/models");
  release.release();
  server.join();
  close(listener);
  if (stalls) {
    Expect(!result.transport_ok && result.transport_code == "request_timeout",
           "stalled request reports a bounded transport timeout");
    Expect(result.elapsed_ms >= 50 && result.elapsed_ms < 1500,
           "deadline applies after connection, including partial response");
  } else {
    Expect(
        result.transport_ok && result.status_code == 200 && result.body == "{}",
        "complete response remains successful");
  }
}

}  // namespace

int main() {
  for (const bool post : {false, true}) {
    CheckTransfer(post, "", true);
    CheckTransfer(post, "HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n{",
                  true);
    CheckTransfer(post, "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}",
                  false);
  }
  std::cout << "Eval request deadlines passed\n";
}
