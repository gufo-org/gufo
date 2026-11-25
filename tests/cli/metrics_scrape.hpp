#ifndef GUFO_TESTS_CLI_METRICS_SCRAPE_HPP_
#define GUFO_TESTS_CLI_METRICS_SCRAPE_HPP_

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

#include "src/cli/serve/http_server.hpp"
#include "src/core/platform/net.hpp"
#include "tests/cli/http_server_test_support.hpp"

namespace gufo::server::testing {

inline std::string Get(const HttpServer& server, std::string_view path) {
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(server.port());
  assert(::inet_pton(AF_INET, "127.0.0.1", &address.sin_addr) == 1);
  const net::Socket fd = net::CreateStreamSocket(AF_INET);
  assert(fd != net::kInvalidSocket);
  assert(net::SetSocketTimeouts(fd, 3'000));
  assert(::connect(fd, reinterpret_cast<const sockaddr*>(&address),
                   sizeof(address)) == 0);
  const std::string request = "GET " + std::string(path) + " HTTP/1.1\r\n\r\n";
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

inline std::string Scrape(const HttpServer& server,
                          std::string_view path = "/metrics") {
  const auto response = Get(server, path);
  ExpectStatus(response, 200);
  const auto split = response.find("\r\n\r\n");
  assert(response.find("Content-Type: text/plain; version=0.0.4\r\n") < split);
  return response.substr(split + 4);
}

inline std::string Sample(const std::string& body, std::string_view series) {
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

inline void Expect(const std::string& body, std::string_view series,
                   std::string_view value) {
  if (Sample(body, series) != value) {
    std::cerr << series << " is " << Sample(body, series) << ", want " << value
              << '\n';
    std::abort();
  }
}

}  // namespace gufo::server::testing

#endif  // GUFO_TESTS_CLI_METRICS_SCRAPE_HPP_
