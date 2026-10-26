// Process signals must never terminate the test runner itself; this suite
// forks itself to prove graceful shutdown, so it stays POSIX-only.
#include <sys/wait.h>
#include <unistd.h>

#include <cassert>
#include <csignal>
#include <iostream>

#include "tests/cli/http_server_test_support.hpp"

namespace {

void TestSignalShutdown() {
  // Process signals must never terminate the test runner itself. Prove that
  // both idle listeners and active generation return through normal cleanup.
  for (const int signal : {SIGINT, SIGTERM}) {
    for (const bool active : {false, true}) {
      const pid_t child = ::fork();
      assert(child >= 0);
      if (child == 0) {
        ::alarm(5);
        {
          RunningServer server({}, true);
          // Accepting a request proves run() installed its handlers.
          ExpectStatus(server.Post("/echo", "ready"), 200);
          int fd = -1;
          if (active) {
            server.backend->wait_for_disconnect = true;
            fd = server.Connect();
            const std::string body = R"({"prompt":"hello"})";
            const std::string request =
                "POST /v1/completions HTTP/1.1\r\nContent-Length: " +
                std::to_string(body.size()) + "\r\n\r\n" + body;
            assert(net::SendNoSignal(fd, request.data(), request.size()) ==
                   static_cast<std::intptr_t>(request.size()));
            assert(server.backend->entered.try_acquire_for(
                std::chrono::seconds(2)));
          }
          assert(::kill(::getpid(), signal) == 0);
          assert(server.run_finished.try_acquire_for(std::chrono::seconds(2)));
          if (active) {
            assert(server.backend->disconnected);
            net::CloseSocket(fd);
          }
        }
        ::_exit(0);
      }
      int status = 0;
      assert(::waitpid(child, &status, 0) == child);
      assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    }
  }
}
}  // namespace

int main() {
  TestSignalShutdown();
  std::cout << "HTTP signal shutdown checks passed.\n";
  return 0;
}
