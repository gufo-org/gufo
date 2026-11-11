#ifndef GUFO_TESTS_CLI_WEBSOCKET_FRAME_REFERENCE_HPP_
#define GUFO_TESTS_CLI_WEBSOCKET_FRAME_REFERENCE_HPP_

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace gufo::websocket_reference {

// "send <opcode> <hex payload>", or "queue 1 <hex message>" for a message
// handed to the consumer.
std::string Action(std::string_view name, unsigned opcode,
                   std::string_view bytes);

// WebSocket's reading side in 0.33.0 over a byte stream.
class ReferenceSocket {
public:
  ReferenceSocket(std::string_view stream, std::size_t max_message)
      : stream_(stream), max_message_(max_message) {
    messages_.socket = this;
  }
  // Runs the frozen ReadLoop over the stream and returns what it sent and
  // queued, in order.
  std::vector<std::string> Run();

private:
  // Records every message pushed; nothing consumes it.
  struct Queue : std::deque<std::string> {
    ReferenceSocket* socket = nullptr;
    void push_back(std::string message) {
      socket->actions_.push_back(Action("queue", 1, message));
      std::deque<std::string>::push_back(std::move(message));
    }
  };

  bool Read(char* data, std::size_t size);
  bool Send(std::uint8_t opcode, std::string_view bytes);
  void Close(std::uint16_t code = 1000);
  bool MarkClosed();
  void ReadLoop();

  std::string_view stream_;
  std::size_t offset_ = 0;
  std::size_t max_message_;
  std::atomic<bool> closed_{false};
  std::vector<std::string> actions_;
  std::mutex queue_mutex_;
  std::condition_variable ready_;
  Queue messages_;
  std::size_t queued_bytes_ = 0;
};

}  // namespace gufo::websocket_reference

#endif  // GUFO_TESTS_CLI_WEBSOCKET_FRAME_REFERENCE_HPP_
