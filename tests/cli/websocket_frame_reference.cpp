// A frozen copy of WebSocket::ReadLoop in 0.33.0
// (src/cli/serve/websocket.cpp), which websocket_frame_test compares with the
// Rust frame reader. Only the socket is replaced: Read takes bytes from a
// stream, Send and Close record the frames they would send, and the 4 MiB
// message limit is a parameter. Do not change ReadLoop to match new behavior.
#include "tests/cli/websocket_frame_reference.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <exception>

#include "src/core/utf8.hpp"

namespace gufo::websocket_reference {
namespace {
constexpr std::size_t kMaximumQueue = 8U << 20;

bool ValidUtf8(std::string_view text) {
  core::Utf8Decoder decoder;
  return decoder.Push(text, true) == text;
}
}  // namespace

std::string Action(std::string_view name, unsigned opcode,
                   std::string_view bytes) {
  std::string action(name);
  action += ' ' + std::to_string(opcode) + ' ';
  constexpr char kHex[] = "0123456789abcdef";
  for (const char c : bytes) {
    action += kHex[static_cast<unsigned char>(c) >> 4];
    action += kHex[static_cast<unsigned char>(c) & 15];
  }
  return action;
}

std::vector<std::string> ReferenceSocket::Run() {
  ReadLoop();
  return actions_;
}

bool ReferenceSocket::Read(char* data, std::size_t size) {
  while (size != 0 && !closed_.load()) {
    if (offset_ == stream_.size())
      return false;
    const auto count = std::min(size, stream_.size() - offset_);
    std::memcpy(data, stream_.data() + offset_, count);
    offset_ += count;
    data += count;
    size -= count;
  }
  return size == 0;
}

bool ReferenceSocket::Send(std::uint8_t opcode, std::string_view bytes) {
  if (closed_.load() && opcode != 8)
    return false;
  actions_.push_back(Action("send", opcode, bytes));
  return true;
}

void ReferenceSocket::Close(std::uint16_t code) {
  if (!MarkClosed())
    return;
  const std::array<char, 2> payload{static_cast<char>(code >> 8),
                                    static_cast<char>(code & 255)};
  (void)Send(8, std::string_view(payload.data(), payload.size()));
}

bool ReferenceSocket::MarkClosed() {
  return !closed_.exchange(true);
}

void ReferenceSocket::ReadLoop() {
  try {
    std::string message;
    bool fragmented = false;
    while (!closed_.load()) {
      std::array<unsigned char, 2> header{};
      if (!Read(reinterpret_cast<char*>(header.data()), 2))
        break;
      const bool final = (header[0] & 128) != 0;
      const auto opcode = header[0] & 15;
      const bool control = opcode >= 8;
      if ((header[0] & 112) != 0 || (header[1] & 128) == 0 ||
          (opcode != 0 && opcode != 1 && opcode != 8 && opcode != 9 &&
           opcode != 10) ||
          (control && !final)) {
        Close(1002);
        return;
      }
      std::uint64_t length = header[1] & 127;
      if (length >= 126) {
        if (control) {
          Close(1002);
          return;
        }
        const unsigned count = length == 126 ? 2 : 8;
        std::array<unsigned char, 8> extended{};
        if (!Read(reinterpret_cast<char*>(extended.data()), count))
          break;
        length = 0;
        for (unsigned i = 0; i < count; ++i)
          length = (length << 8) | extended[i];
        if ((count == 8 && (extended[0] & 128) != 0) ||
            (count == 2 && length < 126) || (count == 8 && length <= 65535)) {
          Close(1002);
          return;
        }
      }
      if (length > max_message_ ||
          (!control && length > max_message_ - message.size())) {
        Close(1009);
        return;
      }
      std::array<unsigned char, 4> mask{};
      if (!Read(reinterpret_cast<char*>(mask.data()), mask.size()))
        break;
      std::string payload(static_cast<std::size_t>(length), '\0');
      if (!Read(payload.data(), payload.size()))
        break;
      for (std::size_t i = 0; i < payload.size(); ++i)
        payload[i] = static_cast<char>(static_cast<unsigned char>(payload[i]) ^
                                       mask[i % 4]);
      if (opcode == 8) {
        if (payload.size() == 1) {
          Close(1002);
          return;
        }
        if (payload.size() >= 2) {
          const unsigned code = (static_cast<unsigned char>(payload[0]) << 8) |
                                static_cast<unsigned char>(payload[1]);
          if (code < 1000 || code >= 5000 || code == 1004 || code == 1005 ||
              code == 1006 || code == 1015 || (code >= 1016 && code < 3000)) {
            Close(1002);
            return;
          }
          if (!ValidUtf8(std::string_view(payload).substr(2))) {
            Close(1007);
            return;
          }
        }
        if (MarkClosed())
          (void)Send(8, payload);
        break;
      }
      if (opcode == 9) {
        if (!Send(10, payload))
          break;
        continue;
      }
      if (opcode == 10)
        continue;
      if ((opcode == 0) != fragmented) {
        Close(1002);
        return;
      }
      message.append(payload);
      fragmented = !final;
      if (!final)
        continue;
      if (!ValidUtf8(message)) {
        Close(1007);
        return;
      }
      {
        std::unique_lock lock(queue_mutex_);
        if (messages_.size() >= 128 ||
            message.size() > kMaximumQueue - queued_bytes_) {
          lock.unlock();
          Close(1009);
          return;
        }
        queued_bytes_ += message.size();
        messages_.push_back(std::move(message));
      }
      message.clear();
      ready_.notify_one();
    }
  } catch (const std::exception&) {
    Close(1011);
  }
  MarkClosed();
}

}  // namespace gufo::websocket_reference
