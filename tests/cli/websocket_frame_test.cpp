#include <gufo/websocket.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::string Action(std::string_view name, unsigned opcode,
                   std::string_view bytes) {
  constexpr char digits[] = "0123456789abcdef";
  std::string out = std::string(name) + " " + std::to_string(opcode) + " ";
  for (const unsigned char byte : bytes) {
    out += digits[byte >> 4];
    out += digits[byte & 15];
  }
  return out;
}

// src/cli/serve/websocket.cpp
constexpr std::size_t kMaximumMessage = 4U << 20;
constexpr std::size_t kMaximumQueue = 8U << 20;

struct Source {
  std::string_view bytes;
  std::size_t offset = 0;
  const bool* closed = nullptr;
};

// WebSocket::Read over a stream: nothing once closed, false when it ends.
int Read(void* opaque, std::uint8_t* data, std::size_t size) {
  auto& source = *static_cast<Source*>(opaque);
  while (size != 0 && !*source.closed) {
    if (source.offset == source.bytes.size())
      return 0;
    const auto count = std::min(size, source.bytes.size() - source.offset);
    std::memcpy(data, source.bytes.data() + source.offset, count);
    source.offset += count;
    data += count;
    size -= count;
  }
  return size == 0 ? 1 : 0;
}

// WebSocket::ReadLoop's handling of the reader's events, with Send, Close and
// the queue recording what they would do, as the reference does.
std::vector<std::string> NewActions(std::string_view stream,
                                    std::size_t max_message) {
  std::vector<std::string> actions;
  bool closed = false;
  const auto mark_closed = [&] {
    const bool changed = !closed;
    closed = true;
    return changed;
  };
  const auto send = [&](std::uint8_t opcode, std::string_view bytes) {
    if (closed && opcode != 8)
      return false;
    actions.push_back(Action("send", opcode, bytes));
    return true;
  };
  const auto close = [&](std::uint16_t code) {
    if (!mark_closed())
      return;
    const std::array<char, 2> payload{static_cast<char>(code >> 8),
                                      static_cast<char>(code & 255)};
    (void)send(8, std::string_view(payload.data(), payload.size()));
  };
  Source source{stream, 0, &closed};
  const std::unique_ptr<GufoWsReader, decltype(&gufo_ws_reader_free)> reader(
      gufo_ws_reader_new(max_message), gufo_ws_reader_free);
  if (!reader)
    throw std::runtime_error("gufo_ws_reader_new failed");
  std::size_t queued = 0;
  std::size_t queued_bytes = 0;
  while (!closed) {
    GufoWsEvent event{};
    gufo_ws_reader_next(reader.get(), Read, &source, &event);
    const std::string_view payload(reinterpret_cast<const char*>(event.data),
                                   event.len);
    if (event.kind == GUFO_WS_END)
      break;
    if (event.kind == GUFO_WS_ERROR) {
      close(event.code);
      return actions;
    }
    if (event.kind == GUFO_WS_CLOSE) {
      if (mark_closed())
        (void)send(8, payload);
      break;
    }
    if (event.kind == GUFO_WS_PING) {
      if (!send(10, payload))
        break;
      continue;
    }
    if (event.kind != GUFO_WS_MESSAGE)
      throw std::runtime_error("internal reader event");
    if (queued >= 128 || payload.size() > kMaximumQueue - queued_bytes) {
      close(1009);
      return actions;
    }
    ++queued;
    queued_bytes += payload.size();
    actions.push_back(Action("queue", 1, payload));
  }
  mark_closed();
  return actions;
}

enum class Length { kShortest, k16, k64, kTopBit };

// A client frame, masked unless `masked` is false. A declared length other
// than the payload's leaves the frame short or long.
std::string Frame(bool fin, unsigned rsv, unsigned opcode,
                  std::string_view payload, Length encoding = Length::kShortest,
                  bool masked = true,
                  std::size_t declared = std::string::npos) {
  if (declared == std::string::npos)
    declared = payload.size();
  constexpr std::array<std::uint8_t, 4> kMask{0x3a, 0x91, 0x07, 0xe4};
  std::string frame;
  frame.push_back(static_cast<char>((fin ? 0x80 : 0) | (rsv << 4) | opcode));
  const char mask_bit = static_cast<char>(masked ? 0x80 : 0);
  if (encoding == Length::kShortest && declared < 126) {
    frame.push_back(static_cast<char>(mask_bit | declared));
  } else if (encoding == Length::k16 ||
             (encoding == Length::kShortest && declared <= 65535)) {
    frame.push_back(static_cast<char>(mask_bit | 126));
    frame.push_back(static_cast<char>(declared >> 8));
    frame.push_back(static_cast<char>(declared & 255));
  } else {
    frame.push_back(static_cast<char>(mask_bit | 127));
    const std::uint64_t length =
        declared | (encoding == Length::kTopBit ? 1ULL << 63 : 0);
    for (int shift = 56; shift >= 0; shift -= 8)
      frame.push_back(static_cast<char>((length >> shift) & 255));
  }
  if (masked)
    frame.append(reinterpret_cast<const char*>(kMask.data()), kMask.size());
  for (std::size_t i = 0; i < payload.size(); ++i) {
    const auto byte = static_cast<std::uint8_t>(payload[i]);
    frame.push_back(static_cast<char>(masked ? byte ^ kMask[i % 4] : byte));
  }
  return frame;
}

std::string Text(std::string_view text) {
  return Frame(true, 0, 1, text);
}

std::string CloseFrame(unsigned code, std::string_view reason = {}) {
  std::string payload{static_cast<char>(code >> 8),
                      static_cast<char>(code & 255)};
  payload.append(reason);
  return Frame(true, 0, 8, payload);
}

void Check(const std::string& stream, std::vector<std::string> expected,
           std::size_t limit = kMaximumMessage) {
  if (NewActions(stream, limit) != expected)
    throw std::runtime_error("WebSocket expected actions differ");
}
int Run() {
  Check(Text("hello"), {"queue 1 68656c6c6f"});
  Check(Frame(false, 0, 1, "he") + Frame(true, 0, 9, "p") +
            Frame(true, 0, 0, "llo"),
        {"send 10 70", "queue 1 68656c6c6f"});
  Check(Text("before") + CloseFrame(1000) + Text("after"),
        {"queue 1 6265666f7265", "send 8 03e8"});
  Check(Frame(true, 0, 1, "x", Length::kShortest, false), {"send 8 03ea"});
  Check(Text(std::string(101, 'x')), {"send 8 03f1"}, 100);
  Check(Text("\xff"), {"send 8 03ef"});
  Check(CloseFrame(1005), {"send 8 03ea"});
  for (const auto length : {125U, 126U, 65535U, 65536U}) {
    const std::string payload(length, 'a');
    Check(Text(payload), {Action("queue", 1, payload)});
  }
  const auto frame = Text("hello");
  for (std::size_t end = 0; end < frame.size(); ++end)
    Check(frame.substr(0, end), {});
  std::string many;
  std::vector<std::string> expected;
  for (int i = 0; i < 129; ++i) {
    many += Text("a");
    if (i < 128)
      expected.push_back("queue 1 61");
  }
  expected.push_back("send 8 03f1");
  Check(many, expected);
  return 0;
}
}  // namespace
int main() {
  try {
    return Run();
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
