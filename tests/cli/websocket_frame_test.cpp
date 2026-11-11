// Compares the Rust WebSocket frame reader (gufo_ws_reader_next), driven as
// WebSocket::ReadLoop drives it, with a frozen copy of ReadLoop from 0.33.0
// (tests/cli/websocket_frame_reference.cpp) on generated client streams:
// every length encoding and limit, a grid of opcodes, FIN, RSV and mask bits,
// every close code from 0 to 5100, UTF-8 within and across fragments, the
// message queue's limits, a stream cut at every byte and random mutations.
// The frames sent back and the messages queued must match. The socket side
// of WebSocket is covered by audio_websocket_test.
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

#include "tests/cli/websocket_frame_reference.hpp"

namespace {

namespace ref = gufo::websocket_reference;

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
    actions.push_back(ref::Action("send", opcode, bytes));
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
    actions.push_back(ref::Action("queue", 1, payload));
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

struct Totals {
  std::size_t cases = 0;
  std::size_t mismatches = 0;
};

std::string Hex(std::string_view bytes) {
  return ref::Action("", 0, bytes.substr(0, 24)).substr(3);
}

void Compare(Totals& totals, const std::string& stream,
             std::size_t max_message = kMaximumMessage) {
  ++totals.cases;
  const auto expected = ref::ReferenceSocket(stream, max_message).Run();
  const auto actual = NewActions(stream, max_message);
  if (expected == actual || ++totals.mismatches > 10)
    return;
  std::fprintf(stderr, "stream %s... (%zu bytes, limit %zu):\n",
               Hex(stream).c_str(), stream.size(), max_message);
  for (std::size_t i = 0; i < std::max(expected.size(), actual.size()); ++i) {
    std::fprintf(stderr, "  expected %.60s | got %.60s\n",
                 i < expected.size() ? expected[i].c_str() : "-",
                 i < actual.size() ? actual[i].c_str() : "-");
  }
}

int Run() {
  Totals totals;
  std::mt19937_64 random(6455);

  // Length encodings: shortest, padded to 16 or 64 bits, the top bit set.
  for (const std::size_t length :
       {0, 1, 3, 4, 5, 125, 126, 127, 300, 65535, 65536, 70000}) {
    const std::string text(length, 'a');
    for (const auto encoding :
         {Length::kShortest, Length::k16, Length::k64, Length::kTopBit}) {
      if (encoding == Length::k16 && length > 65535)
        continue;
      Compare(totals, Frame(true, 0, 1, text, encoding));
      Compare(totals, Frame(true, 0, 9, text, encoding));
    }
  }
  // The message limit, in one frame and across fragments.
  for (const std::size_t length : {299, 300, 301}) {
    Compare(totals, Text(std::string(length, 'b')), 300);
    Compare(totals,
            Frame(false, 0, 1, std::string(150, 'b')) +
                Frame(true, 0, 0, std::string(length - 150, 'b')),
            300);
    Compare(totals, Frame(true, 0, 9, std::string(length % 126, 'p')), 100);
  }
  Compare(totals, Text(std::string(kMaximumMessage, 'c')));
  Compare(totals, Text(std::string(kMaximumMessage + 1, 'c')));
  // A declared length longer than the stream.
  Compare(totals, Frame(true, 0, 1, "short", Length::kShortest, true, 9));

  // Opcode, FIN, RSV and mask bits, alone and inside a fragmented message,
  // then a text frame.
  const std::array<std::string, 3> payloads{"", "ab", std::string(126, 'q')};
  for (unsigned opcode = 0; opcode < 16; ++opcode) {
    for (const bool fin : {false, true}) {
      for (const unsigned rsv : {0U, 1U, 2U, 4U}) {
        for (const bool masked : {false, true}) {
          for (const std::string& payload : payloads) {
            const std::string frame =
                Frame(fin, rsv, opcode, payload, Length::kShortest, masked);
            Compare(totals, frame + Text("z"));
            Compare(totals, Frame(false, 0, 1, "x") + frame + Text("z"));
            Compare(totals,
                    Frame(false, 0, 1, "x") + frame + Frame(true, 0, 0, "y"));
          }
        }
      }
    }
  }

  // Every close code, with an empty, a UTF-8 and a broken reason.
  for (unsigned code = 0; code <= 5100; ++code) {
    Compare(totals, CloseFrame(code));
    if (code % 97 == 0 || code == 1000 || code == 3000 || code == 4999) {
      Compare(totals, CloseFrame(code, "ok \xc3\xa9"));
      Compare(totals, CloseFrame(code, "\xff"));
    }
  }
  Compare(totals, Frame(true, 0, 8, "\x03"));
  Compare(totals, Frame(true, 0, 8, std::string(125, 'r')));
  Compare(totals, Frame(true, 0, 8, std::string(126, 'r')));
  Compare(totals, Text("before") + CloseFrame(1000) + Text("after"));

  // UTF-8 whole, split across two fragments at every byte, and one byte per
  // fragment.
  for (const std::string_view text :
       {std::string_view("plain"), std::string_view("\xc3\xa9t\xc3\xa9"),
        std::string_view("\xe2\x82\xac 5"),
        std::string_view("\xf0\x9d\x84\x9e"), std::string_view("\xc3"),
        std::string_view("\xc0\x80"), std::string_view("\xed\xa0\x80"),
        std::string_view("\xf4\x90\x80\x80"), std::string_view("a\xff"),
        std::string_view("\xe2\x82")}) {
    Compare(totals, Text(text));
    for (std::size_t at = 0; at <= text.size(); ++at) {
      Compare(totals, Frame(false, 0, 1, text.substr(0, at)) +
                          Frame(true, 0, 0, text.substr(at)));
    }
    std::string bytes = Frame(text.size() <= 1, 0, 1, text.substr(0, 1));
    for (std::size_t at = 1; at < text.size(); ++at)
      bytes += Frame(at + 1 == text.size(), 0, 0, text.substr(at, 1));
    Compare(totals, bytes);
  }

  // Control frames between fragments, and the queue's limits.
  Compare(totals, Frame(false, 0, 1, "fr") + Frame(true, 0, 9, "ping") +
                      Frame(true, 0, 10, "pong") + Frame(true, 0, 0, "ag") +
                      Text("next"));
  std::string many;
  for (int i = 0; i < 130; ++i)
    many += Text("m" + std::to_string(i));
  Compare(totals, many);
  const std::string three_mib(3U << 20, 'd');
  Compare(totals, Text(three_mib) + Text(three_mib) + Text(three_mib));

  // A mixed stream cut at every byte, then mutated at random.
  const std::string mixed = Text("hello") + Frame(false, 0, 1, "wo") +
                            Frame(true, 0, 9, "p") + Frame(true, 0, 0, "rld") +
                            Text(std::string(130, 'e')) +
                            Frame(true, 0, 10, "") + CloseFrame(1001, "bye");
  for (std::size_t length = 0; length < mixed.size(); ++length)
    Compare(totals, mixed.substr(0, length));
  for (int mutation = 0; mutation < 3000; ++mutation) {
    std::string mutated = mixed;
    const std::size_t at = random() % mutated.size();
    if (mutation % 3 == 0)
      mutated[at] = static_cast<char>(mutated[at] ^ (1 << (random() % 8)));
    else if (mutation % 3 == 1)
      mutated[at] = static_cast<char>(random());
    else
      mutated.erase(at, 1 + random() % 4);
    Compare(totals, mutated, mutation % 2 == 0 ? kMaximumMessage : 120);
  }
  for (int stream = 0; stream < 1000; ++stream) {
    std::string bytes(random() % 48, '\0');
    for (char& byte : bytes)
      byte = static_cast<char>(random());
    if (bytes.size() >= 2 && stream % 2 == 0)
      bytes[1] = static_cast<char>(bytes[1] | 0x80);
    Compare(totals, bytes);
  }

  std::printf("websocket frames: %zu streams, %zu mismatches\n", totals.cases,
              totals.mismatches);
  return totals.mismatches == 0 ? 0 : 1;
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
