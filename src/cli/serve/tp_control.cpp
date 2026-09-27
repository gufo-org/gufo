#include "src/cli/serve/tp_control.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>

namespace gufo::server {
namespace {

constexpr std::uint32_t kMagic = 0x54504331U;  // "TPC1"
constexpr std::uint16_t kVersion = 13;         // requests and rank-1 executor
                                               // instructions for concurrent
                                               // requests
constexpr std::uint16_t kHello = 1;
constexpr std::uint16_t kCommand = 2;
constexpr std::uint16_t kResponse = 3;
// A request carries its images' pixels: up to 16 images of up to 16 MP each.
constexpr std::size_t kMaxPayloadBytes = std::size_t{1} << 30;
// Before the handshake a peer is not yet authenticated.
constexpr std::size_t kMaxHelloBytes = 64U << 10;
constexpr std::size_t kMaxPromptContextBytes = kMaxPayloadBytes - (64U << 20);
constexpr std::size_t kMaxPromptTokens = 1U << 20;
constexpr std::size_t kMaxBatchMembers = 8;
constexpr std::size_t kMaxClientIdBytes = 4096;
constexpr std::size_t kMaxErrorBytes = 1U << 20;
constexpr std::size_t kMaxAuthTokenBytes = 4096;
constexpr auto kIoTimeout = std::chrono::seconds(30);

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

std::string SystemError(const char* operation) {
  return std::string(operation) + ": " + std::strerror(errno);
}

/// Bounds blocking sends and pre-handshake receives, turns on TCP keepalive so
/// a peer whose host died is reported even on an idle channel, and disables
/// Nagle's algorithm: a cache instruction is a round trip, and a small frame
/// held back for a delayed ACK stalls it by tens of milliseconds.
void SetStartupSocketOptions(int fd, std::chrono::milliseconds io_timeout) {
  const auto milliseconds = std::max<std::int64_t>(io_timeout.count(), 1);
  timeval timeout{};
  timeout.tv_sec = static_cast<time_t>(milliseconds / 1000);
  timeout.tv_usec = static_cast<suseconds_t>((milliseconds % 1000) * 1000);
  (void)::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  (void)::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
  const int enabled = 1;
  (void)::setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &enabled, sizeof(enabled));
  (void)::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &enabled, sizeof(enabled));
#if defined(TCP_KEEPIDLE) && defined(TCP_KEEPINTVL) && defined(TCP_KEEPCNT)
  // Probe after one idle minute, then every ten seconds: a dead host is
  // reported about two minutes after it stops answering.
  const int idle_seconds = 60;
  const int interval_seconds = 10;
  const int probes = 6;
  (void)::setsockopt(fd, IPPROTO_TCP, TCP_KEEPIDLE, &idle_seconds,
                     sizeof(idle_seconds));
  (void)::setsockopt(fd, IPPROTO_TCP, TCP_KEEPINTVL, &interval_seconds,
                     sizeof(interval_seconds));
  (void)::setsockopt(fd, IPPROTO_TCP, TCP_KEEPCNT, &probes, sizeof(probes));
#endif
}

/// After the handshake a rank waits for the next command or response for as
/// long as the server stays idle, so a receive timeout would end the pair.
void ClearReceiveTimeout(int fd) {
  const timeval none{};
  (void)::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &none, sizeof(none));
}

void AppendU32(std::vector<std::uint8_t>* out, std::uint32_t value) {
  for (unsigned shift = 0; shift < 32; shift += 8) {
    out->push_back(static_cast<std::uint8_t>(value >> shift));
  }
}

void AppendU64(std::vector<std::uint8_t>* out, std::uint64_t value) {
  for (unsigned shift = 0; shift < 64; shift += 8) {
    out->push_back(static_cast<std::uint8_t>(value >> shift));
  }
}

bool ReadU32(std::span<const std::uint8_t> data, std::size_t* offset,
             std::uint32_t* value, std::string* error) {
  if (*offset > data.size() || data.size() - *offset < sizeof(std::uint32_t)) {
    SetError(error, "TP control payload is truncated");
    return false;
  }
  std::uint32_t result = 0;
  for (unsigned shift = 0; shift < 32; shift += 8) {
    result |= static_cast<std::uint32_t>(data[(*offset)++]) << shift;
  }
  *value = result;
  return true;
}

bool ReadU64(std::span<const std::uint8_t> data, std::size_t* offset,
             std::uint64_t* value, std::string* error) {
  if (*offset > data.size() || data.size() - *offset < sizeof(std::uint64_t)) {
    SetError(error, "TP control payload is truncated");
    return false;
  }
  std::uint64_t result = 0;
  for (unsigned shift = 0; shift < 64; shift += 8) {
    result |= static_cast<std::uint64_t>(data[(*offset)++]) << shift;
  }
  *value = result;
  return true;
}

void AppendBytes(std::vector<std::uint8_t>* out,
                 std::span<const std::uint8_t> bytes) {
  out->insert(out->end(), bytes.begin(), bytes.end());
}

bool ReadBytes(std::span<const std::uint8_t> data, std::size_t* offset,
               std::span<std::uint8_t> bytes, std::string* error) {
  if (*offset > data.size() || bytes.size() > data.size() - *offset) {
    SetError(error, "TP control payload is truncated");
    return false;
  }
  std::memcpy(bytes.data(), data.data() + *offset, bytes.size());
  *offset += bytes.size();
  return true;
}

[[nodiscard]] bool IsDefaultSampling(const sampling::SamplingConfig& config) {
  const sampling::SamplingConfig defaults;
  return config.temperature == defaults.temperature &&
         config.top_k == defaults.top_k && config.top_p == defaults.top_p &&
         config.min_p == defaults.min_p &&
         config.min_keep == defaults.min_keep && config.seed == defaults.seed &&
         config.repeat_penalty == defaults.repeat_penalty &&
         config.repeat_last_n == defaults.repeat_last_n &&
         config.frequency_penalty == defaults.frequency_penalty &&
         config.presence_penalty == defaults.presence_penalty;
}

[[nodiscard]] bool HasRequestFields(const TpControlCommand& command) {
  return command.max_tokens != 0 || command.cache_prompt ||
         command.cache_prefix_tokens != 0 || !command.prompt_tokens.empty() ||
         !command.client_id.empty() || !command.prompt_context.empty() ||
         !IsDefaultSampling(command.sampling);
}

[[nodiscard]] std::uint32_t FloatBits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

[[nodiscard]] float BitsFloat(std::uint32_t bits) {
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

/// An instruction names one model call and its arguments; every argument the
/// call does not take must be zero, so two encodings of one call cannot differ.
[[nodiscard]] bool ValidateInstruction(std::uint64_t sequence,
                                       const TpInstruction& instruction,
                                       std::string* error) {
  const auto only = [&](bool token, bool offset, bool count, bool prompt_size,
                        bool digest) {
    return (token || instruction.token == 0) &&
           (offset || instruction.offset == 0) &&
           (count || instruction.count == 0) &&
           (prompt_size || instruction.prompt_size == 0) &&
           (digest || instruction.digest == 0);
  };
  // Only a decode carries a draw state.
  if (instruction.op != TpInstructionOp::kDecode &&
      (instruction.rng != 0 || instruction.pending != -1)) {
    SetError(error, "TP instruction carries a draw state it does not use");
    return false;
  }
  const bool snapshot_op = instruction.op == TpInstructionOp::kSnapshot ||
                           instruction.op == TpInstructionOp::kRestore ||
                           instruction.op == TpInstructionOp::kDrop;
  if (snapshot_op != (instruction.snapshot_id != 0)) {
    SetError(error, "TP instruction snapshot ID is invalid");
    return false;
  }
  if ((instruction.op == TpInstructionOp::kAdvanceBatch) !=
      !instruction.batch.empty()) {
    SetError(error, "TP instruction batch does not match its operation");
    return false;
  }
  bool valid = false;
  switch (instruction.op) {
    case TpInstructionOp::kAdvanceBatch:
      valid = instruction.state == 0 &&
              only(false, false, false, false, false) &&
              instruction.batch.size() >= 2 &&
              instruction.batch.size() <= kMaxBatchMembers && sequence == 0;
      for (std::size_t index = 0; valid && index < instruction.batch.size();
           ++index) {
        const auto& member = instruction.batch[index];
        valid = member.sequence != 0 && member.token >= 0;
        for (std::size_t previous = 0; valid && previous < index; ++previous) {
          valid = instruction.batch[previous].state != member.state;
        }
      }
      break;
    case TpInstructionOp::kDrop:
      valid = instruction.state == 0 && only(false, false, false, false, false);
      break;
    case TpInstructionOp::kReuse:
      valid = only(false, false, false, true, false) &&
              instruction.prompt_size <= kMaxPromptTokens;
      break;
    case TpInstructionOp::kSnapshot:
    case TpInstructionOp::kRestore:
    case TpInstructionOp::kCancelPrepare:
    case TpInstructionOp::kInvalidate:
    case TpInstructionOp::kPromptContext:
      valid = only(false, false, false, false, false);
      break;
    case TpInstructionOp::kPrefill:
      valid = only(false, true, true, true, false) && instruction.count != 0 &&
              instruction.prompt_size > instruction.offset &&
              instruction.prompt_size <= kMaxPromptTokens;
      break;
    case TpInstructionOp::kAdvance:
      valid = only(true, false, false, false, false) && instruction.token >= 0;
      break;
    case TpInstructionOp::kDecode:
      valid = only(false, false, true, false, false) && instruction.count > 1 &&
              instruction.pending >= -1;
      break;
    case TpInstructionOp::kEnd:
      valid = only(false, false, true, false, true);
      break;
    case TpInstructionOp::kNone:
      break;
  }
  if (!valid) {
    SetError(error, "TP instruction has invalid arguments for its operation");
    return false;
  }
  // Only a reset, a snapshot drop or clearing a prompt context can happen
  // between requests: the continuation cache makes them on its own schedule.
  // A batch names its members' requests. Every other call belongs to a
  // request.
  if (sequence == 0 && instruction.op != TpInstructionOp::kInvalidate &&
      instruction.op != TpInstructionOp::kDrop &&
      instruction.op != TpInstructionOp::kPromptContext &&
      instruction.op != TpInstructionOp::kAdvanceBatch) {
    SetError(error,
             "TP instruction outside a request must be a reset, a snapshot "
             "drop or a prompt-context release");
    return false;
  }
  return true;
}

[[nodiscard]] bool HasResultFields(const TpControlResponse& response) {
  return !response.tokens.empty() || response.draft_tokens != 0 ||
         response.draft_accepted_tokens != 0 ||
         response.cached_prompt_tokens != 0 ||
         response.cache_snapshot_bytes != 0;
}

}  // namespace

bool ValidateTpControlCommand(const TpControlCommand& command,
                              std::string* error) {
  if (command.kind != TpControlCommandKind::kInstruction &&
      command.instruction != TpInstruction{}) {
    SetError(error, "TP request command carries an instruction");
    return false;
  }
  if (command.kind == TpControlCommandKind::kSingle) {
    if (command.max_tokens == 0 || command.prompt_tokens.empty() ||
        command.prompt_tokens.size() > kMaxPromptTokens ||
        command.cache_prefix_tokens > command.prompt_tokens.size() ||
        command.client_id.size() > kMaxClientIdBytes ||
        command.prompt_context.size() > kMaxPromptContextBytes ||
        std::ranges::any_of(command.prompt_tokens,
                            [](std::int32_t token) { return token < 0; })) {
      SetError(error, "TP control request is invalid");
      return false;
    }
    try {
      command.sampling.Validate();
    } catch (const std::exception& exception) {
      SetError(error, std::string("TP control request sampling is invalid: ") +
                          exception.what());
      return false;
    }
    return true;
  }
  if (command.kind == TpControlCommandKind::kInstruction) {
    // An instruction is one model call within a request that a previous
    // kSingle already described. Anything else here would mean the two
    // messages disagree about the request, so every request field must be
    // empty.
    if (HasRequestFields(command)) {
      SetError(error, "TP instruction carries request fields");
      return false;
    }
    return ValidateInstruction(command.sequence, command.instruction, error);
  }
  SetError(error, "TP control command kind is invalid");
  return false;
}

bool ValidateTpControlResponse(const TpControlResponse& response,
                               std::string* error) {
  if (response.error.size() > kMaxErrorBytes) {
    SetError(error, "TP control response error is too large");
    return false;
  }
  if (response.kind == TpControlResponseKind::kInstruction) {
    if (response.sequence == 0 || HasResultFields(response)) {
      SetError(error, "TP cache acknowledgement is invalid");
      return false;
    }
    return true;
  }
  if (response.kind != TpControlResponseKind::kSingle) {
    SetError(error, "TP control response kind is invalid");
    return false;
  }
  if (response.instruction_index != 0) {
    SetError(error, "TP final response carries an instruction index");
    return false;
  }
  if (response.tokens.size() > kMaxPromptTokens ||
      response.cached_prompt_tokens > kMaxPromptTokens) {
    SetError(error, "TP control response is too large");
    return false;
  }
  return true;
}

TpControlChannel::TpControlChannel(int fd, std::uint32_t rank,
                                   std::uint16_t port)
    : fd_(fd), port_(port), rank_(rank) {
  if (port_ == 0) {
    sockaddr_in address{};
    socklen_t length = sizeof(address);
    if (::getsockname(fd_, reinterpret_cast<sockaddr*>(&address), &length) ==
        0) {
      port_ = ntohs(address.sin_port);
    }
  }
}

std::shared_ptr<TpControlChannel> TpControlChannel::Listen(
    std::uint16_t port, std::string* error,
    std::chrono::milliseconds io_timeout) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    SetError(error, SystemError("TP control socket"));
    return nullptr;
  }
  int reuse = 1;
  (void)::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_ANY);
  address.sin_port = htons(port);
  if (::bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 ||
      ::listen(fd, 1) != 0) {
    SetError(error, SystemError("TP control bind/listen"));
    ::close(fd);
    return nullptr;
  }
  pollfd descriptor{.fd = fd, .events = POLLIN, .revents = 0};
  const int poll_result = ::poll(&descriptor, 1, 30000);
  if (poll_result <= 0) {
    SetError(error, poll_result == 0 ? "TP control accept timed out"
                                     : SystemError("TP control poll"));
    ::close(fd);
    return nullptr;
  }
  int peer = -1;
  do {
    peer = ::accept(fd, nullptr, nullptr);
  } while (peer < 0 && errno == EINTR);
  ::close(fd);
  if (peer < 0) {
    SetError(error, SystemError("TP control accept"));
    return nullptr;
  }
  SetStartupSocketOptions(peer, io_timeout);
  return std::shared_ptr<TpControlChannel>(new TpControlChannel(peer, 0, port));
}

std::shared_ptr<TpControlChannel> TpControlChannel::Connect(
    const std::string& host, std::uint16_t port, std::string* error,
    std::chrono::milliseconds io_timeout) {
  if (host.empty()) {
    SetError(error, "TP control rank one requires a host");
    return nullptr;
  }
  const auto deadline = std::chrono::steady_clock::now() + kIoTimeout;
  for (;;) {
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    const std::string service = std::to_string(port);
    addrinfo* addresses = nullptr;
    const int resolve =
        ::getaddrinfo(host.c_str(), service.c_str(), &hints, &addresses);
    if (resolve != 0) {
      SetError(error, "TP control getaddrinfo: " +
                          std::string(::gai_strerror(resolve)));
      return nullptr;
    }
    int fd = -1;
    for (auto* address = addresses; address != nullptr;
         address = address->ai_next) {
      fd = ::socket(address->ai_family, address->ai_socktype,
                    address->ai_protocol);
      if (fd < 0) {
        continue;
      }
      const int flags = ::fcntl(fd, F_GETFL, 0);
      if (flags < 0 || ::fcntl(fd, F_SETFL, flags | O_NONBLOCK) != 0) {
        ::close(fd);
        fd = -1;
        continue;
      }
      bool connected =
          ::connect(fd, address->ai_addr, address->ai_addrlen) == 0;
      if (!connected && errno == EINPROGRESS) {
        pollfd descriptor{.fd = fd, .events = POLLOUT, .revents = 0};
        const int poll_result = ::poll(&descriptor, 1, 1000);
        int socket_error = 0;
        socklen_t error_size = sizeof(socket_error);
        connected = poll_result > 0 &&
                    ::getsockopt(fd, SOL_SOCKET, SO_ERROR, &socket_error,
                                 &error_size) == 0 &&
                    socket_error == 0;
      }
      if (connected && ::fcntl(fd, F_SETFL, flags) == 0) {
        break;
      }
      ::close(fd);
      fd = -1;
    }
    ::freeaddrinfo(addresses);
    if (fd >= 0) {
      SetStartupSocketOptions(fd, io_timeout);
      return std::shared_ptr<TpControlChannel>(
          new TpControlChannel(fd, 1, port));
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      SetError(error, "TP control connect timed out");
      return nullptr;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

TpControlChannel::~TpControlChannel() {
  interrupted_.store(true, std::memory_order_release);
  if (fd_ >= 0) {
    ::shutdown(fd_, SHUT_RDWR);
    ::close(fd_);
    fd_ = -1;
  }
}

void TpControlChannel::Interrupt() noexcept {
  if (!interrupted_.exchange(true, std::memory_order_acq_rel) && fd_ >= 0) {
    ::shutdown(fd_, SHUT_RDWR);
  }
}

bool TpControlChannel::SendAll(const void* data, std::size_t bytes,
                               std::string* error) {
  const auto* cursor = static_cast<const std::uint8_t*>(data);
  std::size_t sent = 0;
  while (sent < bytes) {
    const ssize_t result =
        ::send(fd_, cursor + sent, bytes - sent, MSG_NOSIGNAL);
    if (result > 0) {
      sent += static_cast<std::size_t>(result);
      continue;
    }
    if (result < 0 && errno == EINTR) {
      continue;
    }
    SetError(error, SystemError("TP control send"));
    return false;
  }
  return true;
}

bool TpControlChannel::RecvAll(void* data, std::size_t bytes,
                               std::string* error) {
  auto* cursor = static_cast<std::uint8_t*>(data);
  std::size_t received = 0;
  while (received < bytes) {
    const ssize_t result = ::recv(fd_, cursor + received, bytes - received, 0);
    if (result > 0) {
      received += static_cast<std::size_t>(result);
      continue;
    }
    if (result == 0) {
      SetError(error, "TP control peer closed the channel");
      return false;
    }
    if (errno == EINTR) {
      continue;
    }
    SetError(error, SystemError("TP control receive"));
    return false;
  }
  return true;
}

bool TpControlChannel::SendFrame(std::uint16_t type, std::uint64_t sequence,
                                 const std::vector<std::uint8_t>& payload,
                                 std::string* error) {
  const std::lock_guard<std::mutex> lock(send_mutex_);
  if (interrupted_.load(std::memory_order_acquire)) {
    SetError(error, "TP control channel is interrupted");
    return false;
  }
  if (fd_ < 0 || payload.size() > kMaxPayloadBytes ||
      payload.size() > std::numeric_limits<std::uint32_t>::max()) {
    SetError(error, "TP control frame is invalid or too large");
    return false;
  }
  // One write per frame: with Nagle disabled, a separate header write would
  // leave as its own segment.
  std::vector<std::uint8_t> frame;
  frame.reserve(20 + payload.size());
  AppendU32(&frame, kMagic);
  frame.push_back(static_cast<std::uint8_t>(kVersion));
  frame.push_back(static_cast<std::uint8_t>(kVersion >> 8));
  frame.push_back(static_cast<std::uint8_t>(type));
  frame.push_back(static_cast<std::uint8_t>(type >> 8));
  AppendU32(&frame, static_cast<std::uint32_t>(payload.size()));
  AppendU64(&frame, sequence);
  frame.insert(frame.end(), payload.begin(), payload.end());
  return SendAll(frame.data(), frame.size(), error);
}

bool TpControlChannel::ReceiveFrame(std::uint16_t type, std::uint64_t* sequence,
                                    std::vector<std::uint8_t>* payload,
                                    std::string* error) {
  const std::lock_guard<std::mutex> lock(receive_mutex_);
  if (interrupted_.load(std::memory_order_acquire)) {
    SetError(error, "TP control channel is interrupted");
    return false;
  }
  if (fd_ < 0 || sequence == nullptr || payload == nullptr) {
    SetError(error, "TP control receive state is invalid");
    return false;
  }
  std::array<std::uint8_t, 20> header{};
  if (!RecvAll(header.data(), header.size(), error)) {
    return false;
  }
  std::size_t offset = 0;
  std::uint32_t magic = 0;
  if (!ReadU32(header, &offset, &magic, error) || magic != kMagic ||
      header[4] != kVersion || header[5] != 0 ||
      header[6] != static_cast<std::uint8_t>(type) ||
      header[7] != static_cast<std::uint8_t>(type >> 8)) {
    SetError(error, "TP control frame header is invalid");
    return false;
  }
  offset += 4;  // version and frame type
  std::uint32_t bytes = 0;
  if (!ReadU32(header, &offset, &bytes, error) ||
      bytes > (handshaken_ ? kMaxPayloadBytes : kMaxHelloBytes) ||
      !ReadU64(header, &offset, sequence, error)) {
    SetError(error, "TP control frame length is invalid");
    return false;
  }
  std::vector<std::uint8_t> received(bytes);
  if (bytes != 0 && !RecvAll(received.data(), received.size(), error)) {
    return false;
  }
  *payload = std::move(received);
  return true;
}

bool TpControlChannel::Handshake(const TpControlConfig& config,
                                 std::string* error) {
  if (config.world_size != 2 || config.rank != rank_ || config.rank > 1 ||
      config.max_context == 0 || config.prefill_chunk_tokens == 0 ||
      config.sessions == 0 || config.auth_token.empty() ||
      config.auth_token.size() > kMaxAuthTokenBytes || handshaken_ ||
      (config.use_mtp && config.max_draft_tokens == 0)) {
    SetError(error, "TP control configuration is invalid");
    return false;
  }
  auth_token_ = config.auth_token;
  std::vector<std::uint8_t> payload;
  AppendU32(&payload, config.rank);
  AppendU32(&payload, config.world_size);
  AppendU32(&payload, config.max_context);
  AppendU32(&payload, config.prefill_chunk_tokens);
  AppendU32(&payload, config.max_draft_tokens);
  AppendU32(&payload, config.use_mtp ? 1U : 0U);
  AppendU32(&payload, config.sessions);
  AppendU32(&payload, config.vision ? 1U : 0U);
  AppendU64(&payload, config.snapshot_budget_bytes);
  AppendU32(&payload, static_cast<std::uint32_t>(config.auth_token.size()));
  payload.insert(payload.end(), config.auth_token.begin(),
                 config.auth_token.end());
  if (!SendFrame(kHello, 0, payload, error)) {
    return false;
  }
  std::uint64_t sequence = 0;
  std::vector<std::uint8_t> peer;
  // Keep the frame error: a peer that died during model load surfaces here as
  // a receive failure, and reporting only a bad sequence sent looking for a
  // protocol bug instead of the dead peer that caused it.
  if (!ReceiveFrame(kHello, &sequence, &peer, error)) {
    return false;
  }
  if (sequence != 0) {
    SetError(error, "TP control hello sequence is invalid");
    return false;
  }
  std::size_t offset = 0;
  std::uint32_t rank = 0;
  std::uint32_t world = 0;
  std::uint32_t context = 0;
  std::uint32_t prefill_chunk = 0;
  std::uint32_t draft = 0;
  std::uint32_t mtp = 0;
  std::uint32_t sessions = 0;
  std::uint32_t vision = 0;
  std::uint64_t snapshot_budget = 0;
  std::uint32_t auth_size = 0;
  std::string peer_token;
  if (!ReadU32(peer, &offset, &rank, error) ||
      !ReadU32(peer, &offset, &world, error) ||
      !ReadU32(peer, &offset, &context, error) ||
      !ReadU32(peer, &offset, &prefill_chunk, error) ||
      !ReadU32(peer, &offset, &draft, error) ||
      !ReadU32(peer, &offset, &mtp, error) ||
      !ReadU32(peer, &offset, &sessions, error) ||
      !ReadU32(peer, &offset, &vision, error) ||
      !ReadU64(peer, &offset, &snapshot_budget, error) ||
      !ReadU32(peer, &offset, &auth_size, error) ||
      auth_size > kMaxAuthTokenBytes || peer.size() - offset != auth_size) {
    SetError(error, "TP control hello payload is invalid");
    return false;
  }
  peer_token.assign(reinterpret_cast<const char*>(peer.data() + offset),
                    auth_size);
  // Name every setting the ranks disagree on: both must be started with the
  // same model options, and a bare mismatch leaves the operator guessing.
  std::string mismatch;
  const auto differs = [&](bool different, const char* what, std::uint64_t mine,
                           std::uint64_t peers) {
    if (different) {
      mismatch += std::string(mismatch.empty() ? "" : ", ") + what + " " +
                  std::to_string(mine) + " here, " + std::to_string(peers) +
                  " on the peer";
    }
  };
  differs(rank != 1U - config.rank, "rank", config.rank, rank);
  differs(world != config.world_size, "world size", config.world_size, world);
  differs(context != config.max_context, "context", config.max_context,
          context);
  differs(prefill_chunk != config.prefill_chunk_tokens, "prefill chunk",
          config.prefill_chunk_tokens, prefill_chunk);
  differs(mtp != (config.use_mtp ? 1U : 0U), "MTP", config.use_mtp ? 1U : 0U,
          mtp);
  differs(draft != config.max_draft_tokens, "draft tokens",
          config.max_draft_tokens, draft);
  differs(sessions != config.sessions, "sessions", config.sessions, sessions);
  differs(vision != (config.vision ? 1U : 0U), "vision (--mmproj)",
          config.vision ? 1U : 0U, vision);
  if (peer_token != auth_token_) {
    mismatch += std::string(mismatch.empty() ? "" : ", ") + "control token";
  }
  if (!mismatch.empty()) {
    SetError(error, "TP control hello configuration mismatch: " + mismatch);
    return false;
  }
  snapshot_budget_bytes_ =
      std::min(config.snapshot_budget_bytes, snapshot_budget);
  world_size_ = config.world_size;
  max_context_ = config.max_context;
  handshaken_ = true;
  ClearReceiveTimeout(fd_);
  return true;
}

bool TpControlChannel::SendCommand(const TpControlCommand& command,
                                   std::string* error) {
  if (!handshaken_ || rank_ != 0 || !ValidateTpControlCommand(command, error)) {
    if (error != nullptr && error->empty()) {
      SetError(error, "TP control command is invalid");
    }
    return false;
  }
  if (max_context_ == 0 || command.prompt_tokens.size() > max_context_) {
    SetError(error, "TP control command exceeds the negotiated context");
    return false;
  }

  std::vector<std::uint8_t> payload;
  payload.reserve(96 + command.prompt_tokens.size() * sizeof(std::uint32_t) +
                  command.client_id.size() + command.prompt_context.size());
  AppendU64(&payload, command.sequence);
  AppendU32(&payload, static_cast<std::uint32_t>(command.kind));
  // The layout after the kind is a function of the kind, so both sides derive
  // the same framing without a separate encoder.
  if (command.kind == TpControlCommandKind::kSingle) {
    AppendU32(&payload, command.max_tokens);
    AppendU32(&payload, command.cache_prompt ? 1U : 0U);
    AppendU32(&payload, command.cache_prefix_tokens);
    AppendU32(&payload,
              static_cast<std::uint32_t>(command.prompt_tokens.size()));
    AppendU32(&payload, static_cast<std::uint32_t>(command.client_id.size()));
    AppendU32(&payload,
              static_cast<std::uint32_t>(command.prompt_context.size()));
    for (const auto token : command.prompt_tokens) {
      AppendU32(&payload, static_cast<std::uint32_t>(token));
    }
    payload.insert(payload.end(), command.client_id.begin(),
                   command.client_id.end());
    payload.insert(payload.end(), command.prompt_context.begin(),
                   command.prompt_context.end());
    const auto& config = command.sampling;
    AppendU32(&payload, FloatBits(config.temperature));
    AppendU32(&payload, static_cast<std::uint32_t>(config.top_k));
    AppendU32(&payload, FloatBits(config.top_p));
    AppendU32(&payload, FloatBits(config.min_p));
    AppendU64(&payload, config.min_keep);
    AppendU64(&payload, static_cast<std::uint64_t>(config.seed));
    AppendU32(&payload, FloatBits(config.repeat_penalty));
    AppendU64(&payload, config.repeat_last_n);
    AppendU32(&payload, FloatBits(config.frequency_penalty));
    AppendU32(&payload, FloatBits(config.presence_penalty));
  } else {
    const auto& instruction = command.instruction;
    AppendU32(&payload, static_cast<std::uint32_t>(instruction.op));
    AppendU64(&payload, instruction.index);
    AppendU32(&payload, instruction.state);
    AppendU32(&payload, static_cast<std::uint32_t>(instruction.token));
    AppendU32(&payload, instruction.offset);
    AppendU32(&payload, instruction.count);
    AppendU32(&payload, instruction.prompt_size);
    AppendU64(&payload, instruction.snapshot_id);
    AppendU64(&payload, instruction.digest);
    AppendU64(&payload, instruction.rng);
    AppendU32(&payload, static_cast<std::uint32_t>(instruction.pending));
    AppendU32(&payload, static_cast<std::uint32_t>(instruction.batch.size()));
    for (const auto& member : instruction.batch) {
      AppendU64(&payload, member.sequence);
      AppendU32(&payload, member.state);
      AppendU32(&payload, static_cast<std::uint32_t>(member.token));
    }
  }
  return SendFrame(kCommand, command.sequence, payload, error);
}

bool TpControlChannel::ReceiveCommand(TpControlCommand* command,
                                      std::string* error) {
  if (!handshaken_ || rank_ != 1 || command == nullptr) {
    SetError(error, "TP control command role is invalid");
    return false;
  }
  *command = {};
  std::uint64_t sequence = 0;
  if (!ReceiveFrame(kCommand, &sequence, &command_prompt_, error)) {
    return false;
  }
  const auto reject = [&](const char* message) {
    command_prompt_.clear();
    command_prompt_.shrink_to_fit();
    SetError(error, message);
    return false;
  };

  TpControlCommand parsed;
  parsed.sequence = sequence;
  std::size_t offset = 0;
  std::uint64_t embedded = 0;
  std::uint32_t kind = 0;
  if (!ReadU64(command_prompt_, &offset, &embedded, error) ||
      !ReadU32(command_prompt_, &offset, &kind, error) ||
      embedded != sequence) {
    return reject("TP control command header is invalid");
  }
  if (kind == static_cast<std::uint32_t>(TpControlCommandKind::kSingle)) {
    parsed.kind = TpControlCommandKind::kSingle;
    std::uint32_t cache_prompt = 0;
    std::uint32_t prompt_count = 0;
    std::uint32_t client_size = 0;
    std::uint32_t context_size = 0;
    if (!ReadU32(command_prompt_, &offset, &parsed.max_tokens, error) ||
        !ReadU32(command_prompt_, &offset, &cache_prompt, error) ||
        !ReadU32(command_prompt_, &offset, &parsed.cache_prefix_tokens,
                 error) ||
        !ReadU32(command_prompt_, &offset, &prompt_count, error) ||
        !ReadU32(command_prompt_, &offset, &client_size, error) ||
        !ReadU32(command_prompt_, &offset, &context_size, error) ||
        cache_prompt > 1 || prompt_count == 0 ||
        prompt_count > kMaxPromptTokens || client_size > kMaxClientIdBytes ||
        context_size > kMaxPromptContextBytes ||
        command_prompt_.size() - offset <
            std::size_t{prompt_count} * sizeof(std::uint32_t) + client_size +
                context_size) {
      return reject("TP control request is invalid");
    }
    parsed.cache_prompt = cache_prompt != 0;
    parsed.prompt_tokens.resize(prompt_count);
    for (auto& token : parsed.prompt_tokens) {
      std::uint32_t value = 0;
      (void)ReadU32(command_prompt_, &offset, &value, error);
      token = static_cast<std::int32_t>(value);
    }
    parsed.client_id.assign(
        reinterpret_cast<const char*>(command_prompt_.data() + offset),
        client_size);
    offset += client_size;
    parsed.prompt_context.assign(
        command_prompt_.begin() + static_cast<std::ptrdiff_t>(offset),
        command_prompt_.begin() +
            static_cast<std::ptrdiff_t>(offset + context_size));
    offset += context_size;
    auto& config = parsed.sampling;
    std::uint32_t temperature = 0;
    std::uint32_t top_k = 0;
    std::uint32_t top_p = 0;
    std::uint32_t min_p = 0;
    std::uint64_t min_keep = 0;
    std::uint64_t seed = 0;
    std::uint32_t repeat_penalty = 0;
    std::uint64_t repeat_last_n = 0;
    std::uint32_t frequency_penalty = 0;
    std::uint32_t presence_penalty = 0;
    if (!ReadU32(command_prompt_, &offset, &temperature, error) ||
        !ReadU32(command_prompt_, &offset, &top_k, error) ||
        !ReadU32(command_prompt_, &offset, &top_p, error) ||
        !ReadU32(command_prompt_, &offset, &min_p, error) ||
        !ReadU64(command_prompt_, &offset, &min_keep, error) ||
        !ReadU64(command_prompt_, &offset, &seed, error) ||
        !ReadU32(command_prompt_, &offset, &repeat_penalty, error) ||
        !ReadU64(command_prompt_, &offset, &repeat_last_n, error) ||
        !ReadU32(command_prompt_, &offset, &frequency_penalty, error) ||
        !ReadU32(command_prompt_, &offset, &presence_penalty, error)) {
      return reject("TP control request sampling is truncated");
    }
    config.temperature = BitsFloat(temperature);
    config.top_k = static_cast<std::int32_t>(top_k);
    config.top_p = BitsFloat(top_p);
    config.min_p = BitsFloat(min_p);
    config.min_keep = static_cast<std::size_t>(min_keep);
    config.seed = static_cast<std::int64_t>(seed);
    config.repeat_penalty = BitsFloat(repeat_penalty);
    config.repeat_last_n = static_cast<std::size_t>(repeat_last_n);
    config.frequency_penalty = BitsFloat(frequency_penalty);
    config.presence_penalty = BitsFloat(presence_penalty);
  } else if (kind ==
             static_cast<std::uint32_t>(TpControlCommandKind::kInstruction)) {
    parsed.kind = TpControlCommandKind::kInstruction;
    auto& instruction = parsed.instruction;
    std::uint32_t op = 0;
    std::uint32_t token_bits = 0;
    std::uint32_t pending_bits = 0;
    std::uint32_t batch_size = 0;
    if (!ReadU32(command_prompt_, &offset, &op, error) ||
        !ReadU64(command_prompt_, &offset, &instruction.index, error) ||
        !ReadU32(command_prompt_, &offset, &instruction.state, error) ||
        !ReadU32(command_prompt_, &offset, &token_bits, error) ||
        !ReadU32(command_prompt_, &offset, &instruction.offset, error) ||
        !ReadU32(command_prompt_, &offset, &instruction.count, error) ||
        !ReadU32(command_prompt_, &offset, &instruction.prompt_size, error) ||
        !ReadU64(command_prompt_, &offset, &instruction.snapshot_id, error) ||
        !ReadU64(command_prompt_, &offset, &instruction.digest, error) ||
        !ReadU64(command_prompt_, &offset, &instruction.rng, error) ||
        !ReadU32(command_prompt_, &offset, &pending_bits, error) ||
        !ReadU32(command_prompt_, &offset, &batch_size, error) ||
        op > static_cast<std::uint32_t>(TpInstructionOp::kPromptContext) ||
        batch_size > kMaxBatchMembers) {
      return reject("TP control instruction is invalid");
    }
    instruction.op = static_cast<TpInstructionOp>(op);
    instruction.token = static_cast<std::int32_t>(token_bits);
    instruction.pending = static_cast<std::int32_t>(pending_bits);
    instruction.batch.resize(batch_size);
    for (auto& member : instruction.batch) {
      std::uint32_t member_token = 0;
      if (!ReadU64(command_prompt_, &offset, &member.sequence, error) ||
          !ReadU32(command_prompt_, &offset, &member.state, error) ||
          !ReadU32(command_prompt_, &offset, &member_token, error)) {
        return reject("TP control instruction batch is truncated");
      }
      member.token = static_cast<std::int32_t>(member_token);
    }
  } else {
    return reject("TP control command kind is invalid");
  }
  if (offset != command_prompt_.size()) {
    return reject("TP control command payload has trailing bytes");
  }
  if (max_context_ == 0 || parsed.prompt_tokens.size() > max_context_) {
    return reject("TP control command exceeds the negotiated context");
  }
  if (!ValidateTpControlCommand(parsed, error)) {
    command_prompt_.clear();
    command_prompt_.shrink_to_fit();
    return false;
  }
  *command = std::move(parsed);
  // A request with images can be large; do not keep its frame around.
  if (command_prompt_.capacity() > (16U << 20)) {
    command_prompt_.clear();
    command_prompt_.shrink_to_fit();
  }
  return true;
}

bool TpControlChannel::SendResponse(const TpControlResponse& response,
                                    std::string* error) {
  if (!handshaken_ || rank_ != 1 ||
      !ValidateTpControlResponse(response, error)) {
    if (error != nullptr && error->empty()) {
      SetError(error, "TP control response is invalid");
    }
    return false;
  }
  std::vector<std::uint8_t> payload;
  payload.reserve(48 + response.tokens.size() * sizeof(std::uint32_t) +
                  response.error.size());
  AppendU64(&payload, response.sequence);
  AppendU32(&payload, static_cast<std::uint32_t>(response.kind));
  if (response.kind == TpControlResponseKind::kInstruction) {
    AppendU64(&payload, response.instruction_index);
  } else {
    AppendU32(&payload, static_cast<std::uint32_t>(response.tokens.size()));
    AppendU64(&payload, response.draft_tokens);
    AppendU64(&payload, response.draft_accepted_tokens);
    AppendU32(&payload, response.cached_prompt_tokens);
    AppendU64(&payload, response.cache_snapshot_bytes);
    for (const auto token : response.tokens) {
      AppendU32(&payload, static_cast<std::uint32_t>(token));
    }
  }
  AppendU32(&payload, static_cast<std::uint32_t>(response.error.size()));
  payload.insert(payload.end(), response.error.begin(), response.error.end());
  return SendFrame(kResponse, response.sequence, payload, error);
}

bool TpControlChannel::ReceiveResponse(TpControlResponse* response,
                                       std::string* error) {
  if (!handshaken_ || rank_ != 0 || response == nullptr) {
    SetError(error, "TP control response role is invalid");
    return false;
  }
  *response = {};
  std::uint64_t sequence = 0;
  if (!ReceiveFrame(kResponse, &sequence, &response_payload_, error)) {
    return false;
  }
  const auto reject = [&](const char* message) {
    response_payload_.clear();
    response_payload_.shrink_to_fit();
    SetError(error, message);
    return false;
  };

  TpControlResponse parsed;
  parsed.sequence = sequence;
  std::size_t offset = 0;
  std::uint64_t embedded = 0;
  std::uint32_t kind = 0;
  if (!ReadU64(response_payload_, &offset, &embedded, error) ||
      !ReadU32(response_payload_, &offset, &kind, error) ||
      embedded != sequence) {
    return reject("TP control response header is invalid");
  }
  if (kind == static_cast<std::uint32_t>(TpControlResponseKind::kInstruction)) {
    parsed.kind = TpControlResponseKind::kInstruction;
    if (!ReadU64(response_payload_, &offset, &parsed.instruction_index,
                 error)) {
      return reject("TP cache acknowledgement is malformed");
    }
  } else if (kind ==
             static_cast<std::uint32_t>(TpControlResponseKind::kSingle)) {
    parsed.kind = TpControlResponseKind::kSingle;
    std::uint32_t token_count = 0;
    if (!ReadU32(response_payload_, &offset, &token_count, error) ||
        !ReadU64(response_payload_, &offset, &parsed.draft_tokens, error) ||
        !ReadU64(response_payload_, &offset, &parsed.draft_accepted_tokens,
                 error) ||
        !ReadU32(response_payload_, &offset, &parsed.cached_prompt_tokens,
                 error) ||
        !ReadU64(response_payload_, &offset, &parsed.cache_snapshot_bytes,
                 error) ||
        token_count > kMaxPromptTokens ||
        response_payload_.size() - offset <
            std::size_t{token_count} * sizeof(std::uint32_t)) {
      return reject("TP control response is invalid");
    }
    parsed.tokens.resize(token_count);
    for (auto& token : parsed.tokens) {
      std::uint32_t value = 0;
      (void)ReadU32(response_payload_, &offset, &value, error);
      token = static_cast<std::int32_t>(value);
    }
  } else {
    return reject("TP control response kind is invalid");
  }
  std::uint32_t error_size = 0;
  if (!ReadU32(response_payload_, &offset, &error_size, error) ||
      error_size > kMaxErrorBytes ||
      response_payload_.size() - offset != error_size) {
    return reject("TP control response error payload is invalid");
  }
  parsed.error.assign(
      reinterpret_cast<const char*>(response_payload_.data() + offset),
      error_size);
  if (!ValidateTpControlResponse(parsed, error)) {
    response_payload_.clear();
    response_payload_.shrink_to_fit();
    return false;
  }
  *response = std::move(parsed);
  return true;
}

struct TpResponseBroker::Impl {
  struct Pending {
    TpControlResponse response;
    bool ready{false};
    bool delivered{false};
  };

  Impl(std::shared_ptr<TpControlChannel> channel, std::size_t capacity)
      : control(std::move(channel)), capacity(capacity) {
    if (!control || control->rank() != 0 || capacity == 0) {
      throw std::invalid_argument(
          "TP response broker requires a rank-zero channel and capacity");
    }
    reader = std::thread([this] { ReadLoop(); });
  }

  ~Impl() { Stop("TP response broker destroyed"); }

  void Stop(std::string reason) {
    const std::lock_guard<std::mutex> stop_lock(stop_mutex);
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (!stopping) {
        stopping = true;
        poisoned = true;
        failure = std::move(reason);
      }
      condition.notify_all();
    }
    if (control) {
      control->Interrupt();
    }
    if (reader.joinable()) {
      reader.join();
    }
  }

  void ReadLoop() {
    for (;;) {
      TpControlResponse response;
      std::string error;
      if (!control->ReceiveResponse(&response, &error)) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!stopping) {
          stopping = true;
          poisoned = true;
          failure = "TP worker response receive failed: " + error;
        }
        condition.notify_all();
        return;
      }
      {
        std::lock_guard<std::mutex> lock(mutex);
        if (stopping) {
          return;
        }
        if (response.kind == TpControlResponseKind::kInstruction) {
          if (!ack_expected || ack_ready || ack_sequence != response.sequence ||
              ack_index != response.instruction_index) {
            stopping = poisoned = true;
            failure = "TP cache acknowledgement is unknown or duplicate";
            condition.notify_all();
            control->Interrupt();
            return;
          }
          ack_error = std::move(response.error);
          ack_ready = true;
          condition.notify_all();
          continue;
        }
        if (ack_expected && ack_sequence == response.sequence) {
          stopping = poisoned = true;
          failure =
              "TP final response arrived before cache acknowledgement was "
              "consumed";
          condition.notify_all();
          control->Interrupt();
          return;
        }
        const auto found = pending.find(response.sequence);
        if (found == pending.end() || found->second->ready) {
          stopping = true;
          poisoned = true;
          failure = "TP response sequence is unknown or duplicate: " +
                    std::to_string(response.sequence);
          condition.notify_all();
          control->Interrupt();
          return;
        }
        found->second->response = std::move(response);
        found->second->ready = true;
        condition.notify_all();
      }
    }
  }

  std::shared_ptr<TpControlChannel> control;
  const std::size_t capacity;
  std::mutex mutex;
  std::condition_variable condition;
  std::mutex stop_mutex;
  std::unordered_map<std::uint64_t, std::shared_ptr<Pending>> pending;
  bool ack_expected{false};
  bool ack_ready{false};
  std::uint64_t ack_sequence{0}, ack_index{0};
  std::string ack_error;
  bool stopping{false};
  bool poisoned{false};
  std::string failure;
  std::thread reader;
};

TpResponseBroker::TpResponseBroker(std::shared_ptr<TpControlChannel> control,
                                   std::size_t max_pending_responses)
    : impl_(std::make_unique<Impl>(std::move(control), max_pending_responses)) {
}

TpResponseBroker::~TpResponseBroker() {
  impl_.reset();
}

bool TpResponseBroker::RegisterInstruction(std::uint64_t sequence,
                                           std::uint64_t index,
                                           std::string* error) {
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->stopping || impl_->poisoned || impl_->ack_expected ||
      !impl_->pending.contains(sequence)) {
    SetError(error, "TP cache acknowledgement cannot be registered: " +
                        impl_->failure);
    return false;
  }
  impl_->ack_sequence = sequence;
  impl_->ack_index = index;
  impl_->ack_expected = true;
  impl_->ack_ready = false;
  impl_->ack_error.clear();
  return true;
}

bool TpResponseBroker::WaitForInstruction(std::string* error) {
  std::unique_lock<std::mutex> lock(impl_->mutex);
  if (!impl_->ack_expected) {
    SetError(error, "TP cache acknowledgement is not registered");
    return false;
  }
  impl_->condition.wait(lock, [&] {
    return impl_->ack_ready || impl_->stopping || impl_->poisoned;
  });
  impl_->ack_expected = false;
  if (impl_->stopping || impl_->poisoned) {
    SetError(error, impl_->failure);
    return false;
  }
  SetError(error, impl_->ack_error);
  return impl_->ack_error.empty();
}

bool TpResponseBroker::RegisterPendingResponse(std::uint64_t sequence,
                                               std::string* error) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->stopping || impl_->poisoned) {
    SetError(error, impl_->failure.empty() ? "TP response broker is stopped"
                                           : impl_->failure);
    return false;
  }
  if (impl_->pending.size() >= impl_->capacity) {
    SetError(error, "TP response broker capacity is exhausted");
    return false;
  }
  if (!impl_->pending.emplace(sequence, std::make_shared<Impl::Pending>())
           .second) {
    SetError(error, "TP response sequence is already registered");
    return false;
  }
  return true;
}

bool TpResponseBroker::CancelUnsentResponse(std::uint64_t sequence,
                                            std::string* error) {
  bool interrupt = false;
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto found = impl_->pending.find(sequence);
    if (found == impl_->pending.end()) {
      SetError(error, "TP response sequence is not registered");
      return false;
    }
    if (found->second->ready) {
      impl_->stopping = true;
      impl_->poisoned = true;
      impl_->failure = "TP response arrived before command send completed: " +
                       std::to_string(sequence);
      impl_->condition.notify_all();
      SetError(error, impl_->failure);
      interrupt = true;
    } else {
      impl_->pending.erase(found);
    }
  }
  if (interrupt) {
    impl_->control->Interrupt();
  }
  return !interrupt;
}

bool TpResponseBroker::WaitForResponse(std::uint64_t sequence,
                                       TpControlResponse* response,
                                       std::string* error) {
  if (response == nullptr) {
    SetError(error, "TP response output is null");
    return false;
  }
  std::unique_lock<std::mutex> lock(impl_->mutex);
  const auto found = impl_->pending.find(sequence);
  if (found == impl_->pending.end()) {
    SetError(error, "TP response sequence is not registered");
    return false;
  }
  const auto pending = found->second;
  impl_->condition.wait(lock, [&] {
    return pending->ready || impl_->poisoned || impl_->stopping;
  });
  if (pending->delivered) {
    SetError(error, "TP response sequence was already consumed");
    return false;
  }
  if (impl_->poisoned || impl_->stopping) {
    SetError(error, impl_->failure.empty() ? "TP response broker is stopped"
                                           : impl_->failure);
    return false;
  }
  if (pending->ready) {
    pending->delivered = true;
    *response = std::move(pending->response);
    impl_->pending.erase(found);
    return true;
  }
  SetError(error, impl_->failure.empty() ? "TP response broker is stopped"
                                         : impl_->failure);
  return false;
}

void TpResponseBroker::FailAll(std::string reason) {
  if (impl_) {
    impl_->Stop(std::move(reason));
  }
}

std::uint16_t TpControlChannel::port() const noexcept {
  return port_;
}
std::uint32_t TpControlChannel::rank() const noexcept {
  return rank_;
}
std::uint32_t TpControlChannel::world_size() const noexcept {
  return world_size_;
}

}  // namespace gufo::server
