#include "src/models/qwen38_flash_next/kernels/rocm/verbs.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <infiniband/verbs.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <utility>

#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif
#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

constexpr std::uint32_t kWireMagic = 0x47554654U;  // "GUFT"
constexpr std::uint32_t kWireVersion = 6;
constexpr std::uint32_t kCollectiveMagic = 0x47554348U;  // "GUCH"
constexpr std::uint32_t kCollectiveVersion = 5;
constexpr std::uint32_t kReadyMagic = 0x47555244U;  // "GURD"
// Exchanges use the send and receive windows in turn. Three, because the add
// reading an overlapped exchange's partial may still be queued when the
// exchange after the next one lands (see Communicator), and a queued
// exchange's partial is staged while the previous one is in flight. A window
// starts with the writer's header; the partial follows at kDataOffset.
constexpr std::size_t kWindowBytes = 32U << 20;
constexpr std::size_t kWindows = 3;
constexpr std::size_t kBufferBytes = kWindows * kWindowBytes;
constexpr std::size_t kDataOffset = 64;
constexpr std::uint32_t kPostedReceives = 4;
constexpr std::uint64_t kWriteTag = 1ULL << 63;
constexpr std::uintptr_t kSendAddress = 0x0000700000000000ULL;
constexpr std::uintptr_t kRecvAddress = kSendAddress + kBufferBytes;
// Overlapped exchanges started and not yet finished; each has its own
// readiness event.
constexpr std::size_t kStartedExchanges = 4;
// Queued exchanges not yet complete; a forward queues one per layer.
constexpr std::size_t kQueuedExchanges = 4096;
// A GPU wait gives up well after the exchange thread's own timeout.
constexpr std::uint64_t kWaitSeconds = 120;
// Flag words, a cache line apart: written by the GPU, by the host, and by a
// GPU wait that timed out.
constexpr std::size_t kReadyOffset = 0;
constexpr std::size_t kArrivedOffset = 64;
constexpr std::size_t kTimedOutOffset = 128;
constexpr std::size_t kFlagBytes = 4096;
constexpr std::uint32_t kPort = 1;
constexpr auto kCollectiveTimeout = std::chrono::seconds(30);
constexpr auto kReceiveSpinWindow = std::chrono::microseconds(500);
// An exchange waits for a layer of GPU work, or a peer that is a layer behind:
// spinning that long keeps the wake-up latency off every decode exchange.
constexpr auto kExchangeSpinWindow = std::chrono::milliseconds(5);
constexpr int kMemoryAccessFlags =
    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;

// Eases a spin loop off the core's shared resources and the SoC power budget,
// which the GPU draws from too.
inline void CpuRelax() noexcept {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#endif
}

void SetError(std::string* error_msg, std::string message) {
  if (error_msg != nullptr) {
    *error_msg = std::move(message);
  }
}

std::string SystemError(const char* operation) {
  return std::string(operation) + ": " + std::strerror(errno);
}

bool MapFixedBuffer(std::uintptr_t address, std::size_t bytes, void** buffer,
                    std::string* error) {
  void* mapped =
      ::mmap(reinterpret_cast<void*>(address), bytes, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
  if (mapped == MAP_FAILED) {
    SetError(error, "fixed RDMA buffer mmap failed: " +
                        std::string(std::strerror(errno)));
    return false;
  }
  if (reinterpret_cast<std::uintptr_t>(mapped) != address) {
    ::munmap(mapped, bytes);
    SetError(error, "fixed RDMA buffer address mismatch");
    return false;
  }
  if (hipHostRegister(mapped, bytes, hipHostRegisterMapped) != hipSuccess) {
    ::munmap(mapped, bytes);
    SetError(error, "hipHostRegister for fixed RDMA buffer failed");
    return false;
  }
  *buffer = mapped;
  return true;
}

void UnmapFixedBuffer(void* buffer, std::size_t bytes) {
  if (buffer == nullptr) {
    return;
  }
  (void)hipHostUnregister(buffer);
  (void)::munmap(buffer, bytes);
}

class Socket {
public:
  Socket() = default;
  explicit Socket(int fd) : fd_(fd) {}
  Socket(const Socket&) = delete;
  Socket& operator=(const Socket&) = delete;
  Socket(Socket&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
  Socket& operator=(Socket&& other) noexcept {
    if (this != &other) {
      Close();
      fd_ = std::exchange(other.fd_, -1);
    }
    return *this;
  }
  ~Socket() { Close(); }

  [[nodiscard]] bool valid() const noexcept { return fd_ >= 0; }
  [[nodiscard]] int fd() const noexcept { return fd_; }

  void Close() noexcept {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

  [[nodiscard]] static Socket Listen(const std::string& host,
                                     std::uint16_t port, std::string* error) {
    struct addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    const std::string service = std::to_string(port);
    struct addrinfo* addresses = nullptr;
    const int result = ::getaddrinfo(host.empty() ? nullptr : host.c_str(),
                                     service.c_str(), &hints, &addresses);
    if (result != 0) {
      SetError(error,
               "bootstrap getaddrinfo: " + std::string(::gai_strerror(result)));
      return {};
    }
    int fd = -1;
    for (auto* address = addresses; address != nullptr;
         address = address->ai_next) {
      fd = ::socket(address->ai_family, address->ai_socktype,
                    address->ai_protocol);
      if (fd < 0) {
        continue;
      }
      int reuse = 1;
      (void)::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
      if (::bind(fd, address->ai_addr, address->ai_addrlen) == 0 &&
          ::listen(fd, 1) == 0) {
        break;
      }
      ::close(fd);
      fd = -1;
    }
    ::freeaddrinfo(addresses);
    if (fd < 0) {
      SetError(error, SystemError("bootstrap listen"));
      return {};
    }
    SetTimeouts(fd);
    pollfd poll_fd{.fd = fd, .events = POLLIN, .revents = 0};
    const int poll_result = ::poll(&poll_fd, 1, 30000);
    if (poll_result <= 0) {
      SetError(error, poll_result == 0 ? "bootstrap accept timed out"
                                       : SystemError("bootstrap poll"));
      ::close(fd);
      return {};
    }
    int peer = -1;
    do {
      peer = ::accept(fd, nullptr, nullptr);
    } while (peer < 0 && errno == EINTR);
    if (peer < 0) {
      SetError(error, SystemError("bootstrap accept"));
      ::close(fd);
      return {};
    }
    Socket listener(fd);
    SetTimeouts(peer);
    return Socket(peer);
  }

  [[nodiscard]] static Socket Connect(const std::string& host,
                                      std::uint16_t port, std::string* error) {
    if (host.empty()) {
      SetError(error, "rank one requires a bootstrap host");
      return {};
    }
    const auto deadline = std::chrono::steady_clock::now() + kCollectiveTimeout;
    for (;;) {
      struct addrinfo hints{};
      hints.ai_family = AF_INET;
      hints.ai_socktype = SOCK_STREAM;
      const std::string service = std::to_string(port);
      struct addrinfo* addresses = nullptr;
      const int result =
          ::getaddrinfo(host.c_str(), service.c_str(), &hints, &addresses);
      if (result != 0) {
        SetError(error, "bootstrap getaddrinfo: " +
                            std::string(::gai_strerror(result)));
        return {};
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
          pollfd poll_fd{.fd = fd, .events = POLLOUT, .revents = 0};
          const int poll_result = ::poll(&poll_fd, 1, 1000);
          int socket_error = 0;
          socklen_t error_size = sizeof(socket_error);
          connected = poll_result > 0 &&
                      ::getsockopt(fd, SOL_SOCKET, SO_ERROR, &socket_error,
                                   &error_size) == 0 &&
                      socket_error == 0;
        }
        if (connected) {
          if (::fcntl(fd, F_SETFL, flags) != 0) {
            ::close(fd);
            fd = -1;
            continue;
          }
          break;
        }
        ::close(fd);
        fd = -1;
      }
      ::freeaddrinfo(addresses);
      if (fd >= 0) {
        SetTimeouts(fd);
        return Socket(fd);
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        SetError(error, SystemError("bootstrap connect"));
        return {};
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }

  [[nodiscard]] bool SendAll(const void* data, std::size_t bytes,
                             std::string* error) const {
    const auto* cursor = static_cast<const std::uint8_t*>(data);
    std::size_t sent = 0;
    while (sent < bytes) {
      const ssize_t n = ::send(fd_, cursor + sent, bytes - sent, MSG_NOSIGNAL);
      if (n < 0 && errno == EINTR) {
        continue;
      }
      if (n <= 0) {
        SetError(error, SystemError("bootstrap send"));
        return false;
      }
      sent += static_cast<std::size_t>(n);
    }
    return true;
  }

  [[nodiscard]] bool RecvAll(void* data, std::size_t bytes,
                             std::string* error) const {
    auto* cursor = static_cast<std::uint8_t*>(data);
    std::size_t received = 0;
    // A collective peer usually answers within a few hundred microseconds.
    // Polling that long avoids the scheduler wake-up a blocking receive pays
    // on every exchange; a slower peer falls back to the blocking receive.
    const auto spin_deadline =
        std::chrono::steady_clock::now() + kReceiveSpinWindow;
    while (received < bytes) {
      const bool spinning = std::chrono::steady_clock::now() < spin_deadline;
      const ssize_t n = ::recv(fd_, cursor + received, bytes - received,
                               spinning ? MSG_DONTWAIT : 0);
      if (n < 0 && spinning && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        continue;
      }
      if (n < 0 && errno == EINTR) {
        continue;
      }
      if (n <= 0) {
        SetError(error, SystemError("bootstrap receive"));
        return false;
      }
      received += static_cast<std::size_t>(n);
    }
    return true;
  }

private:
  static void SetTimeouts(int fd) noexcept {
    struct timeval timeout{};
    timeout.tv_sec = 30;
    timeout.tv_usec = 0;
    (void)::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    (void)::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    const int no_delay = 1;
    (void)::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &no_delay,
                       sizeof(no_delay));
  }

  int fd_{-1};
};

struct Wire {
  std::uint32_t magic{kWireMagic};
  std::uint32_t version{kWireVersion};
  std::uint32_t rank{0};
  std::uint32_t world_size{0};
  std::uint32_t qp_num{0};
  std::uint16_t lid{0};
  std::uint8_t mtu{0};
  std::uint8_t gid_index{0};
  std::uint8_t link_layer{0};
  std::uint16_t port_num{kPort};
  ibv_gid sgid{};
  std::uint64_t recv_address{0};
  std::uint32_t recv_rkey{0};
};

struct CollectiveHeader {
  std::uint32_t magic{kCollectiveMagic};
  std::uint32_t version{kCollectiveVersion};
  std::uint64_t scope_id{0};
  std::uint64_t operation_id{0};
  std::uint64_t bytes{0};
};

static_assert(sizeof(CollectiveHeader) <= kDataOffset);

class Ibrverbs final : public Communicator {
public:
  explicit Ibrverbs(const IbrverbsConfig& config) : config_(config) {}
  ~Ibrverbs() override { Cleanup(); }

  Ibrverbs(const Ibrverbs&) = delete;
  Ibrverbs& operator=(const Ibrverbs&) = delete;

  [[nodiscard]] bool Initialize(std::string* error) {
    if (config_.world_size != 2 || config_.rank > 1 ||
        config_.bootstrap_port == 0 ||
        config_.gid_index > std::numeric_limits<std::uint8_t>::max() ||
        config_.device_index >
            static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
      SetError(error,
               "verbs communicator requires two ranks and a bootstrap port");
      return false;
    }
    if (hipSetDevice(static_cast<int>(config_.device_index)) != hipSuccess) {
      SetError(error, "hipSetDevice for verbs communicator failed");
      return false;
    }
    int device_count = 0;
    ibv_device** devices = ibv_get_device_list(&device_count);
    if (devices == nullptr || device_count <= config_.device_index) {
      if (devices != nullptr) {
        ibv_free_device_list(devices);
      }
      SetError(error, "no usable InfiniBand device");
      return false;
    }
    context_ = ibv_open_device(devices[config_.device_index]);
    ibv_free_device_list(devices);
    if (context_ == nullptr) {
      SetError(error, "ibv_open_device failed");
      return false;
    }
    pd_ = ibv_alloc_pd(context_);
    if (pd_ == nullptr) {
      SetError(error, "ibv_alloc_pd failed");
      return false;
    }
    completion_channel_ = ibv_create_comp_channel(context_);
    cq_ = ibv_create_cq(context_, 16, nullptr, completion_channel_, 0);
    if (cq_ == nullptr) {
      SetError(error, "ibv_create_cq failed");
      return false;
    }

    ibv_port_attr port{};
    if (ibv_query_port(context_, kPort, &port) != 0 ||
        port.state != IBV_PORT_ACTIVE ||
        port.link_layer != IBV_LINK_LAYER_INFINIBAND) {
      SetError(error, "native InfiniBand port 1 is not active");
      return false;
    }
    if (ibv_query_gid(context_, kPort, config_.gid_index, &local_sgid_) != 0) {
      SetError(error, "ibv_query_gid failed");
      return false;
    }

    if (!MapFixedBuffer(kSendAddress, kBufferBytes, &send_buffer_, error)) {
      return false;
    }
    send_registered_ = true;
    if (!MapFixedBuffer(kRecvAddress, kBufferBytes, &recv_buffer_, error)) {
      return false;
    }
    recv_registered_ = true;
    // The GPU stages queued partials straight into the send windows and adds
    // the peer's partial straight from the receive windows. That read relies
    // on platform coherence: on these APU hosts the GPU sees the NIC's DMA
    // into pinned host memory through the registered mapping with no explicit
    // invalidation (qualified runs agree token-for-token across the ranks).
    // A GPU that caches host mappings would need coherent windows here, as
    // the flags below request explicitly.
    if (hipHostGetDevicePointer(&send_device_, send_buffer_, 0) != hipSuccess ||
        send_device_ == nullptr ||
        hipHostGetDevicePointer(&recv_device_, recv_buffer_, 0) != hipSuccess ||
        recv_device_ == nullptr) {
      SetError(error, "exchange windows have no device mapping");
      return false;
    }
    if (hipHostMalloc(&flags_, kFlagBytes,
                      hipHostMallocMapped | hipHostMallocCoherent) !=
            hipSuccess ||
        hipHostGetDevicePointer(&flags_device_, flags_, 0) != hipSuccess ||
        hipMalloc(&arrivals_, sizeof(std::uint32_t)) != hipSuccess ||
        hipMemset(arrivals_, 0, sizeof(std::uint32_t)) != hipSuccess) {
      SetError(error, "exchange flag allocation failed");
      return false;
    }
    std::memset(flags_, 0, kFlagBytes);
    int clock_khz = 0;
    if (hipDeviceGetAttribute(&clock_khz, hipDeviceAttributeWallClockRate,
                              static_cast<int>(config_.device_index)) !=
            hipSuccess ||
        clock_khz <= 0) {
      SetError(error, "device wall clock rate is unknown");
      return false;
    }
    wait_ticks_ = static_cast<std::uint64_t>(clock_khz) * 1000 * kWaitSeconds;
    send_mr_ = ibv_reg_mr(pd_, send_buffer_, kBufferBytes, kMemoryAccessFlags);
    recv_mr_ = ibv_reg_mr(pd_, recv_buffer_, kBufferBytes, kMemoryAccessFlags);
    if (send_mr_ == nullptr || recv_mr_ == nullptr) {
      SetError(error, "ibv_reg_mr for verbs buffers failed");
      return false;
    }

    struct ibv_qp_init_attr init{};
    init.send_cq = cq_;
    init.recv_cq = cq_;
    init.cap.max_send_wr = 16;
    init.cap.max_recv_wr = 16;
    init.cap.max_send_sge = 1;
    init.cap.max_recv_sge = 1;
    init.qp_type = IBV_QPT_RC;
    init.sq_sig_all = 0;
    qp_ = ibv_create_qp(pd_, &init);
    if (qp_ == nullptr) {
      SetError(error, "ibv_create_qp failed");
      return false;
    }

    struct ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = kPort;
    attr.qp_access_flags = kMemoryAccessFlags;
    if (ibv_modify_qp(qp_, &attr,
                      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                          IBV_QP_ACCESS_FLAGS) != 0) {
      SetError(error, "ibv_modify_qp INIT failed");
      return false;
    }

    Wire local{};
    local.rank = config_.rank;
    local.world_size = config_.world_size;
    local.qp_num = qp_->qp_num;
    local.lid = port.lid;
    local.mtu = static_cast<std::uint8_t>(port.active_mtu);
    local.gid_index = static_cast<std::uint8_t>(config_.gid_index);
    local.link_layer = static_cast<std::uint8_t>(port.link_layer);
    local.sgid = local_sgid_;
    // The peer writes each partial into this rank's receive windows.
    local.recv_address = reinterpret_cast<std::uint64_t>(recv_buffer_);
    local.recv_rkey = recv_mr_->rkey;

    Socket control = config_.rank == 0
                         ? Socket::Listen(config_.bootstrap_host,
                                          config_.bootstrap_port, error)
                         : Socket::Connect(config_.bootstrap_host,
                                           config_.bootstrap_port, error);
    if (!control.valid()) {
      return false;
    }
    if (!control.SendAll(&local, sizeof(local), error)) {
      return false;
    }
    Wire remote{};
    if (!control.RecvAll(&remote, sizeof(remote), error)) {
      return false;
    }
    if (remote.magic != kWireMagic || remote.version != kWireVersion ||
        remote.world_size != config_.world_size ||
        remote.rank != 1U - config_.rank || remote.qp_num == 0 ||
        remote.port_num != kPort ||
        remote.link_layer != IBV_LINK_LAYER_INFINIBAND ||
        remote.mtu < IBV_MTU_256 || remote.mtu > IBV_MTU_4096 ||
        remote.recv_address != kRecvAddress || remote.recv_rkey == 0) {
      SetError(error, "verbs bootstrap metadata is invalid");
      return false;
    }

    const auto mtu = static_cast<ibv_mtu>(std::min(
        static_cast<unsigned>(local.mtu), static_cast<unsigned>(remote.mtu)));
    attr = {};
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = mtu;
    attr.dest_qp_num = remote.qp_num;
    attr.rq_psn = 0;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.dlid = remote.lid;
    attr.ah_attr.grh.hop_limit = 1;
    attr.ah_attr.grh.sgid_index = local.gid_index;
    attr.ah_attr.is_global = 1;
    attr.ah_attr.port_num = kPort;
    std::memcpy(&attr.ah_attr.grh.dgid, &remote.sgid, sizeof(remote.sgid));
    if (ibv_modify_qp(qp_, &attr,
                      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                          IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                          IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) !=
        0) {
      SetError(error, "ibv_modify_qp RTR failed");
      return false;
    }
    attr = {};
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = 0;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.max_rd_atomic = 1;
    if (ibv_modify_qp(qp_, &attr,
                      IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT |
                          IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                          IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
      SetError(error, "ibv_modify_qp RTS failed");
      return false;
    }
    remote_recv_address_ = remote.recv_address;
    remote_recv_rkey_ = remote.recv_rkey;
    // Each exchange arrives as a write with immediate data, which consumes a
    // posted receive. Post them before the peer can be told this rank is
    // ready.
    for (std::uint32_t i = 0; i < kPostedReceives; ++i) {
      if (!PostReceive(error)) {
        return false;
      }
    }
    const std::uint32_t ready = kReadyMagic;
    std::uint32_t peer_ready = 0;
    if (!control.SendAll(&ready, sizeof(ready), error) ||
        !control.RecvAll(&peer_ready, sizeof(peer_ready), error) ||
        peer_ready != kReadyMagic) {
      SetError(error, "verbs post-RTS readiness exchange failed");
      return false;
    }
    control_ = std::make_unique<Socket>(std::move(control));
    return true;
  }

  [[nodiscard]] std::uint32_t rank() const noexcept override {
    return config_.rank;
  }
  [[nodiscard]] std::uint32_t world_size() const noexcept override {
    return config_.world_size;
  }
  [[nodiscard]] int device_index() const noexcept override {
    return static_cast<int>(config_.device_index);
  }

  [[nodiscard]] bool BeginOperation(std::uint64_t scope_id,
                                    std::string* error) override {
    std::lock_guard lock(mutex_);
    if (qp_ == nullptr || control_ == nullptr) {
      SetError(error, "verbs communicator is not initialized");
      return false;
    }
    if (poisoned_) {
      PoisonedError(error);
      return false;
    }
    if (bound_scope_.has_value()) {
      Poison(error, "another verbs operation scope is already bound");
      return false;
    }
    bound_scope_ = scope_id;
    return true;
  }

  [[nodiscard]] bool EndOperation(std::uint64_t scope_id,
                                  std::string* error) override {
    std::lock_guard lock(mutex_);
    if (poisoned_) {
      PoisonedError(error);
      return false;
    }
    if (!bound_scope_.has_value()) {
      SetError(error, "verbs operation scope is not bound");
      return false;
    }
    if (*bound_scope_ != scope_id) {
      Poison(error, "verbs operation scope end mismatch: bound=" +
                        std::to_string(*bound_scope_) +
                        " requested=" + std::to_string(scope_id));
      return false;
    }
    if (started_ != 0 || queued_ != 0) {
      Poison(error, "verbs operation scope ended with exchanges in flight");
      return false;
    }
    bound_scope_.reset();
    return true;
  }

  [[nodiscard]] const float* ExchangePartial(const float* data,
                                             std::size_t bytes,
                                             hipStream_t stream,
                                             std::string* error) override {
    std::lock_guard lock(mutex_);
    if (!CheckExchange(data, bytes, error)) {
      return nullptr;
    }
    if (started_ != 0 || queued_ != 0) {
      Poison(error, "verbs exchange started with overlapped ones in flight");
      return nullptr;
    }
    // Synchronizing retires this rank's adds of earlier exchanges, whose
    // windows the peer's next writes reuse.
    if (bytes != 0 &&
        (hipStreamSynchronize(stream) != hipSuccess ||
         hipMemcpy(SendWindow(next_operation_) + kDataOffset, data, bytes,
                   hipMemcpyDeviceToHost) != hipSuccess)) {
      Poison(error, "HIP device-to-host all-reduce staging failed");
      return nullptr;
    }
    std::string transfer_error;
    const float* peer =
        Transfer(*bound_scope_, next_operation_, bytes, &transfer_error);
    if (peer == nullptr) {
      Poison(error, std::move(transfer_error));
      return nullptr;
    }
    ++next_operation_;
    return peer;
  }

  [[nodiscard]] bool StartPartial(const float* data, std::size_t bytes,
                                  hipStream_t stream,
                                  std::string* error) override {
    std::lock_guard lock(mutex_);
    if (!CheckExchange(data, bytes, error)) {
      return false;
    }
    if (started_ == kStartedExchanges) {
      Poison(error, "too many overlapped verbs exchanges in flight");
      return false;
    }
    if (std::string worker_error; !StartWorker(&worker_error)) {
      Poison(error, std::move(worker_error));
      return false;
    }
    // At most kStartedExchanges are in flight, so the exchange that last used
    // this event has finished and the worker no longer waits on it.
    const std::uint64_t operation_id = next_operation_;
    hipEvent_t ready = ready_events_[operation_id % kStartedExchanges];
    if (hipEventRecord(ready, stream) != hipSuccess) {
      Poison(error, "overlapped exchange event record failed");
      return false;
    }
    {
      std::lock_guard jobs(jobs_mutex_);
      jobs_.push_back({.ready = ready,
                       .data = data,
                       .bytes = bytes,
                       .scope_id = *bound_scope_,
                       .operation_id = operation_id});
    }
    ++next_operation_;
    ++started_;
    jobs_ready_.notify_one();
    return true;
  }

  [[nodiscard]] const float* FinishPartial(std::size_t bytes,
                                           std::string* error) override {
    {
      std::lock_guard lock(mutex_);
      if (started_ == 0) {
        Poison(error, "no overlapped verbs exchange is in flight");
        return nullptr;
      }
    }
    Finished finished;
    {
      std::unique_lock jobs(jobs_mutex_);
      finished_ready_.wait(jobs, [this] { return !finished_.empty(); });
      finished = std::move(finished_.front());
      finished_.pop_front();
    }
    std::lock_guard lock(mutex_);
    --started_;
    if (finished.peer != nullptr && finished.bytes != bytes) {
      finished.peer = nullptr;
      finished.error = "overlapped verbs exchange size mismatch";
    }
    if (finished.peer == nullptr) {
      Poison(error, finished.error);
    }
    return finished.peer;
  }

  [[nodiscard]] bool QueuePartial(std::size_t bytes, QueuedPartial* partial,
                                  std::string* error) override {
    std::lock_guard lock(mutex_);
    if (!CheckExchange(static_cast<const float*>(send_device_), bytes, error)) {
      return false;
    }
    if (started_ != 0 || queued_ == kQueuedExchanges) {
      Poison(error, "verbs exchange queued with too many in flight");
      return false;
    }
    if (std::string worker_error; !StartWorker(&worker_error)) {
      Poison(error, std::move(worker_error));
      return false;
    }
    const std::uint64_t operation_id = next_operation_;
    const std::size_t window = (operation_id % kWindows) * kWindowBytes;
    auto* flags = static_cast<std::uint8_t*>(flags_device_);
    *partial = {
        .send = reinterpret_cast<float*>(
            static_cast<std::uint8_t*>(send_device_) + window + kDataOffset),
        .peer = reinterpret_cast<const float*>(
            static_cast<std::uint8_t*>(recv_device_) + window + kDataOffset),
        .ready = reinterpret_cast<std::uint64_t*>(flags + kReadyOffset),
        .arrived =
            reinterpret_cast<const std::uint64_t*>(flags + kArrivedOffset),
        // Flags start at zero, so exchange k publishes k + 1.
        .value = operation_id + 1,
        .arrivals = static_cast<std::uint32_t*>(arrivals_),
        .timed_out = reinterpret_cast<std::uint32_t*>(flags + kTimedOutOffset),
        .wait_ticks = wait_ticks_,
    };
    {
      std::lock_guard jobs(jobs_mutex_);
      jobs_.push_back({.bytes = bytes,
                       .scope_id = *bound_scope_,
                       .operation_id = operation_id,
                       .queued = true});
    }
    ++next_operation_;
    ++queued_;
    jobs_ready_.notify_one();
    return true;
  }

  [[nodiscard]] bool CheckQueued(std::string* error) override {
    std::lock_guard lock(mutex_);
    if (!poisoned_ &&
        Flag<std::uint32_t>(kTimedOutOffset).load(std::memory_order_acquire) !=
            0) {
      poisoned_ = true;
      if (first_error_.empty()) {
        first_error_ = "a GPU wait for a queued verbs exchange timed out";
      }
    }
    if (poisoned_) {
      PoisonedError(error);
      return false;
    }
    return true;
  }

private:
  struct Job {
    hipEvent_t ready{nullptr};
    const float* data{nullptr};
    std::size_t bytes{0};
    std::uint64_t scope_id{0};
    std::uint64_t operation_id{0};
    /// A QueuePartial exchange: the GPU stages it and waits for it.
    bool queued{false};
  };
  struct Finished {
    const float* peer{nullptr};
    std::size_t bytes{0};
    std::string error;
  };

  /// Poisons the communicator under `mutex_`, keeping the first failure so
  /// later calls report why the sequence cannot continue.
  void Poison(std::string* error, std::string message) {
    poisoned_ = true;
    if (first_error_.empty()) {
      first_error_ = message;
    }
    SetError(error, std::move(message));
  }

  /// Reports a call on the poisoned communicator, under `mutex_`.
  void PoisonedError(std::string* error) const {
    SetError(error, first_error_.empty()
                        ? "verbs communicator is poisoned"
                        : "verbs communicator is poisoned: " + first_error_);
  }

  /// Validates an exchange of `bytes` at `data` under `mutex_`, poisoning the
  /// communicator when the sequence cannot continue.
  [[nodiscard]] bool CheckExchange(const float* data, std::size_t bytes,
                                   std::string* error) {
    if (qp_ == nullptr) {
      SetError(error, "verbs communicator is not initialized");
      return false;
    }
    if (control_ == nullptr) {
      SetError(error, "verbs control channel is not initialized");
      return false;
    }
    if (poisoned_) {
      PoisonedError(error);
      return false;
    }
    if (!bound_scope_.has_value()) {
      Poison(error, "verbs collective has no bound operation scope");
      return false;
    }
    if (bytes % sizeof(float) != 0 || (bytes != 0 && data == nullptr)) {
      Poison(error, "all-reduce buffer is invalid");
      return false;
    }
    if (next_operation_ == std::numeric_limits<std::uint64_t>::max()) {
      Poison(error, "verbs collective operation sequence exhausted");
      return false;
    }
    if (bytes > kWindowBytes - kDataOffset) {
      Poison(error, "all-reduce payload exceeds the receive window");
      return false;
    }
    return true;
  }

  [[nodiscard]] std::uint8_t* SendWindow(std::uint64_t operation_id) const {
    return static_cast<std::uint8_t*>(send_buffer_) +
           (operation_id % kWindows) * kWindowBytes;
  }

  template<typename T>
  [[nodiscard]] std::atomic_ref<T> Flag(std::size_t offset) const {
    return std::atomic_ref<T>(
        *reinterpret_cast<T*>(static_cast<std::uint8_t*>(flags_) + offset));
  }

  /// Waits until the GPU has published the partial of queued exchange
  /// `value - 1`: spins briefly, then polls.
  [[nodiscard]] bool WaitReady(std::uint64_t value, std::string* error) {
    const auto start = std::chrono::steady_clock::now();
    const auto spin_deadline = start + kExchangeSpinWindow;
    const auto deadline = start + kCollectiveTimeout;
    while (Flag<std::uint64_t>(kReadyOffset).load(std::memory_order_acquire) <
           value) {
      if (stopping_.load(std::memory_order_relaxed)) {
        SetError(error, "verbs communicator stopped");
        return false;
      }
      const auto now = std::chrono::steady_clock::now();
      if (now >= deadline) {
        SetError(error, "verbs exchange timed out waiting for the GPU");
        return false;
      }
      if (now >= spin_deadline) {
        std::this_thread::sleep_for(std::chrono::microseconds(20));
      } else {
        CpuRelax();
      }
    }
    return true;
  }

  /// Writes the header and the partial staged in the send window into the
  /// peer's window for `operation_id`, and waits for the peer's write of the
  /// same exchange. The write's immediate data is the readiness signal, so no
  /// round trip precedes it. The peer is done with that window: it wrote the
  /// previous exchange only after its partial of that exchange was ready,
  /// which follows its add of the exchange three before this one (see
  /// Communicator). Returns the peer's partial on the GPU, or null.
  [[nodiscard]] const float* Transfer(std::uint64_t scope_id,
                                      std::uint64_t operation_id,
                                      std::size_t bytes, std::string* error) {
    const std::size_t window = (operation_id % kWindows) * kWindowBytes;
    const CollectiveHeader outgoing{
        .scope_id = scope_id,
        .operation_id = operation_id,
        .bytes = bytes,
    };
    std::uint8_t* send = SendWindow(operation_id);
    std::memcpy(send, &outgoing, sizeof(outgoing));
    ibv_sge sge{};
    sge.addr = reinterpret_cast<std::uint64_t>(send);
    sge.length = static_cast<unsigned>(kDataOffset + bytes);
    sge.lkey = send_mr_->lkey;
    ibv_send_wr wr{};
    wr.wr_id = kWriteTag | operation_id;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    wr.imm_data = htonl(static_cast<std::uint32_t>(operation_id));
    wr.wr.rdma.remote_addr = remote_recv_address_ + window;
    wr.wr.rdma.rkey = remote_recv_rkey_;
    ibv_send_wr* bad = nullptr;
    if (ibv_post_send(qp_, &wr, &bad) != 0) {
      SetError(error, "ibv_post_send failed");
      return nullptr;
    }
    if (!WaitForExchange(operation_id, error)) {
      return nullptr;
    }
    CollectiveHeader incoming{};
    std::memcpy(&incoming,
                static_cast<const std::uint8_t*>(recv_buffer_) + window,
                sizeof(incoming));
    if (incoming.magic != kCollectiveMagic ||
        incoming.version != kCollectiveVersion ||
        incoming.scope_id != outgoing.scope_id ||
        incoming.operation_id != outgoing.operation_id ||
        incoming.bytes != bytes) {
      if (error != nullptr) {
        *error = "verbs collective identity or size mismatch: outgoing=" +
                 std::to_string(outgoing.scope_id) + "/" +
                 std::to_string(outgoing.operation_id) + "/" +
                 std::to_string(outgoing.bytes) +
                 " incoming=" + std::to_string(incoming.scope_id) + "/" +
                 std::to_string(incoming.operation_id) + "/" +
                 std::to_string(incoming.bytes);
      }
      return nullptr;
    }
    return reinterpret_cast<const float*>(
        static_cast<std::uint8_t*>(recv_device_) + window + kDataOffset);
  }

  /// Starts the thread that runs overlapped exchanges, under `mutex_`.
  [[nodiscard]] bool StartWorker(std::string* error) {
    if (worker_.joinable()) {
      return true;
    }
    if (copy_stream_ == nullptr &&
        hipStreamCreateWithFlags(&copy_stream_, hipStreamNonBlocking) !=
            hipSuccess) {
      copy_stream_ = nullptr;
      SetError(error, "overlapped exchange stream creation failed");
      return false;
    }
    for (hipEvent_t& event : ready_events_) {
      if (event == nullptr &&
          hipEventCreateWithFlags(&event, hipEventDisableTiming) !=
              hipSuccess) {
        event = nullptr;
        SetError(error, "overlapped exchange event creation failed");
        return false;
      }
    }
    try {
      worker_ = std::thread([this] { Work(); });
    } catch (const std::system_error&) {
      SetError(error, "overlapped exchange thread creation failed");
      return false;
    }
    return true;
  }

  /// Runs exchanges in order. A started exchange's staging copy runs on a
  /// stream of its own, which waits for the partial without waiting for the
  /// work queued behind it; a queued exchange's GPU stages it and publishes
  /// it, and learns of its arrival through the arrived flag.
  void Work() {
    const bool device =
        hipSetDevice(static_cast<int>(config_.device_index)) == hipSuccess;
    for (;;) {
      Job job;
      {
        std::unique_lock jobs(jobs_mutex_);
        jobs_ready_.wait(jobs, [this] {
          return stopping_.load(std::memory_order_relaxed) || !jobs_.empty();
        });
        if (stopping_.load(std::memory_order_relaxed)) {
          return;
        }
        job = jobs_.front();
        jobs_.pop_front();
      }
      Finished finished{.bytes = job.bytes};
      std::string poisoned;
      {
        std::lock_guard lock(mutex_);
        if (poisoned_) {
          PoisonedError(&poisoned);
        }
      }
      if (!poisoned.empty()) {
        finished.error = std::move(poisoned);
      } else if (job.queued) {
        if (WaitReady(job.operation_id + 1, &finished.error)) {
          finished.peer = Transfer(job.scope_id, job.operation_id, job.bytes,
                                   &finished.error);
        }
      } else if (!device ||
                 hipStreamWaitEvent(copy_stream_, job.ready, 0) != hipSuccess ||
                 (job.bytes != 0 &&
                  hipMemcpyAsync(SendWindow(job.operation_id) + kDataOffset,
                                 job.data, job.bytes, hipMemcpyDeviceToHost,
                                 copy_stream_) != hipSuccess) ||
                 hipStreamSynchronize(copy_stream_) != hipSuccess) {
        finished.error = "HIP device-to-host overlapped staging failed";
      } else {
        finished.peer = Transfer(job.scope_id, job.operation_id, job.bytes,
                                 &finished.error);
      }
      const bool failed = finished.peer == nullptr;
      if (failed || job.queued) {
        std::lock_guard lock(mutex_);
        if (failed) {
          poisoned_ = true;
          if (first_error_.empty()) {
            first_error_ = finished.error;
          }
        }
        // Counted out before the GPU learns of the arrival, so a scope that
        // ends once the stream has drained finds nothing in flight.
        if (job.queued) {
          --queued_;
        }
      }
      if (job.queued) {
        // A failure releases every wait: the stream runs to its end and the
        // caller learns of the failure from CheckQueued.
        Flag<std::uint64_t>(kArrivedOffset)
            .store(failed ? std::numeric_limits<std::uint64_t>::max()
                          : job.operation_id + 1,
                   std::memory_order_release);
        continue;
      }
      {
        std::lock_guard jobs(jobs_mutex_);
        finished_.push_back(std::move(finished));
      }
      finished_ready_.notify_all();
    }
  }

  void StopWorker() noexcept {
    {
      std::lock_guard jobs(jobs_mutex_);
      stopping_.store(true, std::memory_order_relaxed);
    }
    jobs_ready_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
    if (flags_ != nullptr) {
      // Nothing will answer a wait still queued on a stream.
      Flag<std::uint64_t>(kArrivedOffset)
          .store(std::numeric_limits<std::uint64_t>::max(),
                 std::memory_order_release);
    }
    for (hipEvent_t& event : ready_events_) {
      if (event != nullptr) {
        (void)hipEventDestroy(event);
        event = nullptr;
      }
    }
    if (copy_stream_ != nullptr) {
      (void)hipStreamDestroy(copy_stream_);
      copy_stream_ = nullptr;
    }
  }

  [[nodiscard]] bool PostReceive(std::string* error) {
    ibv_recv_wr wr{};
    ibv_recv_wr* bad = nullptr;
    if (ibv_post_recv(qp_, &wr, &bad) != 0) {
      SetError(error, "ibv_post_recv failed");
      return false;
    }
    return true;
  }

  /// Handles one completion for `operation`: this rank's write, or the peer's
  /// write for this exchange or the next one (the peer can be one ahead).
  [[nodiscard]] bool TakeCompletion(const ibv_wc& completion,
                                    std::uint64_t operation, bool* written,
                                    bool* received, std::string* error) {
    if (completion.status != IBV_WC_SUCCESS) {
      SetError(error, "verbs all-reduce completion failed with status " +
                          std::to_string(completion.status));
      return false;
    }
    if (completion.opcode == IBV_WC_RDMA_WRITE) {
      if (completion.wr_id != (kWriteTag | operation)) {
        SetError(error, "verbs write completion is out of order");
        return false;
      }
      *written = true;
      return true;
    }
    if (completion.opcode != IBV_WC_RECV_RDMA_WITH_IMM ||
        (completion.wc_flags & IBV_WC_WITH_IMM) == 0) {
      SetError(error, "verbs completion has an unexpected opcode");
      return false;
    }
    if (!PostReceive(error)) {
      return false;
    }
    const std::uint32_t imm = ntohl(completion.imm_data);
    if (imm == static_cast<std::uint32_t>(operation)) {
      *received = true;
    } else if (imm == static_cast<std::uint32_t>(operation + 1) &&
               !early_receive_) {
      early_receive_ = true;
    } else {
      SetError(error, "verbs peer write is out of order");
      return false;
    }
    return true;
  }

  /// Waits until this rank's write has completed and the peer's write for
  /// the same exchange has arrived. Spins briefly, then sleeps on the
  /// completion channel; a closed bootstrap socket reports a dead peer at
  /// once instead of after the transport's retries.
  [[nodiscard]] bool WaitForExchange(std::uint64_t operation,
                                     std::string* error) {
    bool written = false;
    bool received = std::exchange(early_receive_, false);
    const auto start = std::chrono::steady_clock::now();
    const auto spin_deadline = start + kExchangeSpinWindow;
    const auto deadline = start + kCollectiveTimeout;
    std::array<ibv_wc, 4> completions{};
    const auto drain = [&]() -> int {
      const int n = ibv_poll_cq(cq_, static_cast<int>(completions.size()),
                                completions.data());
      if (n < 0) {
        SetError(error, "ibv_poll_cq failed");
        return -1;
      }
      for (int i = 0; i < n; ++i) {
        if (!TakeCompletion(completions[i], operation, &written, &received,
                            error)) {
          return -1;
        }
      }
      return n;
    };
    while (!(written && received)) {
      const int n = drain();
      if (n < 0) {
        return false;
      }
      if (n > 0) {
        continue;
      }
      if (std::chrono::steady_clock::now() < spin_deadline) {
        CpuRelax();
        continue;
      }
      if (completion_channel_ == nullptr) {
        if (std::chrono::steady_clock::now() >= deadline) {
          SetError(error, "verbs all-reduce timed out");
          return false;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
        continue;
      }
      // Arm, then drain once more: a completion that arrived before arming
      // raises no event.
      if (ibv_req_notify_cq(cq_, 0) != 0) {
        SetError(error, "ibv_req_notify_cq failed");
        return false;
      }
      const int late = drain();
      if (late < 0) {
        return false;
      }
      if (late > 0) {
        continue;
      }
      const auto remaining = deadline - std::chrono::steady_clock::now();
      if (remaining.count() <= 0) {
        SetError(error, "verbs all-reduce timed out");
        return false;
      }
      std::array<pollfd, 2> fds{
          pollfd{.fd = completion_channel_->fd, .events = POLLIN, .revents = 0},
          pollfd{.fd = control_->fd(), .events = POLLIN, .revents = 0}};
      const int result = ::poll(
          fds.data(), fds.size(),
          static_cast<int>(std::max<std::int64_t>(
              1,
              std::chrono::duration_cast<std::chrono::milliseconds>(remaining)
                  .count())));
      if (result < 0 && errno != EINTR) {
        SetError(error, "verbs completion poll failed");
        return false;
      }
      if (fds[1].revents != 0) {
        // The bootstrap socket carries nothing after setup; readable means
        // the peer closed or reset it.
        SetError(error, "verbs peer closed the bootstrap connection");
        return false;
      }
      if (fds[0].revents != 0) {
        ibv_cq* event_cq = nullptr;
        void* event_context = nullptr;
        if (ibv_get_cq_event(completion_channel_, &event_cq, &event_context) !=
                0 ||
            event_cq != cq_) {
          SetError(error, "verbs completion event failed");
          return false;
        }
        ibv_ack_cq_events(cq_, 1);
      }
    }
    return true;
  }

  void Cleanup() noexcept {
    // The worker uses the queue pair and buffers until it stops.
    StopWorker();
    control_.reset();
    if (qp_ != nullptr) {
      ibv_destroy_qp(qp_);
      qp_ = nullptr;
    }
    if (send_mr_ != nullptr) {
      (void)ibv_dereg_mr(send_mr_);
      send_mr_ = nullptr;
    }
    if (recv_mr_ != nullptr) {
      (void)ibv_dereg_mr(recv_mr_);
      recv_mr_ = nullptr;
    }
    if (send_registered_) {
      UnmapFixedBuffer(send_buffer_, kBufferBytes);
      send_registered_ = false;
    }
    send_buffer_ = nullptr;
    if (recv_registered_) {
      UnmapFixedBuffer(recv_buffer_, kBufferBytes);
      recv_registered_ = false;
    }
    recv_buffer_ = nullptr;
    if (arrivals_ != nullptr) {
      (void)hipFree(arrivals_);
      arrivals_ = nullptr;
    }
    if (flags_ != nullptr) {
      (void)hipHostFree(flags_);
      flags_ = nullptr;
    }
    if (completion_channel_ != nullptr) {
      (void)ibv_destroy_comp_channel(completion_channel_);
      completion_channel_ = nullptr;
    }
    if (cq_ != nullptr) {
      ibv_destroy_cq(cq_);
      cq_ = nullptr;
    }
    if (pd_ != nullptr) {
      ibv_dealloc_pd(pd_);
      pd_ = nullptr;
    }
    if (context_ != nullptr) {
      ibv_close_device(context_);
      context_ = nullptr;
    }
  }

  IbrverbsConfig config_;
  ibv_context* context_{nullptr};
  ibv_pd* pd_{nullptr};
  ibv_cq* cq_{nullptr};
  ibv_comp_channel* completion_channel_{nullptr};
  ibv_qp* qp_{nullptr};
  ibv_mr* send_mr_{nullptr};
  ibv_mr* recv_mr_{nullptr};
  void* send_buffer_{nullptr};
  void* recv_buffer_{nullptr};
  void* send_device_{nullptr};
  void* recv_device_{nullptr};
  // Queued exchanges' flags (kReadyOffset...) in host memory the GPU maps,
  // and the stage kernel's block counter.
  void* flags_{nullptr};
  void* flags_device_{nullptr};
  void* arrivals_{nullptr};
  std::uint64_t wait_ticks_{0};
  bool send_registered_{false};
  bool recv_registered_{false};
  ibv_gid local_sgid_{};
  std::uint64_t remote_recv_address_{0};
  std::uint32_t remote_recv_rkey_{0};
  std::unique_ptr<Socket> control_;
  std::optional<std::uint64_t> bound_scope_;
  // Monotonic for the communicator lifetime; scope_id supplies request
  // identity.
  std::uint64_t next_operation_{0};
  // The peer's write for the next exchange, seen while this rank was still
  // waiting for its own write of the current one.
  bool early_receive_{false};
  bool poisoned_{false};
  // The first failure that poisoned the communicator.
  std::string first_error_;
  // Started exchanges not yet finished, and queued exchanges not yet
  // complete.
  std::size_t started_{0};
  std::size_t queued_{0};
  mutable std::mutex mutex_;
  // The overlapped exchanges: the worker takes jobs in order and hands back
  // their results in the same order. Lock order: mutex_ before jobs_mutex_.
  std::thread worker_;
  hipStream_t copy_stream_{nullptr};
  std::array<hipEvent_t, kStartedExchanges> ready_events_{};
  std::deque<Job> jobs_;
  std::deque<Finished> finished_;
  std::atomic<bool> stopping_{false};
  std::mutex jobs_mutex_;
  std::condition_variable jobs_ready_;
  std::condition_variable finished_ready_;
};

}  // namespace

std::shared_ptr<Communicator> CreateIbrverbsCommunicator(
    const IbrverbsConfig& config, std::string* error_msg) {
  auto communicator = std::make_shared<Ibrverbs>(config);
  if (!communicator->Initialize(error_msg)) {
    return nullptr;
  }
  return communicator;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
