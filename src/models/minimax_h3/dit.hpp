#ifndef GUFO_MODELS_MINIMAX_H3_DIT_HPP_
#define GUFO_MODELS_MINIMAX_H3_DIT_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/minimax_h3/runtime.hpp"

namespace gufo::minimax_h3 {

inline constexpr std::size_t kDitBlocks = 50;
inline constexpr std::size_t kDitHiddenSize = 5376;
inline constexpr std::size_t kDitAttentionHeads = 56;
inline constexpr std::size_t kDitHeadDimension = 128;
inline constexpr std::size_t kDitAttentionWidth =
    kDitAttentionHeads * kDitHeadDimension;
inline constexpr std::size_t kDitFeedForwardSize = 14336;
inline constexpr std::size_t kDitRopeHalf = 48;
inline constexpr std::size_t kDitModulationSlots = 6;
inline constexpr float kDitNormEpsilon = 1.0e-5F;

struct DitBlockWorkspaceOptions {
  std::size_t rows{0};
  bool row_parallel_attention{true};
};

// Blocks sharing a workspace must execute sequentially. Scratch and the
// rocBLAS handle are reused across those blocks; independent executions need
// separate workspaces. Concurrent block loading is supported.
class DitBlockWorkspace {
public:
  ~DitBlockWorkspace();

  DitBlockWorkspace(const DitBlockWorkspace&) = delete;
  DitBlockWorkspace& operator=(const DitBlockWorkspace&) = delete;
  DitBlockWorkspace(DitBlockWorkspace&&) = delete;
  DitBlockWorkspace& operator=(DitBlockWorkspace&&) = delete;

  [[nodiscard]] static std::shared_ptr<DitBlockWorkspace> Create(
      const DitBlockWorkspaceOptions& options, std::string* error = nullptr);

  [[nodiscard]] std::size_t rows() const noexcept;
  [[nodiscard]] std::uint64_t bytes() const noexcept;

private:
  struct Impl;
  explicit DitBlockWorkspace(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;

  friend class DitBlockSession;
};

struct DitBlockOptions {
  std::size_t block_index{0};
  std::size_t rows{0};
  std::size_t modulation_rows{0};
  bool row_parallel_attention{true};
  bool validate_gemm_plans{true};
};

struct DitBlockInput {
  std::span<const std::uint16_t> hidden;
  std::span<const std::uint16_t> modulation;
  std::span<const std::uint32_t> row_map;
  std::span<const std::uint16_t> rope_cos;
  std::span<const std::uint16_t> rope_sin;
};

struct DitPreparedInput {
  std::span<const std::uint16_t> hidden;
  std::span<const std::uint32_t> row_map;
};

struct DitBlockRetained {
  std::size_t rows{0};
  std::size_t width{0};
  std::vector<std::uint16_t> modulation_attention;
  std::vector<std::uint16_t> attention_output;
  std::vector<std::uint16_t> modulation_mlp;
  std::vector<std::uint16_t> block_output;
};

struct DitBlockTelemetry {
  std::uint64_t weight_bytes{0};
  std::uint64_t activation_bytes{0};
  // Subset of the shared allocation included in activation_bytes.
  std::uint64_t gemm_workspace_bytes{0};
  std::uint64_t dispatches{0};
  double load_ms{0.0};
  double gpu_ms{0.0};
  std::uintptr_t scratch_address{0};
  std::string attention_backend;
  std::string first_non_finite;
};

class DitBlockSession {
public:
  ~DitBlockSession();

  DitBlockSession(const DitBlockSession&) = delete;
  DitBlockSession& operator=(const DitBlockSession&) = delete;
  DitBlockSession(DitBlockSession&&) noexcept;
  DitBlockSession& operator=(DitBlockSession&&) noexcept;

  [[nodiscard]] static std::unique_ptr<DitBlockSession> Create(
      const ModelInventory& inventory, const DitBlockOptions& options,
      const CancellationToken* cancellation, std::string* error = nullptr);
  [[nodiscard]] static std::unique_ptr<DitBlockSession> Create(
      const ModelInventory& inventory, const DitBlockOptions& options,
      std::shared_ptr<DitBlockWorkspace> workspace,
      const CancellationToken* cancellation, std::string* error = nullptr);

  [[nodiscard]] bool PrepareConstants(std::span<const std::uint16_t> modulation,
                                      std::span<const std::uint16_t> rope_cos,
                                      std::span<const std::uint16_t> rope_sin,
                                      const CancellationToken* cancellation,
                                      std::string* error = nullptr);

  [[nodiscard]] bool RunPrepared(const DitPreparedInput& input,
                                 const CancellationToken* cancellation,
                                 std::span<std::uint16_t> output,
                                 DitBlockTelemetry* telemetry,
                                 std::string* error = nullptr);

  // Device-resident production path. Pointers and stream must belong to the
  // current HIP device; execution is enqueued without synchronizing or
  // copying activations through host memory.
  [[nodiscard]] bool RunPreparedDevice(
      const std::uint16_t* hidden_device, const std::uint32_t* row_map_device,
      std::uint16_t* output_device, void* stream, bool input_rms_prepared,
      bool output_rms_required, const CancellationToken* cancellation,
      DitBlockTelemetry* telemetry, std::string* error = nullptr);

  [[nodiscard]] bool Run(const DitBlockInput& input,
                         const CancellationToken* cancellation,
                         DitBlockRetained* retained,
                         DitBlockTelemetry* telemetry,
                         std::string* error = nullptr);

  [[nodiscard]] std::size_t rows() const noexcept;
  [[nodiscard]] std::size_t block_index() const noexcept;
  [[nodiscard]] std::uintptr_t scratch_address() const noexcept;
  [[nodiscard]] std::uint64_t weight_bytes() const noexcept;
  [[nodiscard]] std::uint64_t activation_bytes() const noexcept;
  [[nodiscard]] std::uint64_t gemm_plan_validations() const noexcept;
  [[nodiscard]] double load_ms() const noexcept;

private:
  struct Impl;
  explicit DitBlockSession(std::unique_ptr<Impl> impl);
  [[nodiscard]] bool RunInternal(
      const DitPreparedInput& input, const CancellationToken* cancellation,
      std::span<std::uint16_t> modulation_attention,
      std::span<std::uint16_t> attention_output,
      std::span<std::uint16_t> modulation_mlp,
      std::span<std::uint16_t> block_output, DitBlockTelemetry* telemetry,
      std::string* error, const std::uint16_t* hidden_device = nullptr,
      const std::uint32_t* row_map_device = nullptr,
      std::uint16_t* output_device = nullptr, void* external_stream = nullptr,
      bool input_rms_prepared = false, bool output_rms_required = false);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::minimax_h3

#endif  // GUFO_MODELS_MINIMAX_H3_DIT_HPP_
