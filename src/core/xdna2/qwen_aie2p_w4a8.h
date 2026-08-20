#ifndef STRIX_CORE_XDNA2_QWEN_AIE2P_W4A8_H_
#define STRIX_CORE_XDNA2_QWEN_AIE2P_W4A8_H_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>

#include "src/core/xdna2/device.h"
#include "src/models/qwen_state.hpp"

namespace strix::xdna2 {

// Baked M=1 eh_proj configuration of the reviewed XCLBIN
// (blocks=40, tpc=10, rounds=1); see qwen_aie2p_w4a8.py and the
// aie-qwen-aie2p-w4a8 Nix derivation that builds this exact shape.
inline constexpr std::size_t kQwenAie2pW4a8InputElements = 10240;   // K
inline constexpr std::size_t kQwenAie2pW4a8OutputElements = 5120;   // N
inline constexpr std::size_t kQwenAie2pW4a8MaxM = 4;  // rounds=1 * 4 rows

struct QwenAie2pW4a8Options {
  std::uint32_t device_index{0};
  std::uint32_t timeout_ms{30000};
  std::filesystem::path program_dir;
};

struct QwenAie2pW4a8ProgramInfo {
  std::string target;
  std::string abi;
  std::string model_kind;
  std::string tensor_contract;
  std::string program_sha256;
  std::string xclbin_sha256;
  std::string elf_sha256;
  std::string device_name;
  std::string device_architecture;
  std::string driver;
  std::string firmware;
  std::string xrt_version;
  std::string mlir_aie_version;
  std::string llvm_aie_version;
  std::string aiebu_revision;
  std::string xclbin_uuid;
  std::string kernel_name;
  std::string context_mode;
  std::uint32_t partition_columns{0};
  std::uint32_t blocks{0};
  std::uint32_t tiles_per_column{0};
  std::uint32_t rounds{0};
  std::uint32_t bo_allocations{0};
  std::size_t packed_weight_bytes{0};
  std::size_t packed_input_bytes{0};
  double setup_ms{0.0};
  double weight_pack_ms{0.0};
  double weight_upload_ms{0.0};
};

struct QwenAie2pW4a8RunMetrics {
  double activation_pack_us{0.0};
  double input_upload_us{0.0};
  double submission_us{0.0};
  double completion_us{0.0};
  double command_us{0.0};
  double output_download_us{0.0};
  double end_to_end_us{0.0};
  bool quarantined{false};
};

struct QwenAie2pW4a8Failure {
  std::string category;
  std::string message;
};

class QwenAie2pW4a8Session final {
public:
  [[nodiscard]] static std::unique_ptr<QwenAie2pW4a8Session> Create(
      const QwenAie2pW4a8Options& options, const XrtDeviceInfo& device_info,
      const models::QwenTensorRef& q4k_weights,
      QwenAie2pW4a8Failure* failure = nullptr);

  ~QwenAie2pW4a8Session();

  QwenAie2pW4a8Session(const QwenAie2pW4a8Session&) = delete;
  QwenAie2pW4a8Session& operator=(const QwenAie2pW4a8Session&) = delete;
  QwenAie2pW4a8Session(QwenAie2pW4a8Session&&) noexcept;
  QwenAie2pW4a8Session& operator=(QwenAie2pW4a8Session&&) noexcept;

  // `input` is row-major M x K (M <= kQwenAie2pW4a8MaxM); `output` is row-major
  // M x N. The baked rounds=1 xclbin computes each 4-row M chunk into its own
  // output region; padded tail rows/lanes quantize to zero.
  [[nodiscard]] bool Run(std::span<const float> input, std::span<float> output,
                         QwenAie2pW4a8RunMetrics* metrics = nullptr,
                         QwenAie2pW4a8Failure* failure = nullptr);

  [[nodiscard]] const QwenAie2pW4a8ProgramInfo& ProgramInfo() const noexcept {
    return program_info_;
  }

  [[nodiscard]] static std::size_t ActiveSessionCountForDiagnostics() noexcept;
  [[nodiscard]] static std::size_t ActiveBoCountForDiagnostics() noexcept;

private:
  struct Impl;

  QwenAie2pW4a8Session(std::unique_ptr<Impl> impl,
                       QwenAie2pW4a8ProgramInfo program_info,
                       std::uint32_t timeout_ms);

  std::unique_ptr<Impl> impl_;
  QwenAie2pW4a8ProgramInfo program_info_;
  std::uint32_t timeout_ms_{0};
  bool quarantined_{false};
};

}  // namespace strix::xdna2

#endif  // STRIX_CORE_XDNA2_QWEN_AIE2P_W4A8_H_