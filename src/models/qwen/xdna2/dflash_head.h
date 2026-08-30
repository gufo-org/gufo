#ifndef GUFO_MODELS_QWEN_XDNA2_DFLASH_HEAD_H_
#define GUFO_MODELS_QWEN_XDNA2_DFLASH_HEAD_H_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>

#include "src/core/xdna2/device.h"
#include "src/models/qwen/state.hpp"

namespace gufo::xdna2 {

inline constexpr std::size_t kQwenDFlashHeadInputElements = 5120;
inline constexpr std::size_t kQwenDFlashHeadShardOutputElements = 8192;
inline constexpr std::size_t kQwenDFlashHeadShardCount = 1;
inline constexpr std::size_t kQwenDFlashHeadOutputElements =
    kQwenDFlashHeadShardCount * kQwenDFlashHeadShardOutputElements;
inline constexpr std::size_t kQwenDFlashHeadBatchRows = 8;

struct QwenDFlashHeadOptions {
  std::uint32_t device_index{0};
  std::uint32_t timeout_ms{30000};
  std::filesystem::path program_dir;
};

struct QwenDFlashHeadProgramInfo {
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
  std::uint32_t bo_allocations{0};
  std::size_t packed_weight_bytes{0};
  std::size_t packed_input_bytes{0};
  double setup_ms{0.0};
  double weight_pack_ms{0.0};
  double weight_upload_ms{0.0};
};

struct QwenDFlashHeadRunMetrics {
  double activation_pack_us{0.0};
  double input_upload_us{0.0};
  double submission_us{0.0};
  double completion_us{0.0};
  double command_us{0.0};
  double output_download_us{0.0};
  double output_unpack_us{0.0};
  double end_to_end_us{0.0};
  bool quarantined{false};
};

struct QwenDFlashHeadFailure {
  std::string category;
  std::string message;
};

class QwenDFlashHeadSession final {
public:
  [[nodiscard]] static std::unique_ptr<QwenDFlashHeadSession> Create(
      const QwenDFlashHeadOptions& options, const XrtDeviceInfo& device_info,
      const models::QwenTensorRef& q8_weights,
      QwenDFlashHeadFailure* failure = nullptr);

  ~QwenDFlashHeadSession();

  QwenDFlashHeadSession(const QwenDFlashHeadSession&) = delete;
  QwenDFlashHeadSession& operator=(const QwenDFlashHeadSession&) = delete;
  QwenDFlashHeadSession(QwenDFlashHeadSession&&) noexcept;
  QwenDFlashHeadSession& operator=(QwenDFlashHeadSession&&) noexcept;

  [[nodiscard]] bool Run(std::span<const float> input, std::size_t row_count,
                         std::span<float> output,
                         QwenDFlashHeadRunMetrics* metrics = nullptr,
                         QwenDFlashHeadFailure* failure = nullptr);

  [[nodiscard]] const QwenDFlashHeadProgramInfo& ProgramInfo() const noexcept {
    return program_info_;
  }

  [[nodiscard]] static std::size_t ActiveSessionCountForDiagnostics() noexcept;
  [[nodiscard]] static std::size_t ActiveBoCountForDiagnostics() noexcept;

private:
  struct Impl;

  QwenDFlashHeadSession(std::unique_ptr<Impl> impl,
                        QwenDFlashHeadProgramInfo program_info,
                        std::uint32_t timeout_ms);

  std::unique_ptr<Impl> impl_;
  QwenDFlashHeadProgramInfo program_info_;
  std::uint32_t timeout_ms_{0};
  bool quarantined_{false};
};

}  // namespace gufo::xdna2

#endif  // GUFO_MODELS_QWEN_XDNA2_DFLASH_HEAD_H_
