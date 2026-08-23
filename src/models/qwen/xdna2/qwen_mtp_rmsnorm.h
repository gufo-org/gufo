#ifndef STRIX_CORE_XDNA2_QWEN_MTP_RMSNORM_H_
#define STRIX_CORE_XDNA2_QWEN_MTP_RMSNORM_H_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>

#include "src/core/xdna2/device.h"

namespace strix::xdna2 {

inline constexpr std::size_t kQwenMtpRmsNormElements = 5120;

struct QwenMtpRmsNormOptions {
  std::uint32_t device_index{0};
  std::uint32_t timeout_ms{30000};
  std::filesystem::path program_dir;
};

struct QwenMtpRmsNormProgramInfo {
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
  std::string resource_evidence;
  std::uint32_t requested_partition_columns{0};
  std::uint32_t assigned_partition_columns{0};
  std::uint32_t bo_allocations{0};
  bool buffers_reused{false};
  double setup_ms{0.0};
  double device_open_ms{0.0};
  double xclbin_load_ms{0.0};
  double context_create_ms{0.0};
  double program_load_ms{0.0};
  double buffer_setup_ms{0.0};
  double weight_upload_ms{0.0};
};

struct QwenMtpRmsNormRunMetrics {
  double submission_us{0.0};
  double completion_us{0.0};
  double command_us{0.0};
  bool quarantined{false};
};

struct QwenMtpRmsNormFailure {
  std::string category;
  std::string message;
};

class QwenMtpRmsNormSession final {
public:
  [[nodiscard]] static std::unique_ptr<QwenMtpRmsNormSession> Create(
      const QwenMtpRmsNormOptions& options, const XrtDeviceInfo& device_info,
      std::span<const std::uint16_t> weight_bf16,
      QwenMtpRmsNormFailure* failure = nullptr);

  ~QwenMtpRmsNormSession();

  QwenMtpRmsNormSession(const QwenMtpRmsNormSession&) = delete;
  QwenMtpRmsNormSession& operator=(const QwenMtpRmsNormSession&) = delete;
  QwenMtpRmsNormSession(QwenMtpRmsNormSession&&) noexcept;
  QwenMtpRmsNormSession& operator=(QwenMtpRmsNormSession&&) noexcept;

  [[nodiscard]] bool Run(std::span<const std::uint16_t> input_bf16,
                         std::span<std::uint16_t> output_bf16,
                         QwenMtpRmsNormRunMetrics* metrics = nullptr,
                         QwenMtpRmsNormFailure* failure = nullptr);

  [[nodiscard]] const QwenMtpRmsNormProgramInfo& ProgramInfo() const noexcept {
    return program_info_;
  }

  [[nodiscard]] static std::size_t ActiveSessionCountForDiagnostics() noexcept;
  [[nodiscard]] static std::size_t ActiveBoCountForDiagnostics() noexcept;

private:
  struct Impl;

  QwenMtpRmsNormSession(std::unique_ptr<Impl> impl,
                        QwenMtpRmsNormProgramInfo program_info,
                        std::uint32_t timeout_ms);

  std::unique_ptr<Impl> impl_;
  QwenMtpRmsNormProgramInfo program_info_;
  std::uint32_t timeout_ms_{0};
  bool quarantined_{false};
};

}  // namespace strix::xdna2

#endif  // STRIX_CORE_XDNA2_QWEN_MTP_RMSNORM_H_
