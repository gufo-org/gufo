#ifndef STRIX_CORE_XDNA2_QWEN_MTP_EH_PROJ_H_
#define STRIX_CORE_XDNA2_QWEN_MTP_EH_PROJ_H_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>

#include "src/core/xdna2/device.h"
#include "src/models/qwen/state.hpp"

namespace strix::xdna2 {

inline constexpr std::size_t kQwenMtpEhProjInputElements = 10240;
inline constexpr std::size_t kQwenMtpEhProjOutputElements = 5120;

struct QwenMtpEhProjOptions {
  std::uint32_t device_index{0};
  std::uint32_t timeout_ms{30000};
  std::filesystem::path program_dir;
};

struct QwenMtpEhProjProgramInfo {
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

struct QwenMtpEhProjRunMetrics {
  double activation_pack_us{0.0};
  double input_upload_us{0.0};
  double submission_us{0.0};
  double completion_us{0.0};
  double command_us{0.0};
  double output_download_us{0.0};
  double end_to_end_us{0.0};
  bool quarantined{false};
};

struct QwenMtpEhProjFailure {
  std::string category;
  std::string message;
};

class QwenMtpEhProjSession final {
public:
  [[nodiscard]] static std::unique_ptr<QwenMtpEhProjSession> Create(
      const QwenMtpEhProjOptions& options, const XrtDeviceInfo& device_info,
      const models::QwenTensorRef& q4k_weights,
      QwenMtpEhProjFailure* failure = nullptr);

  ~QwenMtpEhProjSession();

  QwenMtpEhProjSession(const QwenMtpEhProjSession&) = delete;
  QwenMtpEhProjSession& operator=(const QwenMtpEhProjSession&) = delete;
  QwenMtpEhProjSession(QwenMtpEhProjSession&&) noexcept;
  QwenMtpEhProjSession& operator=(QwenMtpEhProjSession&&) noexcept;

  [[nodiscard]] bool Run(std::span<const float> input, std::span<float> output,
                         QwenMtpEhProjRunMetrics* metrics = nullptr,
                         QwenMtpEhProjFailure* failure = nullptr);

  [[nodiscard]] const QwenMtpEhProjProgramInfo& ProgramInfo() const noexcept {
    return program_info_;
  }

  [[nodiscard]] static std::size_t ActiveSessionCountForDiagnostics() noexcept;
  [[nodiscard]] static std::size_t ActiveBoCountForDiagnostics() noexcept;

private:
  struct Impl;

  QwenMtpEhProjSession(std::unique_ptr<Impl> impl,
                       QwenMtpEhProjProgramInfo program_info,
                       std::uint32_t timeout_ms);

  std::unique_ptr<Impl> impl_;
  QwenMtpEhProjProgramInfo program_info_;
  std::uint32_t timeout_ms_{0};
  bool quarantined_{false};
};

}  // namespace strix::xdna2

#endif  // STRIX_CORE_XDNA2_QWEN_MTP_EH_PROJ_H_
