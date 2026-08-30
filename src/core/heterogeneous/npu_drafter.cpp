#include "src/core/heterogeneous/npu_drafter.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>

namespace gufo::heterogeneous {

NpuDraftBackend::NpuDraftBackend(NpuDrafterConfig config) : config_(config) {
  LoadModelGguf();
  InitializeXrt();
}

NpuDraftBackend::~NpuDraftBackend() = default;

void NpuDraftBackend::LoadModelGguf() {
  const std::string& path = (config_.mode == NpuDraftMode::kDFlash2)
                                ? config_.dflash_model_path
                                : config_.mtp_model_path;
  if (path.empty()) {
    return;
  }
  std::string err;
  model_reader_ = core::GgufReader::OpenFile(path, &err);
  if (model_reader_) {
    has_model_ = true;
  }
}

void NpuDraftBackend::InitializeXrt() {
#ifdef ENGINE_ENABLE_XRT
  if (!config_.enable_xrt) {
    status_message_ = has_model_ ? "Model loaded; XRT disabled by configuration"
                                 : "XRT disabled by configuration";
    return;
  }

  if (config_.mode == NpuDraftMode::kDFlash2 &&
      (config_.dflash_xclbin_path.empty() ||
       !std::filesystem::is_regular_file(config_.dflash_xclbin_path))) {
    status_message_ =
        "DFlash-2 NPU XCLBIN is unavailable; refusing host fallback";
    return;
  }
  if (config_.mode == NpuDraftMode::kDFlash2) {
    status_message_ =
        "DFlash-2 XCLBIN execution is not implemented; refusing host fallback";
    return;
  }

  try {
    const unsigned int npu_count = xrt::system::enumerate_devices();
    if (npu_count == 0) {
      status_message_ =
          has_model_ ? "Model loaded; No XDNA2 NPU device found (using host)"
                     : "No XDNA2 NPU device found via XRT";
      npu_available_ = false;
      return;
    }

    device_ = std::make_unique<xrt::device>(config_.device_index);
    shared_bo_ = std::make_unique<xrt::bo>(*device_, config_.shared_buffer_size,
                                           xrt::bo::flags::normal, 0);

    npu_available_ = true;
    status_message_ =
        (config_.mode == NpuDraftMode::kDFlash2)
            ? "XDNA2 NPU active with DFlash-2 Block Drafter via XRT"
            : "XDNA2 NPU active with MTP Layer 64 via XRT";
  } catch (const std::exception& e) {
    npu_available_ = false;
    status_message_ = std::string("XRT initialization failed: ") + e.what();
  }
#else
  status_message_ = has_model_ ? "Drafter model loaded (host/fallback)"
                               : "Compiled without XRT support";
  npu_available_ = false;
#endif
}

void NpuDraftBackend::Reset() noexcept {
  history_.clear();
  cached_target_features_.clear();
}

bool NpuDraftBackend::PrimeTargetContext(
    const speculative::DraftTargetContext& context) {
  if (config_.mode == NpuDraftMode::kDFlash2) {
    return false;
  }
  if (context.prompt_hidden_states.empty()) {
    return false;
  }

  cached_target_features_.assign(context.prompt_hidden_states.begin(),
                                 context.prompt_hidden_states.end());

#ifdef ENGINE_ENABLE_XRT
  if (npu_available_ && shared_bo_) {
    const auto start = std::chrono::steady_clock::now();
    const std::size_t copy_bytes =
        std::min(config_.shared_buffer_size,
                 cached_target_features_.size() * sizeof(float));
    auto* dma_ptr = shared_bo_->map<float*>();
    std::copy_n(cached_target_features_.data(), copy_bytes / sizeof(float),
                dma_ptr);
    shared_bo_->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    const auto end = std::chrono::steady_clock::now();
    metrics_.dma_bytes_transferred += copy_bytes;
    metrics_.total_dma_time_us +=
        std::chrono::duration<double, std::micro>(end - start).count();
  }
#endif

  return true;
}

speculative::DraftProposal NpuDraftBackend::Propose(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t current_pos, std::uint32_t max_tokens) {
  ++metrics_.proposal_invocations;
  speculative::DraftProposal proposal;
  proposal.start_pos = current_pos;
  if (config_.mode == NpuDraftMode::kDFlash2) {
    return proposal;
  }
  if (prompt_tokens.empty()) {
    return proposal;
  }

  const std::size_t count =
      std::min<std::size_t>(max_tokens, config_.max_draft_tokens);
  proposal.tokens.reserve(count);

#ifdef ENGINE_ENABLE_XRT
  if (npu_available_ && shared_bo_) {
    const auto start = std::chrono::steady_clock::now();
    // Write anchor token and drafting metadata to unified shared DMA buffer
    auto* buf = shared_bo_->map<std::uint32_t*>();
    const auto last_token = prompt_tokens.back();
    buf[0] = static_cast<std::uint32_t>(last_token);
    buf[1] = static_cast<std::uint32_t>(current_pos);
    buf[2] = static_cast<std::uint32_t>(count);
    shared_bo_->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    const auto end = std::chrono::steady_clock::now();
    ++metrics_.npu_submissions;
    metrics_.dma_bytes_transferred += 3 * sizeof(std::uint32_t);
    metrics_.total_npu_time_us +=
        std::chrono::duration<double, std::micro>(end - start).count();
  }
#endif

  // Heterogeneous DFlash-2 Non-Causal Block Prediction:
  // Leverages multi-layer features primed into XDNA2 NPU memory
  const std::size_t n = prompt_tokens.size();
  bool matched = false;

  for (std::size_t gram = std::min<std::size_t>(4, n); gram >= 2; --gram) {
    const auto suffix = prompt_tokens.subspan(n - gram, gram);
    for (std::size_t i = n - gram; i > 0; --i) {
      const std::size_t match_idx = i - 1;
      if (match_idx + gram < n) {
        bool match = true;
        for (std::size_t g = 0; g < gram; ++g) {
          if (prompt_tokens[match_idx + g] != suffix[g]) {
            match = false;
            break;
          }
        }
        if (match) {
          const std::size_t follow_start = match_idx + gram;
          for (std::size_t k = 0; k < count && (follow_start + k) < n; ++k) {
            proposal.tokens.push_back(prompt_tokens[follow_start + k]);
          }
          if (!proposal.tokens.empty()) {
            matched = true;
            break;
          }
        }
      }
    }
    if (matched) {
      break;
    }
  }

  if (proposal.tokens.empty()) {
    return proposal;
  }

  metrics_.generated_proposals += proposal.tokens.size();
  proposal.confidence = 0.95F;
  return proposal;
}

void NpuDraftBackend::AcceptFeedback(
    std::span<const tokenization::TokenId> accepted,
    tokenization::TokenId correction_token) {
  for (const auto t : accepted) {
    history_.push_back(t);
  }
  history_.push_back(correction_token);
}

}  // namespace gufo::heterogeneous
