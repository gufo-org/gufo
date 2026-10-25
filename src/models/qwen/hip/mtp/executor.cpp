#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "qfn_mmq.h"
#include "src/core/hip/hip_utils.hpp"
#include "src/core/hip/snapshot_transfer.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/mtp.hpp"
#include "src/models/qwen/hip/mtp/detail/allocation.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "src/models/qwen/hip/ops/moe.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace gufo::hip {
namespace {

constexpr std::array<std::uint8_t, 8> kMtpGpuPersistentMagic = {
    'G', 'M', 'T', 'P', 'G', 'P', 'U', '0'};
constexpr std::uint32_t kMtpGpuPersistentVersion = 1;
constexpr std::size_t kMtpGpuPersistentHeaderBytes = 64;

template<typename T>
  requires(std::is_unsigned_v<T>)
void PutLittleEndian(std::span<std::uint8_t> destination, std::size_t offset,
                     T value) {
  if (offset > destination.size() || sizeof(T) > destination.size() - offset) {
    throw std::length_error("MTP GPU persistent header is truncated");
  }
  for (std::size_t byte = 0; byte < sizeof(T); ++byte) {
    destination[offset + byte] =
        static_cast<std::uint8_t>(value >> (byte * 8U));
  }
}

template<typename T>
  requires(std::is_unsigned_v<T>)
[[nodiscard]] T GetLittleEndian(std::span<const std::uint8_t> source,
                                std::size_t offset) {
  if (offset > source.size() || sizeof(T) > source.size() - offset) {
    throw std::invalid_argument("MTP GPU persistent header is truncated");
  }
  T value = 0;
  for (std::size_t byte = 0; byte < sizeof(T); ++byte) {
    value |= static_cast<T>(source[offset + byte]) << (byte * 8U);
  }
  return value;
}

[[nodiscard]] std::size_t CheckedPersistentAdd(std::size_t left,
                                               std::size_t right) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw std::overflow_error("MTP GPU persistent size overflows");
  }
  return left + right;
}

[[nodiscard]] std::size_t PersistentSizeFromU64(std::uint64_t value) {
  if (value > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("MTP GPU persistent size overflows");
  }
  return static_cast<std::size_t>(value);
}

[[nodiscard]] std::size_t MtpCachePlaneElements(const core::ModelConfig& config,
                                                std::uint32_t max_context) {
  return static_cast<std::size_t>(config.num_key_value_heads) * max_context *
         config.head_dim;
}

/// Resident request state: the FP32 attention plane the decode path reads, plus
/// the FP16 plane the fused KV-write kernel also fills.
[[nodiscard]] std::size_t MtpStateBytes(const core::ModelConfig& config,
                                        std::uint32_t max_context) {
  const std::size_t total_kv = MtpCachePlaneElements(config, max_context);
  return 2U * total_kv * sizeof(float) + 2U * total_kv * sizeof(std::uint16_t);
}

/// Bytes of batched-prime scratch for `prime_chunk` rows. Mirrors
/// AllocatePrimeScratch(); the two are asserted equal at construction so the
/// "estimate == actual" gate cannot drift.
[[nodiscard]] std::size_t MtpPrimeScratchBytes(const core::ModelConfig& config,
                                               std::uint32_t prime_chunk) {
  const std::size_t rows = prime_chunk;
  if (rows == 0) {
    return 0;
  }
  const std::size_t hidden = config.hidden_size;
  const std::size_t kv =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  return rows * sizeof(std::uint32_t) +       // d_prime_tokens_
         rows * hidden * sizeof(float) * 4 +  // hidden_, embed_, out_, normed_
         rows * 2 * hidden * sizeof(float) +  // fusion_
         rows * 2 * kv * sizeof(float);       // k_, v_
}

/// Bytes of the draft table's rows, plus the 4 KiB of slack Flash-Next keeps
/// past its table (#603).
[[nodiscard]] std::size_t MtpDraftTableBytes(const QwenMtpGpuModel& model) {
  return (std::size_t{speculative::kMtpDraftExtraRows} *
          model.GetOutputRowBytes()) +
         4096;
}

/// Rows from which a prime projection switches to the WMMA GEMM, matching the
/// DFlash2 context-injection threshold.
constexpr std::size_t kMtpPrimeWmmaMinRows = 128;
/// Rows per exact BF16 GEMM group below that threshold. The exact kernel's
/// accumulation order matches the decode GEMV and does not depend on the batch
/// width, so a one-row prime reproduces the legacy per-token arithmetic.
constexpr std::size_t kMtpPrimeExactRows = 16;

/// `out[rows, m] = x[rows, k] * weight[m, k]^T` for the BF16 MTP weights.
void PrimeGemm(const models::QwenTensorRef& weight, const float* x, float* out,
               std::size_t rows, std::size_t m, std::size_t k,
               hipStream_t stream) {
  if (weight.data == nullptr || rows == 0) {
    return;
  }
  if (weight.type != core::GgmlType::kBF16) {
    throw std::invalid_argument("MTP prime requires BF16 weights");
  }
  if (rows >= kMtpPrimeWmmaMinRows && IsMoeGroupedBf16GemmSupported(m, k, 1)) {
    LaunchBf16WmmaGemm(weight.data, x, out, rows, m, k, stream);
    return;
  }
  for (std::size_t offset = 0; offset < rows; offset += kMtpPrimeExactRows) {
    const std::size_t group = std::min(kMtpPrimeExactRows, rows - offset);
    LaunchExactBf16GEMMFp32SmallBatch(weight.data, x + (offset * k),
                                      out + (offset * m), group, m, k, stream);
  }
}

template<typename T>
void AllocateBuffer(T*& pointer, std::size_t elements) {
  pointer = static_cast<T*>(detail::AllocateDevice(elements * sizeof(T)));
}

}  // namespace

QwenMtpGpuSnapshot::~QwenMtpGpuSnapshot() {
  if (d_kv_ != nullptr) {
    (void)hipFree(d_kv_);
  }
}

std::size_t QwenMtpGpuSnapshot::PersistentPayloadBytes() const {
  return CheckedPersistentAdd(kMtpGpuPersistentHeaderBytes, payload_bytes_);
}

std::size_t QwenMtpGpuSnapshot::SerializePersistent(
    std::span<std::uint8_t> destination) const {
  const std::size_t expected_bytes = PersistentPayloadBytes();
  const std::size_t head_dim = head_dim_;
  const std::size_t num_kv_heads = num_kv_heads_;
  if (destination.size() != expected_bytes || head_dim == 0 ||
      num_kv_heads == 0 || kv_width_ != num_kv_heads * head_dim ||
      max_context_ == 0 || valid_context_ > max_context_ ||
      valid_context_ > std::numeric_limits<std::size_t>::max() / head_dim ||
      valid_context_ > std::numeric_limits<std::size_t>::max() / num_kv_heads) {
    throw std::invalid_argument("MTP GPU snapshot metadata is malformed");
  }
  const std::size_t elements_per_head =
      static_cast<std::size_t>(valid_context_) * head_dim;
  const std::size_t plane_elements = num_kv_heads * elements_per_head;
  if (plane_elements >
      std::numeric_limits<std::size_t>::max() / (2U * sizeof(float))) {
    throw std::overflow_error("MTP GPU persistent size overflows");
  }
  const std::size_t plane_bytes = plane_elements * sizeof(float);
  if (payload_bytes_ != plane_bytes + plane_bytes ||
      (plane_bytes != 0 && d_kv_ == nullptr)) {
    throw std::invalid_argument("MTP GPU snapshot payload is malformed");
  }

  std::fill(destination.begin(), destination.end(), std::uint8_t{0});
  std::copy(kMtpGpuPersistentMagic.begin(), kMtpGpuPersistentMagic.end(),
            destination.begin());
  PutLittleEndian<std::uint32_t>(destination, 8, kMtpGpuPersistentVersion);
  PutLittleEndian<std::uint32_t>(
      destination, 12,
      static_cast<std::uint32_t>(kMtpGpuPersistentHeaderBytes));
  PutLittleEndian<std::uint32_t>(destination, 16, kv_width_);
  PutLittleEndian<std::uint32_t>(destination, 20,
                                 static_cast<std::uint32_t>(head_dim));
  PutLittleEndian<std::uint32_t>(destination, 24,
                                 static_cast<std::uint32_t>(num_kv_heads));
  PutLittleEndian<std::uint32_t>(destination, 28, max_context_);
  PutLittleEndian<std::uint32_t>(destination, 32, valid_context_);
  PutLittleEndian<std::uint32_t>(destination, 36, 0);
  PutLittleEndian<std::uint64_t>(destination, 40,
                                 static_cast<std::uint64_t>(elements_per_head));
  PutLittleEndian<std::uint64_t>(destination, 48,
                                 static_cast<std::uint64_t>(payload_bytes_));
  PutLittleEndian<std::uint64_t>(destination, 56,
                                 static_cast<std::uint64_t>(expected_bytes));

  if (plane_bytes != 0) {
    const auto* packed = static_cast<const std::uint8_t*>(d_kv_);
    SnapshotTransfer transfer;
    transfer.Copy(destination.data() + kMtpGpuPersistentHeaderBytes, packed,
                  plane_bytes);
    transfer.Copy(
        destination.data() + kMtpGpuPersistentHeaderBytes + plane_bytes,
        packed + plane_bytes, plane_bytes);
  }
  return destination.size();
}

QwenMtpGpuExecutor::QwenMtpGpuExecutor(
    std::shared_ptr<const QwenMtpGpuModel> model, std::uint32_t max_context,
    std::uint32_t prime_chunk)
    : model_(std::move(model)),
      max_context_(max_context),
      prime_chunk_(std::min(prime_chunk, max_context)),
      h_last_hidden_(model_->GetConfig().hidden_size),
      h_logits_(model_->GetConfig().vocab_size) {
  try {
    Allocate();
    Reset();
  } catch (...) {
    Free();
    throw;
  }
}

QwenMtpGpuExecutor::~QwenMtpGpuExecutor() {
  if (stream_ != nullptr) {
    (void)hipStreamSynchronize(stream_);
  }
  Free();
}

std::unique_ptr<QwenMtpGpuExecutor> QwenMtpGpuExecutor::Create(
    std::shared_ptr<const QwenMtpGpuModel> model, std::uint32_t max_context,
    std::uint32_t prime_chunk, std::string* error_msg) {
  if (model == nullptr || max_context == 0 ||
      max_context > model->GetConfig().context_length) {
    if (error_msg != nullptr) {
      *error_msg = "MTP GPU executor context is invalid";
    }
    return nullptr;
  }
  if (prime_chunk > max_context) {
    // A chunk wider than the context is a single slice; clamping keeps the
    // scratch proportional to the context, and EstimateMemoryUsage clamps
    // identically.
    prime_chunk = max_context;
  }
  try {
    return std::unique_ptr<QwenMtpGpuExecutor>(
        new QwenMtpGpuExecutor(std::move(model), max_context, prime_chunk));
  } catch (const std::exception& exception) {
    if (error_msg != nullptr) {
      *error_msg = exception.what();
    }
    return nullptr;
  }
}

void QwenMtpGpuExecutor::Allocate() {
  const auto& config = model_->GetConfig();
  const std::size_t hidden = config.hidden_size;
  const std::size_t attention = config.AttentionSize();
  const std::size_t kv =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t intermediate = config.intermediate_size;
  const std::size_t total_kv = MtpCachePlaneElements(config, max_context_);

  const auto stream_error = hipStreamCreate(&stream_);
  if (stream_error != hipSuccess) {
    throw std::runtime_error(std::string("MTP stream creation failed: ") +
                             hipGetErrorString(stream_error));
  }
  scratch_bytes_ = 0;
  const auto allocate_scratch = [this](auto*& pointer, std::size_t elements) {
    AllocateBuffer(pointer, elements);
    scratch_bytes_ += elements * sizeof(*pointer);
  };
  allocate_scratch(d_target_hidden_, hidden);
  allocate_scratch(d_embedding_, hidden);
  allocate_scratch(d_fusion_, 2 * hidden);
  allocate_scratch(d_hidden_, hidden);
  allocate_scratch(d_normed_, hidden);
  allocate_scratch(d_qg_, 2 * attention);
  allocate_scratch(d_q_, attention);
  allocate_scratch(d_k_, kv);
  allocate_scratch(d_v_, kv);
  allocate_scratch(d_gate_, attention);
  allocate_scratch(d_context_, attention);
  allocate_scratch(d_attn_out_, hidden);
  if (config.IsMoE()) {
    const std::size_t moe_bytes = MoeScratchBytes(config, 1);
    d_moe_ = static_cast<std::uint8_t*>(detail::AllocateDevice(moe_bytes));
    moe_ = CarveMoeScratch(d_moe_, config, 1);
    scratch_bytes_ += moe_bytes;
  } else {
    allocate_scratch(d_ffn_act_, intermediate);
  }
  allocate_scratch(d_ffn_out_, hidden);
  allocate_scratch(d_feedback_hidden_, hidden);
  allocate_scratch(d_logits_, config.vocab_size);
  const std::size_t candidate_workspace =
      models::qwen38_flash_next::rocm::MtpCandidateWorkspaceSize(
          config.vocab_size);
  allocate_scratch(d_candidates_, candidate_workspace);
  allocate_scratch(d_candidate_scratch_, candidate_workspace);
  // The attention cache is request state, not scratch.
  AllocateBuffer(d_kv_cache_, 2 * total_kv);
  d_kv_cache_f16_ =
      detail::AllocateDevice(2 * total_kv * sizeof(std::uint16_t));
  allocate_scratch(d_split_k_scratch_,
                   detail::DecodeAttentionScratchElements(
                       config.num_attention_heads, config.head_dim));
  allocate_scratch(d_out_token_, 1);
  allocate_scratch(d_chain_tokens_, kMtpMaxChainTokens);
  AllocatePrimeScratch();
  if (model_->HasDraftVocab()) {
    allocate_scratch(d_draft_rows_, MtpDraftTableBytes(*model_));
    allocate_scratch(d_draft_ids_, speculative::kMtpDraftExtraRows);
    // The table opens with the special tokens; every request keeps them.
    AppendDraftRows(model_->GetDraftSpecials());
    HIP_CHECK(hipStreamSynchronize(stream_));
  }
}

void QwenMtpGpuExecutor::AllocatePrimeScratch() {
  if (prime_chunk_ == 0) {
    return;
  }
  const auto& config = model_->GetConfig();
  const std::size_t rows = prime_chunk_;
  const std::size_t hidden = config.hidden_size;
  const std::size_t kv =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t before = scratch_bytes_;
  const auto allocate = [this](auto*& pointer, std::size_t elements) {
    AllocateBuffer(pointer, elements);
    scratch_bytes_ += elements * sizeof(*pointer);
  };
  allocate(d_prime_tokens_, rows);
  allocate(d_prime_hidden_, rows * hidden);
  allocate(d_prime_embed_, rows * hidden);
  allocate(d_prime_fusion_, rows * 2 * hidden);
  allocate(d_prime_fusion_out_, rows * hidden);
  allocate(d_prime_normed_, rows * hidden);
  allocate(d_prime_k_, rows * kv);
  allocate(d_prime_v_, rows * kv);
  h_prime_tokens_.assign(rows, 0);
  if (scratch_bytes_ - before != MtpPrimeScratchBytes(config, prime_chunk_)) {
    throw std::logic_error("MTP prime scratch disagrees with its estimate");
  }
}

std::size_t QwenMtpGpuExecutor::StateBytes() const noexcept {
  return MtpStateBytes(model_->GetConfig(), max_context_);
}

std::size_t QwenMtpGpuExecutor::SnapshotPayloadBytes() const noexcept {
  const auto& config = model_->GetConfig();
  const std::size_t kv_width =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  return 2U * kv_width * next_position_ * sizeof(float);
}

QwenGpuMemoryUsage QwenMtpGpuExecutor::GetMemoryUsage() const noexcept {
  return {StateBytes(), scratch_bytes_};
}

QwenGpuMemoryUsage QwenMtpGpuExecutor::EstimateMemoryUsage(
    const QwenMtpGpuModel& model, std::uint32_t max_context,
    std::uint32_t prime_chunk) {
  const auto& config = model.GetConfig();
  if (max_context == 0 || max_context > config.context_length) {
    throw std::invalid_argument("MTP memory estimate has invalid limits");
  }
  // Mirrors Allocate(); the snapshot round-trip test asserts the two agree.
  const std::size_t hidden = config.hidden_size;
  const std::size_t attention = config.AttentionSize();
  const std::size_t kv =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  std::size_t scratch = 0;
  const auto add = [&scratch](std::size_t elements, std::size_t element_bytes) {
    scratch += elements * element_bytes;
  };
  add(hidden, sizeof(float));
  add(hidden, sizeof(float));
  add(2 * hidden, sizeof(float));
  add(hidden, sizeof(float));
  add(hidden, sizeof(float));
  add(2 * attention, sizeof(float));
  add(attention, sizeof(float));
  add(kv, sizeof(float));
  add(kv, sizeof(float));
  add(attention, sizeof(float));
  add(attention, sizeof(float));
  add(hidden, sizeof(float));
  if (config.IsMoE()) {
    scratch += MoeScratchBytes(config, 1);
  } else {
    add(config.intermediate_size, sizeof(float));
  }
  add(hidden, sizeof(float));
  add(hidden, sizeof(float));
  add(config.vocab_size, sizeof(float));
  add(2 *
          std::size_t{
              models::qwen38_flash_next::rocm::MtpCandidateWorkspaceSize(
                  config.vocab_size)},
      sizeof(std::uint32_t));
  add(detail::DecodeAttentionScratchElements(config.num_attention_heads,
                                             config.head_dim),
      sizeof(float));
  add(1, sizeof(std::uint32_t));
  add(kMtpMaxChainTokens, sizeof(std::uint32_t));
  scratch += MtpPrimeScratchBytes(config, std::min(prime_chunk, max_context));
  if (model.HasDraftVocab()) {
    add(MtpDraftTableBytes(model), 1);
    add(speculative::kMtpDraftExtraRows, sizeof(std::int32_t));
  }
  return {
      .request_state_bytes = MtpStateBytes(config, max_context),
      .temporary_scratch_bytes = scratch,
  };
}

void QwenMtpGpuExecutor::Free() noexcept {
  const auto free_buffer = [](auto*& pointer) {
    if (pointer != nullptr) {
      (void)hipFree(pointer);
      pointer = nullptr;
    }
  };
  free_buffer(d_target_hidden_);
  free_buffer(d_embedding_);
  free_buffer(d_fusion_);
  free_buffer(d_hidden_);
  free_buffer(d_normed_);
  free_buffer(d_qg_);
  free_buffer(d_q_);
  free_buffer(d_k_);
  free_buffer(d_v_);
  free_buffer(d_gate_);
  free_buffer(d_context_);
  free_buffer(d_attn_out_);
  free_buffer(d_ffn_act_);
  free_buffer(d_ffn_out_);
  free_buffer(d_moe_);
  moe_ = {};
  free_buffer(d_feedback_hidden_);
  free_buffer(d_logits_);
  free_buffer(d_candidates_);
  free_buffer(d_candidate_scratch_);
  free_buffer(d_kv_cache_);
  free_buffer(d_kv_cache_f16_);
  free_buffer(d_split_k_scratch_);
  free_buffer(d_out_token_);
  free_buffer(d_chain_tokens_);
  free_buffer(d_draft_rows_);
  free_buffer(d_draft_ids_);
  draft_rows_ = 0;
  free_buffer(d_prime_tokens_);
  free_buffer(d_prime_hidden_);
  free_buffer(d_prime_embed_);
  free_buffer(d_prime_fusion_);
  free_buffer(d_prime_fusion_out_);
  free_buffer(d_prime_normed_);
  free_buffer(d_prime_k_);
  free_buffer(d_prime_v_);
  if (stream_ != nullptr) {
    qfn_mmq_release_stream(stream_);
    (void)hipStreamDestroy(stream_);
    stream_ = nullptr;
  }
}

void QwenMtpGpuExecutor::Reset() noexcept {
  RequireChainIdle();
  // Attention only reads the valid prefix; new rows overwrite stale KV.
  next_position_ = 0;
}

void QwenMtpGpuExecutor::Synchronize() {
  const auto status = hipStreamSynchronize(stream_);
  if (status != hipSuccess) {
    throw std::runtime_error(std::string("MTP stream synchronize failed: ") +
                             hipGetErrorString(status));
  }
}

void QwenMtpGpuExecutor::Rewind(std::uint32_t position) {
  RequireChainIdle();
  if (position > next_position_) {
    throw std::out_of_range("MTP rewind position exceeds executed context");
  }
  next_position_ = position;
}

tokenization::TokenId QwenMtpGpuExecutor::ForwardTargetHidden(
    tokenization::TokenId input_token, std::span<const float> target_hidden,
    std::uint32_t position, bool compute_logits, bool small_head) {
  RequireChainIdle();
  if (target_hidden.size() != model_->GetConfig().hidden_size) {
    throw std::invalid_argument("MTP target hidden width is invalid");
  }
  const auto copy_error = hipMemcpyAsync(d_target_hidden_, target_hidden.data(),
                                         target_hidden.size_bytes(),
                                         hipMemcpyHostToDevice, stream_);
  if (copy_error != hipSuccess) {
    throw std::runtime_error(std::string("MTP hidden copy failed: ") +
                             hipGetErrorString(copy_error));
  }
  return Run(input_token, d_target_hidden_, position, compute_logits,
             small_head);
}

tokenization::TokenId QwenMtpGpuExecutor::ForwardFeedback(
    tokenization::TokenId input_token, std::uint32_t position,
    bool compute_logits, bool small_head) {
  RequireChainIdle();
  return Run(input_token, d_feedback_hidden_, position, compute_logits,
             small_head);
}

void QwenMtpGpuExecutor::RequireChainIdle() const {
  if (chain_active_) {
    throw std::logic_error("MTP GPU draft chain is still in flight");
  }
}

void QwenMtpGpuExecutor::StartDraftChain(tokenization::TokenId input_token,
                                         std::span<const float> target_hidden,
                                         std::uint32_t count, bool small_head) {
  if (chain_active_) {
    throw std::logic_error("MTP GPU draft chain is already active");
  }
  if (count == 0 || count > kMtpMaxChainTokens) {
    throw std::invalid_argument("MTP GPU draft chain has an invalid depth");
  }
  if (target_hidden.size() != model_->GetConfig().hidden_size) {
    throw std::invalid_argument("MTP target hidden width is invalid");
  }
  if (next_position_ >= max_context_ || next_position_ + count > max_context_) {
    throw std::out_of_range("MTP GPU draft chain exceeds the context budget");
  }
  const auto copy_error = hipMemcpyAsync(d_target_hidden_, target_hidden.data(),
                                         target_hidden.size_bytes(),
                                         hipMemcpyHostToDevice, stream_);
  if (copy_error != hipSuccess) {
    throw std::runtime_error(std::string("MTP hidden copy failed: ") +
                             hipGetErrorString(copy_error));
  }
  // Step 0 equals the serial Propose loop's ForwardTargetHidden call; later
  // steps equal ForwardFeedback but read the predecessor's device argmax
  // token, so every step matches the serial arithmetic exactly.
  RunDraftForward(nullptr, input_token, d_target_hidden_, next_position_, 0,
                  true, small_head);
  for (std::uint32_t step = 1; step < count; ++step) {
    RunDraftForward(d_chain_tokens_ + step - 1, 0, d_feedback_hidden_,
                    next_position_, step, true, small_head);
  }
  chain_active_ = true;
  chain_slots_ = count;
}

std::vector<tokenization::TokenId> QwenMtpGpuExecutor::CollectChainTokens() {
  if (!chain_active_) {
    throw std::logic_error("MTP GPU draft chain is not active");
  }
  chain_active_ = false;
  std::vector<tokenization::TokenId> tokens(chain_slots_);
  const auto copy_error = hipMemcpyAsync(tokens.data(), d_chain_tokens_,
                                         tokens.size() * sizeof(tokens[0]),
                                         hipMemcpyDeviceToHost, stream_);
  chain_slots_ = 0;
  if (copy_error != hipSuccess || hipStreamSynchronize(stream_) != hipSuccess) {
    throw std::runtime_error("MTP GPU chain result synchronization failed");
  }
  for (auto& token : tokens) {
    token = CheckedSampleToken(token, model_->GetConfig().vocab_size);
  }
  return tokens;
}

void QwenMtpGpuExecutor::AbandonChain() noexcept {
  if (!chain_active_) {
    return;
  }
  // Clear the guard first: the abandoned slots are scratch, the cursor rollback
  // belongs to the caller's Rewind, and clearing before the drain keeps a
  // failed synchronization from stranding chain_active_ again.
  chain_slots_ = 0;
  chain_active_ = false;
  (void)hipStreamSynchronize(stream_);
}

tokenization::TokenId QwenMtpGpuExecutor::Run(tokenization::TokenId input_token,
                                              const float* hidden_input,
                                              std::uint32_t position,
                                              bool compute_logits,
                                              bool small_head) {
  if (position != next_position_ || position >= max_context_) {
    throw std::out_of_range("MTP GPU position is not sequential");
  }
  // Serial path: embed the host token, keep the argmax in d_out_token_, and
  // round-trip exactly as before this refactor.
  RunDraftForward(nullptr, input_token, hidden_input, position, kChainNoSlot,
                  compute_logits, small_head);
  if (!compute_logits) {
    return 0;
  }
  tokenization::TokenId result = 0;
  const auto copy_error = hipMemcpyAsync(&result, d_out_token_, sizeof(result),
                                         hipMemcpyDeviceToHost, stream_);
  if (copy_error != hipSuccess || hipStreamSynchronize(stream_) != hipSuccess) {
    throw std::runtime_error("MTP GPU result synchronization failed");
  }
  return CheckedSampleToken(result, model_->GetConfig().vocab_size);
}

void QwenMtpGpuExecutor::RunDraftForward(const std::uint32_t* source_d_token,
                                         tokenization::TokenId source_h_token,
                                         const float* hidden_input,
                                         std::uint32_t position,
                                         std::uint32_t argmax_slot,
                                         bool compute_logits, bool small_head) {
  if (position != next_position_ || position >= max_context_) {
    throw std::out_of_range("MTP GPU position is not sequential");
  }
  if (small_head && d_draft_rows_ == nullptr) {
    throw std::logic_error("MTP GPU small draft head needs a draft table");
  }
  // The attention below writes this position's cache row.
  if (continuation_hooks_)
    continuation_hooks_->BeforeDraftWrite(position, position + 1);
  const auto& weights = model_->GetWeights();
  const auto& config = weights.config;
  const auto& layer = weights.layer;
  const std::size_t hidden = config.hidden_size;
  const std::size_t attention = config.AttentionSize();
  const std::size_t kv =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t total_kv =
      static_cast<std::size_t>(config.num_key_value_heads) * max_context_ *
      config.head_dim;

  if (source_d_token == nullptr) {
    LaunchEmbeddingLookup(weights.token_embedding.data,
                          weights.token_embedding.type, source_h_token,
                          d_embedding_, hidden, stream_);
  } else {
    // Chain step 1..n-1: the input token is the previous step's device-resident
    // argmax result; no host round trip and identical selection arithmetic.
    LaunchEmbeddingLookup(weights.token_embedding.data,
                          weights.token_embedding.type, source_d_token,
                          d_embedding_, hidden, stream_);
  }
  if (vision_input_)
    vision_input_->Inject(d_embedding_, position + 1, 1, hidden, 1, stream_);
  LaunchRMSNorm(d_embedding_,
                static_cast<const float*>(weights.embedding_norm.data),
                d_fusion_, hidden, 1.0e-6F, stream_);
  LaunchRMSNorm(hidden_input,
                static_cast<const float*>(weights.hidden_norm.data),
                d_fusion_ + hidden, hidden, 1.0e-6F, stream_);
  LaunchGEMV(weights.fusion_projection.data, weights.fusion_projection.type,
             d_fusion_, d_hidden_, hidden, 2 * hidden, stream_,
             models::qwen::QwenGemmMode::kHipMtp);

  LaunchRMSNorm(d_hidden_, static_cast<const float*>(layer.attn_norm.data),
                d_normed_, hidden, 1.0e-6F, stream_);
  LaunchFusedQKVProjections(
      layer.attn_q.data, layer.attn_q.type, layer.attn_k.data,
      layer.attn_k.type, layer.attn_v.data, layer.attn_v.type, d_normed_, d_qg_,
      d_k_, d_v_, 2 * attention, kv, hidden, stream_);
  LaunchUnpackQG(d_qg_, d_q_, d_gate_, config.num_attention_heads,
                 config.head_dim, stream_);
  LaunchPerHeadRMSNorm(d_q_, static_cast<const float*>(layer.attn_q_norm.data),
                       d_q_, config.num_attention_heads, config.head_dim,
                       1.0e-6F, stream_);
  LaunchPerHeadRMSNorm(d_k_, static_cast<const float*>(layer.attn_k_norm.data),
                       d_k_, config.num_key_value_heads, config.head_dim,
                       1.0e-6F, stream_);
  LaunchRoPE(d_q_, d_k_, config.num_attention_heads, config.num_key_value_heads,
             config.head_dim, config.rotary_dim, position, config.rope_theta,
             stream_, vision_input_ ? vision_input_->rope() : nullptr);
  LaunchAttention(
      d_q_, d_k_, d_v_, d_gate_, d_kv_cache_, d_kv_cache_ + total_kv,
      d_kv_cache_f16_, static_cast<std::uint16_t*>(d_kv_cache_f16_) + total_kv,
      d_context_, 0, position, max_context_, config.num_attention_heads,
      config.num_key_value_heads, config.head_dim, stream_, d_split_k_scratch_);
  LaunchGEMV(layer.attn_output.data, layer.attn_output.type, d_context_,
             d_attn_out_, hidden, attention, stream_,
             models::qwen::QwenGemmMode::kHipMtp);
  LaunchResidualAdd(d_hidden_, d_attn_out_, d_hidden_, hidden, stream_);

  LaunchRMSNorm(d_hidden_, static_cast<const float*>(layer.ffn_norm.data),
                d_normed_, hidden, 1.0e-6F, stream_);
  if (config.IsMoE()) {
    // The MTP block of a MoE target is a MoE block too; it runs the same
    // single-token kernels as the target's decode step.
    ExecuteMoeDecodeStep(stream_, moe_, layer, config, d_normed_, d_ffn_out_);
  } else {
    LaunchFusedSwiGLUGEMV(layer.ffn_gate.data, layer.ffn_gate.type,
                          layer.ffn_up.data, layer.ffn_up.type, d_normed_,
                          d_ffn_act_, config.intermediate_size, hidden,
                          stream_);
    LaunchGEMV(layer.ffn_down.data, layer.ffn_down.type, d_ffn_act_, d_ffn_out_,
               hidden, config.intermediate_size, stream_,
               models::qwen::QwenGemmMode::kHipMtp);
  }
  LaunchResidualAdd(d_hidden_, d_ffn_out_, d_hidden_, hidden, stream_);
  LaunchRMSNorm(d_hidden_,
                static_cast<const float*>(weights.shared_head_norm.data),
                d_feedback_hidden_, hidden, 1.0e-6F, stream_);
  ++next_position_;

  if (!compute_logits) {
    return;
  }
  // Chain step: land the argmax in this step's proposal slot; the caller
  // performs the single CollectChainTokens synchronization.
  if (argmax_slot != kChainNoSlot && argmax_slot >= kMtpMaxChainTokens) {
    throw std::out_of_range("MTP GPU chain slot is outside the budget");
  }
  std::uint32_t* const out_token = argmax_slot != kChainNoSlot
                                       ? d_chain_tokens_ + argmax_slot
                                       : d_out_token_;
  if (!small_head) {
    LaunchGEMV(weights.output.data, weights.output.type, d_feedback_hidden_,
               d_logits_, config.vocab_size, hidden, stream_,
               models::qwen::QwenGemmMode::kHipMtp);
    LaunchGPUArgmax(d_logits_, out_token, config.vocab_size, stream_);
    return;
  }
  // The small head (mtp_draft_vocab.hpp): the prefix is the output head's
  // leading rows, read in place, and the table rows follow it in the logits.
  // Only the candidate set changes; the remap turns a table index back into
  // its token ID, so chain steps embed the real token.
  const std::uint32_t prefix = speculative::kMtpDraftVocab;
  LaunchGEMV(weights.output.data, weights.output.type, d_feedback_hidden_,
             d_logits_, prefix, hidden, stream_,
             models::qwen::QwenGemmMode::kHipMtp);
  if (draft_rows_ != 0) {
    LaunchGEMV(d_draft_rows_, weights.output.type, d_feedback_hidden_,
               d_logits_ + prefix, draft_rows_, hidden, stream_,
               models::qwen::QwenGemmMode::kHipMtp);
  }
  LaunchGPUArgmax(d_logits_, out_token, prefix + draft_rows_, stream_);
  LaunchRemapDraftToken(out_token, prefix, d_draft_ids_, draft_rows_, stream_);
}

void QwenMtpGpuExecutor::ResetDraftRows() {
  RequireChainIdle();
  draft_rows_ = static_cast<std::uint32_t>(model_->GetDraftSpecials().size());
}

void QwenMtpGpuExecutor::AppendDraftRows(std::span<const std::int32_t> ids) {
  RequireChainIdle();
  if (ids.empty()) {
    return;
  }
  if (d_draft_rows_ == nullptr ||
      ids.size() > speculative::kMtpDraftExtraRows - draft_rows_) {
    throw std::logic_error("MTP GPU draft table append is invalid");
  }
  const auto& output = model_->GetWeights().output;
  const std::size_t vocab = model_->GetConfig().vocab_size;
  const std::size_t row_bytes = model_->GetOutputRowBytes();
  for (const std::int32_t id : ids) {
    if (id < 0 || static_cast<std::size_t>(id) >= vocab) {
      throw std::out_of_range("MTP GPU draft table token is out of range");
    }
    // The head may be mapped host memory, so the runtime picks the copy kind.
    // The ID goes by value: no host buffer has to outlive the call.
    HIP_CHECK(
        hipMemcpyAsync(d_draft_rows_ + std::size_t{draft_rows_} * row_bytes,
                       static_cast<const std::uint8_t*>(output.data) +
                           static_cast<std::size_t>(id) * row_bytes,
                       row_bytes, hipMemcpyDefault, stream_));
    HIP_CHECK(hipMemsetD32Async(
        reinterpret_cast<hipDeviceptr_t>(d_draft_ids_ + draft_rows_), id, 1,
        stream_));
    ++draft_rows_;
  }
}

void QwenMtpGpuExecutor::PrimeTargetHiddenBatch(
    std::uint32_t start_position,
    std::span<const tokenization::TokenId> successor_tokens,
    std::span<const float> hidden_rows) {
  const std::size_t hidden = model_->GetConfig().hidden_size;
  if (hidden_rows.size() != successor_tokens.size() * hidden) {
    throw std::invalid_argument("MTP prime hidden rows are the wrong shape");
  }
  PrimeRowsBatched(start_position, successor_tokens, nullptr, 0,
                   hidden_rows.data(), hidden);
}

void QwenMtpGpuExecutor::PrimeTargetHiddenBatchDevice(
    std::uint32_t start_position,
    std::span<const tokenization::TokenId> successor_tokens,
    const float* hidden_rows_device, std::size_t row_stride) {
  if (hidden_rows_device == nullptr) {
    throw std::invalid_argument("MTP prime device rows are null");
  }
  PrimeRowsBatched(start_position, successor_tokens, hidden_rows_device,
                   row_stride, nullptr, 0);
}

void QwenMtpGpuExecutor::PrimeRowsBatched(
    std::uint32_t start_position,
    std::span<const tokenization::TokenId> successor_tokens,
    const float* device_rows, std::size_t device_row_stride,
    const float* host_rows, std::size_t host_row_stride) {
  RequireChainIdle();
  const std::size_t count = successor_tokens.size();
  if (count == 0) {
    return;
  }
  if (prime_chunk_ == 0) {
    throw std::logic_error("MTP batched prime needs a non-zero chunk");
  }
  if (start_position > max_context_ || count > max_context_ - start_position) {
    throw std::out_of_range("MTP prime exceeds the executor context");
  }
  if ((device_rows == nullptr) == (host_rows == nullptr)) {
    throw std::invalid_argument("MTP prime needs exactly one row source");
  }
  if (continuation_hooks_)
    continuation_hooks_->BeforeDraftWrite(
        start_position, start_position + static_cast<std::uint32_t>(count));
  const auto& weights = model_->GetWeights();
  const auto& config = weights.config;
  const auto& layer = weights.layer;
  const std::size_t hidden = config.hidden_size;
  const std::size_t kv =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t total_kv = kv * max_context_;

  for (std::size_t offset = 0; offset < count; offset += prime_chunk_) {
    const std::size_t rows =
        std::min<std::size_t>(prime_chunk_, count - offset);
    const std::uint32_t row_base =
        start_position + static_cast<std::uint32_t>(offset);
    // The stride-aware copy keeps the slice's activations packed, which every
    // downstream kernel assumes.
    const bool device_source = device_rows != nullptr;
    const float* source_rows = device_source
                                   ? device_rows + (offset * device_row_stride)
                                   : host_rows + (offset * host_row_stride);
    const std::size_t source_stride =
        device_source ? device_row_stride : host_row_stride;
    HIP_CHECK(hipMemcpy2DAsync(
        d_prime_hidden_, hidden * sizeof(float), source_rows,
        source_stride * sizeof(float), hidden * sizeof(float), rows,
        device_source ? hipMemcpyDeviceToDevice : hipMemcpyHostToDevice,
        stream_));
    std::copy_n(successor_tokens.begin() + offset, rows,
                h_prime_tokens_.begin());
    HIP_CHECK(hipMemcpyAsync(d_prime_tokens_, h_prime_tokens_.data(),
                             rows * sizeof(std::uint32_t),
                             hipMemcpyHostToDevice, stream_));

    LaunchBatchedEmbeddingLookup(weights.token_embedding.data,
                                 weights.token_embedding.type, d_prime_tokens_,
                                 d_prime_embed_, rows, hidden, stream_);
    if (vision_input_ != nullptr) {
      // Mirrors Run()'s per-row offset: the MTP row at position `p` injects at
      // `p + 1`, so a slice beginning at `row_base` injects from `row_base +
      // 1`.
      vision_input_->Inject(d_prime_embed_, row_base + 1,
                            static_cast<std::uint32_t>(rows),
                            static_cast<std::uint32_t>(hidden), 1, stream_);
    }
    LaunchBatchedRMSNormStrided(
        d_prime_embed_, static_cast<const float*>(weights.embedding_norm.data),
        d_prime_fusion_, 2 * hidden, rows, hidden, 1.0e-6F, stream_);
    LaunchBatchedRMSNormStrided(
        d_prime_hidden_, static_cast<const float*>(weights.hidden_norm.data),
        d_prime_fusion_ + hidden, 2 * hidden, rows, hidden, 1.0e-6F, stream_);
    PrimeGemm(weights.fusion_projection, d_prime_fusion_, d_prime_fusion_out_,
              rows, hidden, 2 * hidden, stream_);
    LaunchBatchedRMSNorm(
        d_prime_fusion_out_, static_cast<const float*>(layer.attn_norm.data),
        d_prime_normed_, nullptr, rows, hidden, 1.0e-6F, stream_);
    PrimeGemm(layer.attn_k, d_prime_normed_, d_prime_k_, rows, kv, hidden,
              stream_);
    PrimeGemm(layer.attn_v, d_prime_normed_, d_prime_v_, rows, kv, hidden,
              stream_);
    // K/V write only: `num_heads = 0` disables the fused kernel's Q branch, and
    // both f16 planes stay null because the FP32 plane is authoritative on the
    // MTP path. The normed K lands back in `d_prime_k_`; attention reads the
    // cache, not that buffer.
    LaunchBatchedFusedQKNormRoPEKvWrite(
        nullptr, d_prime_k_, d_prime_v_, nullptr,
        static_cast<const float*>(layer.attn_k_norm.data), nullptr, d_prime_k_,
        d_kv_cache_, d_kv_cache_ + total_kv, nullptr, nullptr, 0, row_base,
        rows, max_context_, 0, config.num_key_value_heads, config.head_dim,
        config.rotary_dim, config.rope_theta, 1.0e-6F, stream_,
        vision_input_ != nullptr ? vision_input_->rope() : nullptr);
  }
  next_position_ = start_position + static_cast<std::uint32_t>(count);
}

std::span<const float> QwenMtpGpuExecutor::CopyLastHidden() {
  const auto error = hipMemcpyAsync(h_last_hidden_.data(), d_feedback_hidden_,
                                    h_last_hidden_.size() * sizeof(float),
                                    hipMemcpyDeviceToHost, stream_);
  if (error != hipSuccess || hipStreamSynchronize(stream_) != hipSuccess) {
    throw std::runtime_error("MTP hidden synchronization failed");
  }
  return h_last_hidden_;
}

std::span<const float> QwenMtpGpuExecutor::CopyLastLogits() {
  const auto error = hipMemcpyAsync(h_logits_.data(), d_logits_,
                                    h_logits_.size() * sizeof(float),
                                    hipMemcpyDeviceToHost, stream_);
  if (error != hipSuccess || hipStreamSynchronize(stream_) != hipSuccess) {
    throw std::runtime_error("MTP logit synchronization failed");
  }
  return h_logits_;
}

models::qwen38_flash_next::MtpCandidateLogits
QwenMtpGpuExecutor::TopCandidates() {
  namespace qfn = models::qwen38_flash_next;
  const std::uint32_t vocab = model_->GetConfig().vocab_size;
  auto* scores = reinterpret_cast<float*>(d_candidates_ + qfn::kMtpCandidates);
  qfn::rocm::MtpTopCandidates(d_logits_, d_candidates_, d_candidate_scratch_,
                              scores, vocab, stream_);
  qfn::MtpCandidateLogits candidates;
  candidates.size = std::min<std::size_t>(vocab, qfn::kMtpCandidates);
  static_assert(offsetof(qfn::MtpCandidateLogits, logits) ==
                qfn::kMtpCandidates * sizeof(std::uint32_t));
  const auto bytes = offsetof(qfn::MtpCandidateLogits, logits) +
                     candidates.size * sizeof(float);
  const auto error = hipMemcpyAsync(&candidates, d_candidates_, bytes,
                                    hipMemcpyDeviceToHost, stream_);
  if (error != hipSuccess || hipStreamSynchronize(stream_) != hipSuccess) {
    throw std::runtime_error("MTP candidate synchronization failed");
  }
  return candidates;
}

void QwenMtpGpuExecutor::RestoreCachePlanes(const std::uint8_t* packed,
                                            std::uint32_t valid_context,
                                            hipMemcpyKind kind) {
  if (continuation_hooks_)
    continuation_hooks_->BeforeDraftWrite(0, valid_context);
  const auto& config = model_->GetConfig();
  const std::size_t head_dim = config.head_dim;
  const std::size_t heads = config.num_key_value_heads;
  const std::size_t row_elements =
      static_cast<std::size_t>(valid_context) * head_dim;
  const std::size_t plane_elements = heads * row_elements;
  const std::size_t total_kv = MtpCachePlaneElements(config, max_context_);
  const auto* source = reinterpret_cast<const float*>(packed);
  for (std::size_t head = 0; head < heads; ++head) {
    if (row_elements == 0) {
      continue;
    }
    const std::size_t destination =
        head * static_cast<std::size_t>(max_context_) * head_dim;
    const std::size_t offset = head * row_elements;
    HIP_CHECK(hipMemcpyAsync(d_kv_cache_ + destination, source + offset,
                             row_elements * sizeof(float), kind, stream_));
    HIP_CHECK(hipMemcpyAsync(d_kv_cache_ + total_kv + destination,
                             source + plane_elements + offset,
                             row_elements * sizeof(float), kind, stream_));
  }
  HIP_CHECK(hipStreamSynchronize(stream_));
  // The FP16 plane is rewritten on every decode step but never read while the
  // FP32 plane is supplied, so it is deliberately left stale here.
}

std::unique_ptr<QwenMtpGpuSnapshot> QwenMtpGpuExecutor::SaveSnapshot() const {
  RequireChainIdle();
  const auto& config = model_->GetConfig();
  const std::size_t head_dim = config.head_dim;
  const std::size_t heads = config.num_key_value_heads;
  const std::size_t kv_width = heads * head_dim;

  auto snapshot = std::unique_ptr<QwenMtpGpuSnapshot>(new QwenMtpGpuSnapshot());
  snapshot->valid_context_ = next_position_;
  snapshot->max_context_ = max_context_;
  snapshot->kv_width_ = static_cast<std::uint32_t>(kv_width);
  snapshot->head_dim_ = static_cast<std::uint32_t>(head_dim);
  snapshot->num_kv_heads_ = static_cast<std::uint32_t>(heads);
  snapshot->payload_bytes_ = SnapshotPayloadBytes();

  const std::size_t row_elements =
      static_cast<std::size_t>(next_position_) * head_dim;
  const std::size_t plane_elements = heads * row_elements;
  if (plane_elements == 0) {
    return snapshot;
  }
  const std::size_t plane_bytes = plane_elements * sizeof(float);
  HIP_CHECK(hipMalloc(&snapshot->d_kv_, plane_bytes + plane_bytes));
  auto* destination = static_cast<float*>(snapshot->d_kv_);
  const std::size_t total_kv = MtpCachePlaneElements(config, max_context_);
  // The FP32 cache is head-major: each head owns a run of `max_context`
  // positions, so the valid rows are gathered per head into a packed plane.
  for (std::size_t head = 0; head < heads; ++head) {
    const std::size_t source =
        head * static_cast<std::size_t>(max_context_) * head_dim;
    const std::size_t offset = head * row_elements;
    HIP_CHECK(hipMemcpyAsync(destination + offset, d_kv_cache_ + source,
                             row_elements * sizeof(float),
                             hipMemcpyDeviceToDevice, stream_));
    HIP_CHECK(hipMemcpyAsync(
        destination + plane_elements + offset, d_kv_cache_ + total_kv + source,
        row_elements * sizeof(float), hipMemcpyDeviceToDevice, stream_));
  }
  HIP_CHECK(hipStreamSynchronize(stream_));
  return snapshot;
}

void QwenMtpGpuExecutor::RestoreSnapshot(const QwenMtpGpuSnapshot& snapshot) {
  RequireChainIdle();
  const auto& config = model_->GetConfig();
  const std::size_t heads = config.num_key_value_heads;
  const std::size_t row_elements =
      static_cast<std::size_t>(snapshot.valid_context_) * config.head_dim;
  const std::size_t plane_elements = heads * row_elements;
  if (snapshot.num_kv_heads_ != heads ||
      snapshot.kv_width_ != heads * config.head_dim ||
      snapshot.head_dim_ != config.head_dim ||
      snapshot.max_context_ != max_context_ ||
      snapshot.valid_context_ > max_context_ ||
      snapshot.payload_bytes_ != 2U * plane_elements * sizeof(float)) {
    throw std::invalid_argument(
        "MTP snapshot is incompatible with the executor");
  }
  if (plane_elements != 0 && snapshot.d_kv_ == nullptr) {
    throw std::invalid_argument("MTP snapshot payload is incomplete");
  }
  RestoreCachePlanes(static_cast<const std::uint8_t*>(snapshot.d_kv_),
                     snapshot.valid_context_, hipMemcpyDeviceToDevice);
  next_position_ = snapshot.valid_context_;
}

void QwenMtpGpuExecutor::RestorePersistentSnapshot(
    std::span<const std::uint8_t> payload) {
  RequireChainIdle();
  if (payload.size() < kMtpGpuPersistentHeaderBytes ||
      !std::equal(kMtpGpuPersistentMagic.begin(), kMtpGpuPersistentMagic.end(),
                  payload.begin()) ||
      GetLittleEndian<std::uint32_t>(payload, 8) != kMtpGpuPersistentVersion ||
      GetLittleEndian<std::uint32_t>(payload, 12) !=
          kMtpGpuPersistentHeaderBytes ||
      GetLittleEndian<std::uint32_t>(payload, 36) != 0) {
    throw std::invalid_argument("MTP GPU persistent header is invalid");
  }
  const std::uint32_t kv_width = GetLittleEndian<std::uint32_t>(payload, 16);
  const std::uint32_t head_dim = GetLittleEndian<std::uint32_t>(payload, 20);
  const std::uint32_t num_kv_heads =
      GetLittleEndian<std::uint32_t>(payload, 24);
  const std::uint32_t max_context = GetLittleEndian<std::uint32_t>(payload, 28);
  const std::uint32_t valid_context =
      GetLittleEndian<std::uint32_t>(payload, 32);
  const std::size_t elements_per_head =
      PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 40));
  const std::size_t payload_bytes =
      PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 48));
  const std::size_t total_bytes =
      PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 56));

  const auto& config = model_->GetConfig();
  const std::size_t expected_heads = config.num_key_value_heads;
  const std::size_t expected_width = expected_heads * config.head_dim;
  if (head_dim == 0 || num_kv_heads == 0 || head_dim != config.head_dim ||
      num_kv_heads != expected_heads || kv_width != expected_width ||
      max_context != max_context_ || valid_context > max_context_ ||
      elements_per_head != static_cast<std::size_t>(valid_context) * head_dim ||
      total_bytes != payload.size() ||
      payload_bytes != 2U * num_kv_heads * elements_per_head * sizeof(float) ||
      CheckedPersistentAdd(kMtpGpuPersistentHeaderBytes, payload_bytes) !=
          payload.size()) {
    throw std::invalid_argument("MTP GPU persistent metadata is incompatible");
  }
  RestoreCachePlanes(payload.data() + kMtpGpuPersistentHeaderBytes,
                     valid_context, hipMemcpyHostToDevice);
  next_position_ = valid_context;
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
