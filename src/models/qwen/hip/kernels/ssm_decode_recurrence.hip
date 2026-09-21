#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "src/models/qwen/hip/kernels/deltanet_decode.hpp"
#include "src/models/qwen/hip/ops/ssm.hpp"

namespace gufo::hip {

__global__ void CaptureBatchedSsmReplayKernel(
    const float* __restrict__ qkv, const float* __restrict__ alpha,
    const float* __restrict__ beta, SsmReplayCapture replay_capture,
    std::uint32_t layer_idx, std::uint32_t start_pos, std::size_t batch_size,
    std::size_t qkv_size, std::size_t time_step_rank) {
  if (replay_capture.qkv == nullptr || replay_capture.alpha == nullptr ||
      replay_capture.beta == nullptr || replay_capture.enabled == nullptr ||
      replay_capture.enabled[0] == 0U) {
    return;
  }

  const std::size_t row_width = qkv_size + (2 * time_step_rank);
  const std::size_t flat_index =
      (blockIdx.x * static_cast<std::size_t>(blockDim.x)) + threadIdx.x;
  if (flat_index >= batch_size * row_width) {
    return;
  }

  const std::size_t row = flat_index / row_width;
  const std::size_t element = flat_index - (row * row_width);
  const std::size_t slot =
      (static_cast<std::size_t>(start_pos) + row) % kSsmReplayCapacity;
  const std::size_t layer_slot =
      (static_cast<std::size_t>(layer_idx) * kSsmReplayCapacity) + slot;

  if (element < qkv_size) {
    replay_capture.qkv[(layer_slot * qkv_size) + element] =
        qkv[(row * qkv_size) + element];
    return;
  }

  const std::size_t projection_element = element - qkv_size;
  if (projection_element < time_step_rank) {
    replay_capture.alpha[(layer_slot * time_step_rank) + projection_element] =
        alpha[(row * time_step_rank) + projection_element];
  } else {
    const std::size_t beta_element = projection_element - time_step_rank;
    replay_capture.beta[(layer_slot * time_step_rank) + beta_element] =
        beta[(row * time_step_rank) + beta_element];
  }
}

void LaunchCaptureBatchedSsmReplay(
    const float* qkv, const float* alpha, const float* beta,
    SsmReplayCapture replay_capture, std::uint32_t layer_idx,
    std::uint32_t start_pos, std::size_t batch_size, std::size_t qkv_size,
    std::size_t time_step_rank, hipStream_t stream) {
  if (batch_size == 0 || qkv_size == 0 || time_step_rank == 0 ||
      replay_capture.qkv == nullptr) {
    return;
  }
  const std::size_t elements = batch_size * (qkv_size + (2 * time_step_rank));
  constexpr unsigned int kBlockSize = 256;
  const auto grid_size =
      static_cast<unsigned int>((elements + kBlockSize - 1) / kBlockSize);
  CaptureBatchedSsmReplayKernel<<<grid_size, kBlockSize, 0, stream>>>(
      qkv, alpha, beta, replay_capture, layer_idx, start_pos, batch_size,
      qkv_size, time_step_rank);
}

// opt-c191-ssm-rows: one launch walks `rows` consecutive verification rows
// instead of one launch per row. Each thread owns channel `c` and the four
// conv-state slots of that channel exclusively, so stepping rows inside the
// thread performs exactly the same updates in exactly the same order that
// `rows` separate launches did, and is therefore bit-exact.
template<typename Sequences = std::nullptr_t>
__global__ void SSMConvKernel(const float* __restrict__ qkv_in,
                              const float* __restrict__ conv_weights,
                              float* __restrict__ conv_state,
                              float* __restrict__ conv_out, std::size_t qkv_dim,
                              std::uint32_t layer_idx,
                              SsmReplayCapture replay_capture,
                              std::uint32_t rows, std::size_t row_stride,
                              Sequences sequences = {}) {
  if constexpr (!std::is_same_v<Sequences, std::nullptr_t>) {
    const auto& sequence = sequences.sequences[blockIdx.y];
    qkv_in += sequence.row_offset * qkv_dim;
    conv_out += sequence.row_offset * qkv_dim;
    conv_state = sequence.conv + layer_idx * qkv_dim * 4;
    replay_capture = sequence.replay;
    rows = sequence.rows;
  }
  const std::size_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= qkv_dim)
    return;

  const std::size_t c_off = c * 4;
  const bool capture =
      replay_capture.qkv != nullptr && replay_capture.position != nullptr &&
      replay_capture.enabled != nullptr && replay_capture.enabled[0] != 0U;

  for (std::uint32_t row = 0; row < rows; ++row) {
    const float* row_in = qkv_in + (static_cast<std::size_t>(row) * row_stride);
    float* row_out = conv_out + (static_cast<std::size_t>(row) * row_stride);

    if (capture) {
      const std::size_t slot =
          static_cast<std::size_t>(replay_capture.position[row]) %
          kSsmReplayCapacity;
      const std::size_t offset =
          ((static_cast<std::size_t>(layer_idx) * kSsmReplayCapacity + slot) *
           qkv_dim) +
          c;
      replay_capture.qkv[offset] = row_in[c];
    }

    conv_state[c_off + 0] = conv_state[c_off + 1];
    conv_state[c_off + 1] = conv_state[c_off + 2];
    conv_state[c_off + 2] = conv_state[c_off + 3];
    conv_state[c_off + 3] = row_in[c];

    float dot = 0.0F;
    if (conv_weights != nullptr) {
      for (std::size_t k = 0; k < 4; ++k) {
        dot += conv_state[c_off + k] * conv_weights[c * 4 + k];
      }
    } else {
      dot = row_in[c];
    }

    const float sig = 1.0F / (1.0F + expf(-dot));
    row_out[c] = dot * sig;
  }
}

namespace {
struct SsmSequenceBatch {
  SsmSequenceState sequences[8];
};
}  // namespace

void LaunchSSMConvRecurrenceRows(
    const float* qkv_in, const float* conv_weights, float* conv_state,
    float* conv_out, void* deltanet_state, const float* alpha_buf,
    const float* beta_buf, const float* ssm_a, const float* ssm_dt,
    const float* ssm_norm, const float* gate, float* out_buf,
    std::uint32_t layer_idx, std::size_t qkv_size, std::uint32_t num_key_heads,
    std::uint32_t num_heads, std::uint32_t key_dim, std::uint32_t val_dim,
    std::uint32_t rows, std::size_t projection_row_stride,
    std::size_t inner_row_stride, hipStream_t stream,
    SsmReplayCapture replay_capture, QwenRecurrentStateStorage state_storage) {
  if (rows == 0) {
    return;
  }
  const std::size_t block_size = 256;
  const std::size_t grid_size = (qkv_size + block_size - 1) / block_size;
  float* conv_layer_state =
      conv_state + (static_cast<std::size_t>(layer_idx) * qkv_size * 4);

  SSMConvKernel<><<<grid_size, block_size, 0, stream>>>(
      qkv_in, conv_weights, conv_layer_state, conv_out, qkv_size, layer_idx,
      replay_capture, rows, qkv_size);

  const auto launch = [&]<typename StateT, bool Resident, bool WriteOutput>() {
    constexpr unsigned threads = Resident ? 128 : 256;
    DeltaNetRecurrenceKernel<StateT, Resident, WriteOutput>
        <<<num_heads, threads, 0, stream>>>(
            conv_out, static_cast<StateT*>(deltanet_state), alpha_buf, beta_buf,
            ssm_a, ssm_dt, ssm_norm, gate, out_buf, layer_idx, num_key_heads,
            num_heads, key_dim, val_dim, replay_capture, rows, qkv_size,
            projection_row_stride, inner_row_stride);
  };
  const auto launch_output = [&]<typename StateT, bool Resident>() {
    if (out_buf != nullptr)
      launch.template operator()<StateT, Resident, true>();
    else
      launch.template operator()<StateT, Resident, false>();
  };
  if (state_storage == QwenRecurrentStateStorage::kBf16) {
    launch_output.template operator()<hip_bfloat16, false>();
  } else if (key_dim == 128 && val_dim == 128) {
    launch_output.template operator()<float, true>();
  } else {
    launch_output.template operator()<float, false>();
  }
}

void LaunchSSMConvRecurrence(
    const float* qkv_in, const float* conv_weights, float* conv_state,
    float* conv_out, void* deltanet_state, const float* alpha_buf,
    const float* beta_buf, const float* ssm_a, const float* ssm_dt,
    const float* ssm_norm, const float* gate, float* out_buf,
    std::uint32_t layer_idx, std::size_t qkv_size, std::uint32_t num_key_heads,
    std::uint32_t num_heads, std::uint32_t key_dim, std::uint32_t val_dim,
    hipStream_t stream, SsmReplayCapture replay_capture,
    QwenRecurrentStateStorage state_storage) {
  LaunchSSMConvRecurrenceRows(
      qkv_in, conv_weights, conv_state, conv_out, deltanet_state, alpha_buf,
      beta_buf, ssm_a, ssm_dt, ssm_norm, gate, out_buf, layer_idx, qkv_size,
      num_key_heads, num_heads, key_dim, val_dim, 1, 0, 0, stream,
      replay_capture, state_storage);
}

void LaunchSSMConvRecurrenceBatch(
    const float* qkv_in, const float* conv_weights, float* conv_out,
    const float* alpha_buf, const float* beta_buf, const float* ssm_a,
    const float* ssm_dt, const float* ssm_norm, const float* gate,
    float* out_buf, std::span<const SsmSequenceState> sequences,
    std::uint32_t layer_idx, std::size_t qkv_size, std::uint32_t num_key_heads,
    std::uint32_t num_heads, std::uint32_t key_dim, std::uint32_t val_dim,
    std::size_t projection_row_stride, std::size_t inner_row_stride,
    hipStream_t stream, QwenRecurrentStateStorage state_storage) {
  if (sequences.empty())
    return;
  if (sequences.size() > 8)
    throw std::invalid_argument("SSM batch exceeds eight sequences");
  if (sequences.size() == 1) {
    const auto& sequence = sequences.front();
    const std::size_t offset = sequence.row_offset;
    LaunchSSMConvRecurrenceRows(
        qkv_in + offset * qkv_size, conv_weights, sequence.conv,
        conv_out + offset * qkv_size, sequence.recurrent,
        alpha_buf ? alpha_buf + offset * projection_row_stride : nullptr,
        beta_buf ? beta_buf + offset * projection_row_stride : nullptr, ssm_a,
        ssm_dt, ssm_norm, gate ? gate + offset * inner_row_stride : nullptr,
        out_buf ? out_buf + offset * inner_row_stride : nullptr, layer_idx,
        qkv_size, num_key_heads, num_heads, key_dim, val_dim, sequence.rows,
        projection_row_stride, inner_row_stride, stream, sequence.replay,
        state_storage);
    return;
  }
  SsmSequenceBatch batch{};
  std::copy(sequences.begin(), sequences.end(), batch.sequences);
  const auto count = static_cast<unsigned>(sequences.size());
  SSMConvKernel<SsmSequenceBatch>
      <<<dim3((qkv_size + 255) / 256, count), 256, 0, stream>>>(
          qkv_in, conv_weights, nullptr, conv_out, qkv_size, layer_idx, {}, 0,
          qkv_size, batch);
  const auto launch = [&]<typename StateT, bool Resident, bool WriteOutput>() {
    DeltaNetRecurrenceKernel<StateT, Resident, WriteOutput, SsmSequenceBatch>
        <<<dim3(num_heads, count), Resident ? 128 : 256, 0, stream>>>(
            conv_out, nullptr, alpha_buf, beta_buf, ssm_a, ssm_dt, ssm_norm,
            gate, out_buf, layer_idx, num_key_heads, num_heads, key_dim,
            val_dim, {}, 0, qkv_size, projection_row_stride, inner_row_stride,
            batch);
  };
  const auto launch_output = [&]<typename StateT, bool Resident>() {
    if (out_buf)
      launch.template operator()<StateT, Resident, true>();
    else
      launch.template operator()<StateT, Resident, false>();
  };
  if (state_storage == QwenRecurrentStateStorage::kBf16)
    launch_output.template operator()<hip_bfloat16, false>();
  else if (key_dim == 128 && val_dim == 128)
    launch_output.template operator()<float, true>();
  else
    launch_output.template operator()<float, false>();
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
