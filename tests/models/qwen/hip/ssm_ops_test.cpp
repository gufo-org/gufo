#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "src/core/hip/detail/hip_graph_decode_executor.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/families/qwen/hip_common/attention_policy.hpp"
#include "src/families/qwen/kernels/ops.hpp"
#include "src/families/qwen/kernels/ops/prefill_fp16.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/modules/ffn.hpp"
#include "src/models/qwen/modules/layer_view.hpp"
#include "src/models/qwen/modules/module_ctx.hpp"
#include "src/models/qwen/modules/norm.hpp"
#include "src/models/qwen/modules/quant_gemm.hpp"
#include "src/models/qwen/modules/residual.hpp"
#include "tests/models/qwen/hip/support/bfloat16.hpp"
#include "tests/models/qwen/hip/support/device.hpp"
#include "tests/models/qwen/hip/support/device_buffer.hpp"
#include "tests/models/qwen/support/synthetic_weights.hpp"

void TestConcurrentSsmRecurrence() {
  using gufo::test::DeviceBuffer;
  using Storage = gufo::hip::QwenRecurrentStateStorage;
  constexpr std::size_t count = 8, total_rows = 36, layers = 2;
  constexpr unsigned heads = 4, key_heads = 2, dim = 128, layer = 1;
  constexpr std::size_t inner = heads * dim;
  constexpr std::size_t qkv = 2 * key_heads * dim + inner;
  constexpr std::size_t conv_per_sequence = layers * qkv * 4;
  constexpr std::size_t state_per_sequence = layers * heads * dim * dim;
  constexpr std::size_t replay_qkv =
      layers * gufo::hip::kSsmReplayCapacity * qkv;
  constexpr std::size_t replay_control =
      layers * gufo::hip::kSsmReplayCapacity * heads;
  const auto values = [](std::size_t size, float phase) {
    std::vector<float> result(size);
    for (std::size_t i = 0; i < size; ++i)
      result[i] = 0.1F * std::sin(phase + static_cast<float>(i) * 0.013F);
    return result;
  };
  DeviceBuffer<float> input(values(total_rows * qkv, 1)),
      weights(values(qkv * 4, 2)), controls(values(total_rows * heads * 2, 3)),
      decay(std::vector<float>(heads, -0.15F)), dt(values(heads, 4)),
      norm(std::vector<float>(dim, 1)), gate(values(total_rows * inner, 5)),
      conv_state(count * conv_per_sequence), conv_output(total_rows * qkv),
      output(total_rows * inner), captured_qkv(count * replay_qkv),
      captured_alpha(count * replay_control),
      captured_beta(count * replay_control);
  const auto initial_conv = values(count * conv_per_sequence, 6);
  const auto initial_state = values(count * state_per_sequence, 7);
  std::vector<std::uint32_t> positions(total_rows), enabled(count);
  std::array<gufo::hip::SsmSequenceState, count> sequences{};
  std::size_t offset = 0;
  for (std::size_t i = 0; i < count; ++i) {
    sequences[i].row_offset = static_cast<std::uint32_t>(offset);
    sequences[i].rows = static_cast<std::uint32_t>(i + 1);
    for (std::size_t row = 0; row <= i; ++row)
      positions[offset + row] = static_cast<std::uint32_t>(14 + i + row);
    enabled[i] = i % 2;
    offset += i + 1;
  }
  DeviceBuffer<std::uint32_t> device_positions(positions),
      device_enabled(enabled);
  const auto clear = [](auto& buffer) {
    HIP_CHECK(
        hipMemset(buffer.data(), 0, buffer.size() * sizeof(*buffer.data())));
  };
  const auto expect_equal = [](const auto& expected, const auto& actual) {
    if (expected.size() != actual.size() ||
        std::memcmp(expected.data(), actual.data(),
                    expected.size() * sizeof(expected[0])) != 0)
      throw std::runtime_error(
          "concurrent SSM changed output, state or replay");
  };
  for (const auto storage : {Storage::kFp32, Storage::kBf16}) {
    const std::size_t element_bytes =
        gufo::hip::QwenRecurrentStateElementBytes(storage);
    std::vector<std::uint8_t> state_bytes(initial_state.size() * element_bytes);
    for (std::size_t i = 0; i < initial_state.size(); ++i) {
      if (storage == Storage::kFp32) {
        std::memcpy(state_bytes.data() + i * element_bytes, &initial_state[i],
                    element_bytes);
      } else {
        const auto bits = gufo::test::FloatToBf16Bits(initial_state[i]);
        std::memcpy(state_bytes.data() + i * element_bytes, &bits,
                    element_bytes);
      }
    }
    DeviceBuffer<std::uint8_t> recurrent(state_bytes);
    for (std::size_t i = 0; i < count; ++i) {
      auto& sequence = sequences[i];
      sequence.state = gufo::hip::QwenRecurrentStateIo::InPlace(
          conv_state.data() + i * conv_per_sequence,
          recurrent.data() + i * state_per_sequence * element_bytes);
      sequence.replay = {captured_qkv.data() + i * replay_qkv,
                         captured_alpha.data() + i * replay_control,
                         captured_beta.data() + i * replay_control,
                         device_positions.data() + sequence.row_offset,
                         device_enabled.data() + i};
    }
    const auto reset = [&] {
      conv_state.CopyFrom(initial_conv);
      recurrent.CopyFrom(state_bytes);
      for (auto* buffer : {&conv_output, &output, &captured_qkv,
                           &captured_alpha, &captured_beta})
        clear(*buffer);
    };
    for (const std::size_t width : {1U, 2U, 4U, 6U, 8U}) {
      for (const bool write_output : {false, true}) {
        reset();
        for (const auto& sequence : std::span(sequences).first(width)) {
          const std::size_t row = sequence.row_offset;
          gufo::hip::LaunchSSMConvRecurrenceRows(
              input.data() + row * qkv, weights.data(), sequence.state,
              conv_output.data() + row * qkv, controls.data() + row * heads * 2,
              controls.data() + row * heads * 2 + heads, decay.data(),
              dt.data(), norm.data(), gate.data() + row * inner,
              write_output ? output.data() + row * inner : nullptr, layer, qkv,
              key_heads, heads, dim, dim, sequence.rows, heads * 2, inner,
              nullptr, sequence.replay, storage);
        }
        const auto expected_state = recurrent.CopyToHost();
        std::vector<std::vector<float>> expected;
        for (auto* buffer : {&conv_state, &conv_output, &output, &captured_qkv,
                             &captured_alpha, &captured_beta})
          expected.push_back(buffer->CopyToHost());
        reset();
        gufo::hip::LaunchSSMConvRecurrenceBatch(
            input.data(), weights.data(), conv_output.data(), controls.data(),
            controls.data() + heads, decay.data(), dt.data(), norm.data(),
            gate.data(), write_output ? output.data() : nullptr,
            std::span(sequences).first(width), layer, qkv, key_heads, heads,
            dim, dim, heads * 2, inner, nullptr, storage);
        expect_equal(expected_state, recurrent.CopyToHost());
        std::size_t index = 0;
        for (auto* buffer : {&conv_state, &conv_output, &output, &captured_qkv,
                             &captured_alpha, &captured_beta})
          expect_equal(expected[index++], buffer->CopyToHost());
      }
    }
  }
  std::cout
      << "Concurrent SSM: FP32/BF16 state, ragged rows and replay exact\n";
}

void TestRecurrentRollbackRows(bool large_state) {
  using gufo::hip::QwenRecurrentStateStorage;
  for (const auto storage :
       {QwenRecurrentStateStorage::kFp32, QwenRecurrentStateStorage::kBf16}) {
    for (const auto [layers, interval] : {std::pair{4U, 4U},
                                          {7U, 4U},
                                          {3U, 4U},
                                          {4U, 2U},
                                          {4U, 1U},
                                          {4U, 0U}}) {
      auto config = gufo::models::qwen::make_small_qwen_config();
      if (large_state) {
        config.ssm_time_step_rank = 48;
        config.ssm_state_size = 128;
        config.ssm_inner_size = 48 * 128;
      }
      config.num_layers = layers;
      config.full_attention_interval = interval;
      auto policy = gufo::hip::QwenExecutionPolicy::Production();
      policy.recurrent_state_storage = storage;
      gufo::hip::QwenGpuArena arena(config, 16, policy);
      const std::size_t conv_row =
          config.SsmQkvSize() * config.ssm_conv_kernel * sizeof(float);
      const std::size_t delta_row =
          config.ssm_time_step_rank * config.ssm_state_size *
          config.SsmValueSize() *
          gufo::hip::QwenRecurrentStateElementBytes(storage);
      std::vector<std::uint8_t> conv(config.SsmLayerCount() * conv_row),
          delta(config.SsmLayerCount() * delta_row);
      unsigned packed = 0;
      for (unsigned layer = 0; layer < layers; ++layer) {
        if (interval == 0 || (layer + 1) % interval) {
          if (config.SsmLayerIndex(layer) != packed++)
            throw std::runtime_error("incorrect dense recurrent layer index");
        }
      }
      if (packed != config.SsmLayerCount())
        throw std::runtime_error("incorrect recurrent layer count");
      // Without recurrent layers the state buffers are empty (null), so there
      // is nothing to pin or divert.
      if (config.SsmLayerCount() == 0)
        continue;
      const auto expect = [](bool condition, const char* message) {
        if (!condition)
          throw std::runtime_error(message);
      };
      const auto check = [&](const void* device,
                             const std::vector<std::uint8_t>& expected) {
        std::vector<std::uint8_t> actual(expected.size());
        if (!actual.empty())
          HIP_CHECK(hipMemcpy(actual.data(), device, actual.size(),
                              hipMemcpyDeviceToHost));
        if (actual != expected)
          throw std::runtime_error("rollback changed recurrent state");
      };
      // A save pins the live buffer in place; the pass after it, and the pass
      // after a restore, write the other buffer. A second save must pin that
      // other buffer, so both parities are covered, including a tail group.
      const float* previous_pin = nullptr;
      for (unsigned generation = 0; generation < 2; ++generation) {
        for (auto* values : {&conv, &delta}) {
          for (std::size_t i = 0; i < values->size(); ++i) {
            (*values)[i] = (i * 17U + i / 13U + generation * 23U) % 256U;
          }
        }
        HIP_CHECK(hipMemcpyAsync(arena.d_ssm_conv_state, conv.data(),
                                 conv.size(), hipMemcpyHostToDevice,
                                 arena.stream));
        HIP_CHECK(hipMemcpyAsync(arena.d_ssm_deltanet_state, delta.data(),
                                 delta.size(), hipMemcpyHostToDevice,
                                 arena.stream));
        const float* pinned = arena.d_ssm_conv_state;
        expect(pinned != previous_pin, "second save pinned the same buffer");
        previous_pin = pinned;
        arena.SaveState(7);
        const auto pass = arena.BeginRecurrentWrite();
        expect(pass.Diverges() && pass.conv_in == pinned &&
                   pass.conv_out == arena.d_ssm_conv_state &&
                   pass.recurrent_out == arena.d_ssm_deltanet_state,
               "save did not divert the next pass or move the live mirrors");
        HIP_CHECK(
            hipMemsetAsync(pass.conv_out, 0x5A, conv.size(), arena.stream));
        HIP_CHECK(hipMemsetAsync(pass.recurrent_out, 0x5A, delta.size(),
                                 arena.stream));
        // Later passes in the same window update in place.
        const auto in_place = arena.BeginRecurrentWrite();
        expect(!in_place.Diverges() && in_place.conv_out == pass.conv_out,
               "in-place pass moved the live state");
        arena.RestoreState();
        HIP_CHECK(hipStreamSynchronize(arena.stream));
        expect(arena.d_ssm_conv_state == pinned, "restore missed the pin");
        check(arena.d_ssm_conv_state, conv);
        check(arena.d_ssm_deltanet_state, delta);
        // The replay after a restore reads the pin and writes elsewhere, so a
        // cancellation can restore a second time.
        const auto replay = arena.BeginRecurrentWrite();
        expect(replay.Diverges() && replay.conv_in == pinned &&
                   replay.conv_out == pass.conv_out,
               "replay pass did not read the pin");
        HIP_CHECK(
            hipMemsetAsync(replay.conv_out, 0x5A, conv.size(), arena.stream));
        HIP_CHECK(hipMemsetAsync(replay.recurrent_out, 0x5A, delta.size(),
                                 arena.stream));
        arena.RestoreState();
        HIP_CHECK(hipStreamSynchronize(arena.stream));
        check(arena.d_ssm_conv_state, conv);
        check(arena.d_ssm_deltanet_state, delta);
        // Leave the live state on the other buffer for the next generation.
        (void)arena.BeginRecurrentWrite();
      }
    }
  }
}

// A rollback pin moves the recurrent state between two buffers instead of
// copying it: the pass after SaveState or RestoreState reads the pinned buffer
// and writes the other one, later passes update in place. Every variant must
// produce the same bytes as an unpinned in-place run of the same rows, over
// both buffer parities, a replay followed by a second (cancellation) restore,
// FP32 resident and generic shapes, BF16 storage, launches split within one
// pass, and pinned/unpinned sequences sharing one batched launch.
void TestRecurrentForkPingPong() {
  using gufo::hip::QwenGpuArena;
  using gufo::hip::QwenRecurrentStateIo;
  using gufo::test::DeviceBuffer;
  using Storage = gufo::hip::QwenRecurrentStateStorage;
  struct Shape {
    std::uint32_t state_size;
    std::uint32_t inner_size;
    std::uint32_t time_step_rank;
    Storage storage;
  };
  const auto expect = [](bool condition, const char* message) {
    if (!condition)
      throw std::runtime_error(message);
  };
  const auto values = [](std::size_t size, float scale, float phase) {
    std::vector<float> result(size);
    for (std::size_t i = 0; i < size; ++i)
      result[i] = scale * std::sin(phase + static_cast<float>(i) * 0.013F);
    return result;
  };
  constexpr std::uint32_t total_rows = 16, replayed_rows = 7, split_rows = 5;
  for (const auto& shape :
       {Shape{128, 256, 2, Storage::kFp32}, Shape{128, 256, 2, Storage::kBf16},
        Shape{32, 128, 4, Storage::kFp32}}) {
    auto config = gufo::models::qwen::make_small_qwen_config();
    config.ssm_state_size = shape.state_size;
    config.ssm_inner_size = shape.inner_size;
    config.ssm_time_step_rank = shape.time_step_rank;
    auto policy = gufo::hip::QwenExecutionPolicy::Production();
    policy.recurrent_state_storage = shape.storage;
    const std::uint32_t heads = config.ssm_time_step_rank;
    const std::uint32_t key_heads = config.ssm_group_count;
    const std::uint32_t key_dim = config.ssm_state_size;
    const std::uint32_t val_dim = config.SsmValueSize();
    const std::size_t qkv = config.SsmQkvSize();
    const std::size_t inner = config.ssm_inner_size;
    const std::size_t layers = config.SsmLayerCount();
    const std::size_t conv_bytes = layers * qkv * 4 * sizeof(float);
    const std::size_t element_bytes =
        gufo::hip::QwenRecurrentStateElementBytes(shape.storage);
    const std::size_t delta_elements = layers * heads * key_dim * val_dim;
    const std::size_t delta_bytes = delta_elements * element_bytes;
    std::cout << "Recurrent fork ping-pong: key_dim=" << key_dim
              << " val_dim=" << val_dim
              << (shape.storage == Storage::kBf16 ? " bf16\n" : " fp32\n");

    DeviceBuffer<float> input(values(total_rows * qkv, 1.0F, 1)),
        weights(values(qkv * 4, 0.3F, 2)),
        alpha(values(total_rows * heads, 1.0F, 3)),
        beta(values(total_rows * heads, 1.0F, 4)),
        decay(std::vector<float>(heads, -0.15F)), dt(values(heads, 0.1F, 5)),
        norm(std::vector<float>(val_dim, 1.0F)),
        gate(values(total_rows * inner, 0.5F, 6)),
        conv_output(total_rows * qkv), output(total_rows * inner);
    std::vector<std::uint8_t> seed_bytes(conv_bytes + delta_bytes);
    {
      const auto conv = values(layers * qkv * 4, 0.3F, 7);
      const auto delta = values(delta_elements, 0.02F, 8);
      std::memcpy(seed_bytes.data(), conv.data(), conv_bytes);
      for (std::size_t i = 0; i < delta_elements; ++i) {
        if (shape.storage == Storage::kFp32) {
          std::memcpy(seed_bytes.data() + conv_bytes + i * element_bytes,
                      &delta[i], element_bytes);
        } else {
          const auto bits = gufo::test::FloatToBf16Bits(delta[i]);
          std::memcpy(seed_bytes.data() + conv_bytes + i * element_bytes, &bits,
                      element_bytes);
        }
      }
    }
    const auto seed = [&](QwenGpuArena& arena) {
      HIP_CHECK(hipMemcpyAsync(arena.d_ssm_conv_state, seed_bytes.data(),
                               conv_bytes, hipMemcpyHostToDevice,
                               arena.stream));
      HIP_CHECK(hipMemcpyAsync(arena.d_ssm_deltanet_state,
                               seed_bytes.data() + conv_bytes, delta_bytes,
                               hipMemcpyHostToDevice, arena.stream));
      HIP_CHECK(hipStreamSynchronize(arena.stream));
    };
    // The live mirrors always name the buffer holding the current state.
    const auto state_bytes = [&](const QwenGpuArena& arena) {
      HIP_CHECK(hipStreamSynchronize(arena.stream));
      std::vector<std::uint8_t> bytes(conv_bytes + delta_bytes);
      HIP_CHECK(hipMemcpy(bytes.data(), arena.d_ssm_conv_state, conv_bytes,
                          hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(bytes.data() + conv_bytes, arena.d_ssm_deltanet_state,
                          delta_bytes, hipMemcpyDeviceToHost));
      return bytes;
    };
    // One pass over every recurrent layer for `rows` rows starting at
    // `first_row`, through the given buffer pair.
    const auto run = [&](QwenGpuArena& arena, QwenRecurrentStateIo state,
                         std::size_t first_row, std::uint32_t rows) {
      for (std::uint32_t layer = 0; layer < layers; ++layer) {
        gufo::hip::LaunchSSMConvRecurrenceRows(
            input.data() + first_row * qkv, weights.data(), state,
            conv_output.data() + first_row * qkv,
            alpha.data() + first_row * heads, beta.data() + first_row * heads,
            decay.data(), dt.data(), norm.data(),
            gate.data() + first_row * inner, output.data() + first_row * inner,
            layer, qkv, key_heads, heads, key_dim, val_dim, rows, heads, inner,
            arena.stream, {}, shape.storage);
      }
      HIP_CHECK(hipStreamSynchronize(arena.stream));
    };
    // The destination of a diverging pass may already hold the pinned bytes
    // (second parity, recycled allocations); poison it so a kernel reading
    // its destination instead of the pin cannot pass.
    const auto poison = [&](QwenRecurrentStateIo state, hipStream_t stream) {
      HIP_CHECK(hipMemsetAsync(state.conv_out, 0x5A, conv_bytes, stream));
      HIP_CHECK(hipMemsetAsync(state.recurrent_out, 0x5A, delta_bytes, stream));
    };

    QwenGpuArena reference(config, 16, policy);
    QwenGpuArena candidate(config, 16, policy);
    seed(reference);
    const auto unpinned = reference.BeginRecurrentWrite();
    expect(!unpinned.Diverges() &&
               unpinned.conv_out == reference.d_ssm_conv_state &&
               unpinned.recurrent_out == reference.d_ssm_deltanet_state,
           "an unpinned arena must update in place");
    run(reference, unpinned, 0, total_rows);
    const auto after_all = state_bytes(reference);
    seed(reference);
    run(reference, reference.BeginRecurrentWrite(), 0, replayed_rows);
    const auto after_replay = state_bytes(reference);

    seed(candidate);
    const float* previous_pin = nullptr;
    for (unsigned parity = 0; parity < 2; ++parity) {
      const float* pinned = candidate.d_ssm_conv_state;
      expect(pinned != previous_pin, "second save pinned the same buffer");
      previous_pin = pinned;
      candidate.SaveState(7);
      // The first pass after the pin diverges; a second launch of the same
      // pass continues in place, as a replay split at the ring bound does.
      const auto pass = candidate.BeginRecurrentWrite();
      expect(pass.Diverges() && pass.conv_in == pinned &&
                 pass.conv_out == candidate.d_ssm_conv_state &&
                 pass.recurrent_out == candidate.d_ssm_deltanet_state,
             "save did not divert the next pass");
      poison(pass, candidate.stream);
      run(candidate, pass, 0, split_rows);
      run(candidate, pass.Continued(), split_rows, total_rows - split_rows);
      expect(state_bytes(candidate) == after_all,
             "diverging pass changed the recurrence");
      // A persistent snapshot must read the live buffer, whichever parity
      // holds it, and survive resetting/reusing a different arena.
      auto snapshot = candidate.SaveSnapshot(0);
      reference.RestoreSnapshot(*snapshot);
      expect(state_bytes(reference) == after_all,
             "snapshot of a forked state changed the recurrence");
      std::vector<std::uint8_t> payload(snapshot->CompactPayloadBytes(), 0xA5);
      expect(snapshot->SerializeCompact(payload) == payload.size(),
             "forked snapshot serialization returned the wrong size");
      std::vector<std::uint8_t> overwritten(payload.size(), 0x5A);
      expect(snapshot->SerializeCompact(overwritten) == overwritten.size() &&
                 overwritten == payload,
             "forked snapshot depends on previous destination bytes");
      reference.Reset();
      reference.RestoreCompactSnapshot(payload, 0);
      expect(state_bytes(reference) == after_all,
             "serialized forked snapshot changed the recurrence");
      const auto in_place = candidate.BeginRecurrentWrite();
      expect(!in_place.Diverges() && in_place.conv_out == pass.conv_out,
             "second pass after a pin must update in place");
      // Rejection: restore, then replay fewer rows from the pin.
      candidate.RestoreState();
      expect(candidate.d_ssm_conv_state == pinned &&
                 state_bytes(candidate) == seed_bytes,
             "restore did not return the pinned bytes");
      const auto replay = candidate.BeginRecurrentWrite();
      expect(replay.Diverges() && replay.conv_in == pinned,
             "replay must read the pin and write elsewhere");
      poison(replay, candidate.stream);
      run(candidate, replay, 0, replayed_rows);
      expect(state_bytes(candidate) == after_replay,
             "replay after restore changed the recurrence");
      // Cancellation restores a second time; the pin survived the replay.
      candidate.RestoreState();
      expect(state_bytes(candidate) == seed_bytes,
             "second restore did not return the pinned bytes");
      // Move the live state to the other buffer and reseed it so the next
      // save pins the other parity.
      run(candidate, candidate.BeginRecurrentWrite(), 0, total_rows);
      expect(candidate.d_ssm_conv_state != pinned,
             "pass after the second restore stayed on the pin");
      seed(candidate);
    }

    // Three sessions in one batched launch: pinned and diverging (3 rows),
    // unpinned and in place (5 rows), and pinned but already diverged by an
    // earlier 4-row pass so this launch is in place on its other buffer
    // (2 rows). Each must match its own serial reference.
    QwenGpuArena pinned_arena(config, 16, policy);
    QwenGpuArena plain_arena(config, 16, policy);
    QwenGpuArena diverged_arena(config, 16, policy);
    seed(pinned_arena);
    seed(plain_arena);
    seed(diverged_arena);
    pinned_arena.SaveState(7);
    diverged_arena.SaveState(7);
    run(diverged_arena, diverged_arena.BeginRecurrentWrite(), 0, 4);
    const std::array<gufo::hip::SsmSequenceState, 3> sequences{{
        {pinned_arena.BeginRecurrentWrite(), {}, 0, 3},
        {plain_arena.BeginRecurrentWrite(), {}, 3, 5},
        {diverged_arena.BeginRecurrentWrite(), {}, 8, 2},
    }};
    expect(sequences[0].state.Diverges() && !sequences[1].state.Diverges() &&
               !sequences[2].state.Diverges(),
           "batched sequences did not get the expected buffer pairs");
    poison(sequences[0].state, pinned_arena.stream);
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
      gufo::hip::LaunchSSMConvRecurrenceBatch(
          input.data(), weights.data(), conv_output.data(), alpha.data(),
          beta.data(), decay.data(), dt.data(), norm.data(), gate.data(),
          output.data(), sequences, layer, qkv, key_heads, heads, key_dim,
          val_dim, heads, inner, pinned_arena.stream, shape.storage);
    }
    HIP_CHECK(hipStreamSynchronize(pinned_arena.stream));
    seed(reference);
    run(reference, reference.BeginRecurrentWrite(), 0, 3);
    expect(state_bytes(pinned_arena) == state_bytes(reference),
           "pinned sequence in a batched launch changed the recurrence");
    seed(reference);
    run(reference, reference.BeginRecurrentWrite(), 3, 5);
    expect(state_bytes(plain_arena) == state_bytes(reference),
           "unpinned sequence in a batched launch changed the recurrence");
    seed(reference);
    run(reference, reference.BeginRecurrentWrite(), 0, 4);
    run(reference, reference.BeginRecurrentWrite(), 8, 2);
    expect(state_bytes(diverged_arena) == state_bytes(reference),
           "in-place pinned sequence in a batched launch changed the "
           "recurrence");
    pinned_arena.RestoreState();
    diverged_arena.RestoreState();
    expect(state_bytes(pinned_arena) == seed_bytes &&
               state_bytes(diverged_arena) == seed_bytes,
           "batched launch wrote into a pinned buffer");
  }
  std::cout << "Recurrent fork ping-pong: pinned passes bit-exact\n";
}

void TestBf16RecurrentMemoryAndSnapshot() {
  gufo::core::ModelConfig production;
  production.model_name = "qwen3.8-27b";
  production.num_layers = 64;
  production.hidden_size = 5120;
  production.intermediate_size = 17408;
  production.num_attention_heads = 24;
  production.num_key_value_heads = 4;
  production.head_dim = 256;
  production.vocab_size = 248320;
  production.context_length = 262144;
  production.full_attention_interval = 4;
  production.mtp_num_layers = 1;
  production.ssm_conv_kernel = 4;
  production.ssm_state_size = 128;
  production.ssm_group_count = 16;
  production.ssm_time_step_rank = 48;
  production.ssm_inner_size = 6144;
  production.rotary_dim = 64;
  production.rope_theta = 10000000.0F;

  auto fp32_policy = gufo::hip::QwenExecutionPolicy::Production();
  auto bf16_policy = fp32_policy;
  bf16_policy.recurrent_state_storage =
      gufo::hip::QwenRecurrentStateStorage::kBf16;

  constexpr std::uint32_t context = 4096;
  const auto fp32_usage = gufo::hip::QwenGpuArena::EstimateMemoryUsage(
      production, context, fp32_policy);
  const auto bf16_usage = gufo::hip::QwenGpuArena::EstimateMemoryUsage(
      production, context, bf16_policy);
  const std::size_t recurrent_elements =
      static_cast<std::size_t>(production.SsmLayerCount()) *
      production.ssm_time_step_rank * production.ssm_state_size *
      production.SsmValueSize();
  const std::size_t expected_savings =
      recurrent_elements * (sizeof(float) - sizeof(std::uint16_t));
  if (fp32_usage.request_state_bytes - bf16_usage.request_state_bytes !=
      expected_savings) {
    throw std::runtime_error(
        "BF16 recurrent accounting did not remove exactly two bytes per "
        "state element");
  }
  if (fp32_usage.temporary_scratch_bytes !=
      bf16_usage.temporary_scratch_bytes) {
    throw std::runtime_error(
        "recurrent storage precision changed shared scratch accounting");
  }

  std::cout << "Qwen27B BF16 recurrent fp32_state_bytes="
            << fp32_usage.request_state_bytes
            << " bf16_state_bytes=" << bf16_usage.request_state_bytes
            << " saved_per_state=" << expected_savings << "\n";
  for (const std::size_t concurrency : {1U, 2U, 4U}) {
    std::cout << "  C=" << concurrency
              << " fp32_reserved=" << concurrency * fp32_usage.TotalBytes()
              << " bf16_reserved=" << concurrency * bf16_usage.TotalBytes()
              << " saved=" << concurrency * expected_savings << "\n";
  }

  const auto small = gufo::models::qwen::make_small_qwen_config();
  constexpr std::uint32_t small_context = 32;
  gufo::hip::QwenGpuArena fp32_arena(small, small_context, fp32_policy);
  gufo::hip::QwenGpuArena bf16_arena(small, small_context, bf16_policy);
  auto fp32_snapshot = fp32_arena.SaveSnapshot(7);
  auto bf16_snapshot = bf16_arena.SaveSnapshot(7);
  if (bf16_snapshot->RecurrentStateStorage() !=
      gufo::hip::QwenRecurrentStateStorage::kBf16) {
    throw std::runtime_error(
        "snapshot did not record canonical BF16 recurrent storage");
  }
  if (bf16_snapshot->PayloadBytes() != bf16_arena.SnapshotPayloadBytes(7)) {
    throw std::runtime_error(
        "BF16 snapshot payload disagrees with state-byte accounting");
  }

  bool rejected = false;
  try {
    bf16_arena.RestoreSnapshot(*fp32_snapshot);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected) {
    throw std::runtime_error(
        "BF16 arena accepted an FP32 recurrent-state snapshot");
  }
}

// Independent causal-convolution oracle, including nonzero carried history.
// Only short batches need scalar output checks; all batches check the exact
// tail.
void CheckSSMConvolution(const std::vector<float>& input,
                         const std::vector<float>& weights,
                         const std::vector<float>& history,
                         const float* device_output, const float* device_state,
                         std::size_t batch, std::size_t channels) {
  std::vector<float> state(channels * 4);
  HIP_CHECK(hipMemcpy(state.data(), device_state, state.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  for (std::size_t c = 0; c < channels; ++c) {
    for (std::size_t j = 0; j < 4; ++j) {
      const float expected = batch + j < 4
                                 ? history[(c * 4) + batch + j]
                                 : input[((batch + j - 4) * channels) + c];
      if (std::memcmp(&expected, &state[(c * 4) + j], sizeof(float)) != 0) {
        throw std::runtime_error(
            "convolution history differs from causal tail");
      }
    }
  }
  if (batch > 8) {
    return;
  }
  std::vector<float> output(batch * channels);
  HIP_CHECK(hipMemcpy(output.data(), device_output,
                      output.size() * sizeof(float), hipMemcpyDeviceToHost));
  for (std::size_t t = 0; t < batch; ++t) {
    for (std::size_t c = 0; c < channels; ++c) {
      double dot = 0.0;
      for (std::size_t j = 0; j < 4; ++j) {
        const float value = t + j < 3 ? history[(c * 4) + t + j + 1]
                                      : input[((t + j - 3) * channels) + c];
        dot += static_cast<double>(value) * weights[(c * 4) + j];
      }
      const double expected = dot / (1.0 + std::exp(-dot));
      const float actual = output[(t * channels) + c];
      if (!std::isfinite(actual) ||
          std::abs(actual - expected) > 2e-6 * (1.0 + std::abs(expected))) {
        throw std::runtime_error("convolution differs from causal formula");
      }
    }
  }
}

void TestBatchedSSMConvEquivalence() {
  constexpr std::size_t batch = 8;
  constexpr std::uint32_t num_key_heads = 16;
  constexpr std::uint32_t num_heads = 48;
  constexpr std::uint32_t key_dim = 128;
  constexpr std::uint32_t val_dim = 128;
  constexpr std::size_t qkv_dim =
      (2 * num_key_heads * key_dim) + (num_heads * val_dim);
  constexpr std::size_t inner_size = num_heads * val_dim;

  std::vector<float> h_qkv(batch * qkv_dim);
  for (std::size_t i = 0; i < h_qkv.size(); ++i) {
    h_qkv[i] = std::sin(static_cast<float>(i) * 0.01F);
  }
  std::vector<float> h_weights(qkv_dim * 4, 0.25F);
  std::vector<float> h_ssm_a(num_heads, -0.05F);
  std::vector<float> h_ssm_dt(num_heads, 0.01F);
  std::vector<float> h_ssm_norm(val_dim, 1.0F);
  std::vector<float> h_gate(batch * inner_size, 0.5F);

  float *d_qkv = nullptr, *d_w = nullptr;
  float *d_state_seq = nullptr, *d_state_batch = nullptr;
  float* d_state_bf16 = nullptr;
  float *d_conv_out_seq = nullptr, *d_conv_out_batch = nullptr;
  float* d_conv_out_bf16 = nullptr;
  float *d_delta_seq = nullptr, *d_delta_batch = nullptr;
  void* d_delta_bf16 = nullptr;
  float *d_alpha = nullptr, *d_beta = nullptr;
  float *d_ssm_a = nullptr, *d_ssm_dt = nullptr, *d_ssm_norm = nullptr;
  float* d_gate = nullptr;
  float *d_out_seq = nullptr, *d_out_batch = nullptr;
  float* d_out_bf16 = nullptr;

  const std::size_t delta_size = num_heads * key_dim * val_dim;
  HIP_CHECK(hipMalloc(&d_qkv, batch * qkv_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_w, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_state_seq, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_state_batch, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_state_bf16, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_conv_out_seq, batch * qkv_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_conv_out_batch, batch * qkv_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_conv_out_bf16, batch * qkv_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_delta_seq, delta_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_delta_batch, delta_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_delta_bf16, delta_size * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_alpha, batch * num_heads * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_beta, batch * num_heads * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_a, num_heads * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_dt, num_heads * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_norm, val_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, batch * inner_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_seq, batch * inner_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_batch, batch * inner_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_bf16, batch * inner_size * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_qkv, h_qkv.data(), batch * qkv_dim * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_w, h_weights.data(), qkv_dim * 4 * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_state_seq, 0, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMemset(d_state_batch, 0, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMemset(d_state_bf16, 0, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMemset(d_delta_seq, 0, delta_size * sizeof(float)));
  HIP_CHECK(hipMemset(d_delta_batch, 0, delta_size * sizeof(float)));
  HIP_CHECK(hipMemset(d_delta_bf16, 0, delta_size * sizeof(std::uint16_t)));
  HIP_CHECK(hipMemset(d_alpha, 0, batch * num_heads * sizeof(float)));
  HIP_CHECK(hipMemset(d_beta, 0, batch * num_heads * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_ssm_a, h_ssm_a.data(), num_heads * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_ssm_dt, h_ssm_dt.data(), num_heads * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_ssm_norm, h_ssm_norm.data(), val_dim * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate, h_gate.data(), batch * inner_size * sizeof(float),
                      hipMemcpyHostToDevice));

  // Sequential
  using gufo::hip::QwenRecurrentStateIo;
  for (std::size_t t = 0; t < batch; ++t) {
    gufo::hip::LaunchSSMConvRecurrence(
        d_qkv + t * qkv_dim, d_w,
        QwenRecurrentStateIo::InPlace(d_state_seq, d_delta_seq),
        d_conv_out_seq + t * qkv_dim, d_alpha + t * num_heads,
        d_beta + t * num_heads, d_ssm_a, d_ssm_dt, d_ssm_norm,
        d_gate + t * inner_size, d_out_seq + t * inner_size, 0, qkv_dim,
        num_key_heads, num_heads, key_dim, val_dim);
    gufo::hip::LaunchSSMConvRecurrence(
        d_qkv + t * qkv_dim, d_w,
        QwenRecurrentStateIo::InPlace(d_state_bf16, d_delta_bf16),
        d_conv_out_bf16 + t * qkv_dim, d_alpha + t * num_heads,
        d_beta + t * num_heads, d_ssm_a, d_ssm_dt, d_ssm_norm,
        d_gate + t * inner_size, d_out_bf16 + t * inner_size, 0, qkv_dim,
        num_key_heads, num_heads, key_dim, val_dim, nullptr, {},
        gufo::hip::QwenRecurrentStateStorage::kBf16);
  }

  // Batched
  gufo::hip::LaunchBatchedSSMConvRecurrence(
      d_qkv, d_w, QwenRecurrentStateIo::InPlace(d_state_batch, d_delta_batch),
      d_conv_out_batch, d_alpha, d_beta, d_ssm_a, d_ssm_dt, d_ssm_norm, d_gate,
      d_out_batch, 0, batch, qkv_dim, num_key_heads, num_heads, key_dim,
      val_dim);

  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> res_seq(batch * inner_size);
  std::vector<float> res_batch(batch * inner_size);
  std::vector<float> res_bf16(batch * inner_size);
  HIP_CHECK(hipMemcpy(res_seq.data(), d_out_seq,
                      batch * inner_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_batch.data(), d_out_batch,
                      batch * inner_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_bf16.data(), d_out_bf16,
                      batch * inner_size * sizeof(float),
                      hipMemcpyDeviceToHost));

  float max_diff = 0.0F;
  for (std::size_t i = 0; i < res_seq.size(); ++i) {
    const float d = std::abs(res_seq[i] - res_batch[i]);
    if (d > max_diff)
      max_diff = d;
  }
  std::cout << "DeltaNet Seq vs Batch max diff: " << max_diff << "\n";
  if (max_diff >= 1e-4F) {
    std::cerr << "DeltaNet batched recurrence mismatch\n";
    std::abort();
  }

  float bf16_output_max_diff = 0.0F;
  for (std::size_t i = 0; i < res_seq.size(); ++i) {
    if (!std::isfinite(res_bf16[i])) {
      std::cerr << "BF16 DeltaNet decode produced non-finite output\n";
      std::abort();
    }
    bf16_output_max_diff =
        std::max(bf16_output_max_diff, std::abs(res_seq[i] - res_bf16[i]));
  }

  std::vector<float> delta_fp32(delta_size);
  std::vector<std::uint16_t> delta_bf16(delta_size);
  HIP_CHECK(hipMemcpy(delta_fp32.data(), d_delta_seq,
                      delta_size * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(delta_bf16.data(), d_delta_bf16,
                      delta_size * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  float bf16_state_max_diff = 0.0F;
  for (std::size_t i = 0; i < delta_size; ++i) {
    const float value = gufo::test::Bf16BitsToFloat(delta_bf16[i]);
    if (!std::isfinite(value)) {
      std::cerr << "BF16 DeltaNet decode produced non-finite state\n";
      std::abort();
    }
    bf16_state_max_diff =
        std::max(bf16_state_max_diff, std::abs(delta_fp32[i] - value));
  }
  std::cout << "DeltaNet BF16 decode max output diff: " << bf16_output_max_diff
            << " state diff: " << bf16_state_max_diff << "\n";
  if (bf16_output_max_diff >= 2e-2F || bf16_state_max_diff >= 1e-2F) {
    std::cerr << "BF16 DeltaNet decode exceeded the storage envelope\n";
    std::abort();
  }

  // Verification and rollback replay must preserve every state bit. Reuse the
  // existing buffers to cover both storage formats, per-row and whole-block
  // dispatch, and omission of outputs that replay never consumes.
  const auto expect_device_equal = [](const void* expected, const void* actual,
                                      std::size_t bytes) {
    std::vector<std::uint8_t> reference(bytes), candidate(bytes);
    HIP_CHECK(
        hipMemcpy(reference.data(), expected, bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(candidate.data(), actual, bytes, hipMemcpyDeviceToHost));
    if (reference != candidate) {
      throw std::runtime_error("SSM verification/replay changed stored bits");
    }
  };
  using Storage = gufo::hip::QwenRecurrentStateStorage;
  for (const auto storage : {Storage::kFp32, Storage::kBf16}) {
    const std::size_t state_bytes =
        delta_size * gufo::hip::QwenRecurrentStateElementBytes(storage);
    const void* reference_state = storage == Storage::kFp32
                                      ? static_cast<void*>(d_delta_seq)
                                      : d_delta_bf16;
    const float* reference_output =
        storage == Storage::kFp32 ? d_out_seq : d_out_bf16;
    for (const bool write_output : {true, false}) {
      for (const auto rows_per_launch : {std::size_t{1}, batch}) {
        HIP_CHECK(hipMemset(d_state_batch, 0, qkv_dim * 4 * sizeof(float)));
        HIP_CHECK(hipMemset(d_delta_batch, 0, state_bytes));
        for (std::size_t offset = 0; offset < batch;
             offset += rows_per_launch) {
          gufo::hip::LaunchSSMConvRecurrenceRows(
              d_qkv + offset * qkv_dim, d_w,
              QwenRecurrentStateIo::InPlace(d_state_batch, d_delta_batch),
              d_conv_out_batch, d_alpha + offset * num_heads,
              d_beta + offset * num_heads, d_ssm_a, d_ssm_dt, d_ssm_norm,
              d_gate + offset * inner_size,
              write_output ? d_out_batch + offset * inner_size : nullptr, 0,
              qkv_dim, num_key_heads, num_heads, key_dim, val_dim,
              static_cast<std::uint32_t>(rows_per_launch), num_heads,
              inner_size, nullptr, {}, storage);
        }
        expect_device_equal(reference_state, d_delta_batch, state_bytes);
        expect_device_equal(d_state_seq, d_state_batch,
                            qkv_dim * 4 * sizeof(float));
        if (write_output) {
          expect_device_equal(reference_output, d_out_batch,
                              batch * inner_size * sizeof(float));
        }
      }
    }
  }
  std::cout << "SSM verification and state-only replay: bit-exact\n";

  HIP_CHECK(hipFree(d_qkv));
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_state_seq));
  HIP_CHECK(hipFree(d_state_batch));
  HIP_CHECK(hipFree(d_state_bf16));
  HIP_CHECK(hipFree(d_conv_out_seq));
  HIP_CHECK(hipFree(d_conv_out_batch));
  HIP_CHECK(hipFree(d_conv_out_bf16));
  HIP_CHECK(hipFree(d_delta_seq));
  HIP_CHECK(hipFree(d_delta_batch));
  HIP_CHECK(hipFree(d_delta_bf16));
  HIP_CHECK(hipFree(d_alpha));
  HIP_CHECK(hipFree(d_beta));
  HIP_CHECK(hipFree(d_ssm_a));
  HIP_CHECK(hipFree(d_ssm_dt));
  HIP_CHECK(hipFree(d_ssm_norm));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_out_seq));
  HIP_CHECK(hipFree(d_out_batch));
  HIP_CHECK(hipFree(d_out_bf16));
}

// opt-c170-deltanet-rowsplit: the row-split recurrence spreads the 128 state
// rows of a head over 16 waves and applies the k/q normalization scales to the
// reduced dot products instead of to the 128-wide vectors. That is the same
// arithmetic reassociated, so it must agree with the single-block kernel to
// FP32 rounding over a full chunk -- including the carried state, which is what
// accumulates any real error across tokens.
//
// Both register tiles are exercised: the launcher picks the 32-key/one-row tile
// at or below 2048 tokens and the two-row prefetching tile above it.
void TestBatchedSSMRowSplitRecurrenceEquivalence(std::size_t batch) {
  constexpr std::uint32_t num_key_heads = 16;
  constexpr std::uint32_t num_heads = 48;
  constexpr std::uint32_t key_dim = 128;
  constexpr std::uint32_t val_dim = 128;
  constexpr std::size_t qkv_dim =
      (2 * num_key_heads * key_dim) + (num_heads * val_dim);
  constexpr std::size_t inner_size = num_heads * val_dim;
  const std::size_t delta_size =
      static_cast<std::size_t>(num_heads) * key_dim * val_dim;

  std::cout << "Batched SSM row-split recurrence: batch=" << batch << "\n";

  std::vector<float> h_qkv(batch * qkv_dim);
  for (std::size_t i = 0; i < h_qkv.size(); ++i) {
    h_qkv[i] = 0.6F * std::sin(0.017F * static_cast<float>(i % 997)) +
               0.2F * std::cos(0.003F * static_cast<float>(i % 31));
  }
  std::vector<float> h_weights(qkv_dim * 4);
  for (std::size_t i = 0; i < h_weights.size(); ++i) {
    h_weights[i] = 0.2F + 0.05F * static_cast<float>(i % 7);
  }
  // Non-trivial decay and beta gates: with alpha and beta pinned to zero the
  // recurrence degenerates to one decay value and the state carry is not
  // tested.
  std::vector<float> h_alpha(batch * num_heads);
  std::vector<float> h_beta(batch * num_heads);
  for (std::size_t i = 0; i < h_alpha.size(); ++i) {
    h_alpha[i] = 1.5F + 0.5F * std::sin(0.011F * static_cast<float>(i % 401));
    h_beta[i] = 0.3F * std::cos(0.007F * static_cast<float>(i % 211));
  }
  std::vector<float> h_ssm_a(num_heads);
  std::vector<float> h_ssm_dt(num_heads);
  for (std::uint32_t i = 0; i < num_heads; ++i) {
    h_ssm_a[i] = -0.04F - 0.01F * static_cast<float>(i % 5);
    h_ssm_dt[i] = 0.1F * static_cast<float>(i % 3);
  }
  std::vector<float> h_ssm_norm(val_dim);
  for (std::size_t i = 0; i < val_dim; ++i) {
    h_ssm_norm[i] = 0.9F + 0.05F * static_cast<float>(i % 23);
  }
  std::vector<float> h_gate(batch * inner_size);
  for (std::size_t i = 0; i < h_gate.size(); ++i) {
    h_gate[i] = 0.5F * std::sin(0.013F * static_cast<float>(i + 1));
  }
  std::vector<float> h_history(qkv_dim * 4);
  for (std::size_t i = 0; i < h_history.size(); ++i) {
    h_history[i] = 0.4F * std::sin(0.031F * static_cast<float>(i % 211));
  }
  // A non-zero starting state so the decay path is live from the first token.
  std::vector<float> h_delta0(delta_size);
  std::vector<std::uint16_t> h_delta0_bf16(delta_size);
  for (std::size_t i = 0; i < h_delta0.size(); ++i) {
    h_delta0[i] = 0.01F * std::sin(0.013F * static_cast<float>(i % 577));
    h_delta0_bf16[i] = gufo::test::FloatToBf16Bits(h_delta0[i]);
  }

  float *d_qkv = nullptr, *d_w = nullptr;
  float *d_state_ref = nullptr, *d_state_new = nullptr;
  float* d_state_bf16 = nullptr;
  float *d_conv_ref = nullptr, *d_conv_new = nullptr;
  float* d_conv_bf16 = nullptr;
  float *d_delta_ref = nullptr, *d_delta_new = nullptr;
  void* d_delta_bf16 = nullptr;
  float *d_alpha = nullptr, *d_beta = nullptr;
  float *d_ssm_a = nullptr, *d_ssm_dt = nullptr, *d_ssm_norm = nullptr;
  float* d_gate = nullptr;
  float *d_out_ref = nullptr, *d_out_new = nullptr;
  float* d_out_bf16 = nullptr;
  float *d_kq = nullptr, *d_ab = nullptr;

  HIP_CHECK(hipMalloc(&d_qkv, batch * qkv_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_w, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_state_ref, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_state_new, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_state_bf16, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_conv_ref, batch * qkv_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_conv_new, batch * qkv_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_conv_bf16, batch * qkv_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_delta_ref, delta_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_delta_new, delta_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_delta_bf16, delta_size * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_alpha, batch * num_heads * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_beta, batch * num_heads * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_a, num_heads * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_dt, num_heads * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_norm, val_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, batch * inner_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_ref, batch * inner_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_new, batch * inner_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_bf16, batch * inner_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_kq, batch * num_key_heads * 3 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ab, batch * num_heads * 2 * sizeof(float)));

  const auto upload = [](float* dst, const std::vector<float>& src) {
    HIP_CHECK(hipMemcpy(dst, src.data(), src.size() * sizeof(float),
                        hipMemcpyHostToDevice));
  };
  upload(d_qkv, h_qkv);
  upload(d_w, h_weights);
  upload(d_alpha, h_alpha);
  upload(d_beta, h_beta);
  upload(d_ssm_a, h_ssm_a);
  upload(d_ssm_dt, h_ssm_dt);
  upload(d_ssm_norm, h_ssm_norm);
  upload(d_gate, h_gate);
  upload(d_delta_ref, h_delta0);
  upload(d_delta_new, h_delta0);
  HIP_CHECK(hipMemcpy(d_delta_bf16, h_delta0_bf16.data(),
                      h_delta0_bf16.size() * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice));
  upload(d_state_ref, h_history);
  upload(d_state_new, h_history);
  upload(d_state_bf16, h_history);

  using gufo::hip::QwenRecurrentStateIo;
  gufo::hip::LaunchBatchedSSMConvRecurrence(
      d_qkv, d_w, QwenRecurrentStateIo::InPlace(d_state_ref, d_delta_ref),
      d_conv_ref, d_alpha, d_beta, d_ssm_a, d_ssm_dt, d_ssm_norm, d_gate,
      d_out_ref, 0, batch, qkv_dim, num_key_heads, num_heads, key_dim, val_dim);

  if (!gufo::hip::IsDeltaNetRowSplitSupported(key_dim, val_dim)) {
    std::cerr << "row-split recurrence rejected the production state shape\n";
    std::abort();
  }
  gufo::hip::LaunchBatchedSSMConvRecurrenceRowSplit(
      d_qkv, d_w, QwenRecurrentStateIo::InPlace(d_state_new, d_delta_new),
      d_conv_new, d_alpha, d_beta, d_ssm_a, d_ssm_dt, d_ssm_norm, d_gate,
      d_out_new, /*q8_out=*/nullptr, d_kq, d_ab, 0, batch, qkv_dim,
      num_key_heads, num_heads, key_dim, val_dim);
  gufo::hip::LaunchBatchedSSMConvRecurrenceRowSplit(
      d_qkv, d_w, QwenRecurrentStateIo::InPlace(d_state_bf16, d_delta_bf16),
      d_conv_bf16, d_alpha, d_beta, d_ssm_a, d_ssm_dt, d_ssm_norm, d_gate,
      d_out_bf16, /*q8_out=*/nullptr, d_kq, d_ab, 0, batch, qkv_dim,
      num_key_heads, num_heads, key_dim, val_dim, nullptr,
      gufo::hip::QwenRecurrentStateStorage::kBf16);

  HIP_CHECK(hipDeviceSynchronize());
  CheckSSMConvolution(h_qkv, h_weights, h_history, d_conv_ref, d_state_ref,
                      batch, qkv_dim);
  CheckSSMConvolution(h_qkv, h_weights, h_history, d_conv_new, d_state_new,
                      batch, qkv_dim);
  CheckSSMConvolution(h_qkv, h_weights, h_history, d_conv_bf16, d_state_bf16,
                      batch, qkv_dim);

  const auto download = [](std::vector<float>& dst, const float* src) {
    HIP_CHECK(hipMemcpy(dst.data(), src, dst.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
  };
  std::vector<float> out_ref(batch * inner_size);
  std::vector<float> out_new(batch * inner_size);
  std::vector<float> out_bf16(batch * inner_size);
  std::vector<float> delta_ref(delta_size);
  std::vector<float> delta_new(delta_size);
  std::vector<std::uint16_t> delta_bf16_bits(delta_size);
  std::vector<float> delta_bf16(delta_size);
  download(out_ref, d_out_ref);
  download(out_new, d_out_new);
  download(out_bf16, d_out_bf16);
  download(delta_ref, d_delta_ref);
  download(delta_new, d_delta_new);
  HIP_CHECK(hipMemcpy(delta_bf16_bits.data(), d_delta_bf16,
                      delta_bf16_bits.size() * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  for (std::size_t i = 0; i < delta_bf16.size(); ++i) {
    delta_bf16[i] = gufo::test::Bf16BitsToFloat(delta_bf16_bits[i]);
  }

  const auto compare = [](const char* label, const std::vector<float>& want,
                          const std::vector<float>& got) {
    double max_abs = 0.0;
    double magnitude = 0.0;
    std::size_t non_finite = 0;
    for (std::size_t i = 0; i < want.size(); ++i) {
      if (!std::isfinite(got[i])) {
        ++non_finite;
        continue;
      }
      max_abs = std::fmax(max_abs, std::fabs(static_cast<double>(got[i]) -
                                             static_cast<double>(want[i])));
      magnitude = std::fmax(magnitude, std::fabs(static_cast<double>(want[i])));
    }
    const double rel = (magnitude > 0.0) ? (max_abs / magnitude) : max_abs;
    std::cout << "  " << label << ": max_abs=" << max_abs
              << " magnitude=" << magnitude << " rel=" << rel
              << " non_finite=" << non_finite << "\n";
    if (non_finite != 0 || rel > 1e-5) {
      std::cerr << label
                << ": row-split recurrence disagrees with the single-block "
                   "kernel\n";
      std::abort();
    }
  };
  compare("gated output", out_ref, out_new);
  compare("carried state", delta_ref, delta_new);

  const auto compare_bf16 = [](const char* label,
                               const std::vector<float>& want,
                               const std::vector<float>& got,
                               double relative_tolerance) {
    double max_abs = 0.0;
    double magnitude = 0.0;
    std::size_t non_finite = 0;
    for (std::size_t i = 0; i < want.size(); ++i) {
      if (!std::isfinite(got[i])) {
        ++non_finite;
        continue;
      }
      max_abs = std::fmax(max_abs, std::fabs(static_cast<double>(got[i]) -
                                             static_cast<double>(want[i])));
      magnitude = std::fmax(magnitude, std::fabs(static_cast<double>(want[i])));
    }
    const double relative = magnitude > 0.0 ? max_abs / magnitude : max_abs;
    std::cout << "  BF16 " << label << ": max_abs=" << max_abs
              << " magnitude=" << magnitude << " rel=" << relative
              << " non_finite=" << non_finite << "\n";
    if (non_finite != 0 || relative > relative_tolerance) {
      std::cerr << "BF16 " << label
                << " exceeded the recurrent-storage envelope\n";
      std::abort();
    }
  };
  compare_bf16("gated output", out_ref, out_bf16, 3e-2);
  compare_bf16("carried state", delta_ref, delta_bf16, 3e-2);

  // Direct FP16 output must match the original FP32 epilogue followed by a
  // cast, for both recurrent-state formats and both register-tile sizes.
  // Reuse the existing scratch and restart from the same nonzero state.
  for (const bool bf16_state : {false, true}) {
    gufo::test::DeviceBuffer<std::uint16_t> expected(batch * inner_size);
    gufo::test::DeviceBuffer<std::uint16_t> actual(batch * inner_size);
    float* output = bf16_state ? d_out_bf16 : d_out_new;
    float* conv_state = bf16_state ? d_state_bf16 : d_state_new;
    float* conv_output = bf16_state ? d_conv_bf16 : d_conv_new;
    void* state = bf16_state ? d_delta_bf16 : d_delta_new;
    const auto storage = bf16_state
                             ? gufo::hip::QwenRecurrentStateStorage::kBf16
                             : gufo::hip::QwenRecurrentStateStorage::kFp32;
    gufo::hip::LaunchFloatToFp16(output, expected.data(), batch * inner_size,
                                 nullptr);
    upload(conv_state, h_history);
    if (bf16_state) {
      HIP_CHECK(hipMemcpy(state, h_delta0_bf16.data(),
                          delta_size * sizeof(std::uint16_t),
                          hipMemcpyHostToDevice));
    } else {
      upload(static_cast<float*>(state), h_delta0);
    }
    gufo::hip::LaunchBatchedSSMConvRecurrenceRowSplit(
        d_qkv, d_w, QwenRecurrentStateIo::InPlace(conv_state, state),
        conv_output, d_alpha, d_beta, d_ssm_a, d_ssm_dt, d_ssm_norm, d_gate,
        output, nullptr, d_kq, d_ab, 0, batch, qkv_dim, num_key_heads,
        num_heads, key_dim, val_dim, nullptr, storage, actual.data());
    const auto actual_bits = actual.CopyToHost();
    const auto expected_bits = expected.CopyToHost();
    {
      // This route leaves the pre-normalization recurrence output in scratch.
      // Evaluate the formula independently to distinguish an inaccurate
      // fusion from harmless differences in the separate FP32 calculation.
      std::vector<float> raw(batch * inner_size);
      download(raw, output);
      double fused_squared_error = 0, separate_squared_error = 0;
      double fused_max_error = 0, separate_max_error = 0;
      std::size_t mismatches = 0, fused_closer = 0, separate_closer = 0;
      for (std::size_t head = 0; head < batch * num_heads; ++head) {
        double sum = 0;
        for (std::size_t column = 0; column < val_dim; ++column) {
          const double value = raw[head * val_dim + column];
          sum += value * value;
        }
        const double inv = 1 / std::sqrt(sum / val_dim + double(1e-6F));
        for (std::size_t column = 0; column < val_dim; ++column) {
          const auto i = head * val_dim + column;
          const double g = h_gate[i];
          const double reference = double(raw[i]) * inv *
                                   double(h_ssm_norm[column]) * g /
                                   (1 + std::exp(-g));
          const double fused = gufo::quant::Fp16ToFloat(actual_bits[i]);
          const double separate = gufo::quant::Fp16ToFloat(expected_bits[i]);
          const double fused_error = std::abs(fused - reference);
          const double separate_error = std::abs(separate - reference);
          // FP16 half-ULP rounding plus the existing 1e-5 FP32 envelope.
          // Include half a subnormal step for outputs close to zero.
          const double bound =
              std::abs(reference) * (1.0 / 2048 + 1e-5) + std::ldexp(1.0, -25);
          if (!std::isfinite(fused) || !std::isfinite(separate) ||
              fused_error > bound || separate_error > bound) {
            throw std::runtime_error("SSM FP16 epilogue exceeded FP64 bound");
          }
          fused_squared_error += fused_error * fused_error;
          separate_squared_error += separate_error * separate_error;
          fused_max_error = std::max(fused_max_error, fused_error);
          separate_max_error = std::max(separate_max_error, separate_error);
          if (actual_bits[i] != expected_bits[i]) {
            if (mismatches++ < 4)
              std::cout << "  FP16 difference " << i << ": fused=" << fused
                        << " separate=" << separate << " FP64=" << reference
                        << "\n";
            fused_closer += fused_error < separate_error;
            separate_closer += separate_error < fused_error;
          }
        }
      }
      std::cout << "  FP16 oracle: mismatches=" << mismatches
                << " fused_closer=" << fused_closer
                << " separate_closer=" << separate_closer
                << " fused_rmse=" << std::sqrt(fused_squared_error / raw.size())
                << " separate_rmse="
                << std::sqrt(separate_squared_error / raw.size())
                << " fused_max=" << fused_max_error
                << " separate_max=" << separate_max_error << "\n";
    }
    if (actual_bits != expected_bits) {
      throw std::runtime_error("fused SSM FP16 epilogue changed output bits");
    }
    std::vector<std::uint8_t> state_bits(
        delta_size * gufo::hip::QwenRecurrentStateElementBytes(storage));
    HIP_CHECK(hipMemcpy(state_bits.data(), state, state_bits.size(),
                        hipMemcpyDeviceToHost));
    const void* expected_state =
        bf16_state ? static_cast<void*>(delta_bf16_bits.data())
                   : static_cast<void*>(delta_new.data());
    if (std::memcmp(state_bits.data(), expected_state, state_bits.size()) !=
        0) {
      throw std::runtime_error("FP16 epilogue changed recurrent state");
    }
  }
  // Restore the FP32 reference consumed by the Q8_1 epilogue check below.
  upload(d_out_new, out_new);

  // opt-c174-ssm-epilogue-quant: with a Q8_1 destination the epilogue quantizes
  // the gated row in the same pass instead of storing FP32 for the quantizer to
  // read back. That has to be byte-for-byte identical to the FP32 path followed
  // by an FP32 quantize, including the per-block scales and the tail-tile
  // zeros.
  if (gufo::hip::IsFusedSSMEpilogueQuantizeQ8_1Supported(val_dim, inner_size)) {
    const std::size_t q8_bytes =
        gufo::hip::QuantizedActivationBytes(batch, inner_size) + 4096;
    void* d_q8_ref = nullptr;
    void* d_q8_got = nullptr;
    float* d_state_q8 = nullptr;
    float* d_conv_q8 = nullptr;
    float* d_delta_q8 = nullptr;
    float* d_out_q8 = nullptr;
    HIP_CHECK(hipMalloc(&d_q8_ref, q8_bytes));
    HIP_CHECK(hipMalloc(&d_q8_got, q8_bytes));
    HIP_CHECK(hipMalloc(&d_state_q8, qkv_dim * 4 * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_conv_q8, batch * qkv_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_delta_q8, delta_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out_q8, batch * inner_size * sizeof(float)));
    HIP_CHECK(hipMemset(d_q8_ref, 0xA5, q8_bytes));
    HIP_CHECK(hipMemset(d_q8_got, 0xA5, q8_bytes));

    // Reference: the FP32 epilogue that just ran, quantized separately.
    gufo::hip::LaunchQuantizeActivationQ8_1FromFp32(d_out_new, d_q8_ref, batch,
                                                    inner_size);

    // Candidate: the same recurrence from the same starting state, but with the
    // epilogue writing Q8_1.
    upload(d_state_q8, h_history);
    upload(d_delta_q8, h_delta0);
    gufo::hip::LaunchBatchedSSMConvRecurrenceRowSplit(
        d_qkv, d_w, QwenRecurrentStateIo::InPlace(d_state_q8, d_delta_q8),
        d_conv_q8, d_alpha, d_beta, d_ssm_a, d_ssm_dt, d_ssm_norm, d_gate,
        d_out_q8, d_q8_got, d_kq, d_ab, 0, batch, qkv_dim, num_key_heads,
        num_heads, key_dim, val_dim);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<std::uint8_t> q8_ref(q8_bytes);
    std::vector<std::uint8_t> q8_got(q8_bytes);
    HIP_CHECK(
        hipMemcpy(q8_ref.data(), d_q8_ref, q8_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(q8_got.data(), d_q8_got, q8_bytes, hipMemcpyDeviceToHost));
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < q8_bytes; ++i) {
      if (q8_ref[i] != q8_got[i]) {
        ++mismatches;
      }
    }
    std::cout << "  fused Q8_1 epilogue mismatching bytes: " << mismatches
              << " of " << q8_bytes << "\n";
    if (mismatches != 0) {
      std::cerr << "fused SSM Q8_1 epilogue differs from the FP32 epilogue "
                   "plus a separate quantize\n";
      std::abort();
    }

    HIP_CHECK(hipFree(d_q8_ref));
    HIP_CHECK(hipFree(d_q8_got));
    HIP_CHECK(hipFree(d_state_q8));
    HIP_CHECK(hipFree(d_conv_q8));
    HIP_CHECK(hipFree(d_delta_q8));
    HIP_CHECK(hipFree(d_out_q8));
  }

  HIP_CHECK(hipFree(d_delta_bf16));
  for (float* p :
       {d_qkv,      d_w,        d_state_ref, d_state_new, d_state_bf16,
        d_conv_ref, d_conv_new, d_conv_bf16, d_delta_ref, d_delta_new,
        d_alpha,    d_beta,     d_ssm_a,     d_ssm_dt,    d_ssm_norm,
        d_gate,     d_out_ref,  d_out_new,   d_out_bf16,  d_kq,
        d_ab}) {
    HIP_CHECK(hipFree(p));
  }
}

// A prefill chunk that follows a rollback pin (prefill right after a cancelled
// request's restore) reads the pinned conv history and state and writes the
// other buffer pair. Both prefill recurrences must then produce the same bytes
// as the in-place form and leave the pinned buffers untouched. Short batches
// matter: the new conv history takes its tail from the old one when fewer than
// four tokens arrive.
void TestBatchedRecurrenceDivergingState(std::size_t batch) {
  using gufo::hip::QwenRecurrentStateIo;
  using gufo::test::DeviceBuffer;
  using Storage = gufo::hip::QwenRecurrentStateStorage;
  constexpr std::uint32_t num_key_heads = 16;
  constexpr std::uint32_t num_heads = 48;
  constexpr std::uint32_t key_dim = 128;
  constexpr std::uint32_t val_dim = 128;
  constexpr std::size_t qkv_dim =
      (2 * num_key_heads * key_dim) + (num_heads * val_dim);
  constexpr std::size_t inner_size = num_heads * val_dim;
  constexpr std::size_t delta_size =
      static_cast<std::size_t>(num_heads) * key_dim * val_dim;
  std::cout << "Batched recurrence from a pinned state: batch=" << batch
            << "\n";

  const auto values = [](std::size_t size, float scale, float phase) {
    std::vector<float> result(size);
    for (std::size_t i = 0; i < size; ++i)
      result[i] = scale * std::sin(phase + static_cast<float>(i) * 0.013F);
    return result;
  };
  const auto expect = [](bool condition, const char* message) {
    if (!condition)
      throw std::runtime_error(message);
  };
  const auto same_bytes = [](const auto& expected, const auto& actual) {
    return expected.size() == actual.size() &&
           std::memcmp(expected.data(), actual.data(),
                       expected.size() * sizeof(expected[0])) == 0;
  };
  DeviceBuffer<float> qkv(values(batch * qkv_dim, 0.6F, 1)),
      weights(values(qkv_dim * 4, 0.25F, 2)),
      alpha(values(batch * num_heads, 1.5F, 3)),
      beta(values(batch * num_heads, 0.3F, 4)),
      decay(std::vector<float>(num_heads, -0.05F)),
      dt(values(num_heads, 0.1F, 5)), norm(values(val_dim, 0.9F, 6)),
      gate(values(batch * inner_size, 0.5F, 7)), conv_output(batch * qkv_dim),
      output(batch * inner_size), kq_scales(batch * num_key_heads * 3),
      alpha_beta(batch * num_heads * 2);
  const auto history = values(qkv_dim * 4, 0.4F, 8);
  const auto delta = values(delta_size, 0.01F, 9);
  const std::size_t conv_bytes = history.size() * sizeof(float);

  for (const auto storage : {Storage::kFp32, Storage::kBf16}) {
    const std::size_t element_bytes =
        gufo::hip::QwenRecurrentStateElementBytes(storage);
    std::vector<std::uint8_t> delta_seed(delta_size * element_bytes);
    for (std::size_t i = 0; i < delta_size; ++i) {
      if (storage == Storage::kFp32) {
        std::memcpy(delta_seed.data() + i * element_bytes, &delta[i],
                    element_bytes);
      } else {
        const auto bits = gufo::test::FloatToBf16Bits(delta[i]);
        std::memcpy(delta_seed.data() + i * element_bytes, &bits,
                    element_bytes);
      }
    }
    DeviceBuffer<float> conv_in_place(history), conv_in(history),
        conv_out(history.size());
    DeviceBuffer<std::uint8_t> delta_in_place(delta_seed), delta_in(delta_seed),
        delta_out(delta_seed.size());
    const auto in_place = QwenRecurrentStateIo::InPlace(conv_in_place.data(),
                                                        delta_in_place.data());
    const QwenRecurrentStateIo diverging{conv_in.data(), conv_out.data(),
                                         delta_in.data(), delta_out.data()};
    const auto reseed = [&] {
      conv_in_place.CopyFrom(history);
      delta_in_place.CopyFrom(delta_seed);
      HIP_CHECK(hipMemset(conv_out.data(), 0x5A, conv_bytes));
      HIP_CHECK(hipMemset(delta_out.data(), 0x5A, delta_seed.size()));
      HIP_CHECK(
          hipMemset(output.data(), 0, batch * inner_size * sizeof(float)));
    };
    const auto compare = [&](const char* route) {
      HIP_CHECK(hipDeviceSynchronize());
      expect(
          same_bytes(conv_in_place.CopyToHost(), conv_out.CopyToHost()) &&
              same_bytes(delta_in_place.CopyToHost(), delta_out.CopyToHost()),
          route);
      expect(same_bytes(history, conv_in.CopyToHost()) &&
                 same_bytes(delta_seed, delta_in.CopyToHost()),
             "diverging prefill wrote into the pinned buffers");
    };
    const auto run_plain = [&](QwenRecurrentStateIo state) {
      gufo::hip::LaunchBatchedSSMConvRecurrence(
          qkv.data(), weights.data(), state, conv_output.data(), alpha.data(),
          beta.data(), decay.data(), dt.data(), norm.data(), gate.data(),
          output.data(), 0, batch, qkv_dim, num_key_heads, num_heads, key_dim,
          val_dim, nullptr, storage);
    };
    const auto run_row_split = [&](QwenRecurrentStateIo state) {
      gufo::hip::LaunchBatchedSSMConvRecurrenceRowSplit(
          qkv.data(), weights.data(), state, conv_output.data(), alpha.data(),
          beta.data(), decay.data(), dt.data(), norm.data(), gate.data(),
          output.data(), nullptr, kq_scales.data(), alpha_beta.data(), 0, batch,
          qkv_dim, num_key_heads, num_heads, key_dim, val_dim, nullptr,
          storage);
    };

    reseed();
    run_plain(in_place);
    HIP_CHECK(hipDeviceSynchronize());
    const auto plain_output = output.CopyToHost();
    run_plain(diverging);
    compare("diverging batched recurrence changed the state");
    expect(same_bytes(plain_output, output.CopyToHost()),
           "diverging batched recurrence changed the output");

    reseed();
    run_row_split(in_place);
    HIP_CHECK(hipDeviceSynchronize());
    const auto row_split_output = output.CopyToHost();
    run_row_split(diverging);
    compare("diverging row-split recurrence changed the state");
    expect(same_bytes(row_split_output, output.CopyToHost()),
           "diverging row-split recurrence changed the output");
  }
}

#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
  std::cout << std::unitbuf;
#if defined(ENGINE_ENABLE_HIP)
  const int device_status = gufo::test::GateHipDevice(
      gufo::test::HipDeviceRequirement::kOptional, "Qwen SSM ops test");
  if (device_status != gufo::test::kHipTestSuccess) {
    return device_status;
  }

  TestBatchedSSMConvEquivalence();
  TestConcurrentSsmRecurrence();
  TestRecurrentRollbackRows(false);
  TestRecurrentRollbackRows(true);
  TestRecurrentForkPingPong();
  TestBf16RecurrentMemoryAndSnapshot();
  for (const std::size_t batch : {1U, 2U, 3U, 7U, 96U}) {
    TestBatchedSSMRowSplitRecurrenceEquivalence(batch);
  }
  // Above the launcher's 2048-token crossover, so the two-row prefetching tile
  // runs too.
  TestBatchedSSMRowSplitRecurrenceEquivalence(2080);
  for (const std::size_t batch : {1U, 2U, 3U, 7U, 96U}) {
    TestBatchedRecurrenceDivergingState(batch);
  }
  TestBatchedRecurrenceDivergingState(2080);
  std::cout << "Qwen ssm ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen ssm ops test.\n";
  return 77;
#endif
}
