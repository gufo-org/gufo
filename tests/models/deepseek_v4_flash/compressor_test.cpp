#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "src/models/deepseek_v4_flash/kernels/rocm/resident_api.h"

namespace {
void Check(bool success, const char* message) {
  if (!success) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
  }
}

struct Tensor {
  explicit Tensor(size_t bytes) : data(ds4_gpu_tensor_alloc(bytes)) {
    Check(data != nullptr, "allocate tensor");
  }
  Tensor(ds4_gpu_tensor* source, size_t offset, size_t bytes)
      : data(ds4_gpu_tensor_view(source, offset, bytes)) {
    Check(data != nullptr, "bounded tensor view");
  }
  ~Tensor() { ds4_gpu_tensor_free(data); }
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;
  ds4_gpu_tensor* data;
};

void Write(Tensor& tensor, const std::vector<float>& values) {
  Check(ds4_gpu_tensor_write(tensor.data, 0, values.data(),
                             values.size() * sizeof(float)),
        "write tensor");
}

std::vector<float> Read(Tensor& tensor, size_t count) {
  std::vector<float> values(count);
  Check(ds4_gpu_tensor_read(tensor.data, 0, values.data(),
                            values.size() * sizeof(float)),
        "read tensor");
  return values;
}

void Run(const std::vector<float>& model, unsigned dim, unsigned ratio,
         unsigned phase) {
  constexpr size_t norm_offset = 128 * 1024;
  const unsigned coff = ratio == 4 ? 2 : 1, width = coff * dim;
  const unsigned rows = 2 * ratio + 3, start = ratio * 3 + phase;
  const unsigned outputs = (phase + rows) / ratio;
  const size_t state_size = coff * ratio * width;
  std::vector<float> kv(rows * width), scores(kv.size());
  for (size_t i = 0; i < kv.size(); ++i) {
    kv[i] = float(int((i * 19) % 521) - 260) / 256;
    scores[i] = float(int((i * 13) % 257) - 128) / 32;
  }
  Tensor input(kv.size() * sizeof(float)), sc(scores.size() * sizeof(float));
  Write(input, kv);
  Write(sc, scores);
  std::vector<float> sequential;
  // Mode 0 stores one row at a time, mode 1 stores spans up to each boundary,
  // and mode 2 replays whole windows.
  for (int mode = 0; mode < 3; ++mode) {
    const bool grouped = mode != 0;
    std::vector<float> expected_kv(state_size, 0),
        expected_score(state_size, -100);
    Tensor state(state_size * sizeof(float)), ss(state_size * sizeof(float)),
        output(outputs * dim * sizeof(float));
    Write(state, expected_kv);
    Write(ss, expected_score);
    Check(!ds4_gpu_compressor_store_batch_tensor(
              input.data, sc.data, state.data, ss.data, model.data(),
              model.size() * sizeof(float), 0, 0, dim, ratio, start,
              ratio - phase + 1),
          "a store spanning a pool boundary is rejected");
    unsigned produced = 0;
    for (unsigned t = 0; t < rows;) {
      const unsigned whole = mode == 2 && (start + t) % ratio == 0
                                 ? (rows - t) / ratio * ratio
                                 : 0;
      const unsigned count =
          whole     ? whole
          : grouped ? std::min(rows - t, ratio - (start + t) % ratio)
                    : 1;
      const size_t offset = size_t(t) * width * sizeof(float);
      const size_t bytes = size_t(count) * width * sizeof(float);
      Tensor k(input.data, offset, bytes), s(sc.data, offset, bytes);
      const unsigned position = start + t + count - 1;
      if (whole) {
        Tensor out(output.data, size_t(produced) * dim * sizeof(float),
                   size_t(whole / ratio) * dim * sizeof(float));
        Check(ds4_gpu_compressor_prefill_replay_tensor(
                  out.data, state.data, ss.data, k.data, s.data, model.data(),
                  model.size() * sizeof(float), 0, 0,
                  norm_offset * sizeof(float), 0, dim, ratio, start + t, whole,
                  16, 0, false, 10000, 1, 0, 1, 32, 1, 1e-6F, nullptr),
              "replay whole windows");
      } else {
        if (grouped)
          Check(ds4_gpu_compressor_store_batch_tensor(
                    k.data, s.data, state.data, ss.data, model.data(),
                    model.size() * sizeof(float), 0, 0, dim, ratio, start + t,
                    count),
                "store a disjoint span");
        Check(
            ds4_gpu_compressor_update_tensor(
                k.data, s.data, state.data, ss.data, output.data, model.data(),
                model.size() * sizeof(float), 0, 0, norm_offset * sizeof(float),
                0, dim, ratio, position, produced, 16, 0, 10000, 1, 0, 1, 32, 1,
                1e-6F, grouped, false, false),
            "pool and update compressor state");
      }
      for (unsigned row = t; row < t + count; ++row) {
        const unsigned slot = (start + row) % ratio;
        const unsigned dst = ratio == 4 ? ratio + slot : slot;
        for (unsigned d = 0; d < width; ++d) {
          expected_kv[dst * width + d] = kv[row * width + d];
          expected_score[dst * width + d] =
              scores[row * width + d] + model[slot * width + d];
        }
        if (slot + 1 == ratio) {
          ++produced;
          if (ratio == 4) {
            std::copy_n(expected_kv.begin() + ratio * width, ratio * width,
                        expected_kv.begin());
            std::copy_n(expected_score.begin() + ratio * width, ratio * width,
                        expected_score.begin());
          }
        }
      }
      Check(Read(state, state_size) == expected_kv,
            "state values match the independent row mapping");
      Check(Read(ss, state_size) == expected_score,
            "scores, untouched slots and ratio-four shift match");
      t += count;
    }
    Check(produced == outputs, "all compression boundaries emitted");
    const auto result = Read(output, outputs * dim);
    if (grouped)
      Check(std::memcmp(result.data(), sequential.data(),
                        result.size() * sizeof(float)) == 0,
            "grouped and replayed compressed rows match the sequential rows "
            "bit for bit");
    else
      sequential = result;
  }
}
}  // namespace

int main() {
  if (!ds4_gpu_init())
    return 77;
  std::vector<float> model(128 * 1024 + 512, 1);
  for (size_t i = 0; i < 128 * 1024; ++i)
    model[i] = float(int(i % 257) - 128) / 1024;
  const size_t bytes = model.size() * sizeof(float);
  Check(ds4_gpu_reserve_support_map(model.data(), bytes, bytes) &&
            ds4_gpu_cache_support_range(model.data(), bytes, 0, bytes,
                                        "compressor fixture"),
        "make synthetic APE and norm weights resident");
  for (unsigned ratio : {4u, 128u})
    for (unsigned dim : {17u, 128u, 512u})
      for (unsigned phase : {0u, 1u, ratio - 1})
        Run(model, dim, ratio, phase);
  ds4_gpu_cleanup();
  std::puts("Compressor spans, ragged rows and boundary states pass");
}
