#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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
    Check(data != nullptr, "allocate bounded tensor");
  }
  ~Tensor() { ds4_gpu_tensor_free(data); }
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;
  ds4_gpu_tensor* data;
};

struct Fixture {
  unsigned vocab, rank;
  size_t first, second;
};

double Weight(const std::vector<unsigned char>& model, size_t offset,
              unsigned rank, unsigned row, unsigned column) {
  const size_t block = offset + (size_t(row) * rank + column) / 32 * 34;
  int8_t quant;
  std::memcpy(&quant, &model[block + 2 + column % 32], sizeof(quant));
  // Fixture blocks use exactly representable 2^-5 scales.
  return double(quant) / 32;
}

void Run(const std::vector<unsigned char>& model, const Fixture& f) {
  for (unsigned rows = 1; rows <= 8; ++rows) {
    std::vector<float> logits(size_t(rows) * f.vocab);
    for (size_t i = 0; i < logits.size(); ++i)
      logits[i] = float(int((i * 19 + i / f.vocab * 7) % 31) - 15) / 1024;
    Tensor input(logits.size() * sizeof(float)), output(rows * sizeof(int32_t)),
        key(sizeof(uint64_t));
    Check(ds4_gpu_tensor_write(input.data, 0, logits.data(),
                               logits.size() * sizeof(float)),
          "upload logits");
    const unsigned anchor = f.vocab - 1;
    const auto select = [&](unsigned count, unsigned previous, size_t bytes,
                            size_t offset) {
      return ds4_gpu_dspark_markov_chain_tensor(
          output.data, key.data, input.data, model.data(), bytes, f.first,
          offset, f.vocab, f.rank, previous, count);
    };
    Check(!select(0, anchor, model.size(), f.second) &&
              !select(9, anchor, model.size(), f.second) &&
              !select(rows, f.vocab, model.size(), f.second) &&
              !select(rows, anchor, f.second, f.second) &&
              !select(rows, anchor, model.size(), UINT64_MAX),
          "reject invalid counts, anchors and model ranges");
    if (rows < 8)
      Check(!select(rows + 1, anchor, model.size(), f.second),
            "reject short input/output buffers");
    Check(select(rows, anchor, model.size(), f.second), "enqueue Markov chain");
    std::vector<int32_t> actual(rows), expected(rows);
    Check(ds4_gpu_tensor_read(output.data, 0, actual.data(),
                              actual.size() * sizeof(int32_t)),
          "read completed chain once");
    unsigned previous = anchor;
    for (unsigned row = 0; row < rows; ++row) {
      double best = -std::numeric_limits<double>::infinity();
      unsigned chosen = 0;
      for (unsigned token = 0; token < f.vocab; ++token) {
        double value = logits[size_t(row) * f.vocab + token];
        for (unsigned column = 0; column < f.rank; ++column)
          value += Weight(model, f.first, f.rank, previous, column) *
                   Weight(model, f.second, f.rank, token, column);
        Check(std::isfinite(value), "finite independent scores");
        if (value > best) {
          best = value;
          chosen = token;
        }
      }
      expected[row] = static_cast<int32_t>(chosen);
      previous = chosen;
    }
    Check(actual == expected,
          "chain matches independent FP64 formula and ties");

    // Repeated scalar calls exercise the old host-mediated dependency with
    // the same bounded rows; the full chain must choose identical IDs.
    previous = anchor;
    for (unsigned row = 0; row < rows; ++row) {
      auto* view =
          ds4_gpu_tensor_view(input.data, size_t(row) * f.vocab * sizeof(float),
                              size_t(f.vocab) * sizeof(float));
      Check(view != nullptr, "bounded scalar logit row");
      Check(ds4_gpu_dspark_markov_chain_tensor(
                output.data, key.data, view, model.data(), model.size(),
                f.first, f.second, f.vocab, f.rank, previous, 1),
            "enqueue scalar Markov choice");
      int32_t selected = -1;
      Check(ds4_gpu_tensor_read(output.data, 0, &selected, sizeof(selected)),
            "read scalar choice");
      ds4_gpu_tensor_free(view);
      Check(selected == expected[row], "scalar and chained choices agree");
      previous = static_cast<unsigned>(selected);
    }
  }
}
}  // namespace

int main() {
  if (!ds4_gpu_init())
    return 77;
  std::vector<unsigned char> model;
  std::vector<Fixture> fixtures;
  uint32_t random = 0x251f307bu;
  for (unsigned rank : {32u, 96u, 256u})
    for (unsigned vocab : {1u, 17u, 257u, 523u}) {
      const size_t bytes = size_t(vocab) * rank / 32 * 34;
      const size_t offset = model.size();
      model.resize(offset + 2 * bytes);
      fixtures.push_back({vocab, rank, offset, offset + bytes});
      for (size_t block = offset; block < model.size(); block += 34) {
        const uint16_t scale = 0x2800;  // IEEE FP16 2^-5.
        std::memcpy(model.data() + block, &scale, sizeof(scale));
        for (unsigned column = 0; column < 32; ++column) {
          random ^= random << 13;
          random ^= random >> 17;
          random ^= random << 5;
          const int8_t value = static_cast<int8_t>(int(random % 31) - 15);
          std::memcpy(model.data() + block + 2 + column, &value, sizeof(value));
        }
      }
    }
  // A separate zero-weight table tests global equal-score tie ordering.
  const size_t zero = model.size();
  model.resize(zero + 2 * 523 * 34, 0);
  fixtures.push_back({523, 32, zero, zero + 523 * 34});
  Check(ds4_gpu_reserve_support_map(model.data(), model.size(), model.size()) &&
            ds4_gpu_cache_support_range(model.data(), model.size(), 0,
                                        model.size(), "Markov fixture"),
        "make Q8 tables resident");
  for (const auto& fixture : fixtures)
    Run(model, fixture);
  ds4_gpu_cleanup();
  std::puts("Markov chain: FP64 oracle, scalar replay, ties and bounds pass");
}
