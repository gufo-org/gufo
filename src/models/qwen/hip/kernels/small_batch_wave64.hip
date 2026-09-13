#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/kernels/small_batch_gemm.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"

namespace gufo::hip::detail {
namespace {

template<core::GgmlType Type, std::size_t Batch>
void Launch(const void* w, const float* x, float* y, std::size_t m,
            std::size_t k, hipStream_t stream) {
  const auto launch = [&]<std::uint32_t Waves, std::size_t Rows,
                          std::size_t Tiles, std::uint32_t MinWaves,
                          std::size_t Step = Batch, bool Narrow = true>() {
    const dim3 grid(
        static_cast<unsigned>((m + Waves * Rows - 1) / (Waves * Rows)));
    SmallBatchKQuantExactFp32GEMMKernel<Waves, Batch, Rows, Type, Tiles,
                                        MinWaves, 1, Narrow, Step, 64>
        <<<grid, Waves * 32, 0, stream>>>(w, x, y, m, k);
  };
  const bool down = m == 5120;
  if constexpr (Type == core::GgmlType::kQ4_K) {
    if constexpr (Batch == 5 || Batch == 6) {
      if (!down) {
        launch.template operator()<16, 3, 2, 12>();
        return;
      }
    }
    constexpr std::size_t kTiles = Batch == 5 ? 1 : 2;
    launch.template operator()<16, 4, kTiles, 4>();
  } else if constexpr (Type == core::GgmlType::kQ5_K) {
    if constexpr (Batch == 4) {
      launch.template operator()<16, 4, 1, 4, 2>();
    } else if constexpr (Batch == 5 || Batch == 6) {
      launch.template operator()<16, 4, 2, 4>();
    } else if constexpr (Batch == 8) {
      if (down) {
        launch.template operator()<16, 4, 1, 4, 1>();
      } else {
        launch.template operator()<8, 4, 1, 4, 1>();
      }
    } else {
      launch.template operator()<8, 4, 1, 4>();
    }
  } else if constexpr (Type == core::GgmlType::kIQ4_XS) {
    if constexpr (Batch == 4) {
      launch.template operator()<16, 3, 1, 12, Batch, false>();
    } else if constexpr (Batch == 6) {
      launch.template operator()<16, 3, 2, 12>();
    } else if constexpr (Batch == 7) {
      launch.template operator()<16, 4, 2, 4>();
    } else {
      launch.template operator()<8, 4, 1, 4>();
    }
  } else {
    static_assert(Type == core::GgmlType::kQ6_K);
    launch.template operator()<16, 3, 2, 1>();
  }
}

template<core::GgmlType Type>
void LaunchBatch(const void* w, const float* x, float* y, std::size_t batch,
                 std::size_t m, std::size_t k, hipStream_t stream) {
  switch (batch) {
    case 4:
      if constexpr (Type != core::GgmlType::kQ6_K) {
        Launch<Type, 4>(w, x, y, m, k, stream);
      }
      return;
    case 5:
      if constexpr (Type == core::GgmlType::kQ4_K ||
                    Type == core::GgmlType::kQ5_K) {
        Launch<Type, 5>(w, x, y, m, k, stream);
      }
      return;
    case 6:
      if constexpr (Type != core::GgmlType::kQ6_K) {
        Launch<Type, 6>(w, x, y, m, k, stream);
      }
      return;
    case 7:
      Launch<Type, 7>(w, x, y, m, k, stream);
      return;
    case 8:
      Launch<Type, 8>(w, x, y, m, k, stream);
      return;
    default:
      return;
  }
}

}  // namespace

bool TryLaunchKQuantSmallBatchWave64(core::GgmlType type, const void* w,
                                     const float* x, float* y,
                                     std::size_t batch, std::size_t m,
                                     std::size_t k, hipStream_t stream) {
  if (batch < 4 || batch > 8) {
    return false;
  }
  // These bounded dimensions also satisfy the kernel's 32-bit index limits.
  // A packed gate/up pair has twice the rows of an individual FFN projection.
  const bool up = k == 5120 && (m == 17408 || m == 34816);
  const bool down = m == 5120 && k == 17408;
  switch (type) {
    case core::GgmlType::kQ4_K:
      if (!(up || (down && batch >= 5))) {
        return false;
      }
      LaunchBatch<core::GgmlType::kQ4_K>(w, x, y, batch, m, k, stream);
      return true;
    case core::GgmlType::kQ5_K:
      if (!((up && batch >= 5 && batch != 6) ||
            (down && (batch == 4 || batch >= 6)))) {
        return false;
      }
      LaunchBatch<core::GgmlType::kQ5_K>(w, x, y, batch, m, k, stream);
      return true;
    case core::GgmlType::kIQ4_XS:
      if (!((batch == 4 && down) || (batch >= 6 && (up || down)))) {
        return false;
      }
      LaunchBatch<core::GgmlType::kIQ4_XS>(w, x, y, batch, m, k, stream);
      return true;
    case core::GgmlType::kQ6_K:
      if (batch < 7 || m != 248320 || k != 5120) {
        return false;
      }
      LaunchBatch<core::GgmlType::kQ6_K>(w, x, y, batch, m, k, stream);
      return true;
    default:
      return false;
  }
}

}  // namespace gufo::hip::detail
#endif  // defined(ENGINE_ENABLE_HIP)
