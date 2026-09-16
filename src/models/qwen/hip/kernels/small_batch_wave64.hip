#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/kernels/small_batch_gemm.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"

namespace gufo::hip::detail {
namespace {

template<core::GgmlType Type, std::size_t Batch, std::size_t Groups = 1>
void Launch(const void* w, const float* x, float* y, std::size_t m,
            std::size_t k, hipStream_t stream) {
  const auto launch =
      [&]<std::uint32_t Waves, std::size_t Rows, std::size_t Tiles,
          std::uint32_t MinWaves, std::size_t Step = Batch, bool Narrow = true,
          std::uint32_t LogicalLanes = 32>() {
        constexpr std::size_t kRowsPerBlock =
            Waves * Rows * (32 / LogicalLanes);
        const dim3 grid(static_cast<unsigned>(
            ((m + kRowsPerBlock - 1) / kRowsPerBlock) * Groups));
        SmallBatchKQuantExactFp32GEMMKernel<Waves, Batch, Rows, Type, Tiles,
                                            MinWaves, Groups, Narrow, Step, 64,
                                            LogicalLanes>
            <<<grid, Waves * 32, 0, stream>>>(w, x, y, m, k);
      };
  const bool down = m == 5120 && k == 17408;
  if constexpr (Batch > 8) {
    if constexpr (Type == core::GgmlType::kIQ4_XS && Batch == 16) {
      if (down || (k == 5120 && (m == 17408 || m == 34816))) {
        launch.template operator()<8, 4, 1, 4, 1, true, 16>();
        return;
      }
    }
    if constexpr (Type == core::GgmlType::kQ6_K) {
      constexpr std::size_t kRows = Batch == 9 || Batch >= 13 ? 6 : 4;
      constexpr std::size_t kStep = Batch == 10 ? 2 : 1;
      launch.template operator()<8, kRows, 1, 4, kStep>();
    } else {
      constexpr std::size_t kStep =
          Type == core::GgmlType::kIQ4_XS                ? (Batch % 2 == 0 ? 2
                                                            : Batch == 11  ? 1
                                                            : Batch == 13  ? 13
                                                                           : 3)
          : Type == core::GgmlType::kQ4_K && Batch == 12 ? 2
                                                         : 1;
      constexpr std::size_t kRows =
          Type != core::GgmlType::kIQ4_XS &&
                  (Batch == 13 || Batch == 14 || Batch == 16 ||
                   (Batch == 15 && Type == core::GgmlType::kQ5_K))
              ? 6
              : 4;
      launch.template operator()<8, kRows, 1, 4, kStep>();
    }
  } else if constexpr (Type == core::GgmlType::kQ4_K) {
    if constexpr (Batch == 4) {
      if (down) {
        launch.template operator()<8, 4, 2, 4, 1>();
        return;
      }
    }
    if constexpr (Batch >= 5) {
      if (k == 5120 && (m == 6144 || m == 10240 || m == 12288)) {
        constexpr std::uint32_t kMinWaves = Batch == 6 ? 8 : 12;
        launch.template operator()<16, 3, 1, kMinWaves>();
        return;
      }
    }
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
      launch.template operator()<16, 4, 2, 4, 1>();
    } else if constexpr (Batch == 6) {
      launch.template operator()<16, 3, 2, 12>();
    } else if constexpr (Batch == 7) {
      launch.template operator()<16, 4, 2, 4>();
    } else {
      if constexpr (Batch == 8) {
        if (m == 34816 && k == 5120) {
          launch.template operator()<8, 4, 1, 4, 1>();
          return;
        }
      }
      launch.template operator()<8, 4, 1, 4>();
    }
  } else {
    static_assert(Type == core::GgmlType::kQ6_K);
    if constexpr (Batch == 8) {
      if (m == 248320 && k == 5120) {
        launch.template operator()<8, 6, 1, 4, 1>();
        return;
      }
    }
    if constexpr (Batch >= 7) {
      if (m == 5120 && k == 6144) {
        launch.template operator()<16, 4, 1, 4>();
        return;
      }
    }
    launch.template operator()<16, 3, 2, 1>();
  }
}

template<core::GgmlType Type>
void LaunchBatch(const void* w, const float* x, float* y, std::size_t batch,
                 std::size_t m, std::size_t k, hipStream_t stream,
                 std::size_t groups) {
  // Adjacent workgroups reuse the same weight rows. Fourteen-row groups cover
  // paired seven-row verification blocks without a separate tail.
  if (groups > 1) {
    const auto launch_groups = [&]<std::size_t Batch>() {
      switch (groups) {
        case 2:
          Launch<Type, Batch, 2>(w, x, y, m, k, stream);
          break;
        case 3:
          Launch<Type, Batch, 3>(w, x, y, m, k, stream);
          break;
        case 4:
          Launch<Type, Batch, 4>(w, x, y, m, k, stream);
          break;
      }
    };
    if constexpr (Type == core::GgmlType::kQ6_K ||
                  Type == core::GgmlType::kQ5_K) {
      if (batch == 14) {
        launch_groups.template operator()<14>();
        return;
      }
    }
    launch_groups.template operator()<16>();
    return;
  }
  switch (batch) {
    case 4:
      if constexpr (Type != core::GgmlType::kQ6_K) {
        Launch<Type, 4>(w, x, y, m, k, stream);
      }
      return;
    case 5:
      if constexpr (Type == core::GgmlType::kQ4_K ||
                    Type == core::GgmlType::kQ5_K ||
                    Type == core::GgmlType::kQ6_K) {
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
    case 9:
      Launch<Type, 9>(w, x, y, m, k, stream);
      return;
    case 10:
      Launch<Type, 10>(w, x, y, m, k, stream);
      return;
    case 11:
      Launch<Type, 11>(w, x, y, m, k, stream);
      return;
    case 12:
      Launch<Type, 12>(w, x, y, m, k, stream);
      return;
    case 13:
      Launch<Type, 13>(w, x, y, m, k, stream);
      return;
    case 14:
      Launch<Type, 14>(w, x, y, m, k, stream);
      return;
    case 15:
      Launch<Type, 15>(w, x, y, m, k, stream);
      return;
    case 16:
      Launch<Type, 16>(w, x, y, m, k, stream);
      return;
    default:
      return;
  }
}

}  // namespace

bool TryLaunchQ8SmallBatchWave64(const void* w, const float* x, float* y,
                                 std::size_t batch, std::size_t m,
                                 std::size_t k, hipStream_t stream,
                                 std::size_t groups) {
  if (groups != 1) {
    const bool ffn =
        ((m == 17408 || m == 34816) && k == 5120) || (m == 5120 && k == 17408);
    if (batch != 16 || groups < 2 || groups > 4 ||
        !(ffn || (k == 5120 && (m == 6144 || m == 10240)) ||
          (m == 5120 && k == 6144)))
      return false;
    const auto launch = [&]<std::size_t Groups>() {
      const auto launch_step = [&]<std::size_t Step>() {
        SmallBatchQ8_0ExactFp32VecGEMMKernel<16, 16, 3, true, true, 64, Groups,
                                             Step>
            <<<((m + 47) / 48) * Groups, 512, 0, stream>>>(w, x, y, m, k);
      };
      if (ffn)
        launch_step.template operator()<1>();
      else
        launch_step.template operator()<16>();
    };
    switch (groups) {
      case 2:
        launch.template operator()<2>();
        break;
      case 3:
        launch.template operator()<3>();
        break;
      case 4:
        launch.template operator()<4>();
        break;
    }
    return true;
  }
  if (batch == 6 && m == 5120 && k == 17408) {
    SmallBatchQ8_0ExactFp32VecGEMMKernel<8, 6, 4, true, false, 64>
        <<<(m + 31) / 32, 256, 0, stream>>>(w, x, y, m, k);
    return true;
  }
  if (batch == 12 && ((k == 5120 && (m == 34816 || m == 10240)) ||
                      (m == 5120 && (k == 17408 || k == 6144)))) {
    SmallBatchQ8_0ExactFp32VecGEMMKernel<16, 12, 4, true, true, 64>
        <<<(m + 63) / 64, 512, 0, stream>>>(w, x, y, m, k);
    return true;
  }
  return false;
}

bool TryLaunchBf16SmallBatchWave64(const void* w, const float* x, float* y,
                                   std::size_t batch, std::size_t m,
                                   std::size_t k, hipStream_t stream,
                                   std::size_t groups) {
  if (groups > 1) {
    const bool up = k == 5120 && (m == 1024 || m == 12288 || m == 248320);
    const bool down = m == 5120 && k == 6144;
    if (batch != 16 || groups > 4 || !(up || down))
      return false;
    const auto launch = [&]<std::size_t Groups, std::size_t Rows>() {
      // Neighboring token groups reuse the same weight rows through cache.
      BatchedExactBf16GEMMFp32VecKernel<16, 16, Rows, false, 64, Groups>
          <<<((m + 16 * Rows - 1) / (16 * Rows)) * Groups, 512, 0, stream>>>(
              static_cast<const hip_bfloat16*>(w), x, y, m, k);
    };
    const auto launch_shape = [&]<std::size_t Groups>() {
      if (down)
        launch.template operator()<Groups, 2>();
      else
        launch.template operator()<Groups, 3>();
    };
    switch (groups) {
      case 2:
        launch_shape.template operator()<2>();
        break;
      case 3:
        launch_shape.template operator()<3>();
        break;
      case 4:
        launch_shape.template operator()<4>();
        break;
    }
    return true;
  }
  if (m == 0 || k == 0 || groups == 0 ||
      !((batch >= 9 && batch <= 15 && batch != 12) || batch == 24))
    return false;
  const auto launch = [&]<std::size_t Batch, unsigned Waves = 16,
                          std::size_t Rows = 2>() {
    BatchedExactBf16GEMMFp32VecKernel<Waves, Batch, Rows, true, 64>
        <<<(m + Waves * Rows - 1) / (Waves * Rows), Waves * 32, 0, stream>>>(
            static_cast<const hip_bfloat16*>(w), x, y, m, k);
  };
  const auto launch_shape = [&]<std::size_t Batch>() {
    if constexpr (Batch <= 11) {
      launch.template operator()<Batch, 8>();
    } else if constexpr (Batch == 13 || Batch == 14) {
      if (m == 248320 && k == 5120)
        launch.template operator()<Batch, 8, 3>();
      else
        launch.template operator()<Batch>();
    } else {
      launch.template operator()<Batch>();
    }
  };
  switch (batch) {
    case 9:
      launch_shape.template operator()<9>();
      break;
    case 10:
      launch_shape.template operator()<10>();
      break;
    case 11:
      launch_shape.template operator()<11>();
      break;
    case 13:
      launch_shape.template operator()<13>();
      break;
    case 14:
      launch_shape.template operator()<14>();
      break;
    case 15:
      launch_shape.template operator()<15>();
      break;
    case 24:
      launch_shape.template operator()<24>();
      break;
    default:
      return false;
  }
  return true;
}

bool TryLaunchKQuantSmallBatchWave64(core::GgmlType type, const void* w,
                                     const float* x, float* y,
                                     std::size_t batch, std::size_t m,
                                     std::size_t k, hipStream_t stream,
                                     std::size_t groups) {
  if (batch < 4 || batch > 16 || groups == 0 || groups > 4 ||
      (groups > 1 && batch != 16 &&
       !((type == core::GgmlType::kQ6_K || type == core::GgmlType::kQ5_K) &&
         batch == 14))) {
    return false;
  }
  // These bounded dimensions also satisfy the kernel's 32-bit index limits.
  // A packed gate/up pair has twice the rows of an individual FFN projection.
  const bool up = k == 5120 && (m == 17408 || m == 34816);
  const bool down = m == 5120 && k == 17408;
  const bool medium_up = k == 5120 && (m == 6144 || m == 10240 || m == 12288);
  const bool medium_down = m == 5120 && k == 6144;
  if (batch > 8) {
    // Preserve scalar accumulation while reusing each weight row across more
    // verification tokens on the native 64-lane wave.
    if (type == core::GgmlType::kQ6_K) {
      const bool head = m == 248320 && k == 5120;
      const bool projection = up || down || medium_down;
      if (!head && !((batch == 14 || batch == 16) && projection))
        return false;
    } else if (batch != 12 && batch != 16) {
      if (!((m == 34816 && k == 5120) || down ||
            (type == core::GgmlType::kQ5_K && batch == 14 && medium_down)))
        return false;
    } else if (!(up || down || medium_up ||
                 (medium_down &&
                  (type != core::GgmlType::kQ5_K || batch == 16)))) {
      return false;
    }
    switch (type) {
      case core::GgmlType::kQ4_K:
        LaunchBatch<core::GgmlType::kQ4_K>(w, x, y, batch, m, k, stream,
                                           groups);
        return true;
      case core::GgmlType::kQ5_K:
        LaunchBatch<core::GgmlType::kQ5_K>(w, x, y, batch, m, k, stream,
                                           groups);
        return true;
      case core::GgmlType::kIQ4_XS:
        LaunchBatch<core::GgmlType::kIQ4_XS>(w, x, y, batch, m, k, stream,
                                             groups);
        return true;
      case core::GgmlType::kQ6_K:
        LaunchBatch<core::GgmlType::kQ6_K>(w, x, y, batch, m, k, stream,
                                           groups);
        return true;
      default:
        return false;
    }
  }
  switch (type) {
    case core::GgmlType::kQ4_K:
      if (!(up || down || (medium_up && batch >= 5))) {
        return false;
      }
      LaunchBatch<core::GgmlType::kQ4_K>(w, x, y, batch, m, k, stream, groups);
      return true;
    case core::GgmlType::kQ5_K:
      if (!((up && batch >= 5 && batch != 6) ||
            (down && (batch == 4 || batch >= 6)) ||
            (medium_up && batch >= 6 && !(batch == 8 && m == 6144)) ||
            (medium_down && batch >= 7))) {
        return false;
      }
      LaunchBatch<core::GgmlType::kQ5_K>(w, x, y, batch, m, k, stream, groups);
      return true;
    case core::GgmlType::kIQ4_XS:
      if (!((batch == 4 && down) || (batch >= 6 && (up || down)))) {
        return false;
      }
      LaunchBatch<core::GgmlType::kIQ4_XS>(w, x, y, batch, m, k, stream,
                                           groups);
      return true;
    case core::GgmlType::kQ6_K:
      if (!((batch == 5 && down) ||
            (batch >= 7 &&
             (down || up || medium_down || (m == 248320 && k == 5120))))) {
        return false;
      }
      LaunchBatch<core::GgmlType::kQ6_K>(w, x, y, batch, m, k, stream, groups);
      return true;
    default:
      return false;
  }
}

}  // namespace gufo::hip::detail
#endif  // defined(ENGINE_ENABLE_HIP)
