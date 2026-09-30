# Gufo RDNA4 (gfx1201) Port — 進度紀錄

分支：`rdna4-port`（基於上游 main，commit f783fed 起）
機器：192.168.68.67（Nobara），Intel Ultra 7 265K + AMD RX 9070（gfx1201，56CU）
ROCm：7.2.4，`__GFX12__` 宏確認可用。VRAM 16GB discrete（非 Strix Halo 統一記憶體）。

## 已完成（含證據）

1. **授權確認**：上游 MIT，自有碼可改可商用，需保留聲明。
2. **門檻放寬**（commit ac27499，5 檔）：
   - `CMakeLists.txt`：允許 `CMAKE_HIP_ARCHITECTURES=gfx1201`
   - `src/core/diagnostics/system_inventory.cpp`：PCI `1002:7550` 辨識為 gfx1201，
     `gcnArchName` 含 `gfx12` 即視為可用
   - `src/core/diagnostics/compatibility.cpp`、`src/cli/diagnose/diagnose.cpp`、
     `src/cli/bench/kernel_bench_main.cpp`：接受 gfx1201
3. **configure 通過**：`cmake --preset release -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++`
4. **`gufo_diagnostics` 編譯通過**：HIP 工具鏈在 gfx1201 上正常（僅 warnings）。
5. **依賴補齊**：`rocwmma-devel`、`libpng-devel`、`libjpeg-turbo-devel`
   （注意：系統 devel 包 7.1.x vs runtime 7.2.4，混用目前可動）。

## 核心卡點：WMMA fragment 佈局

- gfx1201 後端**選不到**舊 `wmma_*_w32`（探針實測 f16/bf16/iu8 全滅：
  `Cannot select: intrinsic %llvm.amdgcn.wmma.*`）。
- 必須用 `_gfx12` 後綴變體，且 **fragment 佈局不同**：
  f16/bf16 每 lane 8 個 halves（halfx8/bf16x8），舊的是 16 個（v16h）；
  iu8 輸入改為 int32x2 兩組。
- 因此 `LoadSwizzled`/`LoadFrag`/tile 數學都要跟著改，不是換函數名就能動。
- `__GFX12__` 宏可用 → `vendors/hip.h` 的 `RDNA4` 分支會自動啟用。

## 15 檔清單（grep `__builtin_amdgcn_wmma`）

| 檔案 | 用法 | 狀態 |
|---|---|---|
| deepseek_v4_flash/.../mmq/mma.hip.hpp | 完整 `#if RDNA4`/`_gfx12` 分支 | OK（待編譯驗） |
| qwen38_flash_next/.../mmq/mma.hpp | 同上 | OK（待編譯驗） |
| qwen/hip/kernels/attention_wmma.hip | 1× f16_w32 | **進行中**（首啃，本體已 wave32） |
| qwen/hip/kernels/prefill_fp16.hip | 1× f16_w32 | 待修 |
| qwen/hip/kernels/prefill_quant_gemm.hip(.hpp) | iu8_w32＋模板 WaveSize（預設32；_w64 只在 WaveSize==64 具現） | 待修 |
| qwen38_flash_next/.../kernels.hip.cpp | f16_w32＋iu8_w32（Wmma/WmmaI8 helper） | 待修 |
| qwen38_flash_next/.../mmq/mmvq.hip.cpp | 1× iu8_w32 | 待修 |
| deepseek_v4_flash/.../detail/ds4_rocm_q8.hip.hpp | 8× f16_w32，無 RDNA4 處理 | 待修 |
| minimax_h3/attention.hip、dense.hip | bf16_w32 | 延後（非 LLM 路徑） |
| qwen3_asr/hip/convolution.hip | bf16_w32 | 延後 |
| qwen_image_21/hip/{attention,convolution,dense}.hip | bf16_w32 | 延後 |
| wave64 強制檔：qwen `small_batch_wave64.hip`、`prefill_quant_wave64.hip`、`small_batch_quant16_wave64.hip`；qfn 兩處 `-mwavefrontsize64` | gfx12 只有 wave32，需條件化或寫 wave32 版 | 待修 |

## 移植依據（都在 .67 本機）

- `mma.hip.hpp` 兩份的 RDNA4 實作（halfx8/bf16x8/int32x2 餵法＝lane-mapping 活文件）
- `/usr/include/rocwmma/`（gfx12 佈局參考）
- gufo 自帶 quality tests（`BUILD_TESTING=ON`，tests/models/*）做數值驗證

## 下一步順序

1. attention_wmma.hip → 單 TU 編譯驗證 → 對 CPU reference 數值
2. prefill_fp16 / prefill_quant_gemm（Qwen 主路徑）
3. qfn kernels.hip.cpp / mmvq / ds4_rocm_q8
4. wave64 三檔＋qfn flags 條件化
5. 延後模型（minimax/tts/asr/image）的 bf16 站點
6. 全量 build → `gufo diagnose` → 小模型 serve smoke（16GB VRAM，27B Q4 勉強，需小模型先行）

## 風險

- `_gfx12` lane-mapping 若理解錯＝靜默數值錯誤，必須逐檔驗數值，不可只求編過。
- 16GB discrete VRAM 跑不了上游基準的大模型＋大 context，先求小模型通。
- 為 gfx1151 調校的 tile/occupancy 在 gfx1201 上只求正確，不求快；效能調校另案。
