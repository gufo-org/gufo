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

## 驗證狀態（實機，2026-09-30）

- [x] WMMA 指令映射（f16/iu8/bf16）：CPU-oracle 微型 GEMM 誤差 0，
      partition 定案。
- [x] quant GEMM（Qwen hpp＋hip）：CPU-oracle EXACT 過（含 fused SwiGLU）。
      抓到一隻真 bug：scratch 寫回 tok/row 順序反了，連帶修了 fp16
      scratch path 和 fused epilogue。
- [ ] attention：測試仍 rel=0.567。S/O 結構已逐項核對，映射經微型
      GEMM 證明正確；下輪重點查 V staging（v_slice/v_key 語義未驗）
      與 Q feed 細節。
- diagnose PASS；serve 可跑但輸出錯（attention 或 fp16 未驗之故；
      quant 已排除）。
- [x] serve 點火成功（2026-09-30）：Qwen3.8-27B-Q3_K_M，c4096，
      12*12=144 答對，英文故事連貫。tail 16-key 修復後 attention
      五組全過（context rel~1e-4，FP64 oracle rel~2e-4，replay 與
      chunk-boundary byte-identical）。batch=2048 case 中止於上游
      tiled 參考核拒收該形狀（與本移植無關）。
- [ ] decode 僅 1.9 tok/s（prefill ~55 tok/s），GPU 滿載 3.1GHz——
      結構性慢，待查 decode 路徑（小 batch GEMV/attention decode 核
      或 hipblaslt 無 gfx1201 調校）。
- [x] decode 慢根因＝權重住 host（2026-09-30 深夜破案）：Qwen loader 用
      mmap＋hipHostRegister 零拷貝（Strix Halo 統一記憶體假設），離散卡
      上每個 decode step 把 13GB 權重从 PCIe 拖一遍（~20GB/s → 1.9t/s；
      prefill 靠 batch reuse 蓋住）。
      修法：`!prop.integrated` 時 hipMalloc＋memcpy 常駐 VRAM（VRAM 不夠
      自動 fallback；teardown 對應 hipFree，`device_owned` 旗標）。
      成果：prefill 54→252 t/s，decode 1.9→21.8 t/s（11 倍），144 照對。
      代價：16GB 卡上 Q3_K_M（13.2GB）+sessions 只剩 c2048 能載（c4096
      state claim 超標；bench 無 --context 會用 native ctx OOM——已知限制）。
      有效頻寬 ~280GB/s（memcpy 屋頂 521GB/s），K-quant GEMV 微調另案。

`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`（f16）與
`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12`（iu8）行为一致：

- lane j 持 A row (j%16)，K-half (j<16 ? K[0:8] : K[8:16])，8 個元素
- lane j 持 B col (j%16)，同 K-half，8 個元素
- lane l acc slot i = D(row = i + (l>=16 ? 8 : 0), col = l%16)，8×f32/i32

驗證：全1→全16.0；one-hot→全2.0（每 slot 覆蓋 2K）；
A=lane/B=1→128+16i（下半）/256+16i（上半）；
A=1/B=lane→128+16(l%16)。四組一致。探針：/tmp/wmma_map*.hip。

## 移植實績

- [x] attention_wmma.hip：S/O 兩階段改 half-fragment 餵法＋列映射
      `(2i+half)→(i+half*8)`，LDS 邏輯佈局不變，下游免動。
      單 TU 在 gfx1201 編譯通過（commit f0c3f1e）。
- [x] prefill_quant_gemm.hpp＋.hip：iu8，舊 acc 佈局經 epilogue 反推為
      row=sub、col=2l+half；改 acc 新佈局＋epilogue 寫入式＋dw/off
      scale 經 QuantSlot12 轉碼。兩 TU 編譯通過。
- [x] prefill_fp16.hip：f16，paired/direct/scratch 三寫回路徑同改。
      編譯通過。
- [x] wave64 三檔中立化：TryLaunch* 在 gfx12 回 false（wave32 fallback
      接手），-mwavefrontsize64 限 gfx1151。Qwen 全數通過，build 推進到
      minimax。
- [x] 非文字模型 trap-stub（用户決議：先關掉）：minimax attention/dense、
      asr conv、image attn/conv/dense 共 11 個 WMMA kernel 本體換成
      fail-stop trap，C++/CMake/CLI 全不動。誤觸即炸，不靜默錯。
- [x] qfn part1：W8A8GemmWave64 調用點＋本體中立化（wave32 tiles 接手），
      moe/w8a8 的 wave64 flags 限 gfx1151。
- [x] qfn attention（WmmaCausalAttentionKernel）：S/O＋rescale＋tail＋
      epilogue 同 Qwen 模式移植（tail/epilogue 為邏輯讀，只換 row）。
- [x] qfn W8A8/RoutedF16/DenseF16 trap-stub（行號插入法；教訓：大括號
      掃描器遇到 #if 分支不對稱會複製貼上整段，大檔一律行號手術）。
      舊 Wmma/WmmaI8 overload 留作純前端樁（後端靠 unreachable 消除，
      實測成立）。
- [x] Qwen3.8-27B-Q3_K_M 下載完成（~/models，13.23GiB；Flash-Next/DS4
      太大不考慮；Q2 走 sdot4 而 gfx1201 無 dot1-insts，故選 Q3）。
- [x] IQ2_XS enablement（用户決議硬上）：llama.cpp 規格從 ds4 vendored
      vec_dot 反推（grid 9bit＋sign 7bit parity＋nibble scale/4），Sub16
      解碼器＋CPU 參考＋dispatch 接線（route/weights/wmma-gemm/batched-
      decode；fp16/small-batch/wave64 刻意不接：前者會 throw，後者已中立化）。
      grid distinct bytes {8,25,43}，max 0x2b（int8 安全，已斷言）。
      待：編譯＋std 交叉驗證＋serve 實測。
- [ ] ds4_q8（stub 完，日後再說）。
- [x] IQ2_XS enablement（硬上成功，2026-10-01）：解碼器地面真相來自
      gguf-py（非推測）：grid 9bit＋sign parity＋scale nibble，
      db = d*(0.5+ls)/4；Q2_K bit-plane-interleaved qs（每 byte 橫跨
      4 groups）＋scales-first layout（跟其他 K-quant 反過來！）。
      驗證：IQ2_XS r=0.94、Q2_K r=0.96（vs 受信 Q3_K/Q5_K CPU）。
      教訓：vec-dot 的 SIMD shuffling 不能直接讀成 plain layout；
      同 tensor 跨格式相關性是王道驗證法。
- [x] IQ2_XS serve 點火（c8192，VRAM ~13GB）：144 ✓，Hello world ✓，
      ~19 t/s。關鍵：(1) VRAM 常駐（沿用）；(2) prefill 走
      dequant-to-BF16＋hipBLAS（為 IQ2_XS/Q2_K 補 Sub16 dequant
      kernel——原先是 no-op 靜默零權重！）；(3) decode 避開 fused
      kernels（packed_format 限舊格式；fused 核對新格式未驗證，
      先走 proven unfused GEMV，效能債另案）。
- [ ] fused SwiGLU/QKV kernels 的 IQ2_XS/Q2_K 審計＋啟用（效能）。
- [x] P1 結論（WKQuant-PreQuantized 之謎，2026-10-01）：standalone
      harness 下 Q3_K/Q4_K/Q5_K/IQ2_XS 經 PreQuantized 全部輸出偏小
      （r=0.67 但幅度差 300 倍）——但 serve（Q3_K_M）是好的。
      判定：harness 測的是 serve 不走的死路（serve 小 batch 走
      native small-batch／dequant+hipBLAS／GEMV，全部 proven）。
      死路特徵：r 高（結構對）但幅度差百倍（疑似 Q8ActTile 在小
      batch 下多數 tile 被標 dead）。不修（動它風險大於收益），
      但開 speculative verification 前必須先修（verification chunk
      走這條路）。
- [ ] 全量 build → diagnose → Q3 serve 點火。
- [ ] qfn W8A8（WmmaI8，3563）：同 Qwen quant 模式（a0/a1 雙 call＋
      dw＋scratch[tok][row] 雙 tile 寫回）。
- [ ] qfn fused-dequant GEMM ×2（4122/5006，a_lo/a_hi）：K32 雙 call 結構，
      取 half-8 餵新指令；acc 映射待讀 epilogue。
- [ ] qfn mmvq.hip.cpp（iu8，K8-part 需上下半拆零；末參 false 需參數化）。
- [ ] qfn part2：kernels.hip.cpp（5 站點：attention S/O、quant GEMM、
      a_lo/a_hi 兩處）＋mmvq.hip.cpp（iu8，K8-part 需上下半拆零）。
- [ ] ds4_rocm_q8.hip.hpp（8× f16，同 attention 模式）。
- [ ] tts/moE_wave64 等 build 浮現的殘敵。
- [ ] minimax_h3/attention.hip、dense.hip：bf16，同構（acc 2i+half 套路），
      bf16 佈局已驗與 f16/iu8 一致。進行中。
- [ ] prefill_quant_gemm.hpp/.hip：iu8，結構已摸清（a0/a1 雙 call 累加同 c，
      acc=8 與 gfx12 一致，只需拆 K-half），待下輪動手。
      ※ 深入後發現：舊 acc 佈局是 row=sub、col=2l+half（epilogue
      `scratch[sub*16+2l+half]` 反推），跟 gfx12 新佈局（row=i+half*8）
      不同列——不能只換 primitive，必須連 acc→epilogue→dw/off/sx
      scale 索引一起改。epilogue 本體與 scale 形狀是 arch 無關的，
      改寫範圍：WmmaQuant＋a/b 取 half＋acc 佈局＋epilogue 寫入式。
- [ ] prefill_fp16.hip：f16，同 attention 模式。

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
