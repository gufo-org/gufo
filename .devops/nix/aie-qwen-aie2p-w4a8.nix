{
  lib,
  runCommand,
  python312,
  aiebu,
  llvm-aie,
  mlir-aie,
  xrt,
}:

let
  pythonEnv = python312.withPackages (_: [
    llvm-aie
    mlir-aie
  ]);
  sitePackages = python312.sitePackages;
  mlirAieRoot = "${mlir-aie}/${sitePackages}/mlir_aie";
  peanoRoot = "${llvm-aie}/${sitePackages}/llvm-aie";
  # Baked static configuration (see qwen_aie2p_w4a8.py): the M=1 eh_proj
  # shape K=10240, N=5120.
  blocks = 40;
  tpc = 10;
  rounds = 1;
in
runCommand "strix-aie-qwen-aie2p-w4a8-program" {
  nativeBuildInputs = [
    aiebu
    pythonEnv
    xrt
  ];
  src = ../../src/core/xdna2/programs/qwen_aie2p_w4a8;
  passthru = {
    inherit aiebu mlir-aie llvm-aie;
    modelKind = "qwen3.8-27b-mtp";
    target = "npu2";
    tensorContract = "q4_k-u4-dyn-i8-g32-int32-fp32-m1-k10240-n5120-b40-t10-r1";
  };
  meta = {
    description = "Qwen3.8 MTP eh_proj W4A8 SHQ4-T16 AIE2P program for Strix Halo";
    license = lib.licenses.asl20;
    platforms = [ "x86_64-linux" ];
  };
} ''
  export MLIR_AIE_INSTALL_DIR=${mlirAieRoot}
  export PEANO_INSTALL_DIR=${peanoRoot}
  export PATH="$MLIR_AIE_INSTALL_DIR/bin:$PATH"
  export PYTHONPATH="$MLIR_AIE_INSTALL_DIR/python:''${PYTHONPATH:-}"
  export LD_LIBRARY_PATH="$MLIR_AIE_INSTALL_DIR/lib:''${LD_LIBRARY_PATH:-}"

  mkdir -p "$out"
  python "$src/qwen_aie2p_w4a8.py" --output-dir "$out" \
    --blocks ${toString blocks} --tpc ${toString tpc} --rounds ${toString rounds}

  test -s "$out/qwen_aie2p_w4a8.xclbin"
  test -s "$out/qwen_aie2p_w4a8_insts.bin"
  test -s "$out/qwen_aie2p_w4a8.insts.elf"
  test -s "$out/qwen_aie2p_w4a8.pdi"

  xclbinutil \
    --dump-section AIE_PARTITION:JSON:"$out/qwen_aie2p_w4a8.aie-partition.json" \
    --input "$out/qwen_aie2p_w4a8.xclbin" \
    >/dev/null
  partition_columns="$(
    python - "$out/qwen_aie2p_w4a8.aie-partition.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as partition_file:
    metadata = json.load(partition_file)
print(int(metadata["aie_partition"]["partition"]["column_width"]))
PY
  )"
  test "$partition_columns" -eq 8

  xclbin_sha="$(sha256sum "$out/qwen_aie2p_w4a8.xclbin" | cut -d ' ' -f 1)"
  insts_sha="$(sha256sum "$out/qwen_aie2p_w4a8_insts.bin" | cut -d ' ' -f 1)"
  elf_sha="$(sha256sum "$out/qwen_aie2p_w4a8.insts.elf" | cut -d ' ' -f 1)"
  pdi_sha="$(sha256sum "$out/qwen_aie2p_w4a8.pdi" | cut -d ' ' -f 1)"
  program_sha="$(
    cat "$out/qwen_aie2p_w4a8.xclbin" \
      "$out/qwen_aie2p_w4a8.insts.elf" \
      | sha256sum | cut -d ' ' -f 1
  )"

  (
    cd "$out"
    sha256sum qwen_aie2p_w4a8.xclbin qwen_aie2p_w4a8_insts.bin \
      qwen_aie2p_w4a8.insts.elf qwen_aie2p_w4a8.pdi >SHA256SUMS
  )

  mkdir -p "$out/include/strix"
  cat >"$out/include/strix/aie_qwen_aie2p_w4a8_manifest.h" <<EOF
#ifndef STRIX_AIE_QWEN_AIE2P_W4A8_MANIFEST_H_
#define STRIX_AIE_QWEN_AIE2P_W4A8_MANIFEST_H_

#include <string_view>

namespace strix::xdna2::generated {

inline constexpr std::string_view kQwenAie2pW4a8Target = "npu2";
inline constexpr std::string_view kQwenAie2pW4a8Abi = "xrt-elf-v1";
inline constexpr std::string_view kQwenAie2pW4a8ModelKind =
    "qwen3.8-27b-mtp";
inline constexpr std::string_view kQwenAie2pW4a8TensorContract =
    "q4_k-u4-dyn-i8-g32-int32-fp32-m1-k10240-n5120-b40-t10-r1";
inline constexpr unsigned int kQwenAie2pW4a8Blocks = ${toString blocks};
inline constexpr unsigned int kQwenAie2pW4a8TilesPerColumn = ${toString tpc};
inline constexpr unsigned int kQwenAie2pW4a8Rounds = ${toString rounds};
inline constexpr std::string_view kQwenAie2pW4a8XrtVersion =
    "${xrt.version}";
inline constexpr std::string_view kQwenAie2pW4a8MlirAieVersion =
    "${mlir-aie.version}";
inline constexpr std::string_view kQwenAie2pW4a8LlvmAieVersion =
    "${llvm-aie.version}";
inline constexpr std::string_view kQwenAie2pW4a8AiebuRevision =
    "${aiebu.src.rev}";
inline constexpr unsigned int kQwenAie2pW4a8PartitionColumns =
    $partition_columns;
inline constexpr std::string_view kQwenAie2pW4a8ProgramSha256 =
    "$program_sha";
inline constexpr std::string_view kQwenAie2pW4a8XclbinSha256 =
    "$xclbin_sha";
inline constexpr std::string_view kQwenAie2pW4a8InstructionSha256 =
    "$insts_sha";
inline constexpr std::string_view kQwenAie2pW4a8ElfSha256 = "$elf_sha";
inline constexpr std::string_view kQwenAie2pW4a8PdiSha256 = "$pdi_sha";

}  // namespace strix::xdna2::generated

#endif  // STRIX_AIE_QWEN_AIE2P_W4A8_MANIFEST_H_
EOF

  cat >"$out/manifest.json" <<EOF
{
  "schemaVersion": "1.0.0",
  "target": "npu2",
  "abi": "xrt-elf-v1",
  "modelKind": "qwen3.8-27b-mtp",
  "tensorContract": "q4_k-u4-dyn-i8-g32-int32-fp32-m1-k10240-n5120-b40-t10-r1",
  "blocks": ${toString blocks},
  "tilesPerColumn": ${toString tpc},
  "rounds": ${toString rounds},
  "xrtVersion": "${xrt.version}",
  "mlirAieVersion": "${mlir-aie.version}",
  "llvmAieVersion": "${llvm-aie.version}",
  "aiebuRevision": "${aiebu.src.rev}",
  "partitionColumns": $partition_columns,
  "programSha256": "$program_sha",
  "xclbinSha256": "$xclbin_sha",
  "instructionSha256": "$insts_sha",
  "elfSha256": "$elf_sha",
  "pdiSha256": "$pdi_sha"
}
EOF
''