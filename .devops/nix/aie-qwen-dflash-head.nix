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
in
runCommand "gufo-aie-qwen-dflash-head-program" {
  nativeBuildInputs = [
    aiebu
    pythonEnv
    xrt
  ];
  src = ../../src/models/qwen/xdna2/programs/qwen_dflash_head;
  passthru = {
    inherit aiebu mlir-aie llvm-aie;
    modelKind = "qwen3.8-27b-dflash2";
    target = "npu2";
    tensorContract =
      "q8_0-i8-dyn-i8-g32-int32-fp32-b8-m8192-s1x8192-k5120";
  };
  meta = {
    description = "Qwen3.8 DFlash2 split vocabulary-head AIE2P program";
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
  python "$src/qwen_dflash_head.py" --output-dir "$out"

  test -s "$out/qwen_dflash_head.xclbin"
  test -s "$out/qwen_dflash_head_insts.bin"
  test -s "$out/qwen_dflash_head.insts.elf"
  test -s "$out/qwen_dflash_head.pdi"

  xclbinutil \
    --dump-section AIE_PARTITION:JSON:"$out/qwen_dflash_head.aie-partition.json" \
    --input "$out/qwen_dflash_head.xclbin" \
    >/dev/null
  partition_columns="$(
    python - "$out/qwen_dflash_head.aie-partition.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as partition_file:
    metadata = json.load(partition_file)
print(int(metadata["aie_partition"]["partition"]["column_width"]))
PY
  )"
  test "$partition_columns" -eq 8

  xclbin_sha="$(sha256sum "$out/qwen_dflash_head.xclbin" | cut -d ' ' -f 1)"
  insts_sha="$(sha256sum "$out/qwen_dflash_head_insts.bin" | cut -d ' ' -f 1)"
  elf_sha="$(sha256sum "$out/qwen_dflash_head.insts.elf" | cut -d ' ' -f 1)"
  pdi_sha="$(sha256sum "$out/qwen_dflash_head.pdi" | cut -d ' ' -f 1)"
  program_sha="$(
    cat "$out/qwen_dflash_head.xclbin" \
      "$out/qwen_dflash_head.insts.elf" \
      | sha256sum | cut -d ' ' -f 1
  )"

  (
    cd "$out"
    sha256sum qwen_dflash_head.xclbin qwen_dflash_head_insts.bin \
      qwen_dflash_head.insts.elf qwen_dflash_head.pdi >SHA256SUMS
  )

  mkdir -p "$out/include/gufo"
  cat >"$out/include/gufo/aie_qwen_dflash_head_manifest.h" <<EOF
#ifndef GUFO_AIE_QWEN_DFLASH_HEAD_MANIFEST_H_
#define GUFO_AIE_QWEN_DFLASH_HEAD_MANIFEST_H_

#include <string_view>

namespace gufo::xdna2::generated {

inline constexpr std::string_view kQwenDFlashHeadTarget = "npu2";
inline constexpr std::string_view kQwenDFlashHeadAbi = "xrt-elf-v1";
inline constexpr std::string_view kQwenDFlashHeadModelKind =
    "qwen3.8-27b-dflash2";
inline constexpr std::string_view kQwenDFlashHeadTensorContract =
    "q8_0-i8-dyn-i8-g32-int32-fp32-b8-m8192-s1x8192-k5120";
inline constexpr std::string_view kQwenDFlashHeadXrtVersion =
    "${xrt.version}";
inline constexpr std::string_view kQwenDFlashHeadMlirAieVersion =
    "${mlir-aie.version}";
inline constexpr std::string_view kQwenDFlashHeadLlvmAieVersion =
    "${llvm-aie.version}";
inline constexpr std::string_view kQwenDFlashHeadAiebuRevision =
    "${aiebu.src.rev}";
inline constexpr unsigned int kQwenDFlashHeadPartitionColumns =
    $partition_columns;
inline constexpr std::string_view kQwenDFlashHeadProgramSha256 =
    "$program_sha";
inline constexpr std::string_view kQwenDFlashHeadXclbinSha256 =
    "$xclbin_sha";
inline constexpr std::string_view kQwenDFlashHeadInstructionSha256 =
    "$insts_sha";
inline constexpr std::string_view kQwenDFlashHeadElfSha256 = "$elf_sha";
inline constexpr std::string_view kQwenDFlashHeadPdiSha256 = "$pdi_sha";

}  // namespace gufo::xdna2::generated

#endif  // GUFO_AIE_QWEN_DFLASH_HEAD_MANIFEST_H_
EOF

  cat >"$out/manifest.json" <<EOF
{
  "schemaVersion": "1.0.0",
  "target": "npu2",
  "abi": "xrt-elf-v1",
  "modelKind": "qwen3.8-27b-dflash2",
  "tensorContract": "q8_0-i8-dyn-i8-g32-int32-fp32-b8-m8192-s1x8192-k5120",
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
