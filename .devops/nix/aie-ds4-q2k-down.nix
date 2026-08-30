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
runCommand "gufo-aie-ds4-q2k-down-program" {
  nativeBuildInputs = [
    aiebu
    pythonEnv
    xrt
  ];
  src = ../../src/models/deepseek_v4_flash/xdna2/programs/ds4_q2k_down;
  passthru = {
    inherit aiebu mlir-aie llvm-aie;
    modelKind = "deepseek-v4-flash";
    target = "npu2";
    tensorContract = "q2_k-u8-dyn-i8-g16-int32-fp32-m4-n4096-k2048";
  };
  meta = {
    description = "DS4 Q2_K four-row down-projection experiment for XDNA2";
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
  python "$src/ds4_q2k_down.py" --output-dir "$out"

  test -s "$out/ds4_q2k_down.xclbin"
  test -s "$out/ds4_q2k_down_insts.bin"
  test -s "$out/ds4_q2k_down.insts.elf"
  test -s "$out/ds4_q2k_down.pdi"

  xclbinutil \
    --dump-section AIE_PARTITION:JSON:"$out/ds4_q2k_down.aie-partition.json" \
    --input "$out/ds4_q2k_down.xclbin" \
    >/dev/null
  partition_columns="$(
    python - "$out/ds4_q2k_down.aie-partition.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as partition_file:
    metadata = json.load(partition_file)
print(int(metadata["aie_partition"]["partition"]["column_width"]))
PY
  )"
  test "$partition_columns" -eq 8

  xclbin_sha="$(sha256sum "$out/ds4_q2k_down.xclbin" | cut -d ' ' -f 1)"
  insts_sha="$(sha256sum "$out/ds4_q2k_down_insts.bin" | cut -d ' ' -f 1)"
  elf_sha="$(sha256sum "$out/ds4_q2k_down.insts.elf" | cut -d ' ' -f 1)"
  pdi_sha="$(sha256sum "$out/ds4_q2k_down.pdi" | cut -d ' ' -f 1)"
  program_sha="$(
    cat "$out/ds4_q2k_down.xclbin" "$out/ds4_q2k_down.insts.elf" \
      | sha256sum | cut -d ' ' -f 1
  )"

  (
    cd "$out"
    sha256sum ds4_q2k_down.xclbin ds4_q2k_down_insts.bin \
      ds4_q2k_down.insts.elf ds4_q2k_down.pdi >SHA256SUMS
  )

  mkdir -p "$out/include/gufo"
  cat >"$out/include/gufo/aie_ds4_q2k_down_manifest.h" <<EOF
#ifndef GUFO_AIE_DS4_Q2K_DOWN_MANIFEST_H_
#define GUFO_AIE_DS4_Q2K_DOWN_MANIFEST_H_

#include <string_view>

namespace gufo::xdna2::generated {

inline constexpr std::string_view kDs4Q2kDownTarget = "npu2";
inline constexpr std::string_view kDs4Q2kDownAbi = "xrt-elf-v1";
inline constexpr std::string_view kDs4Q2kDownModelKind =
    "deepseek-v4-flash";
inline constexpr std::string_view kDs4Q2kDownTensorContract =
    "q2_k-u8-dyn-i8-g16-int32-fp32-m4-n4096-k2048";
inline constexpr std::string_view kDs4Q2kDownXrtVersion = "${xrt.version}";
inline constexpr std::string_view kDs4Q2kDownMlirAieVersion =
    "${mlir-aie.version}";
inline constexpr std::string_view kDs4Q2kDownLlvmAieVersion =
    "${llvm-aie.version}";
inline constexpr std::string_view kDs4Q2kDownAiebuRevision =
    "${aiebu.src.rev}";
inline constexpr unsigned int kDs4Q2kDownPartitionColumns =
    $partition_columns;
inline constexpr std::string_view kDs4Q2kDownProgramSha256 = "$program_sha";
inline constexpr std::string_view kDs4Q2kDownXclbinSha256 = "$xclbin_sha";
inline constexpr std::string_view kDs4Q2kDownInstructionSha256 = "$insts_sha";
inline constexpr std::string_view kDs4Q2kDownElfSha256 = "$elf_sha";
inline constexpr std::string_view kDs4Q2kDownPdiSha256 = "$pdi_sha";

}  // namespace gufo::xdna2::generated

#endif  // GUFO_AIE_DS4_Q2K_DOWN_MANIFEST_H_
EOF

  cat >"$out/manifest.json" <<EOF
{
  "schemaVersion": "1.0.0",
  "target": "npu2",
  "abi": "xrt-elf-v1",
  "modelKind": "deepseek-v4-flash",
  "tensorContract": "q2_k-u8-dyn-i8-g16-int32-fp32-m4-n4096-k2048",
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
