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
runCommand "strix-aie-qwen-mtp-eh-proj-program" {
  nativeBuildInputs = [
    aiebu
    pythonEnv
    xrt
  ];
  src = ../../src/models/qwen/xdna2/programs/qwen_mtp_eh_proj;
  passthru = {
    inherit aiebu mlir-aie llvm-aie;
    modelKind = "qwen3.8-27b-mtp";
    target = "npu2";
    tensorContract = "q4_k-u4-dyn-i8-g32-int32-fp32-m5120-k10240";
  };
  meta = {
    description = "Qwen3.8 MTP eh_proj W4A8 AIE2P program for Strix Halo";
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
  python "$src/qwen_mtp_eh_proj.py" --output-dir "$out"

  test -s "$out/qwen_mtp_eh_proj.xclbin"
  test -s "$out/qwen_mtp_eh_proj_insts.bin"
  test -s "$out/qwen_mtp_eh_proj.insts.elf"
  test -s "$out/qwen_mtp_eh_proj.pdi"

  xclbinutil \
    --dump-section AIE_PARTITION:JSON:"$out/qwen_mtp_eh_proj.aie-partition.json" \
    --input "$out/qwen_mtp_eh_proj.xclbin" \
    >/dev/null
  partition_columns="$(
    python - "$out/qwen_mtp_eh_proj.aie-partition.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as partition_file:
    metadata = json.load(partition_file)
print(int(metadata["aie_partition"]["partition"]["column_width"]))
PY
  )"
  test "$partition_columns" -eq 8

  xclbin_sha="$(sha256sum "$out/qwen_mtp_eh_proj.xclbin" | cut -d ' ' -f 1)"
  insts_sha="$(sha256sum "$out/qwen_mtp_eh_proj_insts.bin" | cut -d ' ' -f 1)"
  elf_sha="$(sha256sum "$out/qwen_mtp_eh_proj.insts.elf" | cut -d ' ' -f 1)"
  pdi_sha="$(sha256sum "$out/qwen_mtp_eh_proj.pdi" | cut -d ' ' -f 1)"
  program_sha="$(
    cat "$out/qwen_mtp_eh_proj.xclbin" \
      "$out/qwen_mtp_eh_proj.insts.elf" \
      | sha256sum | cut -d ' ' -f 1
  )"

  (
    cd "$out"
    sha256sum qwen_mtp_eh_proj.xclbin qwen_mtp_eh_proj_insts.bin \
      qwen_mtp_eh_proj.insts.elf qwen_mtp_eh_proj.pdi >SHA256SUMS
  )

  mkdir -p "$out/include/strix"
  cat >"$out/include/strix/aie_qwen_mtp_eh_proj_manifest.h" <<EOF
#ifndef STRIX_AIE_QWEN_MTP_EH_PROJ_MANIFEST_H_
#define STRIX_AIE_QWEN_MTP_EH_PROJ_MANIFEST_H_

#include <string_view>

namespace strix::xdna2::generated {

inline constexpr std::string_view kQwenMtpEhProjTarget = "npu2";
inline constexpr std::string_view kQwenMtpEhProjAbi = "xrt-elf-v1";
inline constexpr std::string_view kQwenMtpEhProjModelKind =
    "qwen3.8-27b-mtp";
inline constexpr std::string_view kQwenMtpEhProjTensorContract =
    "q4_k-u4-dyn-i8-g32-int32-fp32-m5120-k10240";
inline constexpr std::string_view kQwenMtpEhProjXrtVersion =
    "${xrt.version}";
inline constexpr std::string_view kQwenMtpEhProjMlirAieVersion =
    "${mlir-aie.version}";
inline constexpr std::string_view kQwenMtpEhProjLlvmAieVersion =
    "${llvm-aie.version}";
inline constexpr std::string_view kQwenMtpEhProjAiebuRevision =
    "${aiebu.src.rev}";
inline constexpr unsigned int kQwenMtpEhProjPartitionColumns =
    $partition_columns;
inline constexpr std::string_view kQwenMtpEhProjProgramSha256 =
    "$program_sha";
inline constexpr std::string_view kQwenMtpEhProjXclbinSha256 =
    "$xclbin_sha";
inline constexpr std::string_view kQwenMtpEhProjInstructionSha256 =
    "$insts_sha";
inline constexpr std::string_view kQwenMtpEhProjElfSha256 = "$elf_sha";
inline constexpr std::string_view kQwenMtpEhProjPdiSha256 = "$pdi_sha";

}  // namespace strix::xdna2::generated

#endif  // STRIX_AIE_QWEN_MTP_EH_PROJ_MANIFEST_H_
EOF

  cat >"$out/manifest.json" <<EOF
{
  "schemaVersion": "1.0.0",
  "target": "npu2",
  "abi": "xrt-elf-v1",
  "modelKind": "qwen3.8-27b-mtp",
  "tensorContract": "q4_k-u4-dyn-i8-g32-int32-fp32-m5120-k10240",
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
