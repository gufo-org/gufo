#!/usr/bin/env python3
"""Regenerate iq_tables.hip.hpp from src/core/quant/ggml_iq_grids.inc.

The device tables must stay byte-identical to the shared CPU tables; the
gemv kernel target includes only the generated header.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
SRC = ROOT / "src/core/quant/ggml_iq_grids.inc"
DST = Path(__file__).resolve().parent / "iq_tables.hip.hpp"

HEADER = """#ifndef GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_IQ_TABLES_HIP_HPP_
#define GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_IQ_TABLES_HIP_HPP_

// Device-side IQ codebook tables. Values are byte-identical to the shared
// CPU tables in src/core/quant/ggml_iq_grids.inc; regenerate with
//   python3 staging/models/glm5_next/kernels/rocm/gen_iq_tables.py
// whenever the core file changes (the parity test fails otherwise). Like the
// DS4 iq2_tables.hip.hpp precedent, the definitions live in a header included
// by exactly one translation unit (gemv.hip.cpp).
#include <cstdint>

namespace gufo::models::glm5_next::rocm {
"""

FOOTER = """}  // namespace gufo::models::glm5_next::rocm

#endif  // GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_IQ_TABLES_HIP_HPP_
"""

SYMS = ("kKsignsIq2xs", "kIq2xxsGrid", "kIq3xxsGrid", "kIq1sGridGpu")


def grab(src: str, sym: str) -> tuple[str, int, str]:
    m = re.search(
        r"constexpr\s+std::(uint8_t|uint32_t|uint64_t)\s+"
        + sym
        + r"\[(\d+)\]\s*=\s*\{(.*?)\};",
        src,
        re.S,
    )
    if not m:
        sys.exit(f"{sym} not found in {SRC}")
    return m.group(1), int(m.group(2)), re.sub(r"\s+", " ", m.group(3).strip())


def main() -> None:
    src = SRC.read_text()
    parts = [HEADER]
    for sym in SYMS:
        ctype, count, body = grab(src, sym)
        body = body.rstrip().rstrip(",")
        parts.append(f"__device__ __constant__ {ctype} {sym}[{count}] = {{\n    {body},\n}};\n")
    parts.append(FOOTER)
    DST.write_text("\n".join(parts))
    print(f"wrote {DST}")


if __name__ == "__main__":
    main()
