#!/usr/bin/env python3
"""Patch TheRock/ROCm clang's HIP math headers for MSVC 14.40+ hosts.

MSVC 14.40+ declares the C99 comparison functions (isgreater, isless, ...)
as constexpr builtins for clang via _CLANG_BUILTIN2 in <cmath>. Clang's HIP
wrapper re-declares them as __device__ only, which MSVC's host+device
builtins cannot overload. The MSVC declarations already work on device, so
the __device__ re-declarations are dropped for _MSC_VER targets.

Affects two headers under lib/llvm/lib/clang/<version>/include/:
  __clang_cuda_math_forward_declares.h  (declarations)
  __clang_hip_cmath.h                    (definitions)

Run once after installing the TheRock gfx1151 distribution:

    python tools/windows/patch_sdk_math.py [HIP_PATH]

Defaults to %HIP_PATH% or C:/TheRock/build. Idempotent.
"""

from pathlib import Path
import os
import sys

FUNCTIONS = [
    "isgreater",
    "isgreaterequal",
    "isless",
    "islessequal",
    "islessgreater",
    "isunordered",
]
TYPES = ["float", "double"]
GUARD_OPEN = "#if !defined(_MSC_VER)  // GUFO_MSVC_CMATH_PATCH\n"
GUARD_CLOSE = "#endif\n"


def patch_clang_version(version_dir: Path) -> bool:
    changed = False
    # Clang resource layouts differ: the headers sit directly under the
    # version directory in some SDKs and under include/ in others.
    for base in (version_dir, version_dir / "include"):
        changed |= patch_headers(base)
    return changed


def patch_headers(header_dir: Path) -> bool:
    changed = False
    header = header_dir / "__clang_cuda_math_forward_declares.h"
    if header.exists():
        text = header.read_text(encoding="utf-8")
        if "GUFO_MSVC_CMATH_PATCH" not in text:
            patched = text
            for function in FUNCTIONS:
                for type_name in TYPES:
                    declaration = f"__DEVICE__ bool {function}({type_name}, {type_name});\n"
                    guarded = GUARD_OPEN + declaration + GUARD_CLOSE
                    if declaration in patched:
                        patched = patched.replace(declaration, guarded, 1)
            if patched != text:
                header.write_text(patched, encoding="utf-8", newline="\n")
                changed = True
                print(f"patched declarations: {header}")

    header = header_dir / "__clang_hip_cmath.h"
    if header.exists():
        text = header.read_text(encoding="utf-8")
        if "GUFO_MSVC_CMATH_PATCH" not in text:
            patched = text
            for function in FUNCTIONS:
                for type_name in TYPES:
                    definition = (
                        f"__DEVICE__ __CONSTEXPR__ bool {function}"
                        f"({type_name} __x, {type_name} __y) {{\n"
                        f"  return __builtin_{function}(__x, __y);\n"
                        f"}}\n"
                    )
                    guarded = GUARD_OPEN + definition + GUARD_CLOSE
                    if definition in patched:
                        patched = patched.replace(definition, guarded, 1)
            if patched != text:
                header.write_text(patched, encoding="utf-8", newline="\n")
                changed = True
                print(f"patched definitions: {header}")
    return changed


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1
                else os.environ.get("HIP_PATH", "C:/TheRock/build"))
    include_root = root / "lib/llvm/lib/clang"
    if not include_root.exists():
        print(f"error: {include_root} does not exist")
        return 1
    changed = False
    for version_dir in sorted(include_root.iterdir()):
        if (version_dir / "__clang_hip_cmath.h").exists() or \
           (version_dir / "include/__clang_hip_cmath.h").exists():
            changed |= patch_clang_version(version_dir)
    if not changed:
        print("all HIP math headers already patched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
