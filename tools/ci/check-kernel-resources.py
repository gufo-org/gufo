#!/usr/bin/env python3
"""Check the scratch memory of each GPU kernel in a HIP binary.

A kernel that uses scratch memory has a register spill, or a private array
that the compiler did not keep in registers. The build does not show this.
A compiler update can cause it without a change to the source code. For
example, AMD clang 23 increased the scratch of the Qwen3.8-27B Q8 prefill GEMM
from 0 to 480 bytes, and prefill became 36% slower.

This tool reads the code objects in the .hip_fatbin section of the binary. It
decodes the AMDGPU metadata note of each code object. It does not use a GPU.

Modes:
  check-kernel-resources.py BINARY --baseline FILE
      Compare each kernel with the allowance in FILE. The default allowance is
      0 bytes. Exit status 1 if a kernel uses more scratch than its allowance.
  check-kernel-resources.py BINARY --write-baseline FILE
      Write FILE with an allowance for each kernel that uses scratch now.
  check-kernel-resources.py BINARY --report [FILTER]
      List the kernels that use scratch.
  check-kernel-resources.py BINARY --compare BASE_BINARY
      Compare two builds, for example two compilers. List each kernel that
      has more scratch, more spilled VGPRs or less VGPR occupancy than in
      BASE_BINARY. Exit status 1 if the scratch of a kernel increases.

The allowances use the mangled kernel names. The Itanium ABI keeps these names
the same across compilers. The tool removes the version from the rocPRIM
inline namespace, so a baseline stays valid after a ROCm update.

Compressed offload bundles (CCOB) need Python zstd or zlib. If they are not
available, the tool uses clang-offload-bundler (--bundler, the HIP toolchain,
or PATH).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile

BUNDLE_MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"
CCOB_MAGIC = b"CCOB"
NT_AMDGPU_METADATA = 32


# ---- ELF --------------------------------------------------------------------------------
def elf_sections(data: bytes) -> dict[str, tuple[int, int, int]]:
    """Return name -> (type, offset, size) for each section of an ELF64 LE file."""
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise ValueError("not an ELF64 little-endian file")
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)

    def header(i: int) -> tuple:
        return struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize)

    strtab = header(shstrndx)[4]
    out = {}
    for i in range(shnum):
        name_off, sh_type, _, _, off, size = header(i)[:6]
        end = data.index(b"\0", strtab + name_off)
        out[data[strtab + name_off:end].decode()] = (sh_type, off, size)
    return out


def elf_notes(data: bytes):
    """Yield (type, name, desc) for each note in each SHT_NOTE section."""
    for sh_type, off, size in elf_sections(data).values():
        if sh_type != 7:  # SHT_NOTE
            continue
        pos, end = off, off + size
        while pos + 12 <= end:
            namesz, descsz, ntype = struct.unpack_from("<III", data, pos)
            pos += 12
            name = data[pos:pos + namesz].rstrip(b"\0")
            pos += (namesz + 3) & ~3
            desc = data[pos:pos + descsz]
            pos += (descsz + 3) & ~3
            yield ntype, name, desc


# ---- msgpack (only the part that the AMDGPU metadata uses) ------------------------------
def unpack(buf: bytes, pos: int = 0):
    b = buf[pos]
    pos += 1
    if b <= 0x7F:
        return b, pos
    if 0x80 <= b <= 0x8F:
        return unpack_map(buf, pos, b & 0x0F)
    if 0x90 <= b <= 0x9F:
        return unpack_array(buf, pos, b & 0x0F)
    if 0xA0 <= b <= 0xBF:
        n = b & 0x1F
        return buf[pos:pos + n].decode(errors="replace"), pos + n
    if b >= 0xE0:
        return b - 0x100, pos
    fixed = {0xC0: None, 0xC2: False, 0xC3: True}
    if b in fixed:
        return fixed[b], pos
    fmt = {0xCC: "B", 0xCD: ">H", 0xCE: ">I", 0xCF: ">Q", 0xD0: "b", 0xD1: ">h",
           0xD2: ">i", 0xD3: ">q", 0xCA: ">f", 0xCB: ">d"}
    if b in fmt:
        v, = struct.unpack_from(fmt[b], buf, pos)
        return v, pos + struct.calcsize(fmt[b])
    if b in (0xD9, 0xDA, 0xDB, 0xC4, 0xC5, 0xC6):
        width = {0xD9: 1, 0xDA: 2, 0xDB: 4, 0xC4: 1, 0xC5: 2, 0xC6: 4}[b]
        n = int.from_bytes(buf[pos:pos + width], "big")
        pos += width
        raw = buf[pos:pos + n]
        return (raw.decode(errors="replace") if b >= 0xD9 else raw), pos + n
    if b in (0xDC, 0xDD):
        width = 2 if b == 0xDC else 4
        return unpack_array(buf, pos + width, int.from_bytes(buf[pos:pos + width], "big"))
    if b in (0xDE, 0xDF):
        width = 2 if b == 0xDE else 4
        return unpack_map(buf, pos + width, int.from_bytes(buf[pos:pos + width], "big"))
    raise ValueError(f"unsupported msgpack byte 0x{b:02x}")


def unpack_array(buf, pos, n):
    out = []
    for _ in range(n):
        v, pos = unpack(buf, pos)
        out.append(v)
    return out, pos


def unpack_map(buf, pos, n):
    out = {}
    for _ in range(n):
        k, pos = unpack(buf, pos)
        v, pos = unpack(buf, pos)
        out[k] = v
    return out, pos


def code_object_kernels(co: bytes) -> list[dict]:
    for ntype, name, desc in elf_notes(co):
        if ntype == NT_AMDGPU_METADATA and name == b"AMDGPU":
            meta, _ = unpack(desc)
            return meta.get("amdhsa.kernels", [])
    return []


# ---- offload bundles --------------------------------------------------------------------
def split_bundles(fatbin: bytes):
    """Yield each offload bundle (uncompressed or CCOB) in a .hip_fatbin section."""
    pos = 0
    while True:
        cands = [p for p in (fatbin.find(BUNDLE_MAGIC, pos), fatbin.find(CCOB_MAGIC, pos)) if p >= 0]
        if not cands:
            return
        p = min(cands)
        if fatbin.startswith(BUNDLE_MAGIC, p):
            count, = struct.unpack_from("<Q", fatbin, p + 24)
            q, end = p + 32, p
            for _ in range(count):
                off, size, tlen = struct.unpack_from("<QQQ", fatbin, q)
                q += 24 + tlen
                end = max(end, p + off + size)
            yield fatbin[p:end]
            pos = end
        else:
            version, = struct.unpack_from("<H", fatbin, p + 4)
            total = struct.unpack_from("<Q" if version >= 3 else "<I", fatbin, p + 8)[0]
            yield fatbin[p:p + total]
            pos = p + total


def bundle_entries(bundle: bytes):
    """Yield (triple, payload) for each entry of an uncompressed offload bundle."""
    count, = struct.unpack_from("<Q", bundle, 24)
    q = 32
    for _ in range(count):
        off, size, tlen = struct.unpack_from("<QQQ", bundle, q)
        triple = bundle[q + 24:q + 24 + tlen].decode()
        q += 24 + tlen
        yield triple, bundle[off:off + size]


def inflate_ccob(blob: bytes, bundler: str | None) -> bytes:
    version, method = struct.unpack_from("<HH", blob, 4)
    header = 32 if version >= 3 else 24
    payload = blob[header:]
    try:
        if method == 0:  # llvm::compression::Format::Zlib
            import zlib
            return zlib.decompress(payload)
        if method == 1:  # llvm::compression::Format::Zstd
            try:
                from compression import zstd  # Python 3.14+
            except ImportError:
                import zstandard as zstd  # type: ignore
            return zstd.decompress(payload)
    except Exception:
        pass
    if not bundler:
        raise RuntimeError("compressed offload bundle: needs Python zstd or --bundler")
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in")
        Path(src).write_bytes(blob)
        targets = subprocess.run([bundler, "--list", "--type=o", f"--input={src}"],
                                 check=True, capture_output=True, text=True).stdout.split()
        outs = [os.path.join(tmp, f"t{i}") for i in range(len(targets))]
        subprocess.run([bundler, "--unbundle", "--type=o", f"--input={src}",
                        "--targets=" + ",".join(targets),
                        *[f"--output={path}" for path in outs]],
                       check=True, capture_output=True)
        # Make an uncompressed bundle again. Then one code path reads the two types.
        entries = [(t, Path(o).read_bytes()) for t, o in zip(targets, outs)]
        return make_bundle(entries)


def make_bundle(entries):
    head = BUNDLE_MAGIC + struct.pack("<Q", len(entries))
    table_len = sum(24 + len(t.encode()) for t, _ in entries)
    off = len(head) + table_len
    table, body = b"", b""
    for t, payload in entries:
        table += struct.pack("<QQQ", off + len(body), len(payload), len(t.encode())) + t.encode()
        body += payload
    return head + table + body


def key(mangled: str) -> str:
    """Return the key of a kernel in the baseline: the mangled name without the
    version in the rocPRIM inline namespace (for example ROCPRIM_400200_NS)."""
    return re.sub(r"\d+ROCPRIM_\d+_NS", "ROCPRIM_NS", mangled)


def binary_kernels(path: str, bundler: str | None) -> dict[str, dict]:
    data = Path(path).read_bytes()
    sections = elf_sections(data)
    if ".hip_fatbin" not in sections:
        raise SystemExit(f"{path}: no .hip_fatbin section")
    _, off, size = sections[".hip_fatbin"]
    kernels: dict[str, dict] = {}
    for blob in split_bundles(data[off:off + size]):
        if blob.startswith(CCOB_MAGIC):
            blob = inflate_ccob(blob, bundler)
        for triple, payload in bundle_entries(blob):
            if not triple.startswith(("hip", "hipv4")) or "amdgcn" not in triple or not payload:
                continue
            arch = triple.rsplit("-", 1)[-1]
            for k in code_object_kernels(payload):
                kernels[key(k[".name"])] = {
                    "arch": arch,
                    "scratch": int(k.get(".private_segment_fixed_size", 0)),
                    "vgpr": int(k.get(".vgpr_count", 0)),
                    "vgpr_spill": int(k.get(".vgpr_spill_count", 0)),
                    "sgpr_spill": int(k.get(".sgpr_spill_count", 0)),
                }
    if not kernels:
        raise SystemExit(f"{path}: no AMDGPU kernels found")
    return kernels


# ---- CLI --------------------------------------------------------------------------------
def find_bundler(explicit: str | None) -> str | None:
    if explicit:
        if not os.path.isfile(explicit):
            raise SystemExit(f"offload bundler not found: {explicit}")
        return explicit
    if os.environ.get("HIP_CLANG"):
        candidate = os.path.join(os.path.dirname(os.environ["HIP_CLANG"]), "clang-offload-bundler")
        if os.path.isfile(candidate):
            return candidate
    hipconfig = shutil.which("hipconfig")
    if hipconfig:
        result = subprocess.run([hipconfig, "-l"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            candidate = os.path.join(result.stdout.strip(), "clang-offload-bundler")
            if os.path.isfile(candidate):
                return candidate
    return shutil.which("clang-offload-bundler")


def demangle(names):
    tool = shutil.which("c++filt") or shutil.which("llvm-cxxfilt")
    if not tool or not names:
        return {n: n for n in names}
    out = subprocess.run([tool], input="\n".join(names), capture_output=True, text=True).stdout
    return dict(zip(names, out.splitlines()))


def waves(vgpr: int) -> int:
    """Return the VGPR-limited waves per SIMD on gfx1151 (1536 VGPRs, granule 24, maximum 16)."""
    granule = -(-max(vgpr, 1) // 24) * 24
    return min(16, 1536 // granule)


def compare(base: dict, cand: dict) -> int:
    rows, counts = [], {"worse": 0, "better": 0, "same": 0, "one side only": 0}
    for n in sorted(set(base) | set(cand)):
        b, c = base.get(n), cand.get(n)
        if not b or not c:
            counts["one side only"] += 1
            continue
        worse = (c["scratch"] > b["scratch"] or c["vgpr_spill"] > b["vgpr_spill"]
                 or waves(c["vgpr"]) < waves(b["vgpr"]))
        better = (c["scratch"] < b["scratch"] or c["vgpr_spill"] < b["vgpr_spill"]
                  or waves(c["vgpr"]) > waves(b["vgpr"]))
        counts["worse" if worse else "better" if better else "same"] += 1
        if worse:
            rows.append((n, b, c))
    dm = demangle([n for n, _, _ in rows])
    grew = 0
    for n, b, c in sorted(rows, key=lambda r: -(r[2]["scratch"] - r[1]["scratch"])):
        grew += c["scratch"] > b["scratch"]
        print(f"{b['vgpr']:3d} vgpr {b['vgpr_spill']:4d} spill {b['scratch']:4d} B {waves(b['vgpr']):2d}w"
              f"  ->  {c['vgpr']:3d} vgpr {c['vgpr_spill']:4d} spill {c['scratch']:4d} B {waves(c['vgpr']):2d}w"
              f"  {dm[n]}")
    print(", ".join(f"{v} {k}" for k, v in counts.items()) + f"; scratch increased in {grew}")
    return 1 if grew else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("binary")
    ap.add_argument("--baseline", help="JSON file with the scratch allowances to check")
    ap.add_argument("--write-baseline", help="write a JSON file that allows the current scratch of each kernel")
    ap.add_argument("--report", nargs="?", const="", help="list the kernels that use scratch (optional name filter)")
    ap.add_argument("--bundler", default=None, help="path of clang-offload-bundler, for compressed bundles")
    ap.add_argument("--note", default="", help="text that --write-baseline records in the file")
    ap.add_argument("--compare", metavar="BASE_BINARY", help="list the kernels that are worse than in BASE_BINARY")
    args = ap.parse_args()

    bundler = find_bundler(args.bundler)
    kernels = binary_kernels(args.binary, bundler)
    using = {n: k for n, k in kernels.items() if k["scratch"] > 0}

    if args.compare:
        return compare(binary_kernels(args.compare, bundler), kernels)

    if args.write_baseline:
        allow = {n: {"max_scratch": k["scratch"]} for n, k in sorted(using.items())}
        Path(args.write_baseline).write_text(json.dumps(
            {"schema": "gufo.kernel-resources.v1", "note": args.note,
             "default_max_scratch": 0, "allow": allow}, indent=1) + "\n")
        print(f"wrote {len(allow)} allowances ({len(kernels)} kernels) to {args.write_baseline}")
        return 0

    if args.report is not None:
        dm = demangle(list(using))
        for n, k in sorted(using.items(), key=lambda kv: -kv[1]["scratch"]):
            if args.report in dm[n]:
                print(f"{k['scratch']:6d} B  vgpr {k['vgpr']:3d}  spill {k['vgpr_spill']:4d}  {dm[n]}")
        print(f"{len(using)} of {len(kernels)} kernels use scratch")
        return 0

    if not args.baseline:
        ap.error("one of --baseline, --write-baseline or --report is required")
    budget = json.loads(Path(args.baseline).read_text())
    default = int(budget.get("default_max_scratch", 0))
    allow = budget.get("allow", {})
    bad = [(n, k["scratch"], int(allow.get(n, {}).get("max_scratch", default)))
           for n, k in kernels.items() if k["scratch"] > int(allow.get(n, {}).get("max_scratch", default))]
    stale = sorted(set(allow) - set(kernels))
    dm = demangle([n for n, _, _ in bad] + stale)
    for n, have, limit in sorted(bad, key=lambda t: -t[1]):
        k = kernels[n]
        print(f"FAIL {have:5d} B scratch (limit {limit}), vgpr {k['vgpr']}, spill {k['vgpr_spill']}: {dm[n]}")
    for n in stale:
        print(f"note: allowance for a kernel that no longer exists: {dm[n]}")
    print(f"{len(kernels)} kernels checked, {len(using)} use scratch, {len(bad)} over budget")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
