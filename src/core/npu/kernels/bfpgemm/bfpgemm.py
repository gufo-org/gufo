"""BFP16 GEMM on the whole NPU (8 columns x 4 rows): C = A W^T for many rows
of A, the prefill-sized counterpart of q8gemm.py.

The cores are mlir-aie's asymmetric-tile-buffering design
(programming_examples/ml/block_datatypes/gemm_asymmetric_tile_buffering,
config1, bfpgemm.cc), with A pre-encoded to bfp16ebs8 by the host as config3
does, so each core's A stream carries 9 bytes per 8 values instead of 16 and
the cores skip the conversion. W is bfp16ebs8; C accumulates and leaves as
bf16. Each core owns an M_T x 128 C tile; a group is 1024 outputs (8 core
columns x 128). Two programs: bfpgemm (M_T = 128, bands of 512 rows) and
bfpgemm_s (M_T = 32, bands of 128 rows, for short products). Shapes are set at
dispatch time, so one xclbin serves every matrix.

A job is one band by up to `ng` output groups, or one group by up to `nb`
bands (the W matrix advancing by b_rep words per band: 0 within one product,
a matrix stride for a batch of products). Host layouts (bfp_gemm.cpp):
  A   [core row][band][K / 512][k block 8][32-row sub-tile][row pair 2]
      [column block 8][half 2][row 8] groups of 8 k values (bfp16ebs8): the
      order the cores' MAC loop reads, forwarded by the memtiles unchanged.
  W   [core column c][group][K / 64][64 x 16 bfp16ebs8]: the L1-tiled, 1x2
      8x8 block shuffle of mlir-aie's gemm_atb_layout.h.
  C   [core column c][tile][4 M_T][128] bf16.

Dispatch-time values: ng, nb, kb = K / 64, tiles = ng nb, nb kc, ng kb, b_rep,
a_base and a_row (A words of the job and of one core row's strip), b_base and
b_col (W words of the job and of one column's groups), c_base and c_col (C
words), t1, t2 and t3 (the tiles core columns 2-3, 4-5 and 6-7 compute and
drain: all, or 1 when those columns have no outputs; they then only pass A
and B along).
Instruction streams of several jobs concatenate into one dispatch; the cores
restart at every barrier.
Usage: bfpgemm.py <out dir>: writes bfpgemm.xclbin, bfpgemm_s.xclbin and their
dispatch libraries, the files gufo::npu::BfpGemm::Open loads
(src/core/npu/bfp_gemm.hpp). Needs an mlir-aie (IRON) Python with Peano.
"""
import os
import sys

import aie.iron as iron
import aie.iron.kernels as kernels
import numpy as np
from aie.dialects.aiex import v8bfp16ebs8
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (Buffer, DispatchTime, ExternalFunction, In, InOut, Kernel, ObjectFifo, Program, Runtime,
                      TaskGroup, Worker, WorkerRuntimeBarrier)
from aie.iron.controlflow import range_
from aie.iron.device import Tile, from_name
from ml_dtypes import bfloat16

COLS, ROWS = 8, 4
K_T, N_T, MTK = 64, 128, 512
A_WORDS, W_WORDS, C_WORDS = 64 << 20, 512 << 20, 64 << 20  # W: up to a 2 GiB arena
B_TILE_WORDS = K_T * (N_T // 8) * 9 // 4       # one (64, 128) bfp16 tile: 2304 words
HERE = os.path.dirname(os.path.abspath(__file__))


def tap(total, offset, sizes, strides):
    return TensorAccessPattern((total,), offset, sizes, strides)


def design(M_T):
    """The design for C tiles of M_T rows per core (128: 512-row bands; 32:
    128-row bands for short inputs, the `bfpgemm_s` program)."""
    SUBS = M_T // 32                          # 32-row A sub-tiles per k block
    A_CHUNK_WORDS = M_T * MTK // 8 * 9 // 4   # one core row's (M_T, 512) bfp16 A chunk

    @iron.jit(aiecc_flags=["--dynamic-objFifos"])
    def bfpgemm(A: In, W: In, C: InOut, *, ng: DispatchTime[np.int32], nb: DispatchTime[np.int32],
                kb: DispatchTime[np.int32], tiles: DispatchTime[np.int32], nbkc: DispatchTime[np.int32],
                ngkb: DispatchTime[np.int32], b_rep: DispatchTime[np.int32], a_base: DispatchTime[np.int32], a_row: DispatchTime[np.int32],
                b_base: DispatchTime[np.int32], b_col: DispatchTime[np.int32], c_base: DispatchTime[np.int32],
                c_col: DispatchTime[np.int32], t1: DispatchTime[np.int32], t2: DispatchTime[np.int32],
                t3: DispatchTime[np.int32]):
        # A comes pre-encoded as bfp16 sub-tiles in the cores' order ([k block]
        # [32-row sub-tile]), so the memtile forwards it without reordering.
        a_l2_ty = np.ndarray[(M_T, MTK // 8), np.dtype[v8bfp16ebs8]]
        a_l1_ty = np.ndarray[(32, K_T // 8), np.dtype[v8bfp16ebs8]]
        b_ty = np.ndarray[(K_T, N_T // 8), np.dtype[v8bfp16ebs8]]
        c_l2_ty = np.ndarray[(ROWS * M_T, N_T), np.dtype[bfloat16]]
        c_l1_ty = np.ndarray[(M_T, N_T), np.dtype[bfloat16]]
        rtp_ty = np.ndarray[(4,), np.dtype[np.int32]]

        inc = os.path.join(HERE, "vendor")  # aie_kernel_utils.h, aie_arch.h from mlir-aie
        zero = kernels.zero((M_T, N_T), bfloat16)
        matmul = ExternalFunction("matmul_vectorized_different_datatypes", source_file=os.path.join(HERE, "bfpgemm.cc"),
                                  arg_types=[a_l1_ty, b_ty, c_l1_ty], object_file_name=f"bfpgemm_{M_T}.o",
                                  compile_flags=[f"-I{inc}", f"-DBFP_ROWS={M_T}"])

        a_fifos, a_l1 = [], []
        for row in range(ROWS):
            f = ObjectFifo(a_l2_ty, name=f"A_L3L2_{row}", depth=2)
            a_fifos.append(f)
            a_l1.append(f.cons().forward(obj_type=a_l1_ty, name=f"A_L2L1_{row}", depth=2))
        b_fifos, b_l1 = [], []
        for col in range(COLS):
            f = ObjectFifo(b_ty, name=f"B_L3L2_{col}", depth=2)
            b_fifos.append(f)
            b_l1.append(f.cons().forward(obj_type=b_ty, name=f"B_L2L1_{col}", depth=2))
        c_fifos, c_l1 = [], [[None] * COLS for _ in range(ROWS)]
        c_dims = TensorAccessPattern.full((M_T, N_T)).tile((8, 8)).inverse()
        for col in range(COLS):
            f = ObjectFifo(c_l2_ty, name=f"C_L2L3_{col}", depth=2, to_stream=c_dims)
            c_fifos.append(f)
            parts = f.prod().join([M_T * N_T * i for i in range(ROWS)], obj_types=[c_l1_ty] * ROWS,
                                  names=[f"C_L1L2_{col}_{row}" for row in range(ROWS)], depths=[1] * ROWS)
            for row in range(ROWS):
                c_l1[row][col] = parts[row]

        keys = [(row, col) for row in range(ROWS) for col in range(COLS)]
        rtps = {k: Buffer(rtp_ty, name=f"rtp_{k[0]}_{k[1]}", use_write_rtp=True) for k in keys}
        barriers = {k: WorkerRuntimeBarrier() for k in keys}

        def body(in_a, in_b, out_c, rtp, bar, zero, matmul):
            bar.wait_for_value(1)
            bar.release_with_value(1)
            # Idle tiles: a column with no outputs in the job still takes its A
            # and B objects (shared streams) but computes and drains nothing.
            for _ in range_(rtp[2]):
                for _ in range_(rtp[1]):
                    b = in_b.acquire(1)
                    for _ in range(SUBS):
                        in_a.acquire(1)
                        in_a.release(1)
                    in_b.release(1)
            for _ in range_(rtp[0]):
                c = out_c.acquire(1)
                zero(c)
                for _ in range_(rtp[1]):
                    b = in_b.acquire(1)
                    for _ in range(SUBS):
                        a = in_a.acquire(1)
                        matmul(a, b, c)
                        in_a.release(1)
                    in_b.release(1)
                out_c.release(1)

        workers = [Worker(body, [a_l1[r].cons(), b_l1[c].cons(), c_l1[r][c].prod(), rtps[(r, c)], barriers[(r, c)],
                                 zero, matmul], tile=Tile(c, 2 + r), stack_size=0xA00) for r, c in keys]

        a_ty = np.ndarray[(A_WORDS,), np.dtype[np.int32]]
        w_ty = np.ndarray[(W_WORDS,), np.dtype[np.int32]]
        c_ty = np.ndarray[(C_WORDS,), np.dtype[np.int32]]

        def sequence(A, W, C, ng, nb, kb, tiles, nbkc, ngkb, b_rep, a_base, a_row, b_base, b_col, c_base, c_col, t1, t2,
                     t3, ah, bh, ch):
            # Column classes 0-1 (always active), 2-3 (t1 tiles), 4-5 (t2) and
            # 6-7 (t3).
            active = [tiles, tiles, t1, t1, t2, t2, t3, t3]
            for k in keys:
                rtps[k][0] = active[k[1]]
                rtps[k][1] = kb
                rtps[k][2] = tiles - active[k[1]] if k[1] >= 2 else 0
                barriers[k].set(1)
            tg = TaskGroup()
            for r in range(ROWS):
                # The band's strip for core row r, once per group (stride 0).
                ah[r].fill(A, tap(A_WORDS, a_base + a_row * r, [ng, nbkc, A_CHUNK_WORDS // 256, 256],
                                  [0, A_CHUNK_WORDS, 256, 1]),
                           group=tg)
            for c in range(COLS):
                bh[c].fill(W, tap(W_WORDS, b_base + b_col * c, [nb, ngkb, 9, B_TILE_WORDS // 9], [b_rep, B_TILE_WORDS, B_TILE_WORDS // 9, 1]),
                           group=tg)
            for c in range(COLS):
                ch[c].drain(C, tap(C_WORDS, c_base + c_col * c, [1, active[c], ROWS * M_T, N_T // 2], [0, ROWS * M_T * N_T // 2, N_T // 2, 1]),
                            wait=True, group=tg)
            tg.finish()

        rt = Runtime(sequence, [a_ty, w_ty, c_ty, ng, nb, kb, tiles, nbkc, ngkb, b_rep, a_base, a_row, b_base, b_col, c_base,
                                c_col, t1, t2, t3,
                                [f.prod() for f in a_fifos], [f.prod() for f in b_fifos], [f.cons() for f in c_fifos]])
        return Program(from_name("npu2", n_cols=None), rt, workers=workers).resolve_program()


    return bfpgemm


if __name__ == "__main__":
    import shutil
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    iron.set_current_device(from_name("npu2", n_cols=None))
    for name, m_t in (("bfpgemm", 128), ("bfpgemm_s", 32)):
        p = design(m_t).specialize()
        p.compile(xclbin_path=os.path.join(out, name + ".xclbin"))
        shutil.copyfile(p.get_dispatch_lib_path(), os.path.join(out, name + "_dispatch.so"))
