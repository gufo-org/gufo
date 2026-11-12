"""Generic Q8_0 GEMM on the whole NPU (8 columns x 4 rows): C = W x for up to
8 G rows of x (q8gemm.cc has the arithmetic and record formats). Every core
holds the A rows (G groups of 8 rows, KC chunks of 512 codes, two A records
each) and computes NT tiles of 8 output columns. Each of the 16 shim streams
broadcasts to two cores: both keep the A records, and they take alternate B
records. Each column joins its four cores' output records.

Dispatch-time values: g, kc, nt, ga = 2 g kc (A records), gt = g nt (output
records per core), nt2 = 2 nt, kt = 2 kc nt (B records per stream) and the
word offsets a_base, b_base and c_base of the rows in A, the matrix in W and
the output in C. Instruction streams of several products concatenate into one
dispatch (the cores restart at every barrier).
Usage: q8gemm.py <out dir>: writes q8gemm.xclbin and q8gemm_dispatch.so, the
files gufo::npu::Q8Gemm::Open loads (src/core/npu/q8_gemm.hpp). Needs an
mlir-aie (IRON) Python with Peano.
"""
import os
import sys

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (Buffer, DispatchTime, ExternalFunction, In, InOut, Kernel, ObjectFifo, Program, Runtime,
                      TaskGroup, Worker, WorkerRuntimeBarrier)
from aie.iron.controlflow import range_
from aie.iron.device import Tile, from_name

COLS, ROWS = 8, 4
OBJ = 1088                      # words per record (4352 bytes)
OUT = 128                       # words per output record (one core's 8 x 8 tile)
MAX_A, MAX_C = 8, 32            # A records (4096 bytes kept) and 8 x 8 accumulator tiles per core
A_WORDS, W_WORDS, C_WORDS = 64 << 20, 256 << 20, 64 << 20
HERE = os.path.dirname(os.path.abspath(__file__))


def tap(total, offset, sizes, strides):
    return TensorAccessPattern((total,), offset, sizes, strides)


@iron.jit
def q8gemm(A: In, W: In, C: InOut, *, g: DispatchTime[np.int32], kc: DispatchTime[np.int32],
           nt: DispatchTime[np.int32], ga: DispatchTime[np.int32], gt: DispatchTime[np.int32],
           nt2: DispatchTime[np.int32], kt: DispatchTime[np.int32], a_base: DispatchTime[np.int32],
           b_base: DispatchTime[np.int32], c_base: DispatchTime[np.int32]):
    rec_ty = np.ndarray[(OBJ,), np.dtype[np.int32]]
    out_ty = np.ndarray[(OUT,), np.dtype[np.int32]]
    out4_ty = np.ndarray[(ROWS * OUT,), np.dtype[np.int32]]
    abuf_ty = np.ndarray[(MAX_A * 1024,), np.dtype[np.int32]]
    rtp_ty = np.ndarray[(8,), np.dtype[np.int32]]
    obj = "q8gemm.o"
    begin = ExternalFunction("gemm_begin", source_file=os.path.join(HERE, "q8gemm.cc"), arg_types=[rtp_ty, np.int32],
                             object_file_name=obj, compile_flags=["-O2"] + os.environ.get("GEMM_FLAGS", "").split())
    copy = Kernel("gemm_copy", obj, [rec_ty, abuf_ty])
    mac = Kernel("gemm_mac", obj, [abuf_ty, rec_ty])
    store = Kernel("gemm_store", obj, [out_ty])

    streams, l1, outs, joins = [], {}, [], {}
    for c in range(COLS):
        for h in range(2):
            f = ObjectFifo(rec_ty, name=f"w_{c}_{h}", depth=4)
            streams.append(f)
            fwd = f.cons().forward(obj_type=rec_ty, name=f"l1_{c}_{h}", depth=2)
            l1[(c, 2 * h)] = l1[(c, 2 * h + 1)] = fwd
        o = ObjectFifo(out4_ty, name=f"y_{c}", depth=2)
        outs.append(o)
        parts = o.prod().join([OUT * j for j in range(ROWS)], obj_types=[out_ty] * ROWS,
                              names=[f"y_{c}_{j}" for j in range(ROWS)], depths=[2] * ROWS)
        for j in range(ROWS):
            joins[(c, j)] = parts[j]

    rtps = {k: Buffer(rtp_ty, name=f"rtp_{k[0]}_{k[1]}", use_write_rtp=True) for k in l1}
    barriers = {k: WorkerRuntimeBarrier() for k in l1}

    def body(w_in, y_out, a, rtp, bar, parity, begin, copy, mac, store):
        bar.wait_for_value(1)
        bar.release_with_value(1)
        begin(rtp, parity)
        for _ in range_(rtp[3]):
            r = w_in.acquire(1)
            copy(r, a)
            w_in.release(1)
        for _ in range_(rtp[1]):
            for _ in range_(rtp[5]):
                r = w_in.acquire(1)
                mac(a, r)
                w_in.release(1)
        for _ in range_(rtp[4]):
            y = y_out.acquire(1)
            store(y)
            y_out.release(1)

    workers = [Worker(body, [l1[k].cons(), joins[k].prod(), Buffer(abuf_ty, name=f"a_{k[0]}_{k[1]}"), rtps[k],
                             barriers[k], k[1] % 2, begin, copy, mac, store],
                      tile=Tile(k[0], 2 + k[1]), stack_size=0x800, dynamic_objfifo_lowering=True) for k in sorted(l1)]

    a_ty = np.ndarray[(A_WORDS,), np.dtype[np.int32]]
    w_ty = np.ndarray[(W_WORDS,), np.dtype[np.int32]]
    c_ty = np.ndarray[(C_WORDS,), np.dtype[np.int32]]

    def sequence(A, W, C, g, kc, nt, ga, gt, nt2, kt, a_base, b_base, c_base, wh, yh):
        for k in sorted(l1):
            for i, v in enumerate((g, kc, nt, ga, gt, nt2)):
                rtps[k][i] = v
            barriers[k].set(1)
        tg = TaskGroup()
        for s in range(2 * COLS):
            wh[s].fill(A, tap(A_WORDS, a_base, [1, 1, ga, OBJ], [0, 0, OBJ, 1]), group=tg)
            wh[s].fill(W, tap(W_WORDS, b_base + kt * (s * OBJ), [1, 1, kt, OBJ], [0, 0, OBJ, 1]), group=tg)
        for c in range(COLS):
            yh[c].drain(C, tap(C_WORDS, c_base + gt * (c * ROWS * OUT), [1, 1, gt, ROWS * OUT], [0, 0, ROWS * OUT, 1]),
                        wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [a_ty, w_ty, c_ty, g, kc, nt, ga, gt, nt2, kt, a_base, b_base, c_base, [f.prod() for f in streams],
                            [o.cons() for o in outs]])
    return Program(from_name("npu2", n_cols=None), rt, workers=workers).resolve_program()


if __name__ == "__main__":
    import shutil
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    iron.set_current_device(from_name("npu2", n_cols=None))
    p = q8gemm.specialize()
    p.compile(xclbin_path=os.path.join(out, "q8gemm.xclbin"))
    shutil.copyfile(p.get_dispatch_lib_path(), os.path.join(out, "q8gemm_dispatch.so"))
