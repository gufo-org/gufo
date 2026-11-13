"""EmbeddingGemma 2 encoder program on the whole NPU (8 columns x 4 rows):
milestone 1, the addnorm row phase alone (docs/internal/design/
npu-embedding-encoder.md).

S holds one record per token row, [t | s_hi | s_lo | xn] of 512 bf16 each
(1024 words). Column c takes the objects of R rows c, c + 8, ...: its B
stream carries [t | s_hi | s_lo] of R = 4 rows to its four cores; core q
computes row q and writes its [s_hi' | s_lo' | xn'], and the memtile joins the
four rows in record order, so the drain writes slots 1..3 of the records in
place. The A stream carries the
phase's parameters [w_post | w_next | scale | eps] to every core once.

Dispatch-time values: n_obj (objects per column) and s_base (S words of the
first record).
Usage: embenc.py <out dir>: writes embenc.xclbin and embenc_dispatch.so.
"""
import os
import sys

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (Buffer, DispatchTime, ExternalFunction, In, InOut, ObjectFifo, Program, Runtime, TaskGroup,
                      Worker, WorkerRuntimeBarrier)
from aie.iron.controlflow import range_
from aie.iron.device import Tile, from_name
from ml_dtypes import bfloat16

COLS, ROWS = 8, 4
W = 512                       # hidden width
R = 4                         # rows per object
REC = 4 * W                   # bf16 per record
REC_W = REC // 2              # words per record
PARAMS = 2 * W + 64           # bf16: w_post, w_next, scale and eps (f32), pad
S_WORDS, P_WORDS = 64 << 20, 1 << 16
HERE = os.path.dirname(os.path.abspath(__file__))


def tap(total, offset, sizes, strides):
    return TensorAccessPattern((total,), offset, sizes, strides)


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def embenc(S: InOut, P: In, *, n_obj: DispatchTime[np.int32], s_base: DispatchTime[np.int32]):
    in_ty = np.ndarray[(R * 3 * W,), np.dtype[bfloat16]]
    out_ty = np.ndarray[(3 * W,), np.dtype[bfloat16]]          # one row's [s_hi | s_lo | xn]
    out4_ty = np.ndarray[(ROWS * 3 * W,), np.dtype[bfloat16]]
    par_ty = np.ndarray[(PARAMS,), np.dtype[bfloat16]]
    rtp_ty = np.ndarray[(4,), np.dtype[np.int32]]
    addnorm = ExternalFunction("embenc_addnorm", source_file=os.path.join(HERE, "embenc.cc"),
                               arg_types=[par_ty, in_ty, out_ty, np.int32], compile_flags=["-O2"])

    rows_in, outs, joins = [], [], {}
    for c in range(COLS):
        f = ObjectFifo(in_ty, name=f"rows_{c}", depth=2)
        rows_in.append(f.cons().forward(obj_type=in_ty, name=f"rows_l1_{c}", depth=2))
        rows_in[-1] = (f, rows_in[-1])
        o = ObjectFifo(out4_ty, name=f"out_{c}", depth=2)
        outs.append(o)
        parts = o.prod().join([3 * W * q for q in range(ROWS)], obj_types=[out_ty] * ROWS,
                              names=[f"out_{c}_{q}" for q in range(ROWS)], depths=[2] * ROWS)
        for q in range(ROWS):
            joins[(c, q)] = parts[q]
    pars = []
    for q in range(ROWS):
        f = ObjectFifo(par_ty, name=f"par_{q}", depth=1)
        pars.append((f, f.cons().forward(obj_type=par_ty, name=f"par_l1_{q}", depth=1)))

    keys = [(c, q) for c in range(COLS) for q in range(ROWS)]
    rtps = {k: Buffer(rtp_ty, name=f"rtp_{k[0]}_{k[1]}", use_write_rtp=True) for k in keys}
    barriers = {k: WorkerRuntimeBarrier() for k in keys}

    def body(rows, out, par, rtp, bar, q, addnorm):
        bar.wait_for_value(1)
        bar.release_with_value(1)
        p = par.acquire(1)
        for _ in range_(rtp[0]):
            x = rows.acquire(1)
            y = out.acquire(1)
            addnorm(p, x, y, q)
            rows.release(1)
            out.release(1)
        par.release(1)

    workers = [Worker(body, [rows_in[c][1].cons(), joins[(c, q)].prod(), pars[q][1].cons(), rtps[(c, q)],
                             barriers[(c, q)], q, addnorm], tile=Tile(c, 2 + q), stack_size=0x800)
               for c, q in keys]

    s_ty = np.ndarray[(S_WORDS,), np.dtype[np.int32]]
    p_ty = np.ndarray[(P_WORDS,), np.dtype[np.int32]]

    def sequence(S, P, n_obj, s_base, rh, oh, ph):
        for k in keys:
            rtps[k][0] = n_obj
            barriers[k].set(1)
        tg = TaskGroup()
        for q in range(ROWS):
            ph[q].fill(P, tap(P_WORDS, 0, [1, 1, 1, PARAMS // 2], [0, 0, 0, 1]), group=tg)
        stride = COLS * R * REC_W  # words between a column's objects
        for c in range(COLS):
            base = s_base + c * R * REC_W
            rh[c].fill(S, tap(S_WORDS, base, [1, n_obj, R, 3 * W // 2], [0, stride, REC_W, 1]), group=tg)
        for c in range(COLS):
            base = s_base + c * R * REC_W + W // 2
            oh[c].drain(S, tap(S_WORDS, base, [1, n_obj, R, 3 * W // 2], [0, stride, REC_W, 1]), wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [s_ty, p_ty, n_obj, s_base, [f.prod() for f, _ in rows_in], [o.cons() for o in outs],
                            [f.prod() for f, _ in pars]])
    return Program(from_name("npu2", n_cols=None), rt, workers=workers).resolve_program()


if __name__ == "__main__":
    import shutil
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    iron.set_current_device(from_name("npu2", n_cols=None))
    p = embenc.specialize()
    p.compile(xclbin_path=os.path.join(out, "embenc.xclbin"))
    shutil.copyfile(p.get_dispatch_lib_path(), os.path.join(out, "embenc_dispatch.so"))
