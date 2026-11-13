"""One Qwen3 decode token on the whole NPU (8 columns x 4 rows), Q8_0 weights:
the 28-layer dense stack of Qwen3-ASR's text decoder and Qwen3-TTS's talker.

The cores run a phase interpreter (qwen3dec.cc, *_next): 142 phases per token
(input row, 28 layers x 5, head). Rows 2-3 (j = 0, 1) also attend: core (c, j)
holds query head 2c + j over KV head c. The runtime sequence feeds every
phase's inputs at the head of the weight streams and drains its outputs into
the state buffer S. `pos` (the position), the KV stream's four record counts
and the head's chunk count (rows / 256) are dispatch-time values; layout.py
lays out S and W, and writes qwen3dec_layout.txt for the host.

The same program also runs the 5 x 1024 stack of Qwen3-TTS's code predictor
(FFN 3072, 15 heads of 2048 rows, the head chosen by a dispatch-time offset):
the cores read the token's phases and dimensions from their runtime
parameters, so only the runtime sequence differs. Its sequence is the second
dispatch library, qwen3dec_p_dispatch.so, run on qwen3dec.xclbin's hardware
context (qwen3dec_p.xclbin is the identical program, kept for checks).

Usage: qwen3dec.py <out dir>: writes qwen3dec{,_p}.xclbin,
qwen3dec{,_p}_dispatch.so and qwen3dec{,_p}_layout.txt
(gufo::npu::Qwen3Decoder, src/core/npu/qwen3_decoder.hpp).
"""
import os
import shutil
import sys

import aie.iron as iron
import numpy as np
from aie.iron import (Buffer, DispatchTime, ExternalFunction, In, InOut, Kernel, ObjectFifo, Program, Runtime, TaskGroup,
                      Worker, WorkerRuntimeBarrier)
from aie.iron.controlflow import range_
from aie.iron.device import Tile, from_name

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import layout as L  # noqa: E402

COLS, ROWS, REC, OUT = L.COLS, L.ROWS, L.REC, 32


def design(model, S, W, scalars):
    lay = L.Layout(model)
    in_ty = np.ndarray[(12 * 2048,), np.dtype[np.int8]]  # the largest phase input (hidden 2048)
    rec_ty = np.ndarray[(REC,), np.dtype[np.int8]]
    out_ty = np.ndarray[(OUT,), np.dtype[np.int8]]
    out4_ty = np.ndarray[(ROWS * OUT,), np.dtype[np.int8]]
    q8_ty = np.ndarray[(8192,), np.dtype[np.int8]]
    scratch_ty = np.ndarray[(512,), np.dtype[np.float32]]
    cnt_ty = np.ndarray[(8,), np.dtype[np.int32]]
    rtp_ty = np.ndarray[(8,), np.dtype[np.int32]]
    src = os.path.join(HERE, "qwen3dec.cc")
    flags = ["-Os"]
    kernels = {}
    # The attention role's code is close to the 16 KB program memory: -Oz.
    for role, extra in (("ffn", []), ("attn", ["-DDECODE_ATTN_ROLE", "-Oz"])):
        obj = f"qwen3dec_{role}.o"
        k = {"next": ExternalFunction(f"{role}_next", source_file=src, arg_types=[cnt_ty, rtp_ty, np.int32, np.int32],
                                      object_file_name=obj, compile_flags=flags + extra)}
        for name, types in {"copy": [rec_ty, in_ty], "begin": [in_ty, q8_ty, scratch_ty], "pre_out": [in_ty, out_ty],
                            "rec": [q8_ty, rec_ty], "flush": [out_ty, cnt_ty], "finish": [],
                            "post_out": [out_ty]}.items():
            k[name] = Kernel(f"{role}_{name}", obj, types)
        kernels[role] = k

    w_l2l1, y_l1l2, w_fifos, y_fifos = [], [], [], []
    for c in range(COLS):
        col = []
        for half in range(2):
            # A stream broadcasts to both of its cores; each core keeps its own records of a
            # weight stream (alternating) and skips the rest. Deep memtile buffering keeps the
            # stream running through a phase's prologue.
            f = ObjectFifo(rec_ty, name=f"wl3_{c}_{half}", depth=16)
            w_fifos.append(f)
            fwd = f.cons().forward(obj_type=rec_ty, name=f"w{c}_{half}", depth=2)
            col += [fwd, fwd]
        w_l2l1.append(col)
        o = ObjectFifo(out4_ty, name=f"y{c}", depth=2)
        y_fifos.append(o)
        y_l1l2.append(o.prod().join([OUT * j for j in range(ROWS)], obj_types=[out_ty] * ROWS,
                                    names=[f"y{c}_{j}" for j in range(ROWS)], depths=[2] * ROWS))

    rtps = [[Buffer(rtp_ty, name=f"rtp_{c}_{j}", use_write_rtp=True) for j in range(ROWS)] for c in range(COLS)]
    barriers = [[WorkerRuntimeBarrier() for _ in range(ROWS)] for _ in range(COLS)]

    def body(w_in, y_out, x, q8, scratch, cnt, rtp, barrier, c, j, nxt, copy, begin, pre_out, rec, flush, finish,
             post_out):
        barrier.wait_for_value(1)
        barrier.release_with_value(1)
        for _ in range_(rtp[3]):
            nxt(cnt, rtp, c, j)
            for _ in range_(cnt[0]):
                w = w_in.acquire(1)
                copy(w, x)
                w_in.release(1)
            begin(x, q8, scratch)
            for _ in range_(cnt[2]):
                o = y_out.acquire(1)
                pre_out(x, o)
                y_out.release(1)
            for _ in range_(cnt[3]):
                o = y_out.acquire(1)
                for _ in range_(cnt[4]):
                    w = w_in.acquire(1)
                    rec(q8, w)
                    w_in.release(1)
                flush(o, cnt)
                y_out.release(1)
            finish()
            for _ in range_(cnt[5]):
                o = y_out.acquire(1)
                post_out(o)
                y_out.release(1)

    workers = []
    for c in range(COLS):
        for j in range(ROWS):
            k = kernels["attn" if j < 2 else "ffn"]
            bufs = [Buffer(in_ty, name=f"x_{c}_{j}"), Buffer(q8_ty, name=f"q8_{c}_{j}"),
                    Buffer(scratch_ty, name=f"scratch_{c}_{j}"), Buffer(cnt_ty, name=f"cnt_{c}_{j}")]
            workers.append(Worker(body, [w_l2l1[c][j].cons(), y_l1l2[c][j].prod(), *bufs, rtps[c][j], barriers[c][j],
                                         c, j, *k.values()],
                                  tile=Tile(c, 2 + j), stack_size=0x800, dynamic_objfifo_lowering=True))

    s_ty = np.ndarray[(lay.s_bytes,), np.dtype[np.int8]]
    w_ty = np.ndarray[(lay.w_bytes,), np.dtype[np.int8]]

    def sequence(S, W, *args):
        # A DMA task repeats at most 64 times, so the KV stream is four tasks of
        # recs_a + ... + recs_d records (each at least 1).
        *values, wh, yh = args
        v = dict(zip(scalars, values))
        recs = [v["recs_a"], v["recs_b"], v["recs_c"], v["recs_d"]]
        d = dict(pos=v["pos"], recs=recs, head_chunks=v["head_chunks"])
        if "head_base" in v:
            d["head_base"] = v["head_base"]
        for c in range(COLS):
            for j in range(ROWS):
                rtps[c][j][0] = v["pos"]
                rtps[c][j][1] = recs[0] + recs[1] + recs[2] + recs[3]
                rtps[c][j][2] = v["head_chunks"]
                rtps[c][j][3] = lay.PHASES
                rtps[c][j][4] = lay.H
                rtps[c][j][5] = lay.I
                rtps[c][j][6] = lay.LAYERS
                barriers[c][j].set(1)
        for phase in lay.phases(d):
            # One task group per phase when every shim tile holds at most 8 tasks: the FIFO
            # already orders a stream's inputs before its weights, so only the drains are
            # waited on. Otherwise the inputs go first in a group of their own.
            per_col = [sum(len(phase.heads(s)) + len(phase.stream(s)) for s in (2 * c, 2 * c + 1)) + len(phase.drains(c))
                       for c in range(COLS)]
            tg = TaskGroup()
            for s in range(2 * COLS):
                for (buf, tp) in phase.heads(s):
                    wh[s].fill(S if buf == "S" else W, tp, wait=max(per_col) > 8, group=tg)
            if max(per_col) > 8:
                tg.finish()
                tg = TaskGroup()
            for s in range(2 * COLS):
                for (buf, tp) in phase.stream(s):
                    wh[s].fill(S if buf == "S" else W, tp, group=tg)
            for c in range(COLS):
                for tp in phase.drains(c):
                    yh[c].drain(S, tp, wait=True, group=tg)
            tg.finish()

    rt = Runtime(sequence, [s_ty, w_ty, *[scalars[k] for k in scalars], [f.prod() for f in w_fifos],
                            [f.cons() for f in y_fifos]])
    return Program(from_name("npu2", n_cols=None), rt, workers=workers).resolve_program()


@iron.jit
def qwen3dec(S: InOut, W: In, *, pos: DispatchTime[np.int32], recs_a: DispatchTime[np.int32],
             recs_b: DispatchTime[np.int32], recs_c: DispatchTime[np.int32], recs_d: DispatchTime[np.int32],
             head_chunks: DispatchTime[np.int32]):
    return design("qwen3dec", S, W, dict(pos=pos, recs_a=recs_a, recs_b=recs_b, recs_c=recs_c, recs_d=recs_d,
                                          head_chunks=head_chunks))


@iron.jit
def qwen3dec_p(S: InOut, W: In, *, pos: DispatchTime[np.int32], recs_a: DispatchTime[np.int32],
               recs_b: DispatchTime[np.int32], recs_c: DispatchTime[np.int32], recs_d: DispatchTime[np.int32],
               head_chunks: DispatchTime[np.int32], head_base: DispatchTime[np.int32]):
    return design("qwen3dec_p", S, W, dict(pos=pos, recs_a=recs_a, recs_b=recs_b, recs_c=recs_c, recs_d=recs_d,
                                            head_chunks=head_chunks, head_base=head_base))


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    iron.set_current_device(from_name("npu2", n_cols=None))
    for name, fn in (("qwen3dec", qwen3dec), ("qwen3dec_p", qwen3dec_p)):
        p = fn.specialize()
        p.compile(xclbin_path=os.path.join(out, name + ".xclbin"))
        shutil.copyfile(p.get_dispatch_lib_path(), os.path.join(out, name + "_dispatch.so"))
        with open(os.path.join(out, name + "_layout.txt"), "w") as f:
            for k, v in L.Layout(name).items().items():
                f.write(f"{k} {v}\n")
