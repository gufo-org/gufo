"""On-device check of embenc's row phases against numpy (milestone 1).
Usage: test_embenc.py <dir with embenc.xclbin and embenc_dispatch.so> [rows]
Needs pyxrt and numpy."""
import ctypes
import sys
import time

import numpy as np
import pyxrt

W, R, COLS = 512, 4, 8
REC = 4 * W


def bf16(x):
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def f32(h):
    return (np.asarray(h, np.uint16).astype(np.uint32) << 16).view(np.float32)


def main():
    d = sys.argv[1]
    M = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    assert M % (COLS * R) == 0
    rng = np.random.default_rng(5)
    t = rng.normal(0, 3, (M, W)).astype(np.float32)
    s = rng.normal(0, 10, (M, W)).astype(np.float32)
    s_hi = bf16(s)
    s_lo = bf16(s - f32(s_hi))
    wpost = rng.normal(1, 0.3, W).astype(np.float32)
    wnext = rng.normal(1, 0.3, W).astype(np.float32)
    scale, eps = 0.75, 1e-6

    S = np.zeros((M, REC), np.uint16)
    S[:, :W], S[:, W:2 * W], S[:, 2 * W:3 * W] = bf16(t), s_hi, s_lo
    P = np.zeros(2 * W + 64, np.uint16)
    P[:W], P[W:2 * W] = bf16(wpost), bf16(wnext)
    P[2 * W:2 * W + 4] = np.array([scale, eps], np.float32).view(np.uint16)

    # numpy reference from the bf16 inputs
    tb, sb = f32(bf16(t)).astype(np.float64), f32(s_hi).astype(np.float64) + f32(s_lo)
    wp, wn = f32(bf16(wpost)).astype(np.float64), f32(bf16(wnext)).astype(np.float64)
    tn = tb / np.sqrt((tb ** 2).mean(1, keepdims=True) + eps) * wp
    s1 = (sb + tn) * scale
    xn = s1 / np.sqrt((s1 ** 2).mean(1, keepdims=True) + eps)  # w_next folds into the next product

    lib = ctypes.CDLL(d + "/embenc_dispatch.so")
    gen = lib.dispatch_generate
    gen.restype = ctypes.c_int
    gen.argtypes = [ctypes.c_int32] * 2 + [ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32))]
    out = ctypes.POINTER(ctypes.c_uint32)()
    n = gen(M // (COLS * R), 0, ctypes.byref(out))
    ins = np.ctypeslib.as_array(out, (n,)).copy()

    dev = pyxrt.device(0)
    xb = pyxrt.xclbin(d + "/embenc.xclbin")
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    kern = pyxrt.kernel(ctx, [k.get_name() for k in xb.get_kernels() if k.get_name().startswith("MLIR_AIE")][0])
    to_dev = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    from_dev = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
    bo_i = pyxrt.bo(dev, ins.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
    bo_s = pyxrt.bo(dev, S.nbytes, pyxrt.bo.host_only, kern.group_id(3))
    bo_p = pyxrt.bo(dev, P.nbytes, pyxrt.bo.host_only, kern.group_id(4))
    for bo, arr in ((bo_i, ins), (bo_s, S), (bo_p, P)):
        bo.write(arr.tobytes(), 0)
        bo.sync(to_dev)
    kern(3, bo_i, n, bo_s, bo_p).wait()
    bo_s.sync(from_dev)
    got = np.frombuffer(bo_s.read(S.nbytes, 0), np.uint16).reshape(M, REC)
    g_s = f32(got[:, W:2 * W]).astype(np.float64) + f32(got[:, 2 * W:3 * W])
    g_x = f32(got[:, 3 * W:]).astype(np.float64)
    rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)
    print(f"addnorm M {M}: state rel err {rel(g_s, s1):.2e}, xn rel err {rel(g_x, xn):.2e}")
    # timing (S is rewritten in place: inputs no longer match, timing only)
    reps = 20
    t0 = time.perf_counter()
    for _ in range(reps):
        kern(3, bo_i, n, bo_s, bo_p).wait()
    dt = (time.perf_counter() - t0) / reps
    print(f"time {dt * 1e6:.0f} us ({M * REC * 2 * 2 / 2 / dt / 1e9:.1f} GB/s of records)")
    floor = rel(f32(bf16(xn)).astype(np.float64), xn)  # bf16 output rounding
    own = g_s / np.sqrt((g_s ** 2).mean(1, keepdims=True) + eps)
    print(f"xn vs the device state's own norm {rel(g_x, own):.2e}; ratio of row scales "
          f"{np.median((np.linalg.norm(g_x, axis=1) / np.linalg.norm(own, axis=1))):.5f}")
    print(f"xn bf16 rounding floor {floor:.2e}")
    ok = rel(g_s, s1) < 1e-4 and rel(g_x, xn) < 1.3 * floor
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


main()
