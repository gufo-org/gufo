"""Phase plan of the Qwen3 prefill xclbin (qwen3pre.py), shared with the host.

Ported from LFM3's prefill (lfm3-gufo-extension npu/iron/prefill).
Buffers (bf16 elements unless noted):
  a: token records, M x REC: [xn | x | o | pad], RoPE tables at 4096 of the qkv rows
  b: GEMM weights, bfp16ebs8 blocks (9 bytes per 8 values), in phase order
  c: scratch: SwiGLU output, qkv rows, per-(layer, pass) K/V pieces, attention
     outputs and A blocks (128 x 512) of small per-phase constants
"""

H, I, REC = 2048, 6144, 8192
M_TILE, N_TILE, K_TILE, MTK = 128, 128, 64, 512
QKV_ROW = 4608
PIECE = 16384                      # one core's C object
LAYERS = 28
HEAD_ROWS = 512
STATE = 6
ABLOCK = M_TILE * MTK
PASSES = 2                         # kv heads 0..3, then 4..7


def narrow_tiles(N):
    """Tiles per column and K step of a narrow GEMM (rows < 512): the largest T <= 4
    dividing the column's tiles; core rows >= T idle."""
    per_col = N // N_TILE // 8
    return next(t for t in (4, 3, 2, 1) if per_col % t == 0)


class Plan:
    """Qwen3 dense prefill (28 x 2048, 16 q / 8 kv heads of 128, SwiGLU 6144).

    The core attention handles 4 kv heads with 4 query heads each, so a layer
    runs attention in two passes (kv heads 0..3, then 4..7); a pass's query
    heads 4 p + j are query heads 2 p + j (j < 2) of its kv head p and zero
    rows otherwise (their outputs get zero o_proj weights). The pass's gate
    rows are zero, so sigmoid(0) = 1/2 scales every head: o_proj doubles.
    o_proj's inputs are pass 1's 16 heads, then pass 0's.
    Norm phases are plain residuals (x += o, then xn = rmsnorm(x)); RMSNorm
    weights fold into the following GEMM.
    """

    def __init__(self, M=512):
        self.M, self.kinds = M, ["attn"] * LAYERS
        self.c_off = 0
        self.b_off = 0
        self.sw = self._c(M * 2 * I)
        self.qs = self._c(M * QKV_ROW + 512)
        self.oh = 16 * M * 128               # one pass's head-major attention output
        self.ablocks = []          # (c offset, what) for the host to fill
        self.pieces = {}           # (layer, pass) -> c offset of its K/V pieces
        # Both passes' outputs adjacent, pass 1 first ([32 heads][M][128]: pass
        # 1's 16 heads, then pass 0's), reused by every layer. Each pass drains
        # three junk quarters oh apart below its output: pass 0's land on pass
        # 1's region before pass 1 writes it, pass 1's below both.
        base = self._c(5 * self.oh)
        ao = [base + 4 * self.oh, base + 3 * self.oh]
        self.phases = [self._norm(("embed",))]
        for l in range(LAYERS):
            for p_ in range(PASSES):
                piece = self.pieces[(l, p_)] = self._c(M // 16 * 4 * PIECE)
                self.phases += [
                    self._gemm(H, 4096, "plain", ("rec",), ("plain", self.qs, QKV_ROW), (l, "qkv", p_)),
                    dict(kind="qkr", w=self._ablock((l, "qkn")), qs=self.qs, piece=piece, mode=0),
                    dict(kind="attn", piece=piece, ao=ao[p_], mode=0),
                ]
            self.phases += [
                self._gemm(2 * H, H, "plain", ("heads", ao[1]), ("o",), (l, "attn_out")),
                self._norm((l, "post")),
                self._gemm(H, 2 * I, "swiglu", ("rec",), ("plain", self.sw, 2 * I), (l, "gate_up")),
                self._gemm(I, H, "plain", ("half", self.sw, 2 * I), ("o",), (l, "down")),
                self._norm((l, "ffn_post")),
            ]
        self.junk = self._c(3 * M_TILE * N_TILE)  # idle core rows of narrow GEMMs drain here
        self.b_size = self.b_off
        self.c_size = self.c_off

    def layer_phases(self, layer):
        """One layer's schedule, with an integer or staged layer index.

        Only weights, norm constants and KV pieces advance between layers;
        activations and attention outputs reuse the same scratch regions.
        """
        count = (len(self.phases) - 1) // LAYERS
        first = self.phases[1:1 + count]
        second = self.phases[1 + count:1 + 2 * count]
        for base, following in zip(first, second):
            ph = dict(base)
            for key in ("b", "w", "piece"):
                if key in base:
                    ph[key] = base[key] + layer * (following[key] - base[key])
            if "what" in base:
                ph["what"] = (layer, *base["what"][1:])
            yield ph

    def write(self, path):
        """The layout gufo::npu::Qwen3Prefill reads (qwen3pre_<M>_plan.txt):
        key-value lines, then one line per GEMM (layer, name, pass, K, N, B
        offset in bfp16 blocks), per A block (C offset, layer, name) and per
        K/V piece (layer, pass, C offset). Offsets are bf16 elements unless
        noted."""
        with open(path, "w") as f:
            f.write(f"M {self.M}\nrec {REC}\nqkv_row {QKV_ROW}\npiece_size {PIECE}\n")
            f.write(f"b_size {self.b_size}\nc_size {self.c_size}\nqs {self.qs}\n")
            for ph in self.phases:
                if ph["kind"] == "gemm":
                    w = ph["what"]
                    f.write(f"gemm {w[0]} {w[1]} {w[2] if len(w) > 2 else 0} {ph['K']} {ph['N']} {ph['b']}\n")
            for off, w in self.ablocks:
                f.write(f"ablock {off} {w[0] if w[0] != 'embed' else -1} {w[1] if len(w) > 1 else 'embed'}\n")
            for (l, p_), off in sorted(self.pieces.items()):
                f.write(f"piece {l} {p_} {off}\n")

    def codes(self):
        """One code per phase for the core interpreter (see prefill_next)."""
        out = []
        for ph in self.phases:
            if ph["kind"] == "gemm":
                pair = 0
                if ph["rows"] == 256:  # pair: two row tiles, T = 2 column tiles each
                    t_, pair = 2, 1
                    tiles = ph["N"] // N_TILE // 8 // t_
                elif ph["rows"] < HEAD_ROWS:  # narrow: groups of T tiles per column
                    t_ = narrow_tiles(ph["N"])
                    tiles = ph["N"] // N_TILE // 8 // t_
                else:
                    t_, tiles = 0, (ph["rows"] // M_TILE) * (ph["N"] // N_TILE) // 32
                assert tiles < 128
                mode = {"plain": 0, "swiglu": 1}[ph["mode"]]
                out.append(0 | mode << 3 | tiles << 5 | (ph["K"] // K_TILE) << 12 | t_ << 20 | pair << 23)
            else:
                out.append({"norm": 1, "qkr": 3, "attn": 4}[ph["kind"]] | ph.get("mode", 0) << 3)
        return out

    def _c(self, n):
        at = self.c_off
        self.c_off += (n + 63) // 64 * 64
        return at

    def _ablock(self, what):
        at = self._c(ABLOCK)
        self.ablocks.append((at, what))
        return at

    def _norm(self, what):
        return dict(kind="norm", w=self._ablock(what))

    def _gemm(self, K, N, mode, a, c, what, rows=None):
        at = self.b_off
        self.b_off += K * N // 8
        return dict(kind="gemm", K=K, N=N, mode=mode, a=a, c=c, b=at, what=what, rows=rows or self.M)


def ablock_put(vec):
    """A 128 x 512 A block whose first L1 objects hold vec linearly, 2048 values each
    (object i: rows 32 i .. 32 i + 31 of columns 0..63, blocked [m/r][k/s][r][s])."""
    import numpy as np
    vec = np.asarray(vec, np.uint16).reshape(-1)
    assert vec.size <= 4 * 2048
    z = np.zeros((M_TILE, MTK), np.uint16)
    p = np.arange(vec.size)
    obj, q = p // 2048, p % 2048
    z[obj * 32 + (q // 512) * 8 + (q // 8) % 8, ((q // 64) % 8) * 8 + q % 8] = vec
    return z.reshape(-1)
