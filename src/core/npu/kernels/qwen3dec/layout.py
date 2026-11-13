"""Buffer layouts and per-phase transfers of qwen3dec.py: one decode token of a
Qwen3 dense block stack (28 layers, hidden 2048, 16 query and 8 KV heads of
128, SwiGLU 6144) on the whole NPU, Q8_0 weights.

Hidden rows: core (c, j) owns rows ((c * 8 + t) * 4 + j) * 8 + e. Every drain
of a column takes one 32-byte object from each of its four cores per step, so
a vector drained this way lies in DDR in natural order.

S (state) holds, per layer, an attention input block and an FFN input block
[h f32 H | o f32 H | unused bf16 H | pre-norm w bf16 H], then the q/k/v rows,
the attention output, the FFN activations, the RoPE table, the KV caches, the
logits and the input row. W holds the weight records: 4 rows of Q8_0 codes and
their bf16 block scales (8704 bytes), per stream and core.
"""

H, Q, KV, I, D = 2048, 2048, 1024, 6144, 128
LAYERS = 28
COLS, ROWS = 8, 4
REC = 4 * H + 4 * (H // 32) * 2   # 8704
BLOCK = 12 * H                    # [h f32 | o f32 | unused | pre-norm w bf16]
PHASES = 1 + 5 * LAYERS + 1
QKV_ROWS = Q + 2 * KV             # 4096: 128 per core
QKV_BYTES = 2 * REC               # rows f32, then q-norm and k-norm bf16
QKV_RECS, OUT_RECS, GU_RECS, DOWN_RECS = 32, 16, 96, 48  # per core
HEAD_MAX_CHUNKS = 594             # 8 rows per chunk and core: 152064 rows
ACT_COL = 12 * ROWS * 32          # gate/up output per column (bf16)
Q_COL = 16 * ROWS * 32            # q/k/v rows per column (f32)
CTX = 4096                        # KV positions
KV_TASKS = 4                      # a DMA task repeats at most 64 records


def t(total, offset, sizes, strides):
    from aie.helpers.taplib import TensorAccessPattern  # the host does not need mlir-aie
    return TensorAccessPattern((total,), offset, sizes, strides)


class Layout:
    def __init__(self, ctx=CTX):
        self.ctx = ctx
        o = 0

        def take(n):
            nonlocal o
            at = o
            o += (n + 63) // 64 * 64
            return at
        self.blocks = take((2 * LAYERS + 1) * BLOCK + REC)
        self.act = take(2 * REC)
        self.att = take(17 * 1024 + 512)
        self.junk = take(COLS * 4 * 64)
        self.qkv = [take(QKV_BYTES + REC) for _ in range(LAYERS)]
        self.rope = take((ctx + 1) * 512 + REC)
        self.kv = [take(COLS * ROWS * ctx * 512 + REC) for _ in range(LAYERS)]
        self.logits = take(COLS * ROWS * HEAD_MAX_CHUNKS * 8 * 4)
        self.input = take(H * 4 + REC)
        self.s_bytes = o
        o = 0
        self.w = {}
        for l in range(LAYERS):
            for name, recs in (("qkv", QKV_RECS), ("out", OUT_RECS), ("gate_up", GU_RECS), ("down", DOWN_RECS)):
                self.w[(l, name)] = take(2 * COLS * 2 * recs * REC)
        self.w["head"] = take(2 * COLS * 2 * HEAD_MAX_CHUNKS * 2 * REC)
        self.w_bytes = o
        assert self.s_bytes < 2 ** 31 and self.w_bytes < 2 ** 31

    def block(self, b):
        return self.blocks + b * BLOCK

    def items(self):
        """Named offsets for the host (qwen3dec_layout.txt)."""
        out = {"s_bytes": self.s_bytes, "w_bytes": self.w_bytes, "ctx": self.ctx, "rec": REC,
               "block_bytes": BLOCK, "blocks": self.blocks, "act": self.act, "att": self.att,
               "rope": self.rope, "logits": self.logits, "input": self.input,
               "head_max_chunks": HEAD_MAX_CHUNKS, "w.head": self.w["head"]}
        for l in range(LAYERS):
            out[f"qkv.{l}"] = self.qkv[l]
            out[f"kv.{l}"] = self.kv[l]
            for name in ("qkv", "out", "gate_up", "down"):
                out[f"w.{l}.{name}"] = self.w[(l, name)]
        return out

    # ---- transfers

    def head_records(self, offset, records):
        """`records` records of the same bytes for both of a stream's cores."""
        return t(self.s_bytes, offset, [1, records, REC // 512, 512], [0, REC, 512, 1])

    def weights(self, base, s, records):
        n = 2 * records * REC
        return t(self.w_bytes, base + s * n, [1, 2 * records, REC // 512, 512], [0, REC, 512, 1])

    def hidden(self, offset, c):
        return t(self.s_bytes, offset + c * 1024, [1, 1, 32, 32], [0, 0, 32, 1])

    @staticmethod
    def kv_tasks(d):
        """(first record, records) of the KV stream's tasks."""
        out, at = [], 0
        for i in range(KV_TASKS):
            out.append((at, d["recs"][i]))
            at = at + d["recs"][i]
        return out

    def phases(self, d):
        """Phases of one token; d holds the dispatch-time scalars."""
        # Embed: the input row (written by the host) becomes block 0's h.
        yield Phase(heads={s: [("S", t(self.s_bytes, self.input, [1, 1, REC // 512, 512], [0, 0, 512, 1]))]
                           for s in range(16)},
                    drains=lambda c: [self.hidden(self.block(0), c)])
        for l in range(LAYERS):
            op_in, ffn_in, nxt = self.block(2 * l), self.block(2 * l + 1), self.block(2 * l + 2)
            q, kv = self.qkv[l], self.kv[l]
            # Attn1: h += o; q/k/v rows of rmsnorm(h) * w.
            yield Phase(
                heads={s: [("S", self.head_records(op_in, 3))] for s in range(16)},
                stream={s: [("W", self.weights(self.w[(l, "qkv")], s, QKV_RECS))] for s in range(16)},
                drains=lambda c, ffn_in=ffn_in, q=q: [
                    self.hidden(ffn_in, c),
                    t(self.s_bytes, q + c * Q_COL, [1, 1, Q_COL // 64, 64], [0, 0, 64, 1])])
            # Attn2: cores (c, j < 2) attend with heads 2c + j over KV head c.
            yield Phase(
                heads={s: [("S", self.head_records(q, 2)),
                           ("S", t(self.s_bytes, self.rope + d["pos"] * 512, [1, 1, REC // 512, 512], [0, 0, 512, 1]))]
                       for s in range(0, 16, 2)},
                stream={s: [("S", t(self.s_bytes, kv + (s // 2) * ROWS * self.ctx * 512 + at * 16 * 512,
                                    [1, n, REC // 512, 512], [0, 16 * 512, 512, 1]))
                            for at, n in self.kv_tasks(d)]
                        for s in range(0, 16, 2)},
                drains=lambda c, kv=kv: [
                    t(self.s_bytes, self.junk + c * 128, [1, 1, 2, 64], [0, 0, 64, 1]),
                    t(self.s_bytes, self.att + c * 1024, [1, 8, 4, 32], [0, 32, 256, 1]),
                    t(self.s_bytes, kv + c * ROWS * self.ctx * 512 + d["pos"] * 512,
                      [1, 16, 4, 32], [0, 32, self.ctx * 512, 1])])
            # Attn3: the output projection into the FFN block's o.
            yield Phase(
                heads={s: [("S", t(self.s_bytes, self.att, [1, 1, 17, 512], [0, 0, 1024, 1]))] for s in range(16)},
                stream={s: [("W", self.weights(self.w[(l, "out")], s, OUT_RECS))] for s in range(16)},
                drains=lambda c, ffn_in=ffn_in: [self.hidden(ffn_in + 4 * H, c)])
            # Ffn1: h += o; SiLU(gate) * up of rmsnorm(h) * w.
            yield Phase(
                heads={s: [("S", self.head_records(ffn_in, 3))] for s in range(16)},
                stream={s: [("W", self.weights(self.w[(l, "gate_up")], s, GU_RECS))] for s in range(16)},
                drains=lambda c, nxt=nxt: [self.hidden(nxt, c),
                                           t(self.s_bytes, self.act + c * ACT_COL, [1, 1, ACT_COL // 64, 64], [0, 0, 64, 1])])
            # Ffn2: the down projection into the next block's o.
            yield Phase(
                heads={s: [("S", self.head_records(self.act, 2))] for s in range(16)},
                stream={s: [("W", self.weights(self.w[(l, "down")], s, DOWN_RECS))] for s in range(16)},
                drains=lambda c, nxt=nxt: [self.hidden(nxt + 4 * H, c)])
        # Head: h += o; rows of rmsnorm(h) * w (the final norm) into the logits.
        chunks = d["head_chunks"]
        yield Phase(
            heads={s: [("S", self.head_records(self.block(2 * LAYERS), 3))] for s in range(16)},
            stream={s: [("W", t(self.w_bytes, self.w["head"] + s * (chunks * 4 * REC),
                                [1, chunks * 4, REC // 512, 512], [0, REC, 512, 1]))] for s in range(16)},
            drains=lambda c: [t(self.s_bytes, self.logits + c * (chunks * 128), [1, 1, chunks, 128], [0, 0, 128, 1])])


class Phase:
    def __init__(self, heads=None, stream=None, drains=None):
        self._heads, self._stream, self._drains = heads or {}, stream or {}, drains or (lambda c: [])

    def heads(self, s):
        return self._heads.get(s, [])

    def stream(self, s):
        return self._stream.get(s, [])

    def drains(self, c):
        return self._drains(c)
