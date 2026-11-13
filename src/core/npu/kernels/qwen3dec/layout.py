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

Q, KV, D = 2048, 1024, 128
COLS, ROWS = 8, 4
REC = 4 * 2048 + 4 * (2048 // 32) * 2  # 8704: 4 rows of 2048 or 8 rows of 1024 codes
QKV_ROWS = Q + 2 * KV             # 4096: 128 per core
QKV_BYTES = 2 * REC               # rows f32, then q-norm and k-norm bf16
Q_COL = 16 * ROWS * 32            # q/k/v rows per column (f32)
KV_TASKS = 4                      # a DMA task repeats at most 64 records

# The stacks one program serves: name -> (hidden, FFN, layers, head max chunks,
# heads, KV positions). The talker/ASR head is up to 152064 rows; the code
# predictor keeps 15 heads of 2048 rows (one per step, chosen at dispatch).
MODELS = {
    "qwen3dec": (2048, 6144, 28, 594, 1, 4096),
    "qwen3dec_p": (1024, 3072, 5, 8, 15, 16),
}


def t(total, offset, sizes, strides):
    from aie.helpers.taplib import TensorAccessPattern  # the host does not need mlir-aie
    return TensorAccessPattern((total,), offset, sizes, strides)


class Layout:
    def __init__(self, model="qwen3dec"):
        H, I, LAYERS, HEAD_MAX_CHUNKS, HEADS, ctx = MODELS[model]
        self.H, self.I, self.LAYERS, self.HEAD_MAX_CHUNKS, self.HEADS = H, I, LAYERS, HEAD_MAX_CHUNKS, HEADS
        self.PHASES = 1 + 5 * LAYERS + 1
        self.BLOCK = 12 * H                   # [h f32 | o f32 | unused | pre-norm w bf16]
        self.IN_RECS = (self.BLOCK + REC - 1) // REC
        rows_per_rec = 4 * (2048 // H)
        self.QKV_RECS = 128 // rows_per_rec                    # per core
        self.OUT_RECS = (H // 32) // 4                         # rows of the 2048 attention outputs
        self.GU_RECS = 2 * (I // 32) // rows_per_rec
        self.DOWN_RECS = (H // 32) * I // 8192
        self.HEAD_RECS_PER_CHUNK = 8 // rows_per_rec          # 8 rows per chunk and core
        self.ACT_COL = (I // 512) * ROWS * 32                  # gate/up output per column (bf16)
        self.ACT_RECS = (2 * I + REC - 1) // REC
        self.HID_COL = H // 2                                  # hidden rows per column (f32 bytes)
        self.ctx = ctx
        BLOCK = self.BLOCK
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
            for name, recs in (("qkv", self.QKV_RECS), ("out", self.OUT_RECS), ("gate_up", self.GU_RECS),
                               ("down", self.DOWN_RECS)):
                self.w[(l, name)] = take(2 * COLS * 2 * recs * REC)
        self.head_bytes = 2 * COLS * 2 * HEAD_MAX_CHUNKS * self.HEAD_RECS_PER_CHUNK * REC
        self.w["head"] = take(HEADS * self.head_bytes)
        self.w_bytes = o
        assert self.s_bytes < 2 ** 31 and self.w_bytes < 2 ** 31

    def block(self, b):
        return self.blocks + b * self.BLOCK

    def items(self):
        """Named offsets for the host (qwen3dec_layout.txt)."""
        out = {"s_bytes": self.s_bytes, "w_bytes": self.w_bytes, "ctx": self.ctx, "rec": REC,
               "block_bytes": self.BLOCK, "blocks": self.blocks, "act": self.act, "att": self.att,
               "rope": self.rope, "logits": self.logits, "input": self.input,
               "head_max_chunks": self.HEAD_MAX_CHUNKS, "w.head": self.w["head"], "head_bytes": self.head_bytes,
               "heads": self.HEADS, "hidden": self.H, "ffn": self.I, "layers": self.LAYERS,
               "phases": self.PHASES}
        for l in range(self.LAYERS):
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
        return t(self.s_bytes, offset + c * self.HID_COL, [1, 1, self.HID_COL // 32, 32], [0, 0, 32, 1])

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
        H, LAYERS = self.H, self.LAYERS
        for l in range(LAYERS):
            op_in, ffn_in, nxt = self.block(2 * l), self.block(2 * l + 1), self.block(2 * l + 2)
            q, kv = self.qkv[l], self.kv[l]
            # Attn1: h += o; q/k/v rows of rmsnorm(h) * w.
            yield Phase(
                heads={s: [("S", self.head_records(op_in, self.IN_RECS))] for s in range(16)},
                stream={s: [("W", self.weights(self.w[(l, "qkv")], s, self.QKV_RECS))] for s in range(16)},
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
                stream={s: [("W", self.weights(self.w[(l, "out")], s, self.OUT_RECS))] for s in range(16)},
                drains=lambda c, ffn_in=ffn_in: [self.hidden(ffn_in + 4 * H, c)])
            # Ffn1: h += o; SiLU(gate) * up of rmsnorm(h) * w.
            yield Phase(
                heads={s: [("S", self.head_records(ffn_in, self.IN_RECS))] for s in range(16)},
                stream={s: [("W", self.weights(self.w[(l, "gate_up")], s, self.GU_RECS))] for s in range(16)},
                drains=lambda c, nxt=nxt: [self.hidden(nxt, c),
                                           t(self.s_bytes, self.act + c * self.ACT_COL,
                                             [1, 1, self.ACT_COL // 64, 64], [0, 0, 64, 1])])
            # Ffn2: the down projection into the next block's o.
            yield Phase(
                heads={s: [("S", self.head_records(self.act, self.ACT_RECS))] for s in range(16)},
                stream={s: [("W", self.weights(self.w[(l, "down")], s, self.DOWN_RECS))] for s in range(16)},
                drains=lambda c, nxt=nxt: [self.hidden(nxt + 4 * H, c)])
        # Head: h += o; rows of rmsnorm(h) * w (the final norm) into the logits.
        chunks = d["head_chunks"]
        per = 2 * self.HEAD_RECS_PER_CHUNK  # records per chunk and stream (both cores)
        head = self.w["head"] + d.get("head_base", 0)
        yield Phase(
            heads={s: [("S", self.head_records(self.block(2 * LAYERS), self.IN_RECS))] for s in range(16)},
            stream={s: [("W", t(self.w_bytes, head + s * (chunks * per * REC),
                                [1, chunks * per, REC // 512, 512], [0, REC, 512, 1]))] for s in range(16)},
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
