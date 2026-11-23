"""Reader for the expert store's routing trace (expert_stream::Policy::trace_path).

`records(path)` yields the raw records with the stdlib only; `read_trace(path)`
returns the seeded snapshot and the prepares as numpy arrays. Version 2 files
start with the magic b"GUFOXTRC" and a uint32 version; version 1 files (the
deepseek41 extension's DS41_EXPERT_TRACE before the store moved into Gufo)
start with the snapshot record. The format is in README.md.
"""
from dataclasses import dataclass, field
from pathlib import Path
import struct
import sys

MAGIC = b"GUFOXTRC"
VERSION = 2
NO = 0xFFFFFFFF
# Version 1 traces record neither the chunk size nor the policy: these are
# the values the store used then.
V1_CHUNK_BYTES = 128 << 20
POLICY_FIELDS = ("use_half_life", "use_cap", "max_class", "clock_rows", "seed_uses", "recent_rows", "sweep_tokens")


@dataclass
class Prepare:
    t_start: int
    t_routes: int
    t_end: int
    layer: int
    n: int
    read_wait: int
    demand: int
    adopted: int
    ids: object   # [n, top_k] int32
    hits: object  # [n, top_k] bool
    pred: dict = field(default_factory=dict)  # lead -> [n, k] ids


@dataclass
class Snapshot:
    n_layer: int
    n_expert: int
    used: int  # top_k
    tick: int
    cache_limit: int
    cache_bytes: int
    available: int
    class_bytes: list
    layer_class: list
    expert_bytes: list  # per layer, all pieces
    slots: list         # (class, layer, expert)
    score: object       # [n_layer, n_expert] seeded uses
    last: object        # [n_layer, n_expert] tick of the last use
    version: int = 1
    n_pieces: int = 0   # 0: not recorded (version 1)
    read_margin: int = 0
    chunk_bytes: int = V1_CHUNK_BYTES
    policy: dict = field(default_factory=dict)  # POLICY_FIELDS; empty for version 1
    piece_bytes: list = field(default_factory=list)  # [layer][piece]
    clock: list = field(default_factory=list)        # per layer token clock


class _Reader:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def take(self, fmt):
        values = struct.unpack_from("<" + fmt, self.data, self.pos)
        self.pos += struct.calcsize("<" + fmt)
        return values

    def raw(self, n):
        chunk = self.data[self.pos:self.pos + n]
        if len(chunk) != n:
            raise SystemExit(f"truncated trace at byte {self.pos}")
        self.pos += n
        return chunk


def records(path):
    """Yields ("H", version), then ("S", dict), ("P", dict) and ("Q", dict)
    records; slots, entries, ids, hits and predictions stay raw bytes."""
    r = _Reader(Path(path).read_bytes())
    version = 1
    if r.data[:len(MAGIC)] == MAGIC:
        r.pos = len(MAGIC)
        (version,) = r.take("I")
        if version != VERSION:
            raise SystemExit(f"{path}: trace version {version}, this reader knows {VERSION}")
    yield ("H", version)
    used = None
    while r.pos < len(r.data):
        tag = r.raw(1)
        if tag == b"S":
            s = {"version": version}
            if version == 1:
                v1, s["n_layer"], s["n_expert"], s["used"] = r.take("IIII")
                if v1 != 1:
                    raise SystemExit(f"{path}: snapshot version {v1}")
                s["tick"], s["cache_limit"], s["cache_bytes"], s["available"] = r.take("QQQQ")
                s["n_pieces"], s["read_margin"], s["chunk_bytes"], s["policy"] = 0, 0, V1_CHUNK_BYTES, {}
            else:
                s["n_layer"], s["n_expert"], s["used"], s["n_pieces"] = r.take("IIII")
                (s["read_margin"], s["tick"], s["cache_limit"], s["cache_bytes"], s["available"],
                 s["chunk_bytes"]) = r.take("QQQQQQ")
                s["policy"] = dict(zip(POLICY_FIELDS, r.take("dIdIdII")))
            (n_class,) = r.take("I")
            s["class_bytes"] = list(r.take(f"{n_class}Q"))
            s["layer_class"], s["clock"], s["piece_bytes"], s["expert_bytes"] = [], [], [], []
            for _ in range(s["n_layer"]):
                if version == 1:
                    c, b = r.take("IQ")
                    pieces = []
                else:
                    c, clock = r.take("IQ")
                    pieces = list(r.take(f"{s['n_pieces']}Q"))
                    b = sum(pieces)
                    s["clock"].append(clock)
                s["layer_class"].append(c)
                s["piece_bytes"].append(pieces)
                s["expert_bytes"].append(b)
            (n_slots,) = r.take("I")
            s["slots"] = r.raw(n_slots * 12)
            s["entries"] = r.raw(s["n_layer"] * s["n_expert"] * 16)
            used = s["used"]
            yield ("S", s)
        elif tag == b"P":
            if used is None:
                raise SystemExit(f"{path}: prepare before the snapshot at byte {r.pos - 1}")
            p = dict(zip(("t_start", "t_routes", "t_end", "layer", "n", "read_wait", "demand", "adopted"),
                         r.take("QQQIIQQQ")))
            p["ids"] = r.raw(p["n"] * used * 4)
            p["hits"] = r.raw(p["n"] * used)
            yield ("P", p)
        elif tag == b"Q":
            q = dict(zip(("layer", "lead", "n", "k"), r.take("IIII")))
            q["ids"] = r.raw(q["n"] * q["k"] * 4)
            yield ("Q", q)
        else:
            raise SystemExit(f"{path}: bad record tag {tag!r} at byte {r.pos - 1}")


def read_trace(path):
    """(Snapshot, [Prepare]); predictions attach to their target's prepare."""
    import numpy as np

    snapshot, prepares = None, []
    for kind, body in records(path):
        if kind == "S":
            if snapshot is not None:
                # Fit, Release or Restore changed the cache outside a
                # prepare: replay from the cache as it left it.
                print(f"{path}: a later snapshot drops {len(prepares)} earlier prepares from "
                      "the replay", file=sys.stderr)
                prepares = []
            s = body
            n_layer, n_expert = s["n_layer"], s["n_expert"]
            flat = np.frombuffer(s["slots"], "<u4").reshape(-1, 3)
            entries = np.frombuffer(s["entries"], np.dtype([("s", "<f8"), ("l", "<u8")]))
            snapshot = Snapshot(
                n_layer, n_expert, s["used"], s["tick"], s["cache_limit"], s["cache_bytes"], s["available"],
                s["class_bytes"], s["layer_class"], s["expert_bytes"], [tuple(map(int, row)) for row in flat],
                entries["s"].reshape(n_layer, n_expert).copy(),
                entries["l"].reshape(n_layer, n_expert).astype(np.int64),
                s["version"], s["n_pieces"], s["read_margin"], s["chunk_bytes"], s["policy"],
                s["piece_bytes"], s["clock"])
        elif kind == "P":
            n, used = body["n"], snapshot.used
            ids = np.frombuffer(body["ids"], "<i4").reshape(n, used)
            hits = np.frombuffer(body["hits"], "u1").reshape(n, used).astype(bool)
            prepares.append(Prepare(body["t_start"], body["t_routes"], body["t_end"], body["layer"], n,
                                    body["read_wait"], body["demand"], body["adopted"], ids, hits))
        elif kind == "Q":
            ids = np.frombuffer(body["ids"], "<i4").reshape(body["n"], body["k"])
            # Predictions are written right after their target layer's prepare.
            target = prepares[-1]
            if target.layer != body["layer"] or target.n != body["n"]:
                raise SystemExit(f"{path}: prediction for layer {body['layer']} after layer {target.layer}")
            target.pred[body["lead"]] = ids
    if snapshot is None:
        raise SystemExit(f"{path}: no snapshot record")
    return snapshot, prepares
