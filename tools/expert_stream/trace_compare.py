#!/usr/bin/env python3
"""Check that two expert-store routing traces record the same decisions.

  trace_compare.py A.bin B.bin

Compares the seeded snapshot, every prepare's layer, routes, hit flags and
demand bytes, and every router prediction, ignoring what legitimately varies
between runs: timestamps, read wait and the free memory at the snapshot.
Prints the first difference and exits 1, or a summary and exits 0.

Traces of different format versions compare on the fields both record.

Both runs must replay the same requests with a fixed cache limit, so the
cache size does not depend on the memory free at startup. Callers that
predict should not do so while tracing, so adopted prefetch bytes are zero.
Stdlib only.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from expert_trace import records as raw_records  # noqa: E402


def version(path):
    return next(raw_records(path))[1]


def records(path, common):
    """The fields compared, per record. With `common`, only those both
    format versions record (the header and the snapshot's version 2 fields
    are skipped), so a version 1 trace compares with a version 2 one."""
    for kind, body in raw_records(path):
        if kind == "H":
            if not common:
                yield ("H", (body,))
        elif kind == "S" and common:
            yield ("S", tuple(body[k] for k in ("n_layer", "n_expert", "used", "tick", "cache_limit", "cache_bytes"))
                   + (tuple(body["class_bytes"]), tuple(body["layer_class"]), tuple(body["expert_bytes"]),
                      body["slots"], body["entries"]))
        elif kind == "S":
            yield ("S", tuple(body[k] for k in (
                "n_layer", "n_expert", "used", "n_pieces", "read_margin", "tick", "cache_limit", "cache_bytes",
                "chunk_bytes")) + (tuple(body["policy"].items()), tuple(body["class_bytes"]),
                                   tuple(body["layer_class"]), tuple(body["clock"]),
                                   tuple(map(tuple, body["piece_bytes"])), body["slots"], body["entries"]))
        elif kind == "P":
            yield ("P", tuple(body[k] for k in ("layer", "n", "demand", "adopted", "ids", "hits")))
        else:
            yield ("Q", tuple(body[k] for k in ("layer", "lead", "n", "k", "ids")))


FIELDS = {"H": "version",
          "S": "layers, experts, top-k, pieces, read margin, tick, limit, cache bytes, chunk bytes, policy, "
               "classes, layer classes, clocks, piece bytes, slots, entries",
          "P": "layer, tokens, demand bytes, adopted bytes, routes, hits",
          "Q": "layer, lead, tokens, k, ids"}


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    common = version(sys.argv[1]) != version(sys.argv[2])
    if common:
        print("versions differ: comparing the fields both record")
    a, b = (records(p, common) for p in sys.argv[1:])
    fields = dict(FIELDS)
    if common:
        fields["S"] = "layers, experts, top-k, tick, limit, cache bytes, classes, layer classes, expert bytes, " \
                      "slots, entries"
    counts = {"H": 0, "S": 0, "P": 0, "Q": 0}
    tokens = hits = routes = 0
    index = 0
    while True:
        ra, rb = next(a, None), next(b, None)
        if ra is None and rb is None:
            break
        if ra is None or rb is None:
            print(f"record {index}: {'A' if ra is None else 'B'} ends first")
            return 1
        if ra != rb:
            print(f"record {index}: {ra[0]} differs from {rb[0]}")
            if ra[0] == rb[0]:
                names = fields[ra[0]].split(", ")
                for name, x, y in zip(names, ra[1], rb[1]):
                    if x != y:
                        shown = (lambda v: v if not isinstance(v, bytes) else f"{len(v)} bytes")
                        print(f"  {name}: {shown(x)} != {shown(y)}")
            return 1
        kind, body = ra
        counts[kind] += 1
        if kind == "P":
            layer, n, _, _, _, hit_flags = body
            tokens += n if layer == 0 else 0
            routes += len(hit_flags)
            hits += sum(hit_flags)
        index += 1
    rate = 100.0 * hits / routes if routes else 0.0
    print(f"identical: {counts['S']} snapshot, {counts['P']} prepares, {counts['Q']} predictions, "
          f"{tokens} tokens, {rate:.2f}% route hits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
