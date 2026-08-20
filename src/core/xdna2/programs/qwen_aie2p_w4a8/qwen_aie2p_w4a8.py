# Copyright (C) 2026 Strix Engine contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Build the Qwen3.8-27B MTP W4A8 SHQ4-T16 GEMM/GEMV program for XDNA2
# (AIE2P). One kernel ELF (gemm.cc) serves every configuration; the
# (K-blocks, N-tiles-per-column, M-rounds) triple only changes the data
# movement graph baked into the xclbin:
#
#   weight tensor   (8*TPC*BLOCKS*64, 3072) u8, tile (64, 3072),
#                   group (TPC*BLOCKS, 1)  -> 8 column patterns holding the
#                   per-column stream ordered (tile, K block)
#   input tensor    (BLOCKS*ROUNDS*5, 256) i8, tile (5, 256),
#                   group (BLOCKS*ROUNDS, 1), pattern_repeat=TPC
#                   -> one shared pattern; each record is (4 code rows +
#                   1 parameter row of scales/int32 sums), stream ordered
#                   (K block, M chunk) per output-tile repetition
#   output tensor   (1, ROUNDS*8*TPC*256) f32 drained per column/tile/round,
#                   joined from 4 cores each producing 64 floats (4 M rows x 16
#                   N lanes); each round's 4-row M chunk is a separate 64-float
#                   accumulator, column p region offset
#                   p*ROUNDS*TPC*256 + t*ROUNDS*256 + r*256
#
# A core (row r of column c) holds one 16-lane weight slice across its
# chunk loop and accumulates the full activation sequence into its 64-float
# output before releasing it: GEMM output tile = sum over K blocks and M
# chunks of per-group (activation scale * (weight scale * uint4 dot -
# zero correction * int32 sum)) with INT32 accumulation into the mmul.
import argparse
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorTiler2D
from aie.iron.controlflow import range_
from aie.iron import (
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    TaskGroup,
    Worker,
)
from aie.iron.device import Tile, from_name
from aie.iron.kernel import ExternalFunction, Kernel
from aie.utils import config

K_BLOCK = 256
ARRAY_COLUMNS = 8
ARRAY_ROWS = 4
CORE_COUNT = ARRAY_COLUMNS * ARRAY_ROWS

# Per-invocation configuration (set in main()).
BLOCKS = 1          # ceil(K / 256)
TPC = 1             # ceil(N / (ARRAY_COLUMNS * ARRAY_ROWS * 16))
ROUNDS = 1          # ceil(M / ARRAY_ROWS)

WEIGHT_RECORD_BYTES = 3072          # codes 2048 + scales 512 + zcorr 512
WEIGHT_RECORD_LANE_BYTES = WEIGHT_RECORD_BYTES // 16   # 192 B per lane
INPUT_RECORD_ROWS = 5               # 4 code rows + 1 parameter row
INPUT_RECORD_BYTES = INPUT_RECORD_ROWS * K_BLOCK
OUTPUT_TILE_BYTES = ARRAY_ROWS * 64 * 4      # 4 rows x 16 lanes f32
WEIGHT_LANE_GROUPS = ARRAY_ROWS * 16          # 64 lanes per column tile
WEIGHT_GROUPS = TPC * BLOCKS


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def qwen_aie2p_w4a8(
    weights: In,
    input_tensor: In,
    output_tensor: Out,
):
    weight_dtype = np.dtype[np.uint8]
    input_dtype = np.dtype[np.int8]
    output_dtype = np.dtype[np.float32]
    weights_type = np.ndarray[
        (ARRAY_COLUMNS * WEIGHT_GROUPS * WEIGHT_LANE_GROUPS, WEIGHT_RECORD_LANE_BYTES),
        weight_dtype,
    ]
    input_type = np.ndarray[
        (BLOCKS * ROUNDS * INPUT_RECORD_ROWS, K_BLOCK), input_dtype
    ]
    output_type = np.ndarray[(1, ROUNDS * ARRAY_COLUMNS * TPC * 256), output_dtype]
    weight_mem_type = np.ndarray[
        (WEIGHT_LANE_GROUPS, WEIGHT_RECORD_LANE_BYTES), weight_dtype
    ]
    weight_core_type = np.ndarray[
        (16, WEIGHT_RECORD_LANE_BYTES), weight_dtype
    ]
    input_record_type = np.ndarray[
        (INPUT_RECORD_ROWS, K_BLOCK), input_dtype
    ]
    output_mem_type = np.ndarray[(ARRAY_ROWS * 64,), output_dtype]
    output_core_type = np.ndarray[(64,), output_dtype]

    gemm = ExternalFunction(
        "qwen_aie2p_w4a8_gemm",
        source_file=str(Path(__file__).with_name("gemm.cc")),
        arg_types=[weight_core_type, input_record_type, output_core_type],
        include_dirs=[config.cxx_header_path()],
        compile_flags=[],
    )
    zero = Kernel(
        "qwen_aie2p_w4a8_zero_f32", gemm.object_file_name, [output_core_type]
    )

    weight_producer_fifos = []
    weight_consumer_fifos = []
    input_producer_fifos = []
    input_consumer_fifos = []
    output_producer_fifos = [
        [None for _ in range(ARRAY_COLUMNS)] for _ in range(ARRAY_ROWS)
    ]
    output_consumer_fifos = []

    for column in range(ARRAY_COLUMNS):
        weight_l3l2 = ObjectFifo(
            weight_mem_type, name=f"weight_l3l2_{column}", depth=2
        )
        weight_producer_fifos.append(weight_l3l2)
        weight_consumer_fifos.append(
            weight_l3l2.cons().split(
                [
                    row * WEIGHT_RECORD_LANE_BYTES
                    for row in range(ARRAY_ROWS)
                ],
                obj_types=[weight_core_type] * ARRAY_ROWS,
                names=[
                    f"weight_l2l1_{row}_{column}"
                    for row in range(ARRAY_ROWS)
                ],
                depths=[2] * ARRAY_ROWS,
                tile=Tile(column, 1),
            )
        )

        input_l3l2 = ObjectFifo(
            input_record_type, name=f"input_l3l2_{column}", depth=4
        )
        input_producer_fifos.append(input_l3l2)
        input_consumer_fifos.append(
            input_l3l2.cons().forward(
                obj_type=input_record_type,
                name=f"input_l2l1_{column}",
                tile=Tile(column, 1),
            )
        )

        output_l2l3 = ObjectFifo(
            output_mem_type, name=f"output_l2l3_{column}", depth=2
        )
        output_consumer_fifos.append(output_l2l3)
        joined = output_l2l3.prod().join(
            [
                row * 64
                for row in range(ARRAY_ROWS)
            ],
            obj_types=[output_core_type] * ARRAY_ROWS,
            depths=[max(2, ROUNDS)] * ARRAY_ROWS,
            tile=Tile(column, 1),
        )
        for row in range(ARRAY_ROWS):
            output_producer_fifos[row][column] = joined[row]

    def core_body(weight_fifo, input_fifo, output_fifo, zero_fn, gemv_fn):
        # TPC and BLOCKS become hardware loops (range_), so the traced
        # instruction stream is constant-size (1 zero + 1 gemv + loop
        # branches) regardless of config, instead of TPC*BLOCKS unrolled
        # kernel calls overflowing program memory. The per-round accumulator
        # subviews (acquire(ROUNDS) with static aie.objectfifo.subview.access
        # indices) must stay unrolled, so ROUNDS remains a Python loop —
        # subview access indices are attrs, not SSA values. ROUNDS is small
        # (1-2 today; up to 32 for the M-bucket-128 case, still <50 calls).
        for _ in range_(TPC):
            # acquire(1) returns a single accumulator object; acquire(N>1)
            # returns a list of N objects (aie.objectfifo.subview.access with
            # static indices, unrolled at trace time). Each round is a
            # different 4-row M chunk, so it needs its own 64-float
            # accumulator region.
            outputs = output_fifo.acquire(ROUNDS)
            if not isinstance(outputs, list):
                outputs = [outputs]
            for m_chunk in range(ROUNDS):
                zero_fn(outputs[m_chunk])
            for _ in range_(BLOCKS):
                weight_tile = weight_fifo.acquire(1)
                for m_chunk in range(ROUNDS):
                    input_tile = input_fifo.acquire(1)
                    gemv_fn(weight_tile, input_tile, outputs[m_chunk])
                    input_fifo.release(1)
                weight_fifo.release(1)
            output_fifo.release(ROUNDS)

    workers = Worker.grid(
        ARRAY_ROWS,
        ARRAY_COLUMNS,
        lambda row, column: Worker(
            core_body,
            [
                weight_consumer_fifos[column][row].cons(),
                input_consumer_fifos[column].cons(),
                output_producer_fifos[row][column].prod(),
                zero,
                gemm,
            ],
            stack_size=2048,
            tile=Tile(column, 2 + row),
        ),
    )
    flat_workers = [worker for row in workers for worker in row]

    weight_taps = TensorTiler2D.group_tiler(
        (
            ARRAY_COLUMNS * WEIGHT_GROUPS * WEIGHT_LANE_GROUPS,
            WEIGHT_RECORD_LANE_BYTES,
        ),
        (WEIGHT_LANE_GROUPS, WEIGHT_RECORD_LANE_BYTES),
        (WEIGHT_GROUPS, 1),
        prune_step=False,
    )
    output_taps = TensorTiler2D.simple_tiler(
        (1, ROUNDS * ARRAY_COLUMNS * TPC * 256),
        (1, ROUNDS * TPC * 256),
        prune_step=False,
    )
    input_tap = TensorTiler2D.group_tiler(
        (BLOCKS * ROUNDS * INPUT_RECORD_ROWS, K_BLOCK),
        (INPUT_RECORD_ROWS, K_BLOCK),
        (BLOCKS * ROUNDS, 1),
        pattern_repeat=TPC,
        prune_step=False,
    )[0]

    weight_handles = [
        fifo.prod(tile=Tile(column, 0))
        for column, fifo in enumerate(weight_producer_fifos)
    ]
    input_handles = [
        fifo.prod(tile=Tile(column, 0))
        for column, fifo in enumerate(input_producer_fifos)
    ]
    output_handles = [
        fifo.cons(tile=Tile(column, 0))
        for column, fifo in enumerate(output_consumer_fifos)
    ]

    def sequence(
        weight_data,
        input_data,
        output_data,
        weight_runtime_handles,
        input_runtime_handles,
        output_runtime_handles,
    ):
        tasks = TaskGroup()
        for column in range(ARRAY_COLUMNS):
            output_runtime_handles[column].drain(
                output_data, output_taps[column], wait=True, group=tasks
            )
            weight_runtime_handles[column].fill(
                weight_data, weight_taps[column], group=tasks
            )
            input_runtime_handles[column].fill(
                input_data, input_tap, group=tasks
            )
        tasks.finish()

    runtime = Runtime(
        sequence,
        [
            weights_type,
            input_type,
            output_type,
            weight_handles,
            input_handles,
            output_handles,
        ],
    )
    return Program(
        iron.get_current_device(), runtime, workers=flat_workers
    ).resolve_program()


def main() -> None:
    global BLOCKS, TPC, ROUNDS
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blocks", type=int, required=True)
    parser.add_argument("--tpc", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    args = parser.parse_args()
    BLOCKS = args.blocks
    TPC = args.tpc
    ROUNDS = args.rounds

    args.output_dir.mkdir(parents=True, exist_ok=True)
    iron.set_current_device(from_name("npu2", n_cols=None))
    program = qwen_aie2p_w4a8.specialize()
    program.compile(
        xclbin_path=args.output_dir / "qwen_aie2p_w4a8.xclbin",
        inst_path=args.output_dir / "qwen_aie2p_w4a8_insts.bin",
        elf_path=args.output_dir / "qwen_aie2p_w4a8.insts.elf",
        pdi_path=args.output_dir / "qwen_aie2p_w4a8.pdi",
    )
    print(
        f"built config blocks={BLOCKS} tpc={TPC} rounds={ROUNDS} into {args.output_dir}"
    )


if __name__ == "__main__":
    main()