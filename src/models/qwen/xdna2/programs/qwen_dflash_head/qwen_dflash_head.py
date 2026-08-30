# Copyright (C) 2026 Gufo Engine contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Build the Qwen3.8 DFlash2 split vocabulary-head program for XDNA2."""

import argparse
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D
from aie.iron import In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile, from_name
from aie.iron.kernel import ExternalFunction, Kernel
from aie.utils import config

INPUT_ELEMENTS = 5120
SHARD_OUTPUT_ELEMENTS = 8192
SHARD_COUNT = 1
OUTPUT_ELEMENTS = SHARD_COUNT * SHARD_OUTPUT_ELEMENTS
BATCH_ROWS = 8
TILE_INPUT_ELEMENTS = 256
TILE_OUTPUT_ELEMENTS = 8
MMUL_OUTPUT_ELEMENTS = 8
WEIGHT_ROW_CHUNK_BYTES = 288
INPUT_RECORD_ROWS = 8
ARRAY_COLUMNS = 8
ARRAY_ROWS = 4
CORE_COUNT = ARRAY_COLUMNS * ARRAY_ROWS
INPUT_TILES = INPUT_ELEMENTS // TILE_INPUT_ELEMENTS
OUTPUT_TILES = OUTPUT_ELEMENTS // TILE_OUTPUT_ELEMENTS
SHARD_OUTPUT_TILES = SHARD_OUTPUT_ELEMENTS // TILE_OUTPUT_ELEMENTS
OUTPUT_TILES_PER_CORE = SHARD_OUTPUT_TILES // CORE_COUNT
OUTPUT_TILES_PER_COLUMN = SHARD_OUTPUT_TILES // ARRAY_COLUMNS
WEIGHT_TILE_ROW_BYTES = INPUT_TILES * WEIGHT_ROW_CHUNK_BYTES
OUTPUT_TILE_ELEMENTS = BATCH_ROWS * TILE_OUTPUT_ELEMENTS


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def qwen_dflash_head(weights: In, input_rows: In, output_rows: Out):
    byte_dtype = np.dtype[np.int8]
    output_dtype = np.dtype[np.float32]
    weights_type = np.ndarray[
        (OUTPUT_ELEMENTS, WEIGHT_TILE_ROW_BYTES), byte_dtype
    ]
    input_type = np.ndarray[
        (INPUT_TILES * INPUT_RECORD_ROWS, WEIGHT_ROW_CHUNK_BYTES),
        byte_dtype,
    ]
    output_type = np.ndarray[
        (OUTPUT_TILES, OUTPUT_TILE_ELEMENTS), output_dtype
    ]
    weight_mem_type = np.ndarray[
        (ARRAY_ROWS * TILE_OUTPUT_ELEMENTS, WEIGHT_ROW_CHUNK_BYTES),
        byte_dtype,
    ]
    weight_core_type = np.ndarray[
        (TILE_OUTPUT_ELEMENTS, WEIGHT_ROW_CHUNK_BYTES), byte_dtype
    ]
    input_tile_type = np.ndarray[
        (INPUT_RECORD_ROWS, WEIGHT_ROW_CHUNK_BYTES), byte_dtype
    ]
    output_mem_type = np.ndarray[
        (ARRAY_ROWS * OUTPUT_TILE_ELEMENTS,), output_dtype
    ]
    output_core_type = np.ndarray[(OUTPUT_TILE_ELEMENTS,), output_dtype]

    gemm = ExternalFunction(
        "qwen_dflash_head_gemm_q8_0_w8a8_f32",
        source_file=str(Path(__file__).with_name("gemm.cc")),
        arg_types=[weight_core_type, input_tile_type, output_core_type],
        include_dirs=[config.cxx_header_path()],
        compile_flags=[
            f"-DDIM_M={MMUL_OUTPUT_ELEMENTS}",
            f"-DDIM_K={TILE_INPUT_ELEMENTS}",
            f"-DBATCH_ROWS={BATCH_ROWS}",
        ],
    )
    zero = Kernel(
        "qwen_dflash_head_zero_f32",
        gemm.object_file_name,
        [output_core_type],
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
                    row * TILE_OUTPUT_ELEMENTS * WEIGHT_ROW_CHUNK_BYTES
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
            input_tile_type, name=f"input_l3l2_{column}", depth=2
        )
        input_producer_fifos.append(input_l3l2)
        input_consumer_fifos.append(
            input_l3l2.cons().forward(
                obj_type=input_tile_type,
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
                row * OUTPUT_TILE_ELEMENTS for row in range(ARRAY_ROWS)
            ],
            obj_types=[output_core_type] * ARRAY_ROWS,
            names=[
                f"output_l1l2_{row}_{column}"
                for row in range(ARRAY_ROWS)
            ],
            depths=[2] * ARRAY_ROWS,
            tile=Tile(column, 1),
        )
        for row in range(ARRAY_ROWS):
            output_producer_fifos[row][column] = joined[row]

    def core_body(weight_fifo, input_fifo, output_fifo, zero_fn, gemm_fn):
        for _ in range_(SHARD_COUNT * OUTPUT_TILES_PER_CORE):
            output = output_fifo.acquire(1)
            zero_fn(output)
            for _ in range_(INPUT_TILES):
                weight_tile = weight_fifo.acquire(1)
                input_tile = input_fifo.acquire(1)
                gemm_fn(weight_tile, input_tile, output)
                weight_fifo.release(1)
                input_fifo.release(1)
            output_fifo.release(1)

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

    shard_weight_taps = TensorTiler2D.group_tiler(
        (SHARD_OUTPUT_ELEMENTS, WEIGHT_TILE_ROW_BYTES),
        (
            ARRAY_ROWS * TILE_OUTPUT_ELEMENTS,
            WEIGHT_ROW_CHUNK_BYTES,
        ),
        (OUTPUT_TILES_PER_CORE, INPUT_TILES),
        prune_step=False,
    )
    weight_taps = [
        [
            TensorAccessPattern(
                (OUTPUT_ELEMENTS, WEIGHT_TILE_ROW_BYTES),
                (shard * SHARD_OUTPUT_ELEMENTS * WEIGHT_TILE_ROW_BYTES)
                + tap.offset,
                tap.sizes,
                tap.strides,
            )
            for tap in shard_weight_taps
        ]
        for shard in range(SHARD_COUNT)
    ]
    input_tap = TensorTiler2D.group_tiler(
        (
            INPUT_TILES * INPUT_RECORD_ROWS,
            WEIGHT_ROW_CHUNK_BYTES,
        ),
        (INPUT_RECORD_ROWS, WEIGHT_ROW_CHUNK_BYTES),
        (INPUT_TILES, 1),
        pattern_repeat=OUTPUT_TILES_PER_CORE,
        prune_step=False,
    )[0]
    shard_output_taps = TensorTiler2D.simple_tiler(
        (SHARD_OUTPUT_TILES, OUTPUT_TILE_ELEMENTS),
        (OUTPUT_TILES_PER_COLUMN, OUTPUT_TILE_ELEMENTS),
        prune_step=False,
    )
    output_taps = [
        [
            TensorAccessPattern(
                (OUTPUT_TILES, OUTPUT_TILE_ELEMENTS),
                (shard * SHARD_OUTPUT_TILES * OUTPUT_TILE_ELEMENTS)
                + tap.offset,
                tap.sizes,
                tap.strides,
            )
            for tap in shard_output_taps
        ]
        for shard in range(SHARD_COUNT)
    ]

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
        for shard in range(SHARD_COUNT):
            tasks = TaskGroup()
            for column in range(ARRAY_COLUMNS):
                output_runtime_handles[column].drain(
                    output_data,
                    output_taps[shard][column],
                    wait=True,
                    group=tasks,
                )
                weight_runtime_handles[column].fill(
                    weight_data, weight_taps[shard][column], group=tasks
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    iron.set_current_device(from_name("npu2", n_cols=None))
    program = qwen_dflash_head.specialize()
    program.compile(
        xclbin_path=args.output_dir / "qwen_dflash_head.xclbin",
        inst_path=args.output_dir / "qwen_dflash_head_insts.bin",
        elf_path=args.output_dir / "qwen_dflash_head.insts.elf",
        pdi_path=args.output_dir / "qwen_dflash_head.pdi",
    )


if __name__ == "__main__":
    main()
