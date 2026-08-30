# Copyright (C) 2026 Gufo Engine contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Build the DS4 Q2_K four-row down-projection experiment for XDNA2."""

import argparse
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorTiler2D
from aie.iron import In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.device import Tile, from_name
from aie.iron.kernel import ExternalFunction, Kernel
from aie.utils import config

INPUT_ELEMENTS = 2048
OUTPUT_ELEMENTS = 4096
ACTIVATION_ROWS = 4
TILE_INPUT_ELEMENTS = 256
TILE_OUTPUT_ELEMENTS = 8
WEIGHT_RECORD_BYTES = 3072
INPUT_RECORD_BYTES = 1536
ARRAY_COLUMNS = 8
ARRAY_ROWS = 4
CORE_COUNT = ARRAY_COLUMNS * ARRAY_ROWS
OUTPUT_TILES = OUTPUT_ELEMENTS // TILE_OUTPUT_ELEMENTS
OUTPUT_TILES_PER_CORE = OUTPUT_TILES // CORE_COUNT
INPUT_TILES = INPUT_ELEMENTS // TILE_INPUT_ELEMENTS
DMA_ROW_BYTES = 256
WEIGHT_ROW_CHUNK_BYTES = WEIGHT_RECORD_BYTES // TILE_OUTPUT_ELEMENTS
WEIGHT_TILE_ROW_BYTES = INPUT_TILES * WEIGHT_ROW_CHUNK_BYTES
INPUT_RECORD_ROWS = INPUT_RECORD_BYTES // DMA_ROW_BYTES
PACKED_OUTPUT_ELEMENTS = ACTIVATION_ROWS * OUTPUT_ELEMENTS
OUTPUT_ELEMENTS_PER_COLUMN = PACKED_OUTPUT_ELEMENTS // ARRAY_COLUMNS


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def ds4_q2k_down(weights: In, input_rows: In, output_rows: Out):
    weight_dtype = np.dtype[np.uint8]
    input_dtype = np.dtype[np.int8]
    output_dtype = np.dtype[np.float32]
    weights_type = np.ndarray[
        (OUTPUT_ELEMENTS, WEIGHT_TILE_ROW_BYTES), weight_dtype
    ]
    input_type = np.ndarray[
        (INPUT_TILES * INPUT_RECORD_ROWS, DMA_ROW_BYTES), input_dtype
    ]
    output_type = np.ndarray[(PACKED_OUTPUT_ELEMENTS,), output_dtype]
    weight_mem_type = np.ndarray[
        (ARRAY_ROWS * TILE_OUTPUT_ELEMENTS, WEIGHT_ROW_CHUNK_BYTES),
        weight_dtype,
    ]
    weight_core_type = np.ndarray[
        (TILE_OUTPUT_ELEMENTS, WEIGHT_ROW_CHUNK_BYTES), weight_dtype
    ]
    input_tile_type = np.ndarray[
        (INPUT_RECORD_ROWS, DMA_ROW_BYTES), input_dtype
    ]
    output_mem_type = np.ndarray[
        (ARRAY_ROWS * ACTIVATION_ROWS * TILE_OUTPUT_ELEMENTS,), output_dtype
    ]
    output_core_type = np.ndarray[
        (ACTIVATION_ROWS, TILE_OUTPUT_ELEMENTS), output_dtype
    ]

    gemm = ExternalFunction(
        "ds4_q2k_down_gemm_a8q2_f32",
        source_file=str(Path(__file__).with_name("gemv.cc")),
        arg_types=[weight_core_type, input_tile_type, output_core_type],
        include_dirs=[config.cxx_header_path()],
        compile_flags=[
            f"-DDIM_M={TILE_OUTPUT_ELEMENTS}",
            f"-DDIM_K={TILE_INPUT_ELEMENTS}",
        ],
    )
    zero = Kernel(
        "ds4_q2k_down_zero_f32", gemm.object_file_name, [output_core_type]
    )

    weight_producers = []
    weight_consumers = []
    input_producers = []
    input_consumers = []
    output_producers = [
        [None for _ in range(ARRAY_COLUMNS)] for _ in range(ARRAY_ROWS)
    ]
    output_consumers = []

    for column in range(ARRAY_COLUMNS):
        weight_l3l2 = ObjectFifo(
            weight_mem_type, name=f"weight_l3l2_{column}", depth=2
        )
        weight_producers.append(weight_l3l2)
        weight_consumers.append(
            weight_l3l2.cons().split(
                [
                    row * WEIGHT_RECORD_BYTES
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
        input_producers.append(input_l3l2)
        input_consumers.append(
            input_l3l2.cons().forward(
                obj_type=input_tile_type,
                name=f"input_l2l1_{column}",
                tile=Tile(column, 1),
            )
        )

        output_l2l3 = ObjectFifo(
            output_mem_type, name=f"output_l2l3_{column}", depth=2
        )
        output_consumers.append(output_l2l3)
        joined = output_l2l3.prod().join(
            [
                row * ACTIVATION_ROWS * TILE_OUTPUT_ELEMENTS
                for row in range(ARRAY_ROWS)
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
            output_producers[row][column] = joined[row]

    def core_body(weight_fifo, input_fifo, output_fifo, zero_fn, gemm_fn):
        output = output_fifo.acquire(1)
        zero_fn(output)
        for _ in range(INPUT_TILES):
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
                weight_consumers[column][row].cons(),
                input_consumers[column].cons(),
                output_producers[row][column].prod(),
                zero,
                gemm,
            ],
            stack_size=2048,
            tile=Tile(column, 2 + row),
        ),
    )
    flat_workers = [worker for row in workers for worker in row]

    weight_taps = TensorTiler2D.group_tiler(
        (OUTPUT_ELEMENTS, WEIGHT_TILE_ROW_BYTES),
        (
            ARRAY_ROWS * TILE_OUTPUT_ELEMENTS,
            WEIGHT_ROW_CHUNK_BYTES,
        ),
        (OUTPUT_TILES_PER_CORE, INPUT_TILES),
        prune_step=False,
    )
    input_tap = TensorTiler2D.group_tiler(
        (INPUT_TILES * INPUT_RECORD_ROWS, DMA_ROW_BYTES),
        (INPUT_RECORD_ROWS, DMA_ROW_BYTES),
        (INPUT_TILES, 1),
        pattern_repeat=OUTPUT_TILES_PER_CORE,
        prune_step=False,
    )[0]
    output_taps = TensorTiler2D.simple_tiler(
        (1, PACKED_OUTPUT_ELEMENTS),
        (1, OUTPUT_ELEMENTS_PER_COLUMN),
        prune_step=False,
    )

    weight_handles = [
        fifo.prod(tile=Tile(column, 0))
        for column, fifo in enumerate(weight_producers)
    ]
    input_handles = [
        fifo.prod(tile=Tile(column, 0))
        for column, fifo in enumerate(input_producers)
    ]
    output_handles = [
        fifo.cons(tile=Tile(column, 0))
        for column, fifo in enumerate(output_consumers)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    iron.set_current_device(from_name("npu2", n_cols=None))
    program = ds4_q2k_down.specialize()
    program.compile(
        xclbin_path=args.output_dir / "ds4_q2k_down.xclbin",
        inst_path=args.output_dir / "ds4_q2k_down_insts.bin",
        elf_path=args.output_dir / "ds4_q2k_down.insts.elf",
        pdi_path=args.output_dir / "ds4_q2k_down.pdi",
    )


if __name__ == "__main__":
    main()
