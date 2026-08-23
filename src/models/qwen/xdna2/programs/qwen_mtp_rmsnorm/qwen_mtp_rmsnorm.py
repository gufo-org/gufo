# Copyright (C) 2026 Strix Engine contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Build the Qwen3.8 MTP hidden-width RMSNorm program for XDNA2."""

import argparse
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.device import from_name
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from ml_dtypes import bfloat16

ELEMENT_COUNT = 5120


@iron.jit
def qwen_mtp_rmsnorm(
    input_tensor: In,
    weight_tensor: In,
    output_tensor: Out,
):
    tensor_type = np.ndarray[(ELEMENT_COUNT,), np.dtype[bfloat16]]
    input_fifo = ObjectFifo(tensor_type, depth=1, name="input")
    weight_fifo = ObjectFifo(tensor_type, depth=1, name="weight")
    output_fifo = ObjectFifo(tensor_type, depth=1, name="output")

    kernel = ExternalFunction(
        "qwen_mtp_rmsnorm",
        source_file=str(Path(__file__).with_name("rmsnorm.cc")),
        arg_types=[tensor_type, tensor_type, tensor_type, np.int32],
        include_dirs=[config.cxx_header_path()],
    )

    def core_body(input_consumer, weight_consumer, output_producer, function):
        input_element = input_consumer.acquire(1)
        weight_element = weight_consumer.acquire(1)
        output_element = output_producer.acquire(1)
        function(input_element, weight_element, output_element, ELEMENT_COUNT)
        input_consumer.release(1)
        weight_consumer.release(1)
        output_producer.release(1)

    worker = Worker(
        core_body,
        fn_args=[
            input_fifo.cons(),
            weight_fifo.cons(),
            output_fifo.prod(),
            kernel,
        ],
        stack_size=2048,
    )

    def sequence(
        input_data,
        weight_data,
        output_data,
        input_producer,
        weight_producer,
        output_consumer,
    ):
        input_producer.fill(input_data)
        weight_producer.fill(weight_data)
        output_consumer.drain(output_data, wait=True)

    runtime = Runtime(
        sequence,
        [
            tensor_type,
            tensor_type,
            tensor_type,
            input_fifo.prod(),
            weight_fifo.prod(),
            output_fifo.cons(),
        ],
    )
    return Program(
        iron.get_current_device(), runtime, workers=[worker]
    ).resolve_program()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    iron.set_current_device(from_name("npu2"))

    program = qwen_mtp_rmsnorm.specialize()
    program.compile(
        xclbin_path=args.output_dir / "qwen_mtp_rmsnorm.xclbin",
        inst_path=args.output_dir / "qwen_mtp_rmsnorm.insts.bin",
        elf_path=args.output_dir / "qwen_mtp_rmsnorm.insts.elf",
        pdi_path=args.output_dir / "qwen_mtp_rmsnorm.pdi",
    )


if __name__ == "__main__":
    main()
