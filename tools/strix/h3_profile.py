#!/usr/bin/env python3
"""Manual MiniMax H3 end-to-end profiling harness.

The harness runs the public ``strix video`` path, preserves its
parameter and telemetry reports, and adds bounded process I/O, memory, power,
and clock samples. It never writes the prompt text or model path to its reports.

Complete generations are deliberately opt-in. Development correctness and
optimization use the short production-shape component oracles instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import signal
import statistics
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


PRESETS = ("exact", "fast", "aggressive")
DEFAULT_BENCHMARK_PRESETS = ("exact",)


def balanced_schedule(presets: list[str], rounds: int) -> list[str]:
    """Return a deterministic preset schedule."""
    if not presets or rounds < 1:
        raise ValueError("presets and rounds must be non-empty")
    schedule: list[str] = []
    for round_index in range(rounds):
        shift = round_index % len(presets)
        order = presets[shift:] + presets[:shift]
        if (round_index // len(presets)) % 2:
            order = list(reversed(order))
        schedule.extend(order)
    return schedule


def parse_key_value_lines(text: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue
        token = value.strip().split(maxsplit=1)
        if token and token[0].isdigit():
            result[key] = int(token[0])
    return result


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, OSError):
        return ""


def read_integer(path: Path) -> int | None:
    text = read_text(path).strip()
    try:
        return int(text)
    except ValueError:
        return None


def selected_clock_mhz(path: Path) -> int | None:
    for line in read_text(path).splitlines():
        if "*" not in line:
            continue
        fields = line.replace("*", "").split()
        for field in fields:
            if field.endswith("Mhz") and field[:-3].isdigit():
                return int(field[:-3])
    return None


def discover_amdgpu_hwmon() -> list[Path]:
    roots: list[Path] = []
    for candidate in sorted(
        Path("/sys/class/drm").glob("card*/device/hwmon/hwmon*")
    ):
        if read_text(candidate / "name").strip() == "amdgpu":
            roots.append(candidate)
    return roots


@dataclass
class MonitorSample:
    elapsed_ms: float
    rss_kib: int | None = None
    rss_anon_kib: int | None = None
    rss_file_kib: int | None = None
    rss_shmem_kib: int | None = None
    hwm_kib: int | None = None
    swap_kib: int | None = None
    read_bytes: int | None = None
    write_bytes: int | None = None
    gpu_temp_millic: int | None = None
    gpu_power_microwatts: int | None = None
    gpu_clock_mhz: int | None = None


@dataclass
class ProcessMonitor:
    pid: int
    interval_seconds: float
    started: float = field(default_factory=time.monotonic)
    samples: list[MonitorSample] = field(default_factory=list)
    stop_event: threading.Event = field(default_factory=threading.Event)

    def sample(self) -> None:
        status = parse_key_value_lines(read_text(Path(f"/proc/{self.pid}/status")))
        io = parse_key_value_lines(read_text(Path(f"/proc/{self.pid}/io")))
        temperatures: list[int] = []
        powers: list[int] = []
        clocks: list[int] = []
        for root in discover_amdgpu_hwmon():
            temperature = read_integer(root / "temp1_input")
            power = read_integer(root / "power1_average")
            clock = selected_clock_mhz(root.parent.parent / "pp_dpm_sclk")
            if temperature is not None:
                temperatures.append(temperature)
            if power is not None:
                powers.append(power)
            if clock is not None:
                clocks.append(clock)
        self.samples.append(
            MonitorSample(
                elapsed_ms=(time.monotonic() - self.started) * 1000.0,
                rss_kib=status.get("VmRSS"),
                rss_anon_kib=status.get("RssAnon"),
                rss_file_kib=status.get("RssFile"),
                rss_shmem_kib=status.get("RssShmem"),
                hwm_kib=status.get("VmHWM"),
                swap_kib=status.get("VmSwap"),
                read_bytes=io.get("read_bytes"),
                write_bytes=io.get("write_bytes"),
                gpu_temp_millic=max(temperatures, default=None),
                gpu_power_microwatts=max(powers, default=None),
                gpu_clock_mhz=max(clocks, default=None),
            )
        )

    def run(self) -> None:
        self.sample()
        while not self.stop_event.wait(self.interval_seconds):
            self.sample()
        self.sample()


def maximum(samples: Iterable[MonitorSample], field_name: str) -> int | None:
    values = [
        value
        for sample in samples
        if (value := getattr(sample, field_name)) is not None
    ]
    return max(values) if values else None


def minimum(samples: Iterable[MonitorSample], field_name: str) -> int | None:
    values = [
        value
        for sample in samples
        if (value := getattr(sample, field_name)) is not None
    ]
    return min(values) if values else None


def last(samples: list[MonitorSample], field_name: str) -> int | None:
    for sample in reversed(samples):
        value = getattr(sample, field_name)
        if value is not None:
            return value
    return None


def first(samples: list[MonitorSample], field_name: str) -> int | None:
    for sample in samples:
        value = getattr(sample, field_name)
        if value is not None:
            return value
    return None


def monitor_summary(samples: list[MonitorSample], wall_seconds: float) -> dict[str, Any]:
    read_end = last(samples, "read_bytes")
    read_begin = first(samples, "read_bytes")
    write_end = last(samples, "write_bytes")
    write_begin = first(samples, "write_bytes")
    read_bytes = (
        max(0, read_end - read_begin)
        if read_end is not None and read_begin is not None
        else None
    )
    write_bytes = (
        max(0, write_end - write_begin)
        if write_end is not None and write_begin is not None
        else None
    )
    return {
        "sample_count": len(samples),
        "peak_rss_kib": maximum(samples, "rss_kib"),
        "peak_rss_anon_kib": maximum(samples, "rss_anon_kib"),
        "peak_rss_file_kib": maximum(samples, "rss_file_kib"),
        "peak_rss_shmem_kib": maximum(samples, "rss_shmem_kib"),
        "peak_hwm_kib": maximum(samples, "hwm_kib"),
        "peak_swap_kib": maximum(samples, "swap_kib"),
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "read_mib_per_second": (
            read_bytes / (1024.0 * 1024.0 * wall_seconds)
            if read_bytes is not None and wall_seconds > 0
            else None
        ),
        "write_mib_per_second": (
            write_bytes / (1024.0 * 1024.0 * wall_seconds)
            if write_bytes is not None and wall_seconds > 0
            else None
        ),
        "minimum_gpu_temp_millic": minimum(samples, "gpu_temp_millic"),
        "maximum_gpu_temp_millic": maximum(samples, "gpu_temp_millic"),
        "minimum_gpu_power_microwatts": minimum(
            samples, "gpu_power_microwatts"
        ),
        "maximum_gpu_power_microwatts": maximum(
            samples, "gpu_power_microwatts"
        ),
        "minimum_gpu_clock_mhz": minimum(samples, "gpu_clock_mhz"),
        "maximum_gpu_clock_mhz": maximum(samples, "gpu_clock_mhz"),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    partial = path.with_name(
        f"{path.name}.partial-{os.getpid()}-{time.monotonic_ns()}"
    )
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    partial.replace(path)


def public_descriptor(
    preset: str,
    ordinal: int,
    cache_label: str,
    prompt_sha256: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema": "strix.minimax-h3-profile-run.v1",
        "preset": preset,
        "ordinal": ordinal,
        "cache_label": cache_label,
        "prompt_sha256": prompt_sha256,
        "seed": seed,
        "backend": "rocm-hip-gfx1151",
        "precision": "bf16-f32",
    }


def child_rusage() -> dict[str, float | int]:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "user_seconds": usage.ru_utime,
        "system_seconds": usage.ru_stime,
        "maximum_rss_kib": usage.ru_maxrss,
        "minor_page_faults": usage.ru_minflt,
        "major_page_faults": usage.ru_majflt,
        "block_inputs": usage.ru_inblock,
        "block_outputs": usage.ru_oublock,
        "voluntary_context_switches": usage.ru_nvcsw,
        "involuntary_context_switches": usage.ru_nivcsw,
    }


def rusage_delta(
    before: dict[str, float | int], after: dict[str, float | int]
) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for key, value in after.items():
        result[key] = (
            value
            if key == "maximum_rss_kib"
            else value - before.get(key, 0)
        )
    return result


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stop_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_one(
    args: argparse.Namespace,
    preset: str,
    ordinal: int,
    occurrence: int,
) -> dict[str, Any]:
    run_directory = args.output_root / f"{ordinal:03d}-{preset}"
    run_directory.mkdir()
    output_path = run_directory / "output.mp4"
    parameters_path = run_directory / "parameters.json"
    telemetry_path = run_directory / "telemetry.json"
    progress_path = run_directory / "progress.log"
    cache_label = (
        args.cache_label
        if args.cache_label != "auto"
        else ("first-observation" if occurrence == 0 else "warm-observation")
    )
    prompt_sha256 = hashlib.sha256(args.prompt.encode("utf-8")).hexdigest()
    descriptor = public_descriptor(
        preset, ordinal, cache_label, prompt_sha256, args.seed
    )
    command = [
        str(args.binary),
        "video",
        "--model",
        str(args.model),
        "--preset",
        preset,
        "--seed",
        str(args.seed),
        "--output",
        str(output_path),
        "--parameters",
        str(parameters_path),
        "--report",
        str(telemetry_path),
        "--profile",
        args.prompt,
    ]
    usage_before = child_rusage()
    started = time.monotonic()
    with progress_path.open("wb") as progress:
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=progress,
            start_new_session=True,
        )
        monitor = ProcessMonitor(process.pid, args.sample_interval)
        monitor_thread = threading.Thread(target=monitor.run, daemon=True)
        monitor_thread.start()
        try:
            return_code = process.wait()
        except BaseException:
            stop_process_group(process)
            raise
        finally:
            usage_after = child_rusage()
            monitor.stop_event.set()
            monitor_thread.join()
    wall_seconds = time.monotonic() - started
    descriptor.update(
        {
            "return_code": return_code,
            "wall_ms": wall_seconds * 1000.0,
            "binary_sha256": sha256_file(args.binary),
            "harness_sha256": sha256_file(Path(__file__).resolve()),
            "kernel_release": platform.release(),
            "child_rusage": rusage_delta(usage_before, usage_after),
            "monitor": monitor_summary(monitor.samples, wall_seconds),
            "monitor_samples": [asdict(sample) for sample in monitor.samples],
        }
    )
    if parameters_path.is_file():
        descriptor["parameters"] = load_json(parameters_path)
    if telemetry_path.is_file():
        descriptor["telemetry"] = load_json(telemetry_path)
    if output_path.is_file():
        descriptor["output"] = {
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        }
    atomic_json(run_directory / "profile.json", descriptor)
    return descriptor


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    presets: dict[str, dict[str, Any]] = {}
    phase_keys = (
        "inventory_ms",
        "tokenizer_ms",
        "prompt_ms",
        "denoiser_ms",
        "video_vae_ms",
        "audio_vae_ms",
        "media_ms",
    )
    for preset in sorted({run["preset"] for run in runs}):
        successful = [
            run for run in runs if run["preset"] == preset and run["return_code"] == 0
        ]
        wall_values = [run["wall_ms"] for run in successful]
        total_values = [
            run["telemetry"]["total_ms"]
            for run in successful
            if "telemetry" in run
        ]
        phase_medians: dict[str, float | None] = {}
        for key in phase_keys:
            values = [
                run["telemetry"][key]
                for run in successful
                if key in run.get("telemetry", {})
            ]
            phase_medians[key] = statistics.median(values) if values else None
        total_median = statistics.median(total_values) if total_values else None
        bottlenecks = sorted(
            (
                {
                    "phase": key.removesuffix("_ms"),
                    "median_ms": value,
                    "fraction_of_generation": (
                        value / total_median
                        if total_median is not None and total_median > 0
                        else None
                    ),
                }
                for key, value in phase_medians.items()
                if value is not None
            ),
            key=lambda item: item["median_ms"],
            reverse=True,
        )
        cache_states: dict[str, dict[str, float | int | None]] = {}
        for cache_label in sorted({run["cache_label"] for run in successful}):
            matching = [
                run for run in successful if run["cache_label"] == cache_label
            ]
            cache_wall = [run["wall_ms"] for run in matching]
            cache_total = [
                run["telemetry"]["total_ms"]
                for run in matching
                if "telemetry" in run
            ]
            cache_states[cache_label] = {
                "runs": len(matching),
                "wall_ms_median": statistics.median(cache_wall),
                "generation_ms_median": (
                    statistics.median(cache_total) if cache_total else None
                ),
            }
        presets[preset] = {
            "runs": len(successful),
            "wall_ms_median": (
                statistics.median(wall_values) if wall_values else None
            ),
            "generation_ms_median": total_median,
            "phase_median_ms": phase_medians,
            "bottlenecks": bottlenecks,
            "cache_states": cache_states,
        }
    return {
        "schema": "strix.minimax-h3-profile-summary.v1",
        "runs": runs,
        "presets": presets,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile complete MiniMax H3 CLI generations"
    )
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prompt", default="A red fox walking through snow")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--presets",
        default=",".join(DEFAULT_BENCHMARK_PRESETS),
        help="comma-separated frozen presets; default: exact",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="observations per preset; repeated full generations are rejected",
    )
    parser.add_argument("--cooldown-seconds", type=float, default=0.0)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument(
        "--cache-label",
        choices=("auto", "cold", "warm"),
        default="auto",
        help="operator assertion; auto labels first/repeated observations",
    )
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument(
        "--allow-full-generation",
        action="store_true",
        help="acknowledge that this command runs complete delivery generations",
    )
    args = parser.parse_args()
    args.presets = [value for value in args.presets.split(",") if value]
    unknown = sorted(set(args.presets) - set(PRESETS))
    if unknown:
        parser.error(f"unsupported presets: {', '.join(unknown)}")
    if args.rounds != 1 or args.seed < 0:
        parser.error("--rounds must be exactly 1 and --seed non-negative")
    if args.cooldown_seconds < 0 or args.sample_interval <= 0:
        parser.error("sampling/cooldown values are invalid")
    args.binary = args.binary.resolve()
    args.model = args.model.resolve()
    args.output_root = args.output_root.resolve()
    return args


def main() -> int:
    args = parse_args()
    schedule = balanced_schedule(args.presets, args.rounds)
    if args.plan:
        print(json.dumps({"schedule": schedule}, indent=2))
        return 0
    if not args.allow_full_generation:
        raise SystemExit(
            "complete MiniMax H3 generations are manual release validation; "
            "pass --allow-full-generation after short production-shape "
            "oracles pass"
        )
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        raise SystemExit("profiling binary is missing or not executable")
    if not args.model.is_dir():
        raise SystemExit("model directory is missing")
    if args.output_root.exists():
        raise SystemExit("output root already exists; refusing to replace it")
    args.output_root.mkdir(parents=True)

    occurrences = {preset: 0 for preset in args.presets}
    runs: list[dict[str, Any]] = []
    for ordinal, preset in enumerate(schedule):
        if ordinal > 0 and args.cooldown_seconds > 0:
            time.sleep(args.cooldown_seconds)
        result = run_one(args, preset, ordinal, occurrences[preset])
        occurrences[preset] += 1
        runs.append(result)
        atomic_json(args.output_root / "summary.json", aggregate(runs))
        if result["return_code"] != 0 and not args.keep_going:
            return int(result["return_code"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
