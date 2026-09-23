"""Server lifecycle for Gufo and reference servers."""

from __future__ import annotations

import ctypes
import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_READY_TIMEOUT = 3600.0


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


DROP_CACHES = "sync; echo 3 > /proc/sys/vm/drop_caches"


def drop_file_cache(command: str | None = None) -> None:
    """Drop the page cache so the next launch measures a cold file cache."""
    candidates = [command] if command else [
        DROP_CACHES,
        f"sudo -n sh -c '{DROP_CACHES}'",
        f"doas -n sh -c '{DROP_CACHES}'",
    ]
    for candidate in candidates:
        completed = subprocess.run(candidate, shell=True, capture_output=True, text=True)
        if completed.returncode == 0:
            return
    raise SystemExit(
        "cold-file-cache loading needs privileges to run "
        f"`{DROP_CACHES}`; pass --drop-caches with a command that does it "
        "(for example `doas sh -c '...'`), or skip the loading table"
    )


class HipMemory:
    """Device-global used memory from `hipMemGetInfo`, the counter Gufo's loader logs.

    An external process sees every process's HIP allocations on this driver, so
    both servers are measured the same way; `rocm-smi` VRAM+GTT does not count
    Gufo's weight mapping on unified memory.
    """

    def __init__(self, library: Path):
        self.library = library
        self._lib = ctypes.CDLL(str(library))

    @classmethod
    def for_binary(cls, gufo_binary: Path) -> "HipMemory | None":
        """Resolve libamdhip64 the way the Gufo binary links it."""
        completed = subprocess.run(["ldd", str(gufo_binary)], capture_output=True, text=True)
        for line in completed.stdout.splitlines():
            if "libamdhip64" in line and "=>" in line:
                path = Path(line.split("=>", 1)[1].split()[0])
                if path.exists():
                    return cls(path)
        return None

    def used_gib(self) -> float | None:
        free = ctypes.c_size_t()
        total = ctypes.c_size_t()
        if self._lib.hipMemGetInfo(ctypes.byref(free), ctypes.byref(total)) != 0:
            return None
        return (total.value - free.value) / (1024 ** 3)


class Server:
    """A benchmark server process bound to a private loopback port."""

    def __init__(self, command: list[str], readiness: str, log_path: Path, env: dict[str, str] | None = None):
        self.command = command
        self.readiness = readiness
        self.log_path = log_path
        self.env = env
        self.port = free_port()
        self.process: subprocess.Popen[bytes] | None = None
        self.ready_seconds: float | None = None
        self.failure_exit_code: int | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, timeout: float = DEFAULT_READY_TIMEOUT) -> "Server":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = self.log_path.open("ab")
        env = dict(os.environ)
        if self.env:
            env.update(self.env)
        started = time.perf_counter()
        self.process = subprocess.Popen(
            self.command, stdout=log, stderr=subprocess.STDOUT, env=env,
            start_new_session=True,
        )
        deadline = started + timeout
        url = self.base_url + self.readiness
        while time.perf_counter() < deadline:
            if self.process.poll() is not None:
                self.failure_exit_code = self.process.returncode
                raise RuntimeError(
                    f"server exited with {self.process.returncode} before readiness; see {self.log_path}"
                )
            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    if response.status == 200:
                        self.ready_seconds = time.perf_counter() - started
                        return self
            except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, TimeoutError):
                pass
            time.sleep(0.05)
        self.stop()
        raise RuntimeError(f"server not ready within {timeout:.0f} s; see {self.log_path}")

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        os.killpg(self.process.pid, signal.SIGTERM)
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()

    def __enter__(self) -> "Server":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        if exc and exc[0] is not None and self.process is not None:
            # Preserve the actual failure state before our own SIGTERM.
            # A benchmark validation error is not a server crash.
            self.failure_exit_code = self.process.poll()
        self.stop()


def wait_process_exit(server: Server) -> None:
    server.stop()
    # Give the driver time to release device memory before the next launch.
    time.sleep(2.0)
