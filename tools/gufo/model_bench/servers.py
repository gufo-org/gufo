"""Server lifecycle for Gufo and reference servers."""

from __future__ import annotations

import ctypes
import json
import os
import shlex
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


def _stop_process(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _wait_for_http(
    processes: list[subprocess.Popen[bytes]], url: str, timeout: float,
    log_paths: list[Path], started: float,
) -> float:
    deadline = started + timeout
    while time.perf_counter() < deadline:
        for process in processes:
            if process.poll() is not None:
                logs = ", ".join(str(path) for path in log_paths)
                raise RuntimeError(
                    f"server exited with {process.returncode} before readiness; see {logs}"
                )
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return time.perf_counter() - started
        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, TimeoutError):
            pass
        time.sleep(0.05)
    logs = ", ".join(str(path) for path in log_paths)
    raise RuntimeError(f"server not ready within {timeout:.0f} s; see {logs}")


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
        self._started_at: float | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _spawn(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = self.log_path.open("ab")
        env = dict(os.environ)
        if self.env:
            env.update(self.env)
        self._started_at = time.perf_counter()
        try:
            self.process = subprocess.Popen(
                self.command, stdout=log, stderr=subprocess.STDOUT, env=env,
                start_new_session=True,
            )
        finally:
            log.close()

    def wait_ready(
        self, timeout: float = DEFAULT_READY_TIMEOUT, *,
        extra_processes: list[subprocess.Popen[bytes]] | None = None,
        extra_log_paths: list[Path] | None = None,
    ) -> "Server":
        if self.process is None:
            raise RuntimeError("server process has not been started")
        started = self._started_at or time.perf_counter()
        processes = [self.process, *(extra_processes or [])]
        log_paths = [self.log_path, *(extra_log_paths or [])]
        self.ready_seconds = _wait_for_http(
            processes, self.base_url + self.readiness, timeout, log_paths, started)
        return self

    def start(self, timeout: float = DEFAULT_READY_TIMEOUT) -> "Server":
        self._spawn()
        try:
            return self.wait_ready(timeout)
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        _stop_process(self.process)

    def __enter__(self) -> "Server":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        if exc and exc[0] is not None and self.process is not None:
            # Preserve the actual failure state before our own SIGTERM.
            # A benchmark validation error is not a server crash.
            self.failure_exit_code = self.process.poll()
        self.stop()


class Tp2Server:
    """Rank-0 HTTP server plus a remotely launched rank-1 worker."""

    def __init__(
        self,
        local: Server,
        remote_command: list[str],
        *,
        remote_host: str,
        remote_log_path: Path,
        local_cleanup_command: list[str] | None = None,
        remote_cleanup_command: list[str] | None = None,
    ):
        self.local = local
        self.remote_command = remote_command
        self.remote_host = remote_host
        self.remote_log_path = remote_log_path
        self.local_cleanup_command = local_cleanup_command
        self.remote_cleanup_command = remote_cleanup_command
        self.remote_process: subprocess.Popen[bytes] | None = None
        self.failure_exit_code: int | None = None

    @property
    def command(self) -> list[str]:
        return ["tp2", "rank0", *self.local.command, "rank1", *self.remote_command]

    @property
    def base_url(self) -> str:
        return self.local.base_url

    @property
    def port(self) -> int:
        return self.local.port

    @property
    def log_path(self) -> Path:
        return self.local.log_path

    @property
    def process(self) -> subprocess.Popen[bytes] | None:
        return self.local.process

    @property
    def ready_seconds(self) -> float | None:
        return self.local.ready_seconds

    def _ssh_command(self, command: list[str]) -> list[str]:
        return [
            "ssh", "-o", "BatchMode=yes", self.remote_host,
            "exec " + shlex.join(command),
        ]

    def _spawn_remote(self) -> None:
        self.remote_log_path.parent.mkdir(parents=True, exist_ok=True)
        log = self.remote_log_path.open("ab")
        self.remote_process = subprocess.Popen(
            self._ssh_command(self.remote_command), stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
        log.close()

    def _failure_code(self) -> int | None:
        for process in (self.local.process, self.remote_process):
            if process is not None and process.poll() is not None:
                return process.returncode
        return None

    def start(self, timeout: float = DEFAULT_READY_TIMEOUT) -> "Tp2Server":
        self.local._spawn()
        try:
            self._spawn_remote()
            self.local.wait_ready(
                timeout,
                extra_processes=[self.remote_process],
                extra_log_paths=[self.remote_log_path],
            )
        except Exception:
            self.failure_exit_code = self._failure_code()
            self.stop()
            raise
        return self

    def stop(self) -> None:
        if self.remote_cleanup_command is not None:
            try:
                subprocess.run(
                    self._ssh_command(self.remote_cleanup_command),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=30, check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass
        _stop_process(self.remote_process)
        if self.local_cleanup_command is not None:
            try:
                subprocess.run(
                    self.local_cleanup_command,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=30, check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass
        _stop_process(self.local.process)

    def __enter__(self) -> "Tp2Server":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        if exc and exc[0] is not None and self.failure_exit_code is None:
            # Preserve an actual rank failure before our own cleanup.
            self.failure_exit_code = self._failure_code()
        self.stop()


def wait_process_exit(server: Server | Tp2Server) -> None:
    server.stop()
    # Give the driver time to release device memory before the next launch.
    time.sleep(2.0)
