"""Start and stop a production-like `gufo serve llm` for the experiments."""
import json
import os
import pathlib
import signal
import subprocess
import time
import urllib.request

from paths import BIN, FN_MODEL as FN, FN_MTP as MTP, Q27_DFLASH as DF, Q27_MODEL as Q27, WORK as HERE

# Mirrors the llama-swap production entries, plus the disk tier.
MODELS = {
    "fn": ["--model", FN, "--context", "260000", "--prefill-chunk", "2048",
           "--speculative", "mtp", "--mtp-model", MTP,
           "--temperature", "1.0", "--top-p", "0.95", "--top-k", "20"],
    "q27": ["--model", Q27, "--context", "256000", "--prefill-chunk", "512",
            "--speculative", "dflash2", "--dflash-model", DF,
            "--draft-tokens", "7", "--min-draft-tokens", "1",
            "--temperature", "0.8", "--top-p", "0.95", "--top-k", "20",
            "--preserve-thinking", "auto"],
}


class Server:
    def __init__(self, model, port, log_path, trace_path, cache_dir,
                 disk_bytes=16 << 30, think=None, sessions=2):
        self.port = port
        self.log_path = pathlib.Path(log_path)
        command = [str(BIN), "serve", "--host", "127.0.0.1", "--port", str(port),
                   "--sessions", str(sessions), "llm", *MODELS[model],
                   "--max-pending", "16", "--max-pending-per-client", "4",
                   "--max-tokens", "8192", "--trace", str(trace_path),
                   "--cache-disk", str(cache_dir),
                   "--cache-disk-bytes", str(disk_bytes)]
        if think:
            command += ["--think", think]
        self.command = command
        self.proc = None
        self.model_id = None

    def start(self, timeout=900):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as log:
            log.write("### " + " ".join(self.command) + "\n")
            log.flush()
            self.proc = subprocess.Popen(self.command, stdout=log,
                                         stderr=subprocess.STDOUT,
                                         start_new_session=True)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited with {self.proc.returncode}")
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{self.port}/v1/models", timeout=5) as r:
                    self.model_id = json.load(r)["data"][0]["id"]
                    return self
            except Exception:
                time.sleep(2)
        raise RuntimeError("server did not become ready")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=300)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
