#!/usr/bin/env python3
"""Startup contract while a text model loads, then after the listener handoff.

Polls health, readiness and generation from process start. While loading,
every answer must be llama-server's 503 `Loading model`, `/health` included:
pollers such as llama-swap and `run.py` treat a 200 health check as ready.
After the handoff the real server must answer health, readiness and a
generation. The runner owns the port; pass a production `gufo serve llm`
command after `--`. Add `--cache-disk DIR` to include a disk-tier walk.
"""

import argparse
import hashlib
import http.client
import json
from pathlib import Path
import socket
import subprocess
import time

from run import SERVER_POPEN, interrupt_server, option, provenance, write_json

LOADING = {"message": "Loading model", "type": "unavailable_error",
           "code": "loading"}
# The listener closes before the real server binds; only post-load setup runs
# between them, so a refused connection after loading answers must be brief.
MAX_HANDOFF_S = 10.0


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def probe(port, method, path, body=None):
    started = time.monotonic()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request(method, path, None if body is None else json.dumps(body),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        text = response.read().decode()
        try:
            payload = json.loads(text)
        except ValueError:
            payload = text
        return {"status": response.status, "body": payload,
                "retry_after": response.getheader("Retry-After"),
                "wall_s": time.monotonic() - started}
    except (OSError, http.client.HTTPException) as error:
        return {"status": None, "error": type(error).__name__,
                "wall_s": time.monotonic() - started}
    finally:
        connection.close()


def is_loading(result):
    return (result["status"] == 503 and isinstance(result["body"], dict)
            and result["body"].get("error") == LOADING
            and result["retry_after"] == "1")


def generation(model):
    return {"model": model, "prompt": "The capital of France is",
            "max_tokens": 8, "temperature": 0, "cache_prompt": False}


def observe(process, port, deadline, observations):
    """Probes until /ready answers 200, appending every observation in order."""
    spawned = time.monotonic()

    def record(method, path, body=None):
        result = probe(port, method, path, body)
        result.update(t_s=round(time.monotonic() - spawned, 3),
                      method=method, path=path)
        observations.append(result)
        return result

    while True:
        require(process.poll() is None, f"server exited {process.returncode} during load")
        require(time.monotonic() < deadline, "startup timed out before /ready")
        health = record("GET", "/health")
        if record("GET", "/ready")["status"] == 200:
            return
        # Never send a generation that a ready server would run.
        if health["status"] != 200:
            record("POST", "/v1/completions", generation("probe"))
        time.sleep(0.05)


def check_loading_phase(observations):
    """Returns the loading summary; fails on any answer that claims readiness."""
    # Each round probes /health then /ready: a healthy answer beside an
    # unready one is the false readiness llama-swap and run.py would act on.
    premature = [(health["t_s"], ready["status"])
                 for health, ready in zip(observations, observations[1:])
                 if health["path"] == "/health" and health["status"] == 200
                 and ready["path"] == "/ready" and ready["status"] != 200]
    require(not premature, f"/health answered 200 while /ready did not: {premature[:3]}"
            f" ({len(premature)} rounds)")
    first_real = next(index for index, item in enumerate(observations)
                      if item["status"] is not None and not is_loading(item))
    loading = [item for item in observations[:first_real] if is_loading(item)]
    require(loading, "no loading answer observed; the load window was not tested")
    require(any(item["path"] == "/health" for item in loading),
            "no loading answer to /health observed")
    unexpected = [item for item in observations[:first_real]
                  if item["status"] is not None and not is_loading(item)]
    require(not unexpected, f"unexpected answer while loading: {unexpected}")
    # Refusals are expected before the responder binds and during the handoff,
    # never interleaved with loading answers.
    first_loading = observations.index(loading[0])
    last_loading = max(index for index, item in enumerate(observations)
                       if is_loading(item))
    require(last_loading < first_real, "loading answer after the real server answered")
    interleaved = [item for item in observations[first_loading:last_loading]
                   if item["status"] is None]
    require(not interleaved, f"connection refused between loading answers: {interleaved}")
    handoff_s = observations[first_real]["t_s"] - observations[last_loading]["t_s"]
    require(handoff_s < MAX_HANDOFF_S, f"handoff took {handoff_s:.3f}s")
    return {"loading_answers": len(loading),
            "first_loading_s": loading[0]["t_s"],
            "last_loading_s": observations[last_loading]["t_s"],
            "first_real_s": observations[first_real]["t_s"],
            "handoff_s": round(handoff_s, 3),
            "first_real": {key: observations[first_real][key]
                           for key in ("method", "path", "status")}}


def check_ready(port, model):
    health = probe(port, "GET", "/health")
    require(health["status"] == 200 and health["body"] == {"status": "ok"},
            f"/health after load: {health}")
    ready = probe(port, "GET", "/ready")
    require(ready["status"] == 200 and ready["body"].get("status") == "ready",
            f"/ready after load: {ready}")
    models = probe(port, "GET", "/v1/models")
    require(models["status"] == 200 and models["body"]["data"][0]["id"] == model,
            f"/v1/models after load: {models}")
    completion = probe(port, "POST", "/v1/completions", generation(model))
    require(completion["status"] == 200, f"generation after load: {completion}")
    require(completion["body"]["usage"]["completion_tokens"] > 0,
            "generation produced no tokens")
    return {"health": health, "ready": ready, "completion": completion}


def check_log(log_path):
    lines = log_path.read_text().splitlines()
    loaded = [index for index, line in enumerate(lines)
              if "event=load_completed" in line and "kind=text" in line]
    listening = [index for index, line in enumerate(lines) if "event=listening" in line]
    require(len(loaded) == 1 and len(listening) == 1,
            f"expected one load_completed and one listening line: {loaded} {listening}")
    require(loaded[0] < listening[0], "listener announced before the load completed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if len(command) < 3 or command[1] != "serve" or "llm" not in command:
        parser.error("pass a production gufo serve llm command after --")
    for flag in ("--host", "--port", "--api-key", "--served-model-name"):
        if option(command, flag) is not None:
            parser.error(f"the runner owns {flag}; omit it")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        port = reserve.getsockname()[1]
    model = "startup-loading"
    command += ["--host", "127.0.0.1", "--port", str(port),
                "--served-model-name", model]
    report = {**provenance(), "command": command,
              "binary": str(Path(command[0]).resolve()),
              "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "status": "running"}
    report_path = output / "report.json"
    log_path = output / "server.log"
    write_json(report_path, report)
    with log_path.open("w") as log:
        observations = []
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   **SERVER_POPEN)
        try:
            observe(process, port, time.monotonic() + args.startup_timeout,
                    observations)
            report["loading"] = check_loading_phase(observations)
            report["ready"] = check_ready(port, model)
            interrupt_server(process)
            report["exit_code"] = process.wait(timeout=60)
            require(report["exit_code"] == 0, f"exit status {report['exit_code']}")
            check_log(log_path)
            report["status"] = "passed"
        except BaseException as error:
            report["status"] = "failed"
            report["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            write_json(output / "observations.json", observations)
            if process.poll() is None:
                process.kill()
                process.wait()
            write_json(report_path, report)
    loading = report["loading"]
    print(f"passed: {loading['loading_answers']} loading answers from "
          f"{loading['first_loading_s']}s to {loading['last_loading_s']}s, "
          f"handoff {loading['handoff_s']}s; report {report_path}")


if __name__ == "__main__":
    main()
