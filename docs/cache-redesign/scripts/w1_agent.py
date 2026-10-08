#!/usr/bin/env python3
"""E2 W1: a real Pi coding-agent session grown past 105k tokens through tool
results (tests/functional/agent_long.py), on a production-like server."""
import argparse
import os
import shutil
import subprocess
import sys

import serverctl

from paths import REPO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(serverctl.MODELS), required=True)
    parser.add_argument("--port", type=int, default=18433)
    parser.add_argument("--min-context", type=int, default=105000)
    args = parser.parse_args()
    out = serverctl.HERE / "results" / "e2" / args.model / "w1"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    cache = serverctl.HERE / "cache" / f"e2-{args.model}-w1"
    shutil.rmtree(cache, ignore_errors=True)
    server = serverctl.Server(args.model, args.port, out / "server.log",
                              out / "trace-0.jsonl", cache)
    try:
        server.start()
        result = subprocess.run(
            [os.environ.get("PYTHON", sys.executable), f"{REPO}/tests/functional/agent_long.py",
             "--agent", "pi", "--executable", shutil.which("pi"),
             "--base-url", f"http://127.0.0.1:{args.port}",
             "--model", server.model_id, "--output", str(out / "agent"),
             "--min-context", str(args.min_context),
             "--server-log", str(out / "server.log")],
            cwd=REPO)
        print(f"agent_long exit {result.returncode}", flush=True)
    finally:
        server.stop()
        shutil.rmtree(cache, ignore_errors=True)


if __name__ == "__main__":
    main()
