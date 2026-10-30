#!/usr/bin/env python3
"""Install a separate, disabled rootless Turbo service from a local pinned image."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Full local OCI image ID (sha256:...)")
    parser.add_argument("--model-dir", type=Path, default=Path.home() / "models/Qwen-Image-2.1-Turbo")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter with Pillow")
    args = parser.parse_args()
    if os.getuid() == 0 or not re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", args.image):
        parser.error("run as the unprivileged user with a full immutable local image ID")
    args.image = "sha256:" + args.image.removeprefix("sha256:")
    args.model_dir = args.model_dir.expanduser().resolve()
    subprocess.run([args.python, "-c", "from PIL import Image"], check=True)
    metadata = json.loads((args.model_dir / "model_index.json").read_text())
    if len(metadata.get("sample_sigmas", [])) != 8:
        parser.error("model directory has no eight-step Turbo schedule")
    inspected = json.loads(subprocess.check_output(["podman", "image", "inspect", args.image], text=True))[0]
    if inspected["Id"].removeprefix("sha256:") != args.image.removeprefix("sha256:"):
        raise RuntimeError("image identity mismatch")
    home = Path.home()
    bindir = home / ".local/bin"
    bindir.mkdir(parents=True, exist_ok=True)
    root = home / ".local/share/gufo-image-turbo"
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).with_name("client.py"), root / "client.py")
    engine = bindir / "gufo-turbo"
    podman = ["podman", "run", "--rm", "--network=host", "--userns=keep-id:uid=1000,gid=1000",
              "--device=/dev/kfd", "--device=/dev/dri", "--group-add=keep-groups",
              "--ulimit", "memlock=-1", "-v", f"{args.model_dir}:{args.model_dir}:ro", args.image, "/bin/gufo"]
    engine.write_text("#!/bin/sh\nexec " + shlex.join(podman) + ' "$@"\n')
    engine.chmod(0o755)
    client = bindir / "gufo-image-turbo"
    command = [args.python, str(root / "client.py"), "--url", "http://127.0.0.1:28182",
               "--model", "Qwen-Image-2.1-Turbo"]
    client.write_text("#!/bin/sh\nexec " + shlex.join(command) + ' "$@"\n')
    client.chmod(0o755)
    service_dir = home / ".config/systemd/user"
    service_dir.mkdir(parents=True, exist_ok=True)
    command = [str(engine), "serve", "--host", "127.0.0.1", "--port", "28182",
               "--max-request-bytes", "33554432", "image", "--model", str(args.model_dir),
               "--served-model-name", "Qwen-Image-2.1-Turbo"]
    unit = service_dir / "gufo-image-turbo.service"
    # systemd uses percent specifiers even inside quoted arguments.
    unit.write_text("[Unit]\nDescription=Gufo Qwen Image 2.1 Turbo local generation and editing API\nAfter=network.target\n\n"
                    "[Service]\nType=simple\nExecStart=" + shlex.join(command).replace("%", "%%") +
                    "\nRestart=on-failure\nRestartSec=5\nTimeoutStopSec=45\n\n"
                    "[Install]\nWantedBy=default.target\n")
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    print(json.dumps({"image_id": inspected["Id"], "labels": inspected.get("Labels", {}),
                      "service": str(unit), "client": str(client), "enabled_by_installer": False}))


if __name__ == "__main__":
    main()
