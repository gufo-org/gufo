"""Pi agent configuration and invocation for gufo-agent-eval.

Pi has to exist before the jail starts: the jail runs with ``--unshare-net``,
so nothing can be fetched once it is inside. ``tools/eval/nix/pi.nix`` builds
it, and the flake app and dev shell export ``GUFO_EVAL_PI``; the runner binds
the resolved installation into the jail read-only.

The configuration is what this repository defines:

* **Versioned in ``pi_config/``.** The provider template and settings. These
  define what is being measured, so they hash into the suite identity.
* **Generated per run, into a tmpfs.** The concrete ``models.json`` carrying
  the endpoint URL, model ID, and credential. Never written to the repository
  and never persisted, so it cannot reach a result artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

CONFIG_ASSETS = Path(__file__).resolve().parent / "pi_config"

# Pi's config directory inside the jail. A tmpfs, so nothing survives the run.
JAIL_CONFIG_DIR = "/run/pi"
JAIL_SESSION_DIR = "/out/sessions"

# Everything that keeps Pi from reaching the network or the host's state.
# #153 requires telemetry, update checks, saved sessions, and implicit
# providers to be disabled.
OFFLINE_ENV = {
    "PI_OFFLINE": "1",
    "PI_SKIP_VERSION_CHECK": "1",
    "PI_TELEMETRY": "0",
}

PROVIDER = "gufo-eval"


@dataclass
class PiConfig:
    """Resolved Pi configuration for one run."""

    # The endpoint as seen from the host. Inside the jail it is republished on
    # loopback, so `jail_base_url` is what reaches models.json.
    base_url: str
    model_id: str
    api_key: str
    context_window: int
    max_tokens: int

    @property
    def jail_base_url(self) -> str:
        """The endpoint's address from inside the jail.

        The jail has no route anywhere; the endpoint is bridged onto its
        loopback on the same port, so only the host part changes.
        """
        parsed = urllib.parse.urlparse(self.base_url)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        return parsed._replace(
            scheme="http", netloc=f"127.0.0.1:{port}"
        ).geturl()

    def identity(self) -> str:
        """Hash of the versioned configuration.

        Covers everything in ``pi_config/`` -- the things that change what
        is being measured. Deliberately excludes base URL, model ID, and
        credential, which vary per run without changing the benchmark.
        """
        digest = hashlib.sha256()
        for name in sorted(p.name for p in CONFIG_ASSETS.iterdir()):
            digest.update(name.encode())
            digest.update((CONFIG_ASSETS / name).read_bytes())
        return digest.hexdigest()[:16]


def materialize(config: PiConfig, destination: Path) -> Path:
    """Write a complete Pi config directory for one run.

    The caller mounts `destination` at `JAIL_CONFIG_DIR` inside the jail and
    points `PI_CODING_AGENT_DIR` at it.
    """
    destination.mkdir(parents=True, exist_ok=True)

    template = (CONFIG_ASSETS / "models.json.template").read_text()
    rendered = (
        template.replace("@BASE_URL@", config.jail_base_url)
        .replace("@MODEL_ID@", config.model_id)
        .replace("@API_KEY@", config.api_key)
        .replace("@CONTEXT_WINDOW@", str(config.context_window))
        .replace("@MAX_TOKENS@", str(config.max_tokens))
    )
    # Parse before writing: a malformed template should fail here, not as an
    # opaque Pi startup error inside a jail.
    models = json.loads(rendered)
    models.pop("_comment", None)
    (destination / "models.json").write_text(json.dumps(models, indent=2) + "\n")

    settings = json.loads((CONFIG_ASSETS / "settings.json").read_text())
    settings.pop("_comment", None)
    (destination / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    # Credential-bearing file: readable only by the owner, even though it
    # lives on a tmpfs that is discarded with the run.
    (destination / "models.json").chmod(0o600)

    return destination


def command(pi_binary: Path, config: PiConfig, prompt: str) -> list[str]:
    """argv for one non-interactive Pi run emitting JSON events."""
    return [
        str(pi_binary),
        "--mode",
        "json",
        "--print",
        # provider/id form, so the model resolves against the generated
        # models.json rather than any built-in provider.
        "--model",
        f"{PROVIDER}/{config.model_id}",
        "--session-dir",
        JAIL_SESSION_DIR,
        prompt,
    ]


def environment(config: PiConfig) -> dict[str, str]:
    """Environment for Pi inside the jail.

    The jail clears the host environment, so this is the complete set: no
    host credentials, no host paths.
    """
    env = {
        "HOME": "/root",
        "PATH": "/bin:/usr/bin",
        "PI_CODING_AGENT_DIR": JAIL_CONFIG_DIR,
        # Documented by Pi as the Nix escape hatch: store paths tokenize
        # poorly, so package discovery is pointed somewhere inert.
        "PI_PACKAGE_DIR": f"{JAIL_CONFIG_DIR}/packages",
    }
    env.update(OFFLINE_ENV)
    return env


def resolve_binary() -> Path:
    """Locate the Pi executable.

    `GUFO_EVAL_PI` is what the flake app and the dev shell set, from
    `tools/eval/nix/pi.nix`. `PATH` is a development convenience; a run that
    falls back to it is not reproducible, since the host's Pi is whatever
    happens to be installed.
    """
    override = os.environ.get("GUFO_EVAL_PI")
    if override:
        return Path(override).resolve()

    found = shutil.which("pi")
    if found:
        return Path(found).resolve()

    raise RuntimeError(
        "pi not found. Build it with `nix-build tools/eval/nix/pi.nix` and "
        "export GUFO_EVAL_PI=<result>/bin/pi, or enter the dev shell."
    )
