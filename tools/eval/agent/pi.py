"""Pi agent configuration and invocation for gufo-agent-eval.

Pi itself is not packaged here. Install it however you like -- ``npm i -g
@earendil-works/pi-coding-agent``, ``npx``, or a Nix package -- and the runner
finds it on ``PATH`` or via ``GUFO_EVAL_PI``.

Install it *before* the run, though. The jail runs with ``--unshare-net``, so
Pi cannot fetch anything once it is inside; the runner binds the resolved
installation in read-only.

What this repository does define is the configuration:

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

    base_url: str
    model_id: str
    api_key: str
    context_window: int
    max_tokens: int

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
        template.replace("@BASE_URL@", config.base_url)
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

    `GUFO_EVAL_PI` wins so a run can pin a specific installation; otherwise
    whatever is on `PATH`.
    """
    override = os.environ.get("GUFO_EVAL_PI")
    if override:
        return Path(override).resolve()

    found = shutil.which("pi")
    if found:
        return Path(found).resolve()

    raise RuntimeError(
        "pi not found. Install it (npm i -g @earendil-works/pi-coding-agent) "
        "or set GUFO_EVAL_PI to its path."
    )
