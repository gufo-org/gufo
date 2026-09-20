"""Exercise the installed CLI parser without loading model weights."""

import os
import subprocess
import sys


def main():
    binary = sys.argv[1]
    env = {key: value for key, value in os.environ.items()
           if key not in {"HOST", "PORT", "GUFO_HOST", "GUFO_PORT"}}

    def check(args, status, message):
        result = subprocess.run([binary, *args], env=env, text=True,
                                capture_output=True, timeout=10)
        output = result.stdout + result.stderr
        assert result.returncode == status and message in output, (
            args, result.returncode, output)
        return output

    for modality in ("llm", "audio", "video"):
        check(["serve", "--port", "0", modality, "--help"], 0, "--api-key")
        check(["serve", modality, "--port=0", "--help"], 0, "--api-key")
        if modality == "audio":
            help_text = check(["serve", modality, "--help"], 0, "--tts-context")
            assert "--sessions" not in help_text
            check(["serve", modality, "--sessions", "2"], 2, "Unknown option")
            check(["serve", "--sessions", "2", modality], 2, "Unknown option")
        else:
            check(["serve", modality, "--sessions", "0"], 2,
                  "server limits must be positive")
        check(["serve", modality, "unexpected"], 2, "Unexpected argument")
        check(["serve", modality, "--port", "65536"], 2, "--port must")
        check(["serve", modality, "--host", "bad.address"], 2, "--host must")

    # Values must stay attached to their options, including before a modality
    # was selected. All these deliberately fail at the named file lookup.
    for args in (["--model", "audio"], ["llm", "--model", "audio"],
                 ["--served-model-name", "-v", "--model", "audio"],
                 ["--port", "0", "llm", "--served-model-name", "-v", "--model", "audio"],
                 ["--api-key", "test-key", "llm", "--model", "audio"],
                 ["llm", "--served-model-name", "--verbose", "--model", "audio"]):
        check(["serve", *args], 1, "Error loading model 'audio'")
    check(["serve", "help", "unknown"], 2, "unknown serve command")

    for command in ("prompt", "chat", "bench"):
        check([command, "--help"], 0, "--draft-policy")
        check([command, "--draft-tokens", "0"], 2, "draft-tokens")
    check(["chat", "unexpected"], 2, "Unexpected argument")
    check(["chat", "--prompt", "unused"], 2, "Unknown option")
    print("Serving and text CLI checks passed.")


if __name__ == "__main__":
    main()
