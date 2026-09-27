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

    for modality in ("llm", "tts", "asr", "video", "image"):
        check(["serve", "--port", "0", modality, "--help"], 0, "--api-key")
        check(["serve", modality, "--port=0", "--help"], 0, "--api-key")
        if modality in ("tts", "asr", "image"):
            help_text = check(["serve", modality, "--help"], 0,
                              "--model")
            assert "--sessions" not in help_text
            check(["serve", modality, "--sessions", "2"], 2, "Unknown option")
            check(["serve", "--sessions", "2", modality], 2, "Unknown option")
        else:
            check(["serve", modality, "--sessions", "0"], 2,
                  "server limits must be positive")
        check(["serve", modality, "unexpected"], 2, "Unexpected argument")
        check(["serve", modality, "--port", "65536"], 2, "--port must")
        check(["serve", modality, "--host", "bad.address"], 2, "--host must")
    check(["serve", "image"], 2, "--model <DIR> is required")
    for modality in ("tts", "asr"):
        check(["serve", modality], 2, "--model <DIR> is required")
        check(["serve", modality, "--model", "/missing", "--context", "0"],
              2, "--context must be at least")
        text = check(["serve", modality, "--help"], 0, "--context")
        assert ("--voice " in text) == (modality == "tts")
        assert "--served-model-name" in text
    check(["serve", "asr", "--voice", "x=y"], 2, "Unknown option")
    text = check(["transcribe", "--help"], 0, "--prompt")
    assert "--context" in text


    # Values must stay attached to their options, including before a modality
    # was selected. All these deliberately fail at the named file lookup.
    for args in (["--model", "audio"], ["llm", "--model", "audio"],
                 ["--served-model-name", "-v", "--model", "audio"],
                 ["--port", "0", "llm", "--served-model-name", "-v", "--model", "audio"],
                 ["--api-key", "test-key", "llm", "--model", "audio"],
                 ["llm", "--served-model-name", "--verbose", "--model", "audio"]):
        check(["serve", *args], 1, "Error loading model 'audio'")
    check(["serve", "help", "unknown"], 2, "unknown serve command")
    help_text = check(["serve", "llm", "--help"], 0, "model native context")
    assert "-1 = until EOS or context full" in help_text
    assert "Path to GGUF model file (required)" in help_text
    assert "8589934592" in help_text
    assert "0 = auto, at most 1 GiB and 1/8 available RAM" in help_text
    assert "--log-progress" in help_text
    check(["bench", "--help"], 0, "Path to GGUF model file (required)")
    for args in (["serve"], ["serve", "llm"], ["bench"],
                 ["serve", "llm", "--model", ""], ["bench", "--model", ""]):
        check(args, 2, "--model <PATH> is required")
    check(["bench", "--model", "missing.gguf"], 1,
          "Error loading GGUF model 'missing.gguf'")
    for limit in ("-1", "1", "16384"):
        check(["serve", "llm", "--model", "missing.gguf",
               "--context", "0", "--max-tokens", limit],
              1, "Error loading model")
    for limit in ("0", "-2", "4294967296"):
        check(["serve", "llm", "--max-tokens", limit], 2,
              "sampling and scheduling limits are invalid")
    for staging in (None, "0", "8589934592"):
        args = ["serve", "llm", "--model", "missing.gguf",
                "--cache-disk", "/unused-cache"]
        if staging is not None:
            args += ["--cache-disk-staging-bytes", staging]
        check(args, 1, "Error loading model")
    check(["serve", "llm", "--cache-disk", "/unused-cache",
           "--cache-disk-bytes", "0"], 2,
          "sampling and scheduling limits are invalid")

    for command in ("prompt", "chat", "bench"):
        check([command, "--help"], 0, "--draft-policy")
        check([command, "--draft-tokens", "0"], 2, "draft-tokens")
    check(["chat", "unexpected"], 2, "Unexpected argument")
    check(["chat", "--prompt", "unused"], 2, "Unknown option")
    print("Serving and text CLI checks passed.")


if __name__ == "__main__":
    main()
