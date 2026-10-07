"""Exercise the installed CLI parser without loading model weights."""

import os
import stat
import subprocess
import sys
import tempfile


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

    def run(args):
        return subprocess.run([binary, *args], env=env, text=True,
                              capture_output=True, timeout=10)

    for modality in ("llm", "tts", "asr", "video", "image"):
        check(["serve", "--port", "0", modality, "--help"], 0, "--api-key")
        check(["serve", modality, "--port=0", "--help"], 0, "--api-key")
        check(["serve", modality, "--help"], 0, "--log-level")
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
    assert "0 = auto, at most the disk budget and 1/8 available RAM" in help_text
    assert "--log-progress" in help_text

    # Verbosity must change the output, not merely parse. The config line is
    # emitted at the debug tier before the model file is opened, so these all
    # fail at the named file and differ only in what they logged.
    check(["serve", "--help"], 0, "--log-level")
    load = ["serve", "llm", "--model", "missing.gguf"]
    assert "event=options" not in run(load).stderr
    for spelling in (["-v"], ["--verbose"], ["--log-level", "debug"]):
        result = run(load + spelling)
        assert "[DEBUG] [server] event=options" in result.stderr, spelling
        assert "log_level=debug" in result.stderr, spelling
    for level in ("info", "warn", "error"):
        assert "event=options" not in run(
            load + ["--log-level", level]).stderr, level
    # The queue-budget diagnostic is the first startup line, and it now runs
    # after `--log-level` arms the threshold, so an absolute level drops it
    # with the rest of the boot sequence. It is INFO tier; match the trailing
    # " kind=" so the separate WARN `queue_budget_exceeded` line is not
    # mistaken for it.
    assert "event=queue_budget kind=llm" in run(load).stderr
    for level in ("info", "debug"):
        assert "event=queue_budget kind=llm" in run(
            load + ["--log-level", level]).stderr, level
    for level in ("warn", "error"):
        assert "event=queue_budget kind=" not in run(
            load + ["--log-level", level]).stderr, level
    # -v is shorthand for --log-level=debug, so combining the two is a
    # conflict to report in either order, not a precedence to resolve silently.
    for both in (["--log-level", "info", "-v"], ["-v", "--log-level", "warn"],
                 ["--verbose", "--log-level=debug"]):
        check(load + both, 2, "cannot combine with --log-level")
    # Debug lines must not leak the API key they report as set.
    keyed = run(load + ["-v", "--api-key", "super-secret-value"])
    assert "api_key=set" in keyed.stderr
    assert "super-secret-value" not in keyed.stderr
    for bad in ("trace", "Debug", "", "info,debug"):
        check(["serve", "llm", "--log-level", bad, "--model", "missing.gguf"],
              2, "--log-level must be error, warn, info or debug")
    # --log-progress asks for INFO-tier lines. A quieter threshold would accept
    # the flag and discard every line it produces, so that pairing fails.
    for quiet in ("warn", "error"):
        check(["serve", "llm", "--model", "missing.gguf", "--log-progress",
               "--log-level", quiet], 2, "--log-progress needs --log-level=info")
    # The default tier and the debug shorthand both carry progress lines.
    check(["serve", "llm", "--model", "missing.gguf", "--log-progress"], 1,
          "Error loading model")
    check(["serve", "llm", "--model", "missing.gguf", "--log-progress",
           "--log-level=debug"], 1, "Error loading model")
    check(["serve", "llm", "--model", "missing.gguf", "--log-progress",
           "--log-level=info"], 1, "Error loading model")
    check(["serve", "llm", "--model", "missing.gguf", "--log-progress", "-v"],
          1, "Error loading model")
    # --trace records client content, so it belongs to the text server only,
    # fails before the model opens when the file cannot be written, and the
    # startup log says the file is armed.
    assert "--trace" in help_text
    check(["serve", "tts", "--trace", "trace.jsonl"], 2, "Unknown option")
    check(load + ["--trace", "/nonexistent-gufo-dir/trace.jsonl"], 2,
          "cannot open --trace file")
    with tempfile.TemporaryDirectory() as directory:
        trace = os.path.join(directory, "trace.jsonl")
        result = run(load + ["--trace", trace])
        assert result.returncode == 1, result.stderr
        assert "[WARN] [trace] event=trace_enabled" in result.stderr
        assert stat.S_IMODE(os.stat(trace).st_mode) == 0o600
    # Every help variant groups verbosity the same way: a "Logging:" section,
    # with no hand-written "Server Options:" list to drift from the parser.
    for args in (["serve", "--help"], ["serve", "llm", "--help"],
                 ["serve", "video", "--help"], ["serve", "tts", "--help"]):
        text = check(args, 0, "Logging:")
        assert "Server Options:" not in text, args
        assert "--log-level" in text and "-v, --verbose" in text, args
    check(["serve", "llm", "--log-level=debug", "--model", "missing.gguf"], 1,
          "Error loading model")
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
    # The per-client cap defaults to --max-pending, so lowering the global
    # queue alone is valid. An explicit per-client value is still bounded by it.
    assert ("Maximum queued requests per client IP (default: --max-pending)"
            in help_text)
    for args in (["--max-pending", "2"], ["--max-pending", "1"],
                 ["--max-pending", "2", "--max-pending-per-client", "2"],
                 ["--max-pending-per-client", "16"]):
        check(["serve", "llm", "--model", "missing.gguf", *args], 1,
              "Error loading model")
    for args in (["--max-pending", "2", "--max-pending-per-client", "3"],
                 ["--max-pending-per-client", "17"],
                 ["--max-pending-per-client", "0"], ["--max-pending", "0"]):
        check(["serve", "llm", *args], 2,
              "sampling and scheduling limits are invalid")
    for flag, value in (("--temperature", "2.01"), ("--presence-penalty", "2.01"),
                        ("--frequency-penalty", "-2.01")):
        check(["serve", "llm", flag, value], 2,
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
