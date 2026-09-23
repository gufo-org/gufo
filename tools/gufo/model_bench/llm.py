"""LLM category: loading, single-user depth sweeps, concurrency, memory."""

from __future__ import annotations

import base64
import datetime as dt
import functools
import hashlib
import json
import struct
import zlib
import random
import shutil
import statistics
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

from gufo.serving_bench import (
    PromptCase,
    RequestObservation,
    load_prompt_suite,
    load_reference_report,
    run_corpus_benchmark,
    run_request,
)

from .artifacts import artifact_path, load_artifact, merge_rows, new_artifact, public_command, save_artifact
from .config import BenchConfig, TableSpec
from .servers import HipMemory, Server, drop_file_cache, wait_process_exit

print = functools.partial(print, flush=True)  # progress must reach redirected logs immediately

MODEL_ALIAS = "bench"
CLIENT_ID = "model-bench"
REQUEST_TIMEOUT = 3600.0

WORDS = (
    "the river bends past a quiet town where old mills once ground wheat for "
    "every family along the valley and children still climb the stone bridge "
    "to watch boats carry timber salt and wool toward distant markets while "
    "farmers mend fences count sheep and argue about rain clouds that gather "
    "over the western hills each autumn evening before the harvest festival "
    "brings music lanterns and long tables of bread cheese apples and cider "
    "shared by neighbours who remember older winters when snow closed the "
    "road for weeks and stories were traded by firelight instead of coins"
).split()


class Session:
    """Shared run state: config, target, server factory and identities."""

    def __init__(
        self,
        config: BenchConfig,
        target: str,
        *,
        gufo_binary: Path,
        reference_binary: str,
        source: dict[str, Any],
        fingerprint: dict[str, Any] | None,
        log_dir: Path,
        document: str,
        todo_only: bool,
        drop_caches: str | None = None,
        repetitions: int | None = None,
        fresh: bool = False,
        depths: list[int] | None = None,
        modes: list[str] | None = None,
        context: int | None = None,
    ):
        self.config = config
        self.target = target
        self.gufo_binary = gufo_binary
        self.reference_binary = reference_binary
        self.source = source
        self.fingerprint = fingerprint
        self.log_dir = log_dir
        self.document = document
        self.todo_only = todo_only
        self.drop_caches = drop_caches
        self.repetitions = repetitions
        self.fresh = fresh
        self.depths = depths
        self.modes = modes
        self.context = context
        self.tokenizer_calibration: dict[str | None, tuple[int, float]] = {}

    def reps(self, spec: dict[str, Any], default: int = 1) -> int:
        return self.repetitions or int(spec.get("repetitions", default))

    @property
    def profile(self) -> str:
        return "gufo" if self.target == "gufo" else self.config.data["reference"]["endpoint_profile"]

    @property
    def cache_prompt(self) -> bool | None:
        return None if self.target == "gufo" else True

    @property
    def reference_kind(self) -> str:
        return self.config.data["reference"].get("kind", "llama.cpp")

    # ----- server commands -------------------------------------------------

    def gufo_command(self, table: TableSpec, *, mode: str | None, context: int, sessions: int, port: int) -> list[str]:
        cfg = self.config
        gguf = cfg.file("gguf", table.variant)
        llm_args = list(cfg.data["gufo"].get("llm", []))
        command = [str(self.gufo_binary), "serve", "--port", str(port), "--sessions", str(sessions),
                   *cfg.data["gufo"].get("serve", []), "llm", "--model", str(gguf),
                   "--context", str(context), "--served-model-name", MODEL_ALIAS, *llm_args]
        if table.kind == "image-encoder":
            command += ["--mmproj", str(cfg.file("mmproj", table.variant)), "--max-request-bytes", str(64 << 20)]
        if mode and mode != "ar":
            command += cfg.substitute(cfg.speculative["gufo_args"], table.variant)
        return command

    def reference_binary_for(self, mode: str | None) -> str:
        """The reference executable; a speculative mode may name its own build."""
        if mode and mode != "ar":
            override = self.config.speculative.get("reference", {}).get("server")
            if override:
                return override
        return self.reference_binary

    def reference_command(self, table: TableSpec, *, mode: str | None, context: int, parallel: int, port: int) -> list[str]:
        cfg = self.config
        gguf = cfg.file("gguf", table.variant)
        args = list(cfg.data["reference"]["args"])
        if self.reference_kind == "ds4":
            if parallel > 1 and mode and mode != "ar":
                raise RuntimeError("the pinned ds4 ROCm server disables DSpark in native session batching")
            # The caller supplies total reference context; ds4 allocates per session.
            command = [self.reference_binary_for(mode), "--model", str(gguf),
                       "--ctx", str(context // parallel),
                       "--port", str(port), "--host", "127.0.0.1", *args]
            # Even --batched-session 1 disables DSpark in this upstream pin.
            if parallel > 1:
                command += ["--batched-session", str(parallel)]
        else:
            command = [self.reference_binary_for(mode), "-m", str(gguf), "-c", str(context), "-np", str(parallel),
                       "--port", str(port), "--host", "127.0.0.1", "--alias", MODEL_ALIAS, *args]
        if table.kind == "image-encoder":
            command += ["--mmproj", str(cfg.file("mmproj", table.variant))]
        if mode and mode != "ar":
            args = cfg.reference_speculative
            if args is None:
                raise RuntimeError(f"{cfg.reference_name} has no configured {mode} mode")
            command += cfg.substitute(args, table.variant)
        return command

    def server(self, table: TableSpec, *, mode: str | None, context: int, sessions: int, tag: str) -> Server:
        readiness = self.config.data["gufo"]["readiness"] if self.target == "gufo" else self.config.data["reference"]["readiness"]
        log = self.log_dir / f"{table.id}-{self.target}-{tag}.log"
        placeholder = Server([], readiness, log)
        if self.target == "gufo":
            command = self.gufo_command(table, mode=mode, context=context, sessions=sessions, port=placeholder.port)
        else:
            command = self.reference_command(table, mode=mode, context=context, parallel=sessions, port=placeholder.port)
            if shutil.which(command[0]) is None:
                raise RuntimeError(
                    f"{command[0]} not on PATH; select its optional Nix reference package "
                    "(see docs/BENCHMARKS.md)"
                )
            self.reference_version(mode)
        placeholder.command = command
        return placeholder

    def reference_version(self, mode: str | None = None) -> str | None:
        if self.target != "reference":
            return None
        binary = self.reference_binary_for(mode)
        cache = self.__dict__.setdefault("_reference_versions", {})
        if binary not in cache:
            if self.reference_kind == "ds4":
                executable = Path(shutil.which(binary) or binary).resolve()
                revision = (executable.parent.parent / "share" / "ds4-revision").read_text().strip()
                expected = self.config.data["reference"]["revision"]
                if revision != expected:
                    raise RuntimeError(f"ds4 reference revision {revision} differs from configured {expected}")
                cache[binary] = f"ds4 {revision}"
                return cache[binary]
            completed = subprocess.run([binary, "--version"], capture_output=True, text=True)
            lines = (completed.stdout + completed.stderr).splitlines()
            versions = [line.strip() for line in lines if line.strip().startswith("version:")]
            cache[binary] = f"{Path(binary).name} {versions[0] if versions else 'version unknown'}"
        return cache[binary]

    def artifact(self, table: TableSpec, *, mode: str | None, command: list[str], notes: list[str]) -> dict[str, Any]:
        version = self.reference_version(mode)
        if version:
            notes = [f"{self.config.reference_name}: {version}", *notes]
        return new_artifact(self.config, table, self.target, mode=mode, command=command,
                            source=self.source, fingerprint=self.fingerprint, notes=notes)

    def store(self, path: Path, artifact: dict[str, Any]) -> None:
        existing = None if self.fresh else load_artifact(path)
        save_artifact(path, merge_rows(existing, artifact))
        print(f"artifact: {path}")

    def request(self, base_url: str, prompt: str, max_tokens: int, *, index: int = 0,
                messages: list[dict[str, Any]] | None = None,
                cache_prompt: bool | None = "default",  # type: ignore[assignment]
                extra_body: dict[str, Any] | None = None) -> RequestObservation:
        if self.target == "reference" and self.reference_kind == "ds4":
            extra_body = {"thinking": {"type": "disabled"}, **(extra_body or {})}
        return run_request(
            base_url=base_url, model=MODEL_ALIAS, prompt=prompt, max_tokens=max_tokens,
            temperature=float(self.config.data["sampling"]["temperature"]),
            timeout_seconds=REQUEST_TIMEOUT, client_id=CLIENT_ID, concurrency=1,
            repetition=1, request_index=index, endpoint_profile=self.profile,
            cache_prompt=self.cache_prompt if cache_prompt == "default" else cache_prompt, messages=messages,
            extra_body=extra_body,
        )

    def chat_text(self, base_url: str, messages: list[dict[str, str]], max_tokens: int,
                  extra_body: dict[str, Any] | None = None) -> str:
        """Non-streaming completion text, used to build a reusable conversation prefix."""
        payload = {"model": MODEL_ALIAS, "messages": messages, "max_tokens": max_tokens,
                   "temperature": float(self.config.data["sampling"]["temperature"]), "stream": False}
        if self.target == "reference" and self.reference_kind == "ds4":
            payload["thinking"] = {"type": "disabled"}
        if self.cache_prompt is not None:
            payload["cache_prompt"] = self.cache_prompt
        if extra_body:
            payload.update(extra_body)
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            base_url.rstrip("/") + "/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json", "X-Client-ID": CLIENT_ID}, method="POST")
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            reply = json.loads(response.read().decode("utf-8"))
        content = reply["choices"][0]["message"].get("content")
        return content if isinstance(content, str) else ""


# ----- synthetic prompts ---------------------------------------------------


def synthetic_text(seed: int, words: int) -> str:
    """Deterministic prose-like text with a fixed vocabulary."""
    rng = random.Random(seed)
    out: list[str] = []
    count = 0
    while count < words:
        sentence_len = min(rng.randint(6, 14), words - count)
        sentence = [rng.choice(WORDS) for _ in range(sentence_len)]
        sentence[0] = sentence[0].capitalize()
        out.append(" ".join(sentence) + ".")
        count += sentence_len
        if rng.random() < 0.2:
            out.append("\n\n")
    return " ".join(out).replace(" \n\n ", "\n\n").strip()


class Tokenizer:
    """Estimates token counts for synthetic text from server-reported usage."""

    def __init__(self, session: Session, base_url: str, variant: str | None = None):
        self.session = session
        self.base_url = base_url
        if variant in session.tokenizer_calibration:
            self.overhead, self.ratio = session.tokenizer_calibration[variant]
            return
        overhead = session.request(base_url, "Hi", 1).prompt_tokens - 1
        probe_words = 3000
        probe = session.request(base_url, synthetic_text(7777, probe_words), 1).prompt_tokens
        self.overhead = overhead
        self.ratio = (probe - overhead) / probe_words
        session.tokenizer_calibration[variant] = (self.overhead, self.ratio)

    def words_for(self, tokens: int) -> int:
        return max(1, round(tokens / self.ratio))


def _tolerance(target: int, fraction: float) -> int:
    return max(32, int(target * fraction))


def _mean_sd(values: list[float]) -> tuple[float, float | None]:
    if len(values) < 2:
        return values[0], None
    return statistics.fmean(values), statistics.stdev(values)


# ----- tables ----------------------------------------------------------------


def _selected_rows(session: Session, table: TableSpec, labels: dict[Any, str],
                   display_table: TableSpec | None = None) -> list[Any]:
    """Row keys to measure, honouring --todo against the current document."""
    if not session.todo_only:
        return list(labels)
    from .render import todo_rows

    todo = todo_rows(session.config, session.document, display_table or table, session.target,
                     table.spec["label"] if display_table else None)
    if todo is None:
        return list(labels)
    return [key for key, label in labels.items() if label in todo]


def run_loading(session: Session, table: TableSpec) -> None:
    cfg = session.config
    spec = table.spec
    keys = _selected_rows(session, table, {v: cfg.variant_label(v) for v in cfg.variants})
    if not keys:
        print(f"{table.id}: nothing to do")
        return
    rows: dict[str, Any] = {}
    command: list[str] = []
    for variant in keys:
        sub = TableSpec(table.id, table.base, variant, spec)
        cfg.require_files(variant)
        samples: list[float] = []
        for repetition in range(session.reps(spec)):
            try:
                drop_file_cache(session.drop_caches)
            except SystemExit as reason:
                print(f"{table.id}: skipped; {reason}")
                return
            # Load with the speculative support files on both sides when the reference has the mode.
            mode = cfg.speculative["mode"] if (session.target == "gufo" or cfg.reference_speculative) else None
            if session.modes and mode not in session.modes:
                mode = "ar"
            sessions = int(spec.get("sessions", 1))
            context = session.context or int(spec["context"])
            server = session.server(sub, mode=mode,
                                    context=context if session.target == "gufo" else context * sessions,
                                    sessions=sessions, tag=f"{variant}-{repetition}")
            command = server.command
            with server:
                assert server.ready_seconds is not None
                samples.append(server.ready_seconds)
            wait_process_exit(server)
        mean, sd = _mean_sd(samples)
        rows[variant] = {"ready_s": round(mean, 3), "ready_s_sd": None if sd is None else round(sd, 3),
                         "samples": len(samples), "command": " ".join(public_command(command))}
        print(f"{table.id} {variant}: ready {mean:.2f} s")
    artifact = session.artifact(table, mode=mode, command=command,
                                notes=["file cache reset before each launch with " +
                                       ("the supplied --drop-caches command" if session.drop_caches else
                                        "`echo 3 > /proc/sys/vm/drop_caches`"),
                                       f"readiness = HTTP 200 on {cfg.data['gufo' if session.target == 'gufo' else 'reference']['readiness']}"])
    artifact["rows"] = rows
    session.store(artifact_path(cfg, table, session.target), artifact)


def run_single(session: Session, table: TableSpec, display_table: TableSpec | None = None) -> None:
    workloads = table.workload_tables()
    if workloads:
        for workload in workloads:
            run_single(session, workload, display_table=table)
        return
    cfg = session.config
    spec = table.spec
    if session.modes and ("ar" if not table.speculative else cfg.speculative["mode"]) not in session.modes:
        print(f"{table.id}: skipped by --mode")
        return
    if table.speculative and session.target != "gufo" and cfg.reference_speculative is None:
        print(f"{table.id}: {cfg.reference_name} has no {cfg.speculative['label']} mode; the table compares against its AR column")
        return
    cfg.require_files(table.variant)
    depths = [int(d) for d in spec["depths"]]
    if session.depths:
        depths = [d for d in depths if d in session.depths]
    keys = _selected_rows(session, table, {d: f"{d:,}" for d in depths}, display_table)
    if not keys:
        print(f"{table.id}: nothing to do")
        return
    prompt_tokens = int(spec["prompt_tokens"])
    output_tokens = int(spec["output_tokens"])
    fraction = float(spec.get("depth_tolerance", 0.005))
    repetitions = session.reps(spec)
    base_seed = int(spec.get("prefix", {}).get("seed", 1))
    mode = cfg.speculative["mode"] if table.speculative else "ar"

    context = session.context or int(spec["context"])
    server = session.server(table, mode=mode, context=context, sessions=1, tag="single")
    rows: dict[str, Any] = {}
    with server:
        tokenizer = Tokenizer(session, server.base_url, table.variant)
        # Warm kernels and allocations with an untimed full-size request.
        session.request(server.base_url, synthetic_text(8888, tokenizer.words_for(prompt_tokens)), 16)
        failures: list[str] = []
        for depth in keys:
            pps: list[float] = []
            tgs: list[float] = []
            accepts: list[float] = []
            counts: list[dict[str, Any]] = []
            try:
                observations = [
                    _measure_depth(session, server.base_url, tokenizer, depth=depth,
                                   prompt_tokens=prompt_tokens, output_tokens=output_tokens,
                                   fraction=fraction, seed=base_seed, repetition=repetition,
                                   task=spec.get("workload", "prose"))
                    for repetition in range(repetitions)
                ]
            except (RuntimeError, OSError) as failure:  # OSError: server died mid-request
                exit_code = server.process.poll() if server.process else None
                if exit_code is not None:
                    # The server was killed (SIGKILL = the kernel OOM killer on this host):
                    # this depth and every deeper one are unmeasurable here, not TODO.
                    reason = (f"server exited with {exit_code} while measuring d{depth} "
                              f"(context {context}); {'OOM-killed' if exit_code == -9 else 'crashed'}")
                    for remaining in keys[keys.index(depth):]:
                        rows[str(remaining)] = {"unavailable": reason}
                        print(f"{table.id} d{remaining}: n/a, {reason}")
                    failures.append(reason)
                    break
                print(f"{table.id} d{depth}: FAILED, row left as is: {failure}")
                failures.append(f"d{depth}: {failure}")
                continue
            per_step: list[float] = []
            for observation in observations:
                if observation.prefill_tokens_per_second is not None:
                    pps.append(observation.prefill_tokens_per_second)
                if observation.decode_tokens_per_second is not None:
                    tgs.append(observation.decode_tokens_per_second)
                if observation.draft_acceptance is not None:
                    accepts.append(observation.draft_acceptance * 100.0)
                accepted = observation.draft_accepted_tokens
                # Every verification step yields the accepted draft tokens plus one
                # target token, so steps = completion - accepted.
                if observation.draft_tokens > 0 and observation.completion_tokens > accepted:
                    per_step.append(accepted / (observation.completion_tokens - accepted))
                counts.append({"cache_n": observation.cached_prompt_tokens,
                               "prompt_n": observation.prefill_tokens,
                               "predicted_n": observation.completion_tokens,
                               "draft_n": observation.draft_tokens,
                               "draft_n_accepted": accepted,
                               "completion_sha256": observation.completion_sha256})
            row: dict[str, Any] = {"counts": counts, "samples": repetitions,
                                   "command": " ".join(public_command(server.command))}
            for name, values in (("pp", pps), ("tg", tgs), ("acceptance", accepts), ("accepted_per_step", per_step)):
                if values:
                    mean, sd = _mean_sd(values)
                    row[name] = round(mean, 2)
                    row[f"{name}_sd"] = None if sd is None else round(sd, 2)
            rows[str(depth)] = row
            print(f"{table.id} d{depth}: pp {row.get('pp')} tg {row.get('tg')} accepted/step {row.get('accepted_per_step')}")
    wait_process_exit(server)
    artifact = session.artifact(table, mode=(mode if table.speculative else None), command=server.command, notes=[
        f"pp{prompt_tokens}/tg{output_tokens}; depth is a cached conversation prefix of synthetic text "
        f"({PREFIX_REPLY_TOKENS}-token reply); measured turn task: {spec.get('workload', 'prose')}; "
        f"tolerance max(32, {fraction:.3%}) on cache_n and prompt_n; actual counts per sample in rows",
        f"synthetic text: {tokenizer.ratio:.3f} tokens/word, template overhead {tokenizer.overhead} tokens",
        *[f"not measured, {failure}" for failure in failures],
    ])
    artifact["rows"] = rows
    if rows:
        session.store(artifact_path(cfg, table, session.target), artifact)
    if failures:
        raise SystemExit(f"{table.id}: {len(failures)} depth(s) failed; see the artifact notes")


PREFIX_REPLY_TOKENS = 8
# Asks for natural prose (not a continuation of the synthetic word salad) so the
# generated tokens resemble real use and speculative drafting is representative;
# the length request keeps greedy decoding from stopping at EOS before tg128.
TASKS = {
    # Generic prose: what a drafter sees in ordinary chat.
    "prose": ("\n\nSummarize the passage above in detail, then write a short story inspired by it. "
              "Write at least 500 words."),
    # Fully predictable output: the single-user analogue of the `repetition` corpus.
    "repetition": "\n\nRepeat the passage above word for word, from the beginning.",
    # Reasoning output: the generated tokens are chain-of-thought rather than prose.
    # Thinking is enabled per request so the cached prefix turn renders unchanged.
    "thinking": ("\n\nHow many distinct words appear in the passage above, and which three are "
                 "the most frequent? Work through it carefully before answering."),
}
THINKING_WORKLOAD = "thinking"


def turn_prompt(new_target: int, ratio: float, *, task: str, depth: int = 0,
                repetition: int = 0, attempt: int = 0) -> str:
    """One shared prompt recipe for depth sweeps and pp-matched concurrency."""
    instruction = TASKS[task]
    words = max(1, round(new_target / ratio) - len(instruction.split()))
    return synthetic_text(100_000 + depth * 10 + repetition * 100 + attempt, words) + instruction


def _measure_depth(session: Session, base_url: str, tokenizer: Tokenizer, *, depth: int, prompt_tokens: int,
                   output_tokens: int, fraction: float, seed: int, repetition: int,
                   task: str = "prose") -> RequestObservation:
    """Time pp/tg after a cached conversation prefix of about `depth` tokens.

    Both servers reuse a prior turn's state: the prefix is sent as its own turn
    (a short generated reply), and the measured request continues that
    conversation with a new user turn of about `prompt_tokens` tokens that asks
    for a long natural-prose answer, so greedy decoding does not stop at EOS
    before the requested output length on either server and the generated
    text is representative for speculative drafting.
    """
    new_target = prompt_tokens - tokenizer.overhead
    ratio = tokenizer.ratio
    # The template renders earlier turns differently when thinking is enabled, so the
    # prefix turn must carry the same options for its cached tokens to match.
    extra = ({"chat_template_kwargs": {"enable_thinking": True}, "reasoning_effort": "high"}
             if task == THINKING_WORKLOAD else None)
    for attempt in range(4):
        messages: list[dict[str, str]] = []
        if depth > 0:
            prefix_target = depth - tokenizer.overhead - PREFIX_REPLY_TOKENS
            prefix = synthetic_text(seed + depth, max(1, round(prefix_target / ratio)))
            reply = session.chat_text(base_url, [{"role": "user", "content": prefix}], PREFIX_REPLY_TOKENS,
                                      extra_body=extra)
            messages = [{"role": "user", "content": prefix}, {"role": "assistant", "content": reply}]
        new_text = turn_prompt(new_target, ratio, task=task, depth=depth,
                               repetition=repetition, attempt=attempt)
        messages.append({"role": "user", "content": new_text})
        observation = session.request(base_url, new_text, output_tokens, index=attempt, messages=messages,
                                      extra_body=extra)
        if observation.completion_tokens < output_tokens:
            raise RuntimeError(
                f"depth {depth}: server generated {observation.completion_tokens} of {output_tokens} tokens "
                f"(cache_n {observation.cached_prompt_tokens}, prompt_n {observation.prefill_tokens}); either the "
                "model stopped at EOS despite the continuation request, or prefix + prompt + output exceed the "
                "table's `context`"
            )
        cache_ok = abs(observation.cached_prompt_tokens - depth) <= _tolerance(depth, fraction)
        prefill_ok = abs(observation.prefill_tokens - prompt_tokens) <= _tolerance(prompt_tokens, fraction)
        if cache_ok and prefill_ok:
            return observation
        print(f"  d{depth} attempt {attempt}: cache_n {observation.cached_prompt_tokens} prompt_n "
              f"{observation.prefill_tokens}; recalibrating")
        measured = observation.prefill_tokens + observation.cached_prompt_tokens
        expected_words = sum(len(m["content"].split()) for m in messages if m["role"] == "user")
        ratio = (measured - tokenizer.overhead) / expected_words
        # Long texts tokenize slightly differently from the 3000-word probe; keep the
        # corrected ratio so later (deeper) points do not repeat the miss.
        tokenizer.ratio = ratio
    raise RuntimeError(f"depth {depth}: cached prefix outside tolerance after 4 attempts")


def run_multi(session: Session, table: TableSpec, display_table: TableSpec | None = None) -> None:
    workloads = table.workload_tables()
    if workloads:
        for workload in workloads:
            run_multi(session, workload, display_table=table)
        return
    cfg = session.config
    spec = table.spec
    cfg.require_files(table.variant)
    levels = [int(c) for c in spec["concurrency"]]
    keys = _selected_rows(session, table, {c: str(c) for c in levels}, display_table)
    if not keys:
        print(f"{table.id}: nothing to do")
        return
    matched_prompt = spec.get("prompt_tokens")
    prefill_first = bool(spec.get("prefill_first", False))
    if matched_prompt:
        task = spec["workload"]
        case_ids = {f"synthetic_{task}_pp{matched_prompt}"}
        cases: list[PromptCase] = []
        suite_bytes = b""
    else:
        suite = (cfg.model_dir / "artifacts" / spec["suite"]).resolve()
        cases = load_prompt_suite(suite, selected=set(spec["cases"]))
        case_ids = {case.identifier for case in cases}
        suite_bytes = suite.read_bytes()
    modes = list(spec.get("modes", ["ar"]))
    if session.target != "gufo":
        modes = [m for m in modes if m == "ar" or cfg.reference_speculative is not None]
    if session.modes:
        modes = [m for m in modes if m in session.modes]
    for mode in modes:
        # Reference AR keeps the unsuffixed name; its speculative run is suffixed like Gufo's.
        path = artifact_path(cfg, table, session.target, None if (session.target != "gufo" and mode == "ar") else mode)
        reference = None
        ar_path = artifact_path(cfg, table, "gufo", "ar")
        if path != ar_path and ar_path.exists():
            reference = load_reference_report(ar_path)
        if mode != "ar":
            missing = case_ids - set((reference or {}).get("hashes", {}))
            if missing:
                raise RuntimeError(
                    f"{table.id}: missing isolated AR completion hashes for {', '.join(sorted(missing))}; "
                    f"qualify C1 AR once and save its report to {ar_path}"
                )
        combined = None if session.fresh else load_artifact(path)
        for users in keys:
            if path == ar_path and reference is None and combined is not None and "c1" in combined.get("results", {}):
                # Gufo AR C2+ compares against this same artifact's C1 completions.
                reference = load_reference_report(ar_path) if ar_path.exists() else None
            context = int(spec["context"])
            server = session.server(table, mode=mode, context=context if session.target == "gufo" else context * users,
                                    sessions=users, tag=f"{mode or 'ref'}-c{users}")
            try:
                with server:
                    if matched_prompt:
                        tokenizer = Tokenizer(session, server.base_url, table.variant)
                        prompt = turn_prompt(int(matched_prompt) - tokenizer.overhead,
                                             tokenizer.ratio, task=task)
                        cases = [PromptCase(next(iter(case_ids)), task, prompt)]
                        suite_bytes = json.dumps(
                            {"recipe": "single-user-d0", "task": task, "prompt": prompt},
                            sort_keys=True).encode()
                    report = run_corpus_benchmark(
                        base_url=server.base_url, model=MODEL_ALIAS, cases=cases,
                        workload_id=f"{cfg.model}-{table.id}-{session.target}-{mode}",
                        max_tokens=int(spec["output_tokens"]),
                        temperature=float(cfg.data["sampling"]["temperature"]),
                        concurrency_levels=[users], warmup_rounds=int(spec.get("warmup", 1)),
                        repetitions=session.reps(spec), timeout_seconds=REQUEST_TIMEOUT,
                        fingerprint=session.fingerprint or {}, source_revision=session.source["revision"],
                        source_dirty=session.source["dirty"], suite_bytes=suite_bytes,
                        corpus_layout=spec.get("corpus_layout", "distinct"), endpoint_profile=session.profile,
                        cache_prompt=(True if prefill_first else
                                      (False if not spec.get("cache_prompt", False) else None)),
                        prefill_first=prefill_first,
                        pin_slots=prefill_first and session.target == "reference",
                        reference=reference,
                        notes=[note for note in (
                            session.reference_version(mode), " ".join(public_command(server.command)),
                            "fresh server per concurrency level") if note],
                    )
            except (RuntimeError, OSError):
                exit_code = server.failure_exit_code
                wait_process_exit(server)
                if exit_code is None:
                    raise
                # Killed mid-cohort (SIGKILL = the kernel OOM killer): this level and the
                # larger ones are unmeasurable on this host.
                reason = f"server exited with {exit_code} at C{users}; {'OOM-killed' if exit_code == -9 else 'crashed'}"
                combined = combined or {"artifactType": "model-bench-unavailable", "results": {}}
                for remaining in keys[keys.index(users):]:
                    combined.setdefault("results", {})[f"c{remaining}"] = {"unavailable": reason}
                    print(f"{table.id} {session.target} {mode} C{remaining}: n/a, {reason}")
                combined["modelBench"] = {"table": table.id, "target": session.target, "mode": mode,
                                          "measuredOn": dt.date.today().isoformat()}
                save_artifact(path, combined)
                break
            wait_process_exit(server)
            if matched_prompt:
                result = report["results"][f"c{users}"]
                for sample in result["samples"]:
                    actual = sample["prompt_tokens"] if prefill_first else sample["prefill_tokens"]
                    if abs(actual - int(matched_prompt)) > _tolerance(int(matched_prompt), 0.005):
                        raise RuntimeError(
                            f"{table.id}: expected pp{matched_prompt}, got {actual}; recalibrate the prompt")
                    if sample["completion_tokens"] != int(spec["output_tokens"]):
                        raise RuntimeError(f"{table.id}: incomplete generated output at C{users}")
                if session.target == "gufo" and reference and result["completionExactness"]["exactRate"] != 1.0:
                    raise RuntimeError(f"{table.id}: C{users} output differs from its isolated AR reference")
                report["workload"]["promptGenerator"] = {
                    "recipe": "single-user-d0", "task": task,
                    "requestedTokens": int(matched_prompt),
                    "templateOverhead": tokenizer.overhead,
                    "tokensPerWord": tokenizer.ratio,
                    "promptSha256": hashlib.sha256(prompt.encode()).hexdigest(),
                }
            if combined is None or combined.get("artifactType") != report.get("artifactType"):
                combined = report
            else:
                combined["results"].update(report["results"])
                combined["notes"] = report.get("notes", combined.get("notes"))
            combined["workload"]["concurrency"] = [
                int(key[1:]) for key in combined["results"] if key.startswith("c")]
            combined["modelBench"] = {"table": table.id, "target": session.target, "mode": mode,
                                      "measuredOn": dt.date.today().isoformat()}
            save_artifact(path, combined)
            from .render import _serving_rate

            rate = _serving_rate(report, users)
            formatted = f"{rate:.2f}" if rate is not None else "unavailable"
            print(f"{table.id} {session.target} {mode} C{users}: {formatted} decode tok/s -> {path}")


class MemoryPoller:
    """Samples device-global HIP memory use on a thread and keeps the peak."""

    def __init__(self, hip: HipMemory) -> None:
        self.hip = hip
        self.peak: float | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            used = self.hip.used_gib()
            if used is not None and (self.peak is None or used > self.peak):
                self.peak = used
            time.sleep(0.25)

    def __enter__(self) -> "MemoryPoller":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        self._thread.join()


def run_memory(session: Session, table: TableSpec) -> None:
    cfg = session.config
    spec = table.spec
    cfg.require_files(table.variant)
    workloads = {w["id"]: w for w in spec["workloads"]}
    keys = _selected_rows(session, table, {k: k for k in workloads})
    if not keys:
        print(f"{table.id}: nothing to do")
        return
    hip = HipMemory.for_binary(session.gufo_binary)
    if hip is None:
        raise SystemExit("memory table needs libamdhip64 (resolved through `ldd` of the Gufo binary)")
    idle = hip.used_gib()
    mode = spec.get("mode", "ar")
    server = session.server(table, mode=mode, context=int(spec["context"]), sessions=1, tag="memory")
    rows: dict[str, Any] = {}
    with server:
        tokenizer = Tokenizer(session, server.base_url, table.variant)
        for key in keys:
            workload = workloads[key]
            depth = int(workload["depth"])
            with MemoryPoller(hip) as poller:
                _measure_depth(session, server.base_url, tokenizer, depth=depth,
                               prompt_tokens=int(workload["prompt_tokens"]),
                               output_tokens=int(workload["output_tokens"]), fraction=0.02, seed=1, repetition=0)
            rows[key] = {"gib": None if poller.peak is None else round(poller.peak, 2),
                         "idle_gib": None if idle is None else round(idle, 2),
                         "command": " ".join(public_command(server.command))}
            print(f"{table.id} {key}: {rows[key]['gib']} GiB")
    wait_process_exit(server)
    artifact = session.artifact(table, mode=None, command=server.command,
                                notes=["peak device-global hipMemGetInfo used bytes (total - free), sampled every 250 ms "
                                       "during the request; idle_gib is the same counter before the server started",
                                       f"server mode: {mode}"])
    artifact["rows"] = rows
    session.store(artifact_path(cfg, table, session.target), artifact)


def _png(size: int, seed: int) -> bytes:
    """A deterministic RGB gradient PNG (compresses well, differs per seed)."""
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    rows = bytearray()
    for y in range(size):
        rows.append(0)  # filter: none
        for x in range(size):
            rows += bytes((((x + seed) * 255 // size) & 0xFF, ((y + 3 * seed) * 255 // size) & 0xFF,
                           ((x ^ y) + 7 * seed) & 0xFF))
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(bytes(rows), 6)) + chunk(b"IEND", b"")


def run_image_encoder(session: Session, table: TableSpec) -> None:
    """Prefill time of a request carrying one image: encoder plus the image tokens' prefill.

    Both servers are timed on the same scope (`prompt_ms` of the request); the
    projector's encode alone is not separable over HTTP.
    """
    cfg = session.config
    spec = table.spec
    cfg.require_files(table.variant)
    cfg.file("mmproj", table.variant)
    sizes = [int(v) for v in spec["sizes"]]
    keys = _selected_rows(session, table, {v: f"{v}×{v}" for v in sizes})
    if not keys:
        print(f"{table.id}: nothing to do")
        return
    warmup = int(spec.get("warmup", 1))
    repetitions = session.reps(spec, int(spec.get("repetitions", 3)))
    context = session.context or int(spec.get("context", 8192))
    server = session.server(table, mode="ar", context=context, sessions=1, tag="image")
    rows: dict[str, Any] = {}
    with server:
        for size in keys:
            samples: list[float] = []
            tokens: list[int] = []
            for index in range(warmup + repetitions):
                image = base64.b64encode(_png(size, index)).decode("ascii")
                messages = [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
                    {"type": "text", "text": "Describe this image in one word."}]}]
                observation = session.request(server.base_url, "", 1, index=index, messages=messages, cache_prompt=False)
                if index >= warmup and observation.prefill_ms is not None:
                    samples.append(observation.prefill_ms)
                    tokens.append(observation.prefill_tokens)
            mean, sd = _mean_sd(samples)
            rows[str(size)] = {"ms": round(mean, 1), "ms_sd": None if sd is None else round(sd, 1),
                               "prompt_n": tokens, "samples": len(samples),
                               "command": " ".join(public_command(server.command))}
            print(f"{table.id} {size}×{size}: {mean:.1f} ms prefill ({tokens[0]} prompt tokens)")
    wait_process_exit(server)
    artifact = session.artifact(table, mode=None, command=server.command, notes=[
        "prompt_ms of a request with one gradient PNG and a one-line text turn, max_tokens 1, cache_prompt=false, "
        f"{warmup} warm-up then {repetitions} timed samples per size; includes the projector encode and the "
        "prefill of the image and text tokens on both servers"])
    artifact["rows"] = rows
    session.store(artifact_path(cfg, table, session.target), artifact)


RUNNERS = {
    "loading": run_loading,
    "single": run_single,
    "multi": run_multi,
    "memory": run_memory,
    "image-encoder": run_image_encoder,
}


def run_table(session: Session, table: TableSpec) -> None:
    if (session.target == "reference" and table.kind != "loading"
            and not session.config.data["reference"].get("stage_timings", True)):
        raise RuntimeError(
            f"{session.config.reference_name} does not expose qualified per-request stage timings; "
            "qualify timing and prepared-session reuse before running its throughput/memory tables. "
            "Whole-request wall time cannot substitute for decode time."
        )
    RUNNERS[table.kind](session, table)
