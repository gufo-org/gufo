#!/usr/bin/env python3
"""E2 synthetic workloads W2-W4 against a production-like server.

W2  subagents: a parent agent plus fresh subagents sharing its system prompt
    and forks sharing its whole history.
W3  multi-user chat: six conversations alternating on two sessions.
W4  restart: a long agent and a paused chat across a server restart, then a
    new subagent sharing the agent's system prompt.

Every request is logged to client.jsonl; the server writes its own log and a
content trace. Thinking is off so the synthetic histories replay exactly.
"""
import argparse
import json
import pathlib
import random
import shutil
import threading
import time
import urllib.request

import serverctl

from paths import REPO


def load_text(patterns):
    parts = []
    for pattern in patterns:
        for path in sorted(REPO.glob(pattern)):
            if path.is_file():
                parts.append(path.read_text(errors="replace"))
    return "\n".join(parts)


CODE = load_text(["src/**/*.cpp", "src/**/*.hpp"])
DOCS = load_text(["docs/**/*.md", "README.md", "tests/functional/*.py"])


class Text:
    """Deterministic, non-repeating slices so prompts never collide by accident."""

    def __init__(self, source, seed):
        self.source = source
        self.offset = random.Random(seed).randrange(len(source) // 2)

    def take(self, chars):
        if self.offset + chars > len(self.source):
            self.offset = 0
        out = self.source[self.offset:self.offset + chars]
        self.offset += chars
        return out


class Client:
    def __init__(self, server, out_path):
        self.server = server
        self.out = open(out_path, "a")
        self.lock = threading.Lock()

    def chat(self, conv, turn, messages, max_tokens):
        body = {"model": self.server.model_id, "messages": messages,
                "max_tokens": max_tokens}
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.server.port}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        started = time.time()
        with urllib.request.urlopen(request, timeout=7200) as response:
            result = json.load(response)
        record = {"conv": conv, "turn": turn, "start": started,
                  "wall_s": round(time.time() - started, 3),
                  "usage": result.get("usage"), "timings": result.get("timings")}
        with self.lock:
            self.out.write(json.dumps(record) + "\n")
            self.out.flush()
        print(f"{conv:>10} t{turn:<3} prompt={record['usage']['prompt_tokens']:>7} "
              f"cached={(record['timings'] or {}).get('cache_n')} "
              f"wall={record['wall_s']}s", flush=True)
        return result["choices"][0]["message"].get("content") or ""


class Conversation:
    def __init__(self, name, messages):
        self.name = name
        self.messages = list(messages)
        self.turn = 0
        self.busy = threading.Lock()

    def step(self, client, next_user, max_tokens):
        """Caller must hold `busy`."""
        reply = client.chat(self.name, self.turn, self.messages, max_tokens)
        self.messages.append({"role": "assistant", "content": reply})
        self.messages.append({"role": "user", "content": next_user()})
        self.turn += 1


WORKERS = 2


def run_schedule(client, schedule, workers=None):
    """Runs (conversation, next_user, max_tokens) steps in order on N workers;
    a worker waits while the step's conversation is still running."""
    workers = workers or WORKERS
    queue = list(schedule)
    lock = threading.Lock()

    def worker():
        while True:
            with lock:
                if not queue:
                    return
                index = next((i for i, (conv, _, _) in enumerate(queue)
                              if not conv.busy.locked()), None)
                if index is None:
                    step = None
                else:
                    step = queue.pop(index)
                    # Claim the conversation before releasing the queue lock.
                    step[0].busy.acquire()
            if step is None:
                time.sleep(0.2)
                continue
            conv, next_user, max_tokens = step
            try:
                conv.step(client, next_user, max_tokens)
            finally:
                conv.busy.release()

    threads = [threading.Thread(target=worker) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def tool_output(text, chars, label):
    return f"Tool result ({label}):\n```\n{text.take(chars)}\n```\nContinue the task."


def system_prompt(seed, chars):
    return ("You are a careful coding agent. Use the tool results to answer. "
            "Reference material follows.\n" + Text(DOCS, seed).take(chars))


def w2(client):
    code = Text(CODE, 2)
    system = {"role": "system", "content": system_prompt(20, 20000)}
    parent = Conversation("parent", [system, {"role": "user", "content":
                          "Task: audit the cache code for eviction bugs.\n"
                          + tool_output(code, 6000, "ls")}])
    out = lambda n: (lambda: tool_output(code, n, "read"))
    run_schedule(client, [(parent, out(10000), 96)] * 4, workers=1)
    fresh = [Conversation(f"fresh{i}", [system, {"role": "user", "content":
             f"Subtask {i}: inspect module {i}.\n" + tool_output(code, 4000, "ls")}])
             for i in range(3)]
    schedule = []
    for turn in range(5):
        schedule += [(sub, out(6000), 96) for sub in fresh]
        if turn < 4:
            schedule.append((parent, out(10000), 96))
    run_schedule(client, schedule)
    forks = [Conversation(f"fork{i}", parent.messages[:-1] + [{
             "role": "user", "content": f"Fork {i}: investigate hypothesis {i} in "
             "depth.\n" + tool_output(code, 4000, "grep")}]) for i in range(2)]
    schedule = []
    for turn in range(4):
        schedule += [(fork, out(6000), 96) for fork in forks]
    schedule += [(parent, out(10000), 96)] * 2
    run_schedule(client, schedule)
    run_schedule(client, [(parent, out(10000), 96)] * 3, workers=1)


USERS, STEPS = 6, 60


def w3(client):
    rng = random.Random(7)
    docs = Text(DOCS, 3)
    default_system = {"role": "system", "content": system_prompt(30, 3000)}
    users = []
    for i in range(USERS):
        if i < 3:
            system = default_system
        else:
            system = {"role": "system", "content": system_prompt(31 + i, 5000)}
        users.append(Conversation(f"user{i}", [system, {"role": "user",
                     "content": f"Hi, I am user {i}. " + docs.take(600)}]))

    def message():
        text = f"Question: {docs.take(rng.randrange(200, 600))}"
        if rng.random() < 0.2:
            text += "\nHere is a document:\n" + docs.take(rng.randrange(12000, 26000))
        return text

    weights = ([3, 3, 2, 2, 1, 1] * ((USERS + 5) // 6))[:USERS]
    schedule = []
    for _ in range(STEPS):
        user = rng.choices(users, weights)[0]
        schedule.append((user, message, 200))
    run_schedule(client, schedule)


def w4(client_factory, server_factory):
    code = Text(CODE, 4)
    docs = Text(DOCS, 5)
    system = {"role": "system", "content": system_prompt(40, 20000)}
    agent = Conversation("agent", [system, {"role": "user", "content":
                         "Task: port the snapshot code.\n" + tool_output(code, 8000, "ls")}])
    chat = Conversation("chat", [{"role": "user", "content": docs.take(2000)}])
    out = lambda n: (lambda: tool_output(code, n, "read"))
    msg = lambda: "Follow-up: " + docs.take(8000)
    server = server_factory().start()
    client = client_factory(server)
    schedule = []
    for turn in range(14):
        schedule.append((agent, out(13000), 96))
        if turn % 2 == 0:
            schedule.append((chat, msg, 160))
    run_schedule(client, schedule)
    server.stop()
    server = server_factory().start()
    client = client_factory(server)
    sub = Conversation("subagent", [system, {"role": "user", "content":
                       "Subtask: check the restore path.\n" + tool_output(code, 4000, "ls")}])
    schedule = [(agent, out(13000), 96)] * 6 + [(chat, msg, 160)] * 3 + \
               [(sub, out(6000), 96)] * 3
    run_schedule(client, schedule)
    server.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workload", choices=("w2", "w3", "w4"))
    parser.add_argument("--model", choices=tuple(serverctl.MODELS), required=True)
    parser.add_argument("--port", type=int, default=18432)
    parser.add_argument("--disk-bytes", type=int, default=16 << 30)
    parser.add_argument("--sessions", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--users", type=int, default=6)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--context", default=None)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    global WORKERS, USERS, STEPS
    WORKERS, USERS, STEPS = args.workers, args.users, args.steps
    name = args.workload + (f"-{args.tag}" if args.tag else "")
    out = serverctl.HERE / "results" / "e2" / args.model / name
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    cache = serverctl.HERE / "cache" / f"e2-{args.model}-{name}"
    shutil.rmtree(cache, ignore_errors=True)
    traces = iter(range(100))

    def server_factory():
        server = serverctl.Server(args.model, args.port, out / "server.log",
                                  out / f"trace-{next(traces)}.jsonl", cache,
                                  disk_bytes=args.disk_bytes, think="off",
                                  sessions=args.sessions)
        if args.context:
            index = server.command.index("--context")
            server.command[index + 1] = args.context
        return server

    def client_factory(server):
        return Client(server, out / "client.jsonl")

    try:
        if args.workload == "w4":
            w4(client_factory, server_factory)
        else:
            server = server_factory().start()
            try:
                {"w2": w2, "w3": w3}[args.workload](client_factory(server))
            finally:
                server.stop()
    finally:
        shutil.rmtree(cache, ignore_errors=True)


if __name__ == "__main__":
    main()
