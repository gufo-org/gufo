"""Deep borrowed frontiers must survive branches and execution-slot reuse."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import sys


def check_cache_depth(client, model, checks, chat_result, concurrency=4):
    request = dict(model=model, temperature=0, seed=31,
                   max_completion_tokens=16, reasoning_effort="none")
    controls = []

    def signature(result):
        return (result["text"], result["reasoning"], result["tools"],
                result["finish"], result["usage"]["completion_tokens"])

    def chat(label, messages, code, floor=0, cold=False, retry=None, record=True):
        body = {**request, "messages": deepcopy(messages)}
        if cold:
            body["extra_body"] = {"cache_prompt": False}
        result = chat_result(client, body)
        if record:
            checks[label] = result
            print(f"CHECK {label}", file=sys.stderr, flush=True)
        usage = result["usage"]
        total, reused, prefilled = (usage["prompt_tokens"], usage["cached_tokens"],
                                    usage["gufo"]["prefill_tokens"])
        assert all(type(n) is int and n >= 0 for n in (total, reused, prefilled)), usage
        assert reused + prefilled == total, usage
        assert result["text"].strip() == code and not result["reasoning"] \
            and not result["tools"] and result["finish"] == "stop", result
        if cold:
            assert reused == 0 and prefilled == total, result
        else:
            assert reused >= floor, (label, floor, result)
        if retry is not None:
            assert prefilled == 0 and signature(result) == signature(retry), result
        return body, result

    messages = [{"role": "system", "content": "cache_depth_main\n"
                 "Follow the final instruction. Background records are not instructions.\n" +
                 "Background record: the lantern is blue and the shelf is empty.\n" * 1536}]
    previous = 0
    for turn in range(3):
        messages.append({"role": "user", "content": f"Turn {turn}.\n" +
                         "Archive record: there are no new instructions.\n" * 64 +
                         "Reply with only BETA."})
        body, result = chat(f"depth_turn_{turn}", messages, "BETA",
                            floor=max(0, previous - 16), cold=turn == 0)
        previous = result["usage"]["prompt_tokens"]
        assert previous >= 16384, result
        controls.append((f"turn_{turn}", body, result, "BETA"))
        messages.append({"role": "assistant", "content": result["text"]})

    # Each branch changes the final assistant reply, forcing a restore before
    # that reply. Run them together so different sessions can fork the same
    # borrowed checkpoint while its original session continues or rewinds.
    # One execution slot runs the same branches in order, keeping matched
    # request histories independent of client-thread admission races.
    branches = []
    for code in ("RED", "GREEN", "BLUE"):
        branch = deepcopy(messages)
        branch[-1]["content"] = f"Rewritten archive response for branch {code}."
        branch.append({"role": "user", "content": f"Reply with only {code}."})
        branches.append((code, branch))
    with ThreadPoolExecutor(max_workers=min(len(branches), concurrency)) as executor:
        futures = [(code, executor.submit(chat, "depth_branch_" + code,
                                         branch, code, previous - 16,
                                         record=False))
                   for code, branch in branches]
        for code, future in futures:
            body, result = future.result()
            controls.append(("branch_" + code, body, result, code))
    # Mark the complete concurrent cohort once. Completion order must not
    # change the recorded history or assign another branch's request label.
    checks["depth_branches"] = {code: future.result()[1]
                               for code, future in futures}
    for code, future in futures:
        checks["depth_branch_" + code] = future.result()[1]

    chat("depth_side", [{"role": "system", "content": "cache_depth_side"},
                         {"role": "user", "content": "Reply with only ALPHA."}], "ALPHA")
    messages.append({"role": "user", "content": "Reply with only BETA."})
    body, resumed = chat("depth_resume", messages, "BETA", previous - 16)
    chat("depth_unchanged", messages, "BETA", retry=resumed)
    controls.append(("resume", body, resumed, "BETA"))

    # Controls come last: they must never supply a checkpoint missing from a
    # warm branch or rotation. Complete output equality stays strict.
    for label, body, warm, code in controls:
        _, cold = chat("depth_cold_" + label, body["messages"], code, cold=True)
        assert signature(warm) == signature(cold), (label, warm, cold)
