"""Stage oracles for the gufo YuE2 port, from the pinned upstream CPU path.

Run with torch, transformers, safetensors, soundfile and tiktoken, and the
upstream `src` directory on PYTHONPATH. It needs upstream `main` (1116bea was
used): the `yue2-v0.1.6` tag predates `yue2.protocol.make_request`. Run it
from outside the upstream clone:
  PYTHONPATH=U/src python run_official.py <stage> --out DIR --model M --vae V --upstream U
then point yue2_hip_oracle_test at DIR (YUE2_ORACLE_DIR).

Stages:
  tokens  tokenizer encodings of fixed strings (tokens.json)
  ar      short plan + semantic generation, then teacher-forced logits
  nar     latents from the saved tokens and the seeded noise
  vae     audio from the saved latents
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

MODEL = VAE = SONG = EXAMPLES = None


def tokens(out):
    from yue2.tokenization_yue2 import YuE2TextTokenizer
    from yue2.protocol import make_request
    tok = YuE2TextTokenizer(MODEL / "qwen.tiktoken")
    song = json.loads(SONG.read_text(encoding="utf-8"))
    request = make_request(song["style"], song["lyrics"], cot=song["cot"], seed=song["seed"])
    texts = {
        "song_text": request.text(),
        "score_abc": (EXAMPLES / "score.abc").read_text(encoding="utf-8"),
        "melody_abc": (EXAMPLES / "melody.abc").read_text(encoding="utf-8"),
        "jazz_abc": (EXAMPLES / "score-jazz.abc").read_text(encoding="utf-8"),
        "unicode": "Café naïve – 日本語の歌詞\r\n  tabs\tand   spaces 1234567 it's I'LL <|endoftext|>",
        "nfc": "e\u0301 A\u030a",
        "empty": "",
        "newlines": "\n\n\n a\n\n b \n",
    }
    result = {name: {"text": text, "ids": tok.encode(text)} for name, text in texts.items()}
    (out / "tokens.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print({k: len(v["ids"]) for k, v in result.items()})


def load_model():
    from yue2.modeling_yue2 import YuE2ForCausalLM
    torch.set_num_threads(16)
    start = time.perf_counter()
    model = YuE2ForCausalLM.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.bfloat16,
                                            low_cpu_mem_usage=True).eval()
    print(f"model load {time.perf_counter() - start:.1f}s", flush=True)
    return model


def ar(out, abc_tokens, semantic_tokens):
    from yue2.tokenization_yue2 import YuE2TextTokenizer
    from yue2.protocol import make_request, token_prefixes, GenerationConfig, Sampling, CODEC_OFFSET
    from yue2.sampling import generate_tokens
    import dataclasses
    tok = YuE2TextTokenizer(MODEL / "qwen.tiktoken")
    song = json.loads(SONG.read_text(encoding="utf-8"))
    request = make_request(song["style"], song["lyrics"], cot=song["cot"], seed=song["seed"])
    model = load_model()
    config = GenerationConfig()
    abc_sampling = dataclasses.replace(config.abc, max_tokens=abc_tokens, min_tokens=min(config.abc.min_tokens, abc_tokens))
    sem_sampling = dataclasses.replace(config.semantic, max_tokens=semantic_tokens,
                                       min_tokens=min(config.semantic.min_tokens, semantic_tokens))
    plan_prefix = token_prefixes(request, tok)
    start = time.perf_counter()
    abc_ids, abc_timing, abc_truncated = generate_tokens(model, plan_prefix, abc_sampling, request.seed, "abc",
                                                         use_cuda_graph=False)
    print(f"abc {len(abc_ids)} tokens {time.perf_counter() - start:.1f}s truncated={abc_truncated}", flush=True)
    prefix = token_prefixes(request, tok, abc_ids)
    start = time.perf_counter()
    sem_ids, sem_timing, sem_truncated = generate_tokens(model, prefix, sem_sampling, request.seed, "semantic",
                                                         use_cuda_graph=False)
    print(f"semantic {len(sem_ids)} tokens {time.perf_counter() - start:.1f}s truncated={sem_truncated}", flush=True)
    np.save(out / "plan_prefix.npy", np.asarray(plan_prefix, dtype=np.int32))
    np.save(out / "abc_ids.npy", np.asarray(abc_ids, dtype=np.int32))
    np.save(out / "prefix.npy", np.asarray(prefix, dtype=np.int32))
    np.save(out / "semantic.npy", np.asarray([t - CODEC_OFFSET for t in sem_ids], dtype=np.int32))
    (out / "plan.json").write_text(json.dumps({
        "request": request.to_dict(), "abc": tok.decode(abc_ids), "abc_truncated": abc_truncated,
        "semantic_truncated": sem_truncated, "abc_sampling": dataclasses.asdict(abc_sampling),
        "semantic_sampling": dataclasses.asdict(sem_sampling)}, ensure_ascii=False, indent=1), encoding="utf-8")

    # Teacher-forced logits: one causal forward over the whole sequence, rows at
    # the last prefix positions and a spread of generated positions.
    sequence = prefix + sem_ids
    rows = sorted({*range(len(plan_prefix) - 4, len(plan_prefix)),
                   *range(len(prefix) - 4, len(prefix)),
                   *range(len(prefix), len(sequence), max(1, len(sem_ids) // 16)),
                   len(sequence) - 1})
    start = time.perf_counter()
    with torch.inference_mode():
        hidden, _ = model.model(input_ids=torch.tensor([sequence]),
                                position_ids=torch.arange(len(sequence))[None], use_cache=False)
        selected = hidden[0, rows]
        logits = model.lm_head(selected).float().numpy()
    print(f"teacher forcing {len(sequence)} tokens {time.perf_counter() - start:.1f}s", flush=True)
    np.save(out / "tf_sequence.npy", np.asarray(sequence, dtype=np.int32))
    np.save(out / "tf_rows.npy", np.asarray(rows, dtype=np.int32))
    np.save(out / "tf_logits.npy", logits)
    np.save(out / "tf_hidden.npy", selected.float().numpy())


def nar(out, steps):
    from yue2.nar import CachedNAR, song_chunks
    prefix = np.load(out / "prefix.npy").tolist()
    codec = np.load(out / "semantic.npy").tolist()
    seed = json.loads((out / "plan.json").read_text(encoding="utf-8"))["request"]["seed"]
    model = load_model()
    chunks = song_chunks(prefix, codec, seed)
    assert len(chunks) == 1
    np.save(out / "noise.npy", chunks[0].noise.numpy())
    start = time.perf_counter()
    engine = CachedNAR(model, chunks[0])
    print(f"nar prefill {time.perf_counter() - start:.1f}s", flush=True)
    state = chunks[0].noise.to(torch.bfloat16)
    first = engine.velocity(state, 20.0 if steps else 0.0)
    np.save(out / "velocity_t1.npy", first.float().numpy())
    raw_mid = torch.logit(torch.tensor(0.75, dtype=torch.float64)).item()
    np.save(out / "velocity_t075.npy", engine.velocity(state, raw_mid).float().numpy())
    start = time.perf_counter()
    latent = engine.solve(steps, on_progress=lambda c, t: print(f"  step {c}/{t}", flush=True))
    print(f"nar solve {time.perf_counter() - start:.1f}s", flush=True)
    np.save(out / "latent.npy", latent.numpy())


def vae(out):
    from yue2.modeling_vae import YuE2VAE
    torch.set_num_threads(16)
    latent = torch.from_numpy(np.load(out / "latent.npy")).float().T.unsqueeze(0)
    model = YuE2VAE.from_pretrained(VAE, decoder_only=True, device="cpu", local_files_only=True)
    start = time.perf_counter()
    with torch.inference_mode():
        audio = model.decode_tiled(latent, core_frames=1024, halo_frames=16, output_device="cpu")
        full = model.decode(latent)
    print(f"vae {time.perf_counter() - start:.1f}s", flush=True)
    np.save(out / "audio_tiled.npy", audio[0].numpy())
    np.save(out / "audio_full.npy", full[0].numpy())
    import soundfile as sf
    sf.write(out / "audio.wav", audio[0].clamp(-1, 1).T.numpy(), 48000, subtype="FLOAT")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["tokens", "ar", "nar", "vae"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--abc-tokens", type=int, default=4096)
    parser.add_argument("--semantic-tokens", type=int, default=300)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--model", type=Path, required=True, help="YuE2-3B directory")
    parser.add_argument("--vae", type=Path, required=True, help="YuE2-Vae directory")
    parser.add_argument("--upstream", type=Path, required=True,
                        help="YuE checkout (for examples/song.json and the ABC examples)")
    args = parser.parse_args()
    global MODEL, VAE, SONG, EXAMPLES
    MODEL, VAE = args.model, args.vae
    EXAMPLES = args.upstream / "examples"
    SONG = EXAMPLES / "song.json"
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == "tokens":
        tokens(args.out)
    elif args.stage == "ar":
        ar(args.out, args.abc_tokens, args.semantic_tokens)
    elif args.stage == "nar":
        nar(args.out, args.steps)
    else:
        vae(args.out)


if __name__ == "__main__":
    sys.exit(main())
