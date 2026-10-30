#!/usr/bin/env python3
"""Independent packed/box reference using pinned LiquidAI source and GGUF weights.

Requires transformers 5.19.0, torch, gguf-py, and a local snapshot of
LiquidAI/d1-Omni-2B at 431f1d667de5afcc9b0c58b2d9c170a7b38b358b.
GGUF weights are dequantized to F32, then loaded strictly into LiquidAI's
unmodified model classes. Runtime Gufo has no dependency on this tool.
"""
import argparse
import base64
import importlib
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import types
import urllib.request

import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize
from PIL import Image
from transformers import AutoTokenizer


PINNED_FILES = {
    'modeling_lfm3.py': '1617c42a7e3ccb97c810867536a8f6bb71f3543db00cafda9b3bd7a571ec72b9',
    'configuration_lfm3.py': 'cd9e8499d5e764e058c7aa1bb62234a0592acdf086126bea36770eb9e45e48cf',
    'modeling_lfm3_vl.py': 'c61dd6d90425d126de8402a9aa0a2a9d0241910100b719b8ab2bfee65e5e974f',
    'configuration_lfm3_vl.py': 'd65e130693c7415a0b20b527e2c8df5322112cd8e7e4b3793348a2ddcd0343be',
    'image_processing_lfm3_vl.py': '4f319e51f80051bce417db52df568a94ff3ea9a4527e1e0844f8ed3e21934b4b',
    'attention.py': '2608cc3f48bc3f906b64244322df81de5a18cd1faa8243680713289d5540fd2c',
    'prompt.py': 'e9d40f98c36ab68b8e3c33fd00a77e58218460fdc88c96b5edeef793a260d20b',
    'box.py': '1a123160ce3c94c25074132407c41d553601a6d8cb7056c814397ad0718dd16f',
    'grid.py': 'a5626b67094eda0a4a22d50e73346f3bc2f1b4b6eb6d95ba64333c6f71592eff',
    'api.py': 'd57845b414b6766cd0410f0437080244320adebd4f9f031a03777acbec243d2a',
    'calibration.py': '04243c3a2c1c727440184901dd60a4b4b04f72204dc577bd9f30b0ea3373d114',
    'media.py': '1b2c76ba9fa13e75bf8b66caa9294682e76fbf463fe56c13b4e1f47c96e81e8e',
    'calibration.json': '182270f679ee9b8ead5d2d3f37ea5f297839126baaced22e4cd57d48cfaf1772',
    'config.json': 'cebf53b6bb8835cefe9b415d9619c23398ab8326d6385ee80f98fef0caad7169',
    'tokenizer.json': '41927bd314d17d7d7fadb44e10cd0919f7b71528d7741596d8cbd15c6945ec44',
}


def load_weights(module, reader, mapper, device):
    tensors = {t.name: t for t in reader.tensors}

    def get(name):
        t = tensors[name]
        return dequantize(t.data, t.tensor_type).astype(np.float32).reshape(tuple(int(d) for d in t.shape[::-1]))

    state = {}
    for name, parameter in module.state_dict().items():
        mapped = mapper(name)
        if isinstance(mapped, tuple):
            first, second = (get(n).reshape(1024, 3, 16, 16) for n in mapped)
            values = np.stack([first, second], axis=2)
        else:
            values = get(mapped)
        state[name] = torch.from_numpy(values.copy().reshape(parameter.shape))
    module.load_state_dict(state, strict=True, assign=True)
    return module.eval().to(device)


def text_name(name):
    if name in ("model.embed_tokens.weight", "lm_head.weight"):
        return "token_embd.weight"
    if name == "model.embedding_norm.weight":
        return "output_norm.weight"
    m = re.fullmatch(r"model.layers.(\d+).(.+)", name)
    if not m:
        raise ValueError(name)
    fields = {"operator_norm": "attn_norm", "ffn_norm": "ffn_norm",
              "post_operator_norm": "post_attention_norm", "post_ffn_norm": "post_ffw_norm",
              "feed_forward.w1": "ffn_gate", "feed_forward.w3": "ffn_up", "feed_forward.w2": "ffn_down",
              "conv.conv": "shortconv.conv", "conv.in_proj": "shortconv.in_proj", "conv.out_proj": "shortconv.out_proj",
              "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
              "self_attn.o_proj": "attn_output", "self_attn.q_norm": "attn_q_norm", "self_attn.k_norm": "attn_k_norm",
              "self_attn.gate_proj": "attn_gate"}
    return f"blk.{m[1]}.{fields[m[2].removesuffix('.weight')]}.weight"


def vision_name(name):
    if name == "patchifier.weight":
        return ("v.patch_embd.weight", "v.patch_embd.weight.1")
    if name == "embedding_norm.weight":
        return "v.embedding_norm.weight"
    m = re.fullmatch(r"layers.(\d+).(.+)", name)
    if not m:
        raise ValueError(name)
    fields = {"operator_norm": "ln1", "ffn_norm": "ln2",
              "feed_forward.w1": "ffn_gate", "feed_forward.w3": "ffn_up", "feed_forward.w2": "ffn_down",
              "conv.in_proj": "conv2d_in_proj", "conv.spatial_conv": "conv2d_spatial", "conv.out_proj": "conv2d_out_proj",
              "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
              "self_attn.out_proj": "attn_out", "self_attn.q_layernorm": "attn_q_norm", "self_attn.k_layernorm": "attn_k_norm"}
    return f"v.blk.{m[1]}.{fields[m[2].removesuffix('.weight')]}.weight"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mmproj", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--candidate", default="http://127.0.0.1:18886")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    for name, expected in PINNED_FILES.items():
        if hashlib.sha256((args.reference_dir / name).read_bytes()).hexdigest() != expected:
            raise ValueError("reference file differs from the pinned revision: " + name)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    sys.path.insert(0, str(args.reference_dir.parent.resolve()))
    package = args.reference_dir.name
    def mod(name):
        return importlib.import_module(package + "." + name)
    attention, prompt, api = mod("attention"), mod("prompt"), mod("api")
    calibration = mod("calibration").Calibration.load(args.reference_dir / "calibration.json")
    recorded = json.loads((args.reference_dir / "config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.reference_dir, trust_remote_code=True)
    tokenizer.d1_codes = 2
    tokenizer.d1_numbers = True
    tokenizer.d1_prompt = 1
    lm, config = mod("modeling_lfm3"), mod("configuration_lfm3")
    cfg = config.Lfm3Config(**recorded["text_config"])
    cfg._attn_implementation = "d1"
    with torch.device("meta"):
        model = lm.Lfm3ForCausalLM(cfg)
    model.model.rotary_emb = lm.Lfm3RotaryEmbedding(config=cfg)
    print("loading reference text weights", flush=True)
    model = load_weights(model, GGUFReader(str(args.model)), text_name, args.device)
    model.lm_head.weight = model.model.embed_tokens.weight
    vl, vc = mod("modeling_lfm3_vl"), mod("configuration_lfm3_vl")
    vision_cfg = vc.Lfm3VlVisionConfig(**recorded["vision_config"])
    vision_cfg._attn_implementation = "sdpa"
    with torch.device("meta"):
        vision = vl.Lfm3VlVisionModel(vision_cfg)
        projector = vl.Lfm3VlMultiModalProjector(types.SimpleNamespace(
            downsample_factor=2, vision_config=vision_cfg, projector_layer_norm_eps=1e-6,
            projector_hidden_size=4096, projector_bias=True, projector_hidden_act="gelu", text_config=cfg))
    vision.rotary_pos_emb = vl.Lfm3VlVisionRotaryEmbedding(vision_cfg)
    mm = GGUFReader(str(args.mmproj))
    print("loading reference vision weights", flush=True)
    vision = load_weights(vision, mm, vision_name, args.device)
    names = {"layer_norm.weight": "mm.input_norm.weight", "layer_norm.bias": "mm.input_norm.bias",
             "linear_1.weight": "mm.0.weight", "linear_1.bias": "mm.0.bias",
             "linear_2.weight": "mm.2.weight", "linear_2.bias": "mm.2.bias"}
    projector = load_weights(projector, mm, lambda name: names[name], args.device)
    processor = mod("image_processing_lfm3_vl").Lfm3VlImageProcessor(min_pixels=65536, max_pixels=1638400)
    image = mod("media").cap_pixels(Image.open(args.image), 1638400)
    features = processor(images=[image], return_tensors="pt")
    with torch.inference_mode():
        image_rows = projector(vision(pixel_values=features["pixel_values"].to(args.device),
                                      grid_thw=features["image_grid_thw"].to(args.device)).last_hidden_state)
    image_token = tokenizer.convert_tokens_to_ids("<|image|>")
    markup = "<|vision_start|>" + "<|image|>" * image_rows.shape[0] + "<|vision_end|>"
    image_uri = "data:image/png;base64," + base64.b64encode(args.image.read_bytes()).decode()
    binary = {"type": "noul", "instructions": "Is the customer asking for a refund?"}
    primitive = {"refund": binary,
                 "team": {"type": "choice", "instructions": "Which team should handle this?",
                          "criteria": {"billing": "Charges and refunds", "shipping": "Delivery", "technical": "App faults"}},
                 "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["Can wait", "Today", "Now"]}}
    remote = {"type": "box", "instructions": "Where is the remote control on the left?"}
    cases = [
        ("packed-text", "I was charged twice. Please refund one charge.", primitive, 16, False),
        ("single-control", "I was charged twice. Please refund one charge.", primitive, 1, False),
        ("packed-two", "I was charged twice. Please refund one charge.", primitive, 2, False),
        ("ragged-seventeen", "Please refund my duplicate charge.", {"q" + str(i): binary for i in range(17)}, 16, False),
        ("single-box", None, {"remote": remote}, 16, True),
        ("boxes-one", None, {"cats": {"type": "boxes", "instructions": "Every cat.", "max": 1}}, 16, True),
        ("boxes-four", None, {"cats": {"type": "boxes", "instructions": "Every cat.", "max": 4}}, 16, True),
        ("boxes-sixteen", None, {"cats": {"type": "boxes", "instructions": "Every cat."}}, 16, True),
        ("absent-box", None, {"elephant": {"type": "box", "instructions": "Where is the elephant?"}}, 16, True),
        ("mixed-slots", None, {"remote": remote, "cats": {"type": "boxes", "instructions": "Every cat.", "max": 4},
                                "present": {"type": "noul", "instructions": "Are there cats in the picture?"}}, 16, True),
        ("unicode-box", {"message": "Les chats et la télécommande. €42"}, {"remote": remote}, 16, True),
    ]
    plan, records, errors = [], [], []
    for name, state, raw, set_size, visual in cases:
        if args.only and name not in args.only:
            continue
        print("reference", name, flush=True)
        qs = [prompt.as_question(q) for q in raw.values()]
        suffixes, offsets = zip(*(prompt.suffix_layout(tokenizer, q, "<think></think>") for q in qs))
        prefix = prompt.prefix_text(tokenizer, state, "<|startoftext|>", media=markup if visual else "")
        answers, usage = {}, 0
        for first in range(0, len(qs), set_size):
            own = qs[first:first + set_size]
            ids, state_len, groups, ends = attention.pack_set(tokenizer, prefix, list(suffixes[first:first + set_size]),
                prompt.question_header(tokenizer, state), list(offsets[first:first + set_size]))
            slots = [slot for positions in ends for slot in positions]
            readouts = [prompt.readout_ids(tokenizer, slot) for q in own for slot in prompt.question_slots(q, tokenizer)]
            vocabulary = sorted({token for table in readouts for forms in table for token in forms})
            tokens = torch.tensor([ids], device=args.device)
            own_name = name + "-" + str(first)
            with torch.inference_mode():
                inputs = model.model.embed_tokens(tokens)
                if visual:
                    inputs[0, tokens[0] == image_token] = image_rows
                embeddings = args.output / (own_name + ".embeddings.bin")
                inputs.cpu().numpy().tofile(embeddings)
                with attention.state_attention(model, torch.tensor([state_len]), torch.tensor([groups])):
                    hidden = model.model(inputs_embeds=inputs, attention_mask=torch.ones_like(tokens), use_cache=False).last_hidden_state
                logits = torch.nn.functional.linear(hidden[0, slots], model.lm_head.weight[vocabulary]).cpu().numpy()
            reference_file = args.output / (own_name + ".reference.bin")
            logits.tofile(reference_file)
            native_file = args.output / (own_name + ".native.bin")
            plan.append({"name": own_name, "prefix": prefix, "header": prompt.question_header(tokenizer, state),
                         "questions": dict(list(raw.items())[first:first + set_size]), "tokens": ids, "groups": groups, "slots": slots,
                         "vocabulary": vocabulary, "embeddings": str(embeddings.resolve()), "output": str(native_file.resolve()),
                         "reference": str(reference_file.resolve())})
            at = 0
            column = {token: i for i, token in enumerate(vocabulary)}
            for key, q in zip(list(raw)[first:first + set_size], own):
                ps = []
                for slot in prompt.question_slots(q, tokenizer):
                    row = logits[at]; at += 1
                    table = prompt.readout_ids(tokenizer, slot)
                    scores = [max(float(row[column[t]]) for t in forms) for forms in table]
                    ps.append(prompt.probabilities(slot, scores, None if prompt.typed(q) else calibration))
                answers[key] = api.answer(q, ps if prompt.typed(q) else ps[0], calibration=calibration)
            usage += len(ids)
        request = {"state": state, "questions": raw, "set_size": set_size}
        if visual:
            request["images"] = [image_uri]
        start = time.perf_counter()
        with urllib.request.urlopen(urllib.request.Request(args.candidate + "/v1/systemone",
                data=json.dumps(request, ensure_ascii=False).encode(), headers={"Content-Type": "application/json"}), timeout=180) as reply:
            candidate = json.load(reply)
        records.append({"name": name, "wall_ms": (time.perf_counter() - start) * 1000,
                        "reference": {"answers": answers, "usage": {"input_tokens": usage, "output_tokens": 0}}, "candidate": candidate})
    plan_file = args.output / "plan.json"
    plan_file.write_text(json.dumps(plan, ensure_ascii=False) + "\n")
    subprocess.run([str(args.probe.resolve()), str(args.model.resolve()), str(plan_file.resolve())], check=True)
    for item in plan:
        reference, native = (np.fromfile(item[key], dtype=np.float32) for key in ["reference", "output"])
        maximum = float(np.max(np.abs(reference - native)))
        item["max_logit_error"] = maximum
        if maximum > .002:
            errors.append(f"{item['name']}: logit error {maximum} exceeds 0.002")
    def compare(a, b, path=""):
        if isinstance(b, dict):
            for k, v in b.items():
                if k not in a:
                    errors.append(path + "/" + k + ": missing")
                else:
                    compare(a[k], v, path + "/" + k)
        elif isinstance(b, list):
            if len(a) != len(b):
                errors.append(path + ": length differs")
            else:
                for i, v in enumerate(b): compare(a[i], v, path + "/" + str(i))
        elif isinstance(b, float):
            limit = .002 if path.endswith(("/top", "/bottom", "/left", "/right")) else .02
            if not isinstance(a, (int, float)) or abs(a - b) > limit:
                errors.append(f"{path}: {a} vs {b} exceeds {limit}")
        elif a != b:
            errors.append(f"{path}: {a} != {b}")
    for record in records:
        compare(record["candidate"], record["reference"], record["name"])
    result = {"reference_revision": "431f1d667de5afcc9b0c58b2d9c170a7b38b358b", "records": records,
              "logits": [{k: v for k, v in p.items() if k in ["name", "max_logit_error"]} for p in plan], "errors": errors}
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    for error in errors: print("FAIL", error)
    print("PASS" if not errors else "FAIL", len(records), "reference cases")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
