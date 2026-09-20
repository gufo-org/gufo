"""Dump Qwen3-TTS talker per-layer tensors for byte-exact port validation.

Captures, for talker layer 0 (and a later layer) during prefill:
  - input to self_attn (post input_layernorm)
  - q/k/v pre-rope (post q_norm/k_norm), cos/sin, q/k post-rope
  - attn output (pre o_proj) and post o_proj
  - MLP input (post post_attention_layernorm), gate/up/down outputs
  - layer output, norm output, final logits
Also dumps prefill inputs_embeds, position_ids, attention_mask.
"""
import argparse
import collections
import functools
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import torch

TEXT = (
    "The boy who lived. Mr. and Mrs. Dursley, of number four, Privet Drive, were "
    "proud to say that they were perfectly normal, thank you very much. They were "
    "the last people you'd expect to be involved in anything strange or mysterious, "
    "because they just didn't hold with such nonsense."
)

_OUT = None
CAPTURE = collections.OrderedDict()
_TALKER_FORWARD_ACTIVE = False
_LAYERS_SEEN = set()


def _dump(name, tensor):
    CAPTURE[name] = tensor.detach().cpu().to(torch.float32).numpy()


# ---- patch talker model forward to grab hidden states at layer boundaries ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--reference-root", default="/home/fbozzo/projects/Qwen3-TTS"
    )
    ap.add_argument(
        "--dependency-root",
        help=(
            "site-packages containing the official pinned Qwen dependencies; "
            "Torch and NumPy are imported from the selected interpreter first"
        ),
    )
    ap.add_argument("--model", default="/home/fbozzo/projects/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    ap.add_argument("--out", default="/home/fbozzo/projects/gufo/artifacts/qwen3_tts/internals")
    ap.add_argument("--speaker", default="vivian")
    ap.add_argument("--language", default="english")
    ap.add_argument(
        "--device",
        default="cpu",
        help="official PyTorch device map, for example cpu or cuda:0 on ROCm",
    )
    ap.add_argument("--max-new-tokens", type=int, default=4)
    args = ap.parse_args()

    compat_root = os.path.join(os.path.dirname(__file__), "compat")
    sys.path.insert(0, compat_root)
    next_path = 1
    if args.dependency_root:
        sys.path.insert(next_path, args.dependency_root)
        next_path += 1
    sys.path.insert(next_path, args.reference_root)
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    from qwen_tts.core.models.modeling_qwen3_tts import (
        Qwen3TTSTalkerAttention,
        Qwen3TTSTalkerDecoderLayer,
        Qwen3TTSTalkerForConditionalGeneration,
        Qwen3TTSTalkerModel,
    )

    orig_model_forward = Qwen3TTSTalkerModel.forward
    orig_attn_forward = Qwen3TTSTalkerAttention.forward
    orig_layer_forward = Qwen3TTSTalkerDecoderLayer.forward
    orig_talker_forward = Qwen3TTSTalkerForConditionalGeneration.forward
    layer_index = {"value": 0}
    predictor_first_code = {"value": None}
    predictor_history = {
        "hidden": [], "codes": [], "logits": [], "projection": [],
        "projection_fp64": [],
    }

    @functools.wraps(orig_model_forward)
    def patched_model_forward(self, *positional, **keyword):
        global _TALKER_FORWARD_ACTIVE
        inputs_embeds = keyword.get("inputs_embeds")
        if inputs_embeds is not None:
            if (
                inputs_embeds.shape[1] == 1
                and "cached_step_inputs_embeds" not in CAPTURE
            ):
                _dump("cached_step_inputs_embeds", inputs_embeds[0])
                position_ids = keyword.get("position_ids")
                cache_position = keyword.get("cache_position")
                if position_ids is not None:
                    _dump("cached_step_position_ids", position_ids)
                if cache_position is not None:
                    _dump("cached_step_cache_position", cache_position)
        _TALKER_FORWARD_ACTIVE = True
        try:
            return orig_model_forward(self, *positional, **keyword)
        finally:
            _TALKER_FORWARD_ACTIVE = False

    @functools.wraps(orig_attn_forward)
    def patched_attn_forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        cache_position=None,
        **keyword,
    ):
        result = orig_attn_forward(
            self,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **keyword,
        )
        if (
            _TALKER_FORWARD_ACTIVE
            and self.layer_idx == 0
            and "layer0_attn_out" not in CAPTURE
        ):
            _dump("layer0_attn_hidden_in", hidden_states)
            cos, sin = position_embeddings
            _dump("layer0_attn_cos", cos)
            _dump("layer0_attn_sin", sin)
            _dump("layer0_attn_out", result[0])
        return result

    @functools.wraps(orig_layer_forward)
    def patched_layer_forward(self, hidden_states, **keyword):
        index = layer_index["value"]
        layer_index["value"] += 1
        if _TALKER_FORWARD_ACTIVE and index == 0:
            _dump("layer0_input", hidden_states)
        result = orig_layer_forward(self, hidden_states, **keyword)
        if _TALKER_FORWARD_ACTIVE and index == 0:
            _dump("layer0_output", result[0])
        if _TALKER_FORWARD_ACTIVE and index == 27:
            _dump("layer27_output", result[0])
        return result

    @functools.wraps(orig_talker_forward)
    def patched_talker_forward(self, *positional, **keyword):
        predictor_first_code["value"] = keyword.get("input_ids")
        inputs_embeds = keyword.get("inputs_embeds")
        position_ids = keyword.get("position_ids")
        attention_mask = keyword.get("attention_mask")
        if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
            if "prefill_inputs_embeds" not in CAPTURE:
                _dump("prefill_inputs_embeds", inputs_embeds[0])
                if position_ids is not None:
                    _dump("prefill_position_ids", position_ids)
                if attention_mask is not None:
                    _dump("prefill_attention_mask", attention_mask)
        out = orig_talker_forward(self, *positional, **keyword)
        if (
            inputs_embeds is not None
            and inputs_embeds.shape[1] > 1
            and "prefill_logits_lastpos" not in CAPTURE
        ):
            _dump("prefill_logits_lastpos", out.logits[0, -1:, :])
            _dump("prefill_past_hidden", out.past_hidden[0])
            _dump("prefill_last_hidden", out.hidden_states[0][-1][0, -1:, :])
        elif (
            keyword.get("input_ids") is not None
            and "cached_step_logits" not in CAPTURE
        ):
            _dump("cached_step_logits", out.logits[0, -1:, :])
            _dump("cached_step_past_hidden", out.past_hidden[0])
        return out

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.model.eval()
    predictor = model.model.talker.code_predictor
    orig_predictor_generate = predictor.generate

    def traced_predictor_generate(*positional, **keyword):
        # Freeze the official input and history, so native checks can distinguish
        # predictor arithmetic from errors propagated by earlier token choices.
        keyword["output_scores"] = True
        result = orig_predictor_generate(*positional, **keyword)
        predictor_history["hidden"].append(
            keyword["inputs_embeds"][0, 0].detach().cpu().float().numpy()
        )
        predictor_history["codes"].append(
            torch.cat((predictor_first_code["value"], result.sequences), dim=-1)
            [0].detach().cpu().float().numpy()
        )
        predictor_history["logits"].append(
            torch.stack(result.scores, dim=1)[0].detach().cpu().float().numpy()
        )
        projection = predictor.small_to_mtp_projection
        inputs = keyword["inputs_embeds"]
        with torch.inference_mode():
            predictor_history["projection"].append(
                projection(inputs)[0].cpu().float().numpy()
            )
            # Distinguish a backend's BF16 rounding boundary from an incorrect
            # implementation. An upstream matmul is not exact arithmetic.
            precise = (
                inputs.cpu().double() @ projection.weight.cpu().double().T
                + projection.bias.cpu().double()
            )
            predictor_history["projection_fp64"].append(
                precise[0].bfloat16().float().numpy()
            )
        return result

    Qwen3TTSTalkerModel.forward = patched_model_forward
    Qwen3TTSTalkerAttention.forward = patched_attn_forward
    Qwen3TTSTalkerDecoderLayer.forward = patched_layer_forward
    Qwen3TTSTalkerForConditionalGeneration.forward = patched_talker_forward
    predictor.generate = traced_predictor_generate
    try:
        model.generate_custom_voice(
            text=TEXT, speaker=args.speaker, language=args.language,
            max_new_tokens=args.max_new_tokens, do_sample=False,
            subtalker_dosample=False,
        )
    finally:
        Qwen3TTSTalkerModel.forward = orig_model_forward
        Qwen3TTSTalkerAttention.forward = orig_attn_forward
        Qwen3TTSTalkerDecoderLayer.forward = orig_layer_forward
        Qwen3TTSTalkerForConditionalGeneration.forward = orig_talker_forward
        predictor.generate = orig_predictor_generate
    for name, rows in predictor_history.items():
        if rows:
            CAPTURE["predictor_history_" + name] = np.stack(rows)

    talker = model.model.talker
    talker_device = next(talker.parameters()).device
    predictor_hidden = (
        torch.from_numpy(CAPTURE["prefill_past_hidden"])
        .unsqueeze(0)
        .to(device=talker_device, dtype=torch.bfloat16)
    )
    first_code = torch.tensor([[1995]], device=talker_device)
    first_code_embedding = talker.get_input_embeddings()(first_code)
    predictor_result = talker.code_predictor.generate(
        inputs_embeds=torch.cat((predictor_hidden, first_code_embedding), dim=1),
        max_new_tokens=talker.config.num_code_groups - 1,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
    )
    CAPTURE["predictor_first_frame_codes"] = (
        torch.cat((first_code, predictor_result.sequences), dim=-1)
        .detach()
        .cpu()
        .to(torch.int32)
        .numpy()
    )
    CAPTURE["predictor_first_frame_logits"] = (
        torch.stack(predictor_result.scores, dim=1)
        .detach()
        .cpu()
        .to(torch.float32)
        .numpy()[0]
    )

    hashes = {}
    for name, arr in CAPTURE.items():
        path = os.path.join(args.out, name + ".npy")
        np.save(path, arr)
        hashes[name] = hashlib.sha256(open(path, "rb").read()).hexdigest()
        print(f"{name}: {arr.shape}")
    reference_commit = subprocess.run(
        ["git", "-C", args.reference_root, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({
            "schema": "gufo.qwen3-tts-probe.v1",
            "reference_commit": reference_commit,
            "torch_version": torch.__version__,
            "hip_version": torch.version.hip,
            "device": args.device,
            "dtype": "bfloat16",
            "attention_implementation": "eager",
            "text": TEXT,
            "speaker": args.speaker,
            "language": args.language,
            "shapes": {k: list(v.shape) for k, v in CAPTURE.items()},
            "sha256": hashes,
        }, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
