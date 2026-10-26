#!/usr/bin/env python3
"""Independent audio oracle from the pinned official Gemma4 feature extractor.

Use the transformers source at TRANSFORMERS_REVISION on PYTHONPATH. No Gufo
module is imported. All feature values and the full validity mask are retained,
including padded invalid frames and cases sharing a longer batch length.
"""
import argparse
import json

import numpy as np
import transformers
from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor

TRANSFORMERS_REVISION = "50e35cc0ca18ec19b80653690de553e6decf74f5"
CHECKPOINT_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"


def waveform(length, kind):
    if kind == "silence":
        return np.zeros(length, dtype=np.float32)
    # Exact binary fractions, reproducible independently in C++.
    i = np.arange(length, dtype=np.int64)
    if kind == "triangle":
        phase = i % 80
        return ((np.minimum(phase, 80 - phase) - 20) / 64).astype(np.float32)
    if kind == "music":
        phase = i % np.where((i // 2000) % 2 == 0, 64, 80)
        period = np.where((i // 2000) % 2 == 0, 64, 80)
        return ((np.minimum(phase, period - phase) - period // 4) / 64).astype(np.float32)
    if kind == "environment":
        burst = np.where(i % 4000 < 800, 1, 0)
        return ((((i * 73 + i * i * 19) % 1024 - 512) * burst) / 1024).astype(np.float32)
    return (((i * 73 + i * i * 19) % 1024 - 512) / 1024).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--transformers-revision", required=True,
                        choices=[TRANSFORMERS_REVISION])
    parser.add_argument("--embeddings-output")
    parser.add_argument("--decoder-output")
    args = parser.parse_args()
    with open(f"{args.model}/processor_config.json", encoding="utf-8") as handle:
        config = json.load(handle)["feature_extractor"]
    extractor = Gemma4AudioFeatureExtractor(**config)
    cases = []
    probes = [(1, "silence", 0), (160, "triangle", 0), (161, "triangle", 0),
              (255, "noise", 0), (256, "noise", 0), (321, "triangle", 0),
              (640, "noise", 0), (641, "noise", 0), (800, "triangle", 0),
              (801, "triangle", 0), (1023, "noise", 0), (1024, "silence", 0),
              (1601, "triangle", 0), (321, "noise", 16000),
              (16000, "triangle", 0)]
    for length, kind, padding in probes:
        audio = waveform(length, kind)
        clips = [audio] if not padding else [audio, np.zeros(padding, dtype=np.float32)]
        result = extractor(clips, return_tensors="np")
        features = result["input_features"][0]
        mask = result["input_features_mask"][0]
        cases.append({"name": f"{kind}_{length}_pad{padding}", "samples": length,
                      "kind": kind, "padding_samples": padding,
                      "frames": len(mask), "mask": mask.astype(int).tolist(),
                      "soft_tokens": int(mask[::2][::2].sum()),
                      "features": features.reshape(-1).tolist()})
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump({"model": "google/embeddinggemma-2", "revision": CHECKPOINT_REVISION,
                   "transformers_revision": args.transformers_revision,
                   "transformers": transformers.__version__, "dtype": "float32",
                   "cases": cases}, handle, separators=(",", ":"))

    if args.decoder_output:
        import base64
        import io
        import librosa
        import soundfile as sf
        import soxr
        from transformers.audio_utils import load_audio_librosa
        records = []
        for rate, channels, length, container in [
                (16000, 1, 801, "WAV"), (8000, 1, 401, "WAV"),
                (24000, 2, 1201, "WAV"), (44100, 2, 2209, "WAV"),
                (48000, 2, 2401, "WAV"), (16000, 1, 801, "FLAC"),
                (16000, 1, 16001, "MP3"), (16000, 2, 16001, "OGG")]:
            mono = waveform(length, "triangle")
            samples = mono if channels == 1 else np.stack((mono, waveform(length, "noise")), axis=1)
            buffer = io.BytesIO()
            sf.write(buffer, samples, rate, format=container)
            encoded = buffer.getvalue()
            decoded = load_audio_librosa(base64.b64encode(encoded).decode(), sampling_rate=16000)
            records.append({"name": f"{container}_{rate}_{channels}_{length}",
                            "encoded_hex": encoded.hex(), "waveform": decoded.tolist()})
        with open(args.decoder_output, "w", encoding="utf-8") as handle:
            json.dump({"transformers_revision": args.transformers_revision,
                       "backend": "librosa", "librosa": librosa.__version__,
                       "soundfile": sf.__version__, "soxr": soxr.__version__,
                       "cases": records}, handle, separators=(",", ":"))
    if args.embeddings_output:
        import torch
        from transformers import AutoConfig, AutoModel, AutoProcessor
        torch.set_num_threads(8)
        config = AutoConfig.from_pretrained(args.model)
        model = AutoModel.from_pretrained(args.model, config=config,
                                         dtype=torch.float32, attn_implementation="eager").eval()
        processor = AutoProcessor.from_pretrained(args.model)
        records = []
        probes = [
            ("triangle_161", "<|audio|>", [(161, "triangle")], []),
            ("noise_801", "<|audio|>", [(801, "noise")], []),
            ("triangle_16000", "<|audio|>", [(16000, "triangle")], []),
            ("silence_16000", "<|audio|>", [(16000, "silence")], []),
            ("music", "<|audio|>", [(16000, "music")], []),
            ("environment", "<|audio|>", [(16000, "environment")], []),
            ("speech", "<|audio|>", [(160000, "speech")], []),
            ("audio_text", "title: a tone | text: <|audio|> a rhythmic sound", [(16000, "triangle")], []),
            ("multiple_audio", "<|audio|> then <|audio|>", [(801, "noise"), (16000, "triangle")], []),
            ("audio_image_text", "sound <|audio|> picture <|image|> combined", [(16000, "triangle")],
             [{"width": 37, "height": 29, "seed": 0}]),
        ]
        from reference_images import image
        for name, text, clips, pictures in probes:
            from transformers.audio_utils import load_audio
            from pathlib import Path
            speech = Path(__file__).resolve().parents[3] / "tests/functional/fixtures/audio/clear-english-voice.wav"
            options = {"text": [text], "audio": [
                load_audio(str(speech), sampling_rate=16000, backend="librosa") if kind == "speech" else waveform(length, kind)
                for length, kind in clips],
                       "return_tensors": "pt"}
            if pictures:
                options["images"] = [[image(p["width"], p["height"], p["seed"]) for p in pictures]]
            inputs = processor(**options)
            with torch.inference_mode():
                encoded = model.audio_tower(inputs["input_features"], inputs["input_features_mask"])
                soft = model.embed_audio(encoded.last_hidden_state)
                states = model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                pooled = (states * mask).sum(1) / mask.sum(1)
                vector = torch.nn.functional.normalize(pooled, dim=-1)[0]
            records.append({"name": name, "samples": clips[0][0], "kind": clips[0][1],
                            "clips": [{"samples": length, "kind": kind} for length, kind in clips],
                            "text": text, "images": pictures,
                            "token_ids": inputs["input_ids"][0].tolist(),
                            "audio_features": soft[encoded.attention_mask].tolist(),
                            "encoder_states": encoded.last_hidden_state[encoded.attention_mask].tolist(),
                            "embedding": vector.tolist()})
            print(f"{name}: {len(records[-1]['audio_features'])} audio tokens", flush=True)
            if len(records) > 4:
                del records[-1]["audio_features"]
                del records[-1]["encoder_states"]
        for record in records:
            for key in ("audio_features", "encoder_states"):
                if key in record:
                    record[key] = [[float(f"{x:.7g}") for x in row] for row in record[key]]
        with open(args.embeddings_output, "w", encoding="utf-8") as handle:
            json.dump({"model": "google/embeddinggemma-2", "revision": CHECKPOINT_REVISION,
                       "transformers_revision": args.transformers_revision,
                       "transformers": transformers.__version__, "dtype": "float32",
                       "cases": records}, handle, separators=(",", ":"))


if __name__ == "__main__":
    main()
