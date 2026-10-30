# Independent d1 packed and box teacher

`reference.py` loads dequantized GGUF weights strictly into LiquidAI's
unmodified text, vision and projector classes, uses its prompt/packing/box
readout code, and compares real Gufo HTTP replies. `d1_forward_probe` also
checks every prompt ID, group tag, answer slot and selected vocabulary logit
against those independent fixtures. Use CPU FP32 for the teacher; Gufo runs
on the GPU. This is numerical qualification, not a teacher speed comparison.

The tested Python environment uses PyTorch 2.14.1, transformers 5.19.0 and
gguf-py from llama.cpp `8642541b255bfb5dc39cd351a178be09013a55d6`. Use an
isolated Python environment for transformer versions different from the
repository's development shell. No Python dependency is linked into Gufo.

Download only the reference source/config/tokenizer files:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    "LiquidAI/d1-Omni-2B",
    revision="431f1d667de5afcc9b0c58b2d9c170a7b38b358b",
    local_dir="reference/d1_reference",
    allow_patterns=["*.py", "config.json", "calibration.json",
                    "tokenizer.json", "tokenizer_config.json", "processor_config.json"],
)
```

Build the opt-in probe and start the production server as in the model guide:

```sh
nix develop -c cmake --build --preset gpu-test --target d1_forward_probe

python3 tools/d1/reference.py \
  --model d1-Omni-2B-Q8_0.gguf --mmproj mmproj-d1-Omni-2B-BF16.gguf \
  --reference-dir reference/d1_reference --image cats.png \
  --probe build/gpu-test/tests/models/d1/d1_forward_probe \
  --candidate http://127.0.0.1:18886 --output artifacts/d1/reference

python3 tests/functional/d1_packed.py \
  --url http://127.0.0.1:18886 --image cats.png --audio jfk.wav \
  --output artifacts/d1/packed-http.json
```

Set `PYTHONPATH` to the chosen environment and llama.cpp's `gguf-py` directory
when they are not installed packages. Use `--only CASE...` for affected
reference reruns. The teacher stores exact prompts, embeddings, reference and
native logits, answers and failures. Never replace its fixed gates to accept
a numerical change. The pinned `calibration.json` is also the source of the
four compiled typed-box calibration tables in `src/models/d1/boxes.cpp`.
