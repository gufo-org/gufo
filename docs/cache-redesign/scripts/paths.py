"""Locations used by every experiment script, overridable by environment.

CACHE_EXP_DIR   work directory holding results/ and cache/ (default: this
                script's directory). Point it at docs/cache-redesign to replay
                the committed traces.
GUFO_REPO       gufo checkout used for corpora and test drivers (default: the
                checkout containing this script, else /home/mixer/gufo)
GUFO_BIN        gufo binary (default: $CACHE_EXP_DIR/bin/gufo)
LLAMA_TOKENIZE  llama.cpp's llama-tokenize (default: found on PATH)
FN_MODEL, FN_MTP, Q27_MODEL, Q27_DFLASH   model files
"""
import os
import pathlib
import shutil

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
WORK = pathlib.Path(os.environ.get("CACHE_EXP_DIR", SCRIPT_DIR)).resolve()


def _repo():
    if os.environ.get("GUFO_REPO"):
        return pathlib.Path(os.environ["GUFO_REPO"])
    for parent in SCRIPT_DIR.parents:
        if (parent / "AGENTS.md").exists() and (parent / "src" / "cli").exists():
            return parent
    return pathlib.Path("/home/mixer/gufo")


REPO = _repo()
BIN = pathlib.Path(os.environ.get("GUFO_BIN", WORK / "bin" / "gufo"))
TOKENIZE = os.environ.get("LLAMA_TOKENIZE") or shutil.which("llama-tokenize") or \
    "/nix/store/26d5y7z8vj2qc6qi0x6dlhcs17ixz1k2-llama-cpp-11382/bin/llama-tokenize"
FN_MODEL = os.environ.get(
    "FN_MODEL", "/persist/models/qwen38-flash-next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf")
FN_MTP = os.environ.get(
    "FN_MTP", "/persist/models/qwen38-flash-next/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf")
Q27_MODEL = os.environ.get(
    "Q27_MODEL", "/persist/models/models--unsloth--Qwen3.8-27B-GGUF/snapshots/"
    "4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q8_K_XL.gguf")
Q27_VOCAB = os.environ.get("Q27_VOCAB", Q27_MODEL)
Q27_DFLASH = os.environ.get("Q27_DFLASH", "/persist/models/Qwen3.8-27B-DFlash2-Q8_0.gguf")
