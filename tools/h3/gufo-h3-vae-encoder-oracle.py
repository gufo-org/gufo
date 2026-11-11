#!/usr/bin/env python3
"""Capture the MiniMax H3 VisualVAE *encoder* oracle for keyframe conditioning.

The engine port (``src/models/minimax_h3/video_vae_encoder.{hpp,cpp,hip}``) has to
reproduce the released single-frame encode path bit-for-bit closely enough to pass
the shared 0.05 relative gates. This script is the reference side of that contract:
it runs the checkpoint's OWN shipped bundle

  ``FL2VA/video_vae/vae_cnn.py:EncoderFCN3D`` + ``klvae.py`` tiling math
  (``split_tiles`` L192-218, ``blend`` L220-250)

against the released F32 weights in ``FL2VA/video_vae/source/model.safetensors``,
cross-checked against the installed diffusers implementation
(``models/autoencoders/autoencoder_kl_minimax_h3.py:_split_tiles/_blend/_stitch_tiles``,
the path ``encoders.py:MiniMaxH3KeyframeVaeEncoderStep`` reaches through
``AutoencoderKLMiniMaxH3._encode_clip``). Both references agree; the script asserts it.

Emitted fixture (schema ``gufo.minimax-h3-video-vae-encoder-golden.v1``):

* a deterministic 512x512 uint8 keyframe (``numpy.random.default_rng(seed).bytes``)
  stored as PNG plus its ImageNet-normalized F32 planes (the engine input),
* the 48-channel tiled moments ``(48, 32, 32)`` for the 512 canvas (tile 256,
  min overlap 64, latent-aligned widening, row-then-col linear blend, trim),
* the canonical first tile's pre-quant/post-quant moments ``(48, 16, 16)``,
* per-stage activations for the top-left 64x64 crop (single-tile geometry):
  ``conv_in``, each of the six ``down`` level outputs (post-downsample where the
  level has one), ``norm_out`` (post GroupNorm+SiLU, i.e. the tensor entering
  ``conv_out``), ``pre_quant`` and the post-``quant_conv`` moments,
* the reference tile split for lengths 256/320/512/768/1024/1344 so the engine
  plan can be compared without torch.

Usage (oracle venv, CPU):

  python tools/h3/gufo-h3-vae-encoder-oracle.py \
      --model-root ~/models/MiniMax-H3 \
      --output tests/fixtures/minimax_h3
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.machinery
import json
import math
import sys
import time
import types
from pathlib import Path

import numpy
import torch


MODEL_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
REFERENCE_REVISION = "8974cc055ea9c02fcd14cc27dfda3e1027c05153"
SCHEMA = "gufo.minimax-h3-video-vae-encoder-golden.v1"

SOURCE_COMPONENT = "FL2VA/video_vae"
SOURCE_CONFIG = "source/config.json"
SOURCE_WEIGHTS = "source/model.safetensors"

# FL2VA/video_vae/config.json: vae_tile_size / vae_tile_overlap_min / vae_ratio.
TILE_PIXELS = 256
TILE_OVERLAP_MINIMUM = 64
SPATIAL_RATIO = 16

DEFAULT_SEED = 20261102
DEFAULT_CANVAS = 512
DEFAULT_STAGE_CANVAS = 64
SPLIT_LENGTHS = (256, 320, 512, 768, 1024, 1344)

# Encoder channel geometry, from source/config.json (ch=128, ch_mult, space_down).
LEVEL_COUNT = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the MiniMax H3 VisualVAE encoder golden fixture"
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--canvas", type=int, default=DEFAULT_CANVAS)
    parser.add_argument("--stage-canvas", type=int, default=DEFAULT_STAGE_CANVAS)
    parser.add_argument(
        "--source-checkpoint-sha256",
        help="reuse a known digest instead of re-hashing the 10 GiB checkpoint",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="torch CPU threads (1 keeps the reduction order reproducible)",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def write_tensor(path: Path, tensor: torch.Tensor) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (
        tensor.detach().contiguous().float().cpu().numpy().astype("<f4", copy=False)
    )
    path.write_bytes(memoryview(numpy.ascontiguousarray(array)).tobytes())
    return {
        "dtype": "F32",
        "shape": list(tensor.shape),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def _stub_module(name: str) -> types.ModuleType:
    """Register an import-only module with a valid ``__spec__``.

    ``importlib.util.find_spec`` raises when a synthetic module has ``__spec__``
    None, and diffusers probes optional backends exactly that way; a spec with no
    loader plus no distribution metadata keeps the package marked unavailable.
    """
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, None, origin=name)
        sys.modules[name] = module
    return module


def install_import_stubs() -> None:
    """Provide the two third-party leaves the shipped bundle imports.

    ``FL2VA/video_vae/normalize.py`` imports ``torchvision.transforms.Normalize``
    and ``vae_processor.py`` imports ``einops.rearrange``; the oracle venv has
    neither. The ``Normalize`` stand-in below is the float-tensor definition
    torchvision uses (constants cast to the tensor dtype, broadcast per channel,
    ``(x - mean) / std``), so the shipped ``NORM_CONFIGS`` constants and the
    shipped ``get_normalize_transform`` wiring stay load-bearing. The einops
    shim is import-only: the encoder path never touches ``VAEProcessor``.
    """

    class Normalize:
        def __init__(self, mean, std, inplace=False):
            self.mean = torch.as_tensor(mean, dtype=torch.float32).view(-1, 1, 1)
            self.std = torch.as_tensor(std, dtype=torch.float32).view(-1, 1, 1)

        def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
            return tensor.sub(self.mean).div(self.std)

    torchvision = _stub_module("torchvision")
    transforms = _stub_module("torchvision.transforms")
    transforms.Normalize = Normalize  # type: ignore[attr-defined]
    torchvision.transforms = transforms  # type: ignore[attr-defined]
    einops = _stub_module("einops")

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("einops is not installed in the oracle venv")

    einops.rearrange = unavailable  # type: ignore[attr-defined]


def load_bundle(model_root: Path):
    """Import the checkpoint's own shipped ``video_vae`` namespace package."""
    component = model_root / SOURCE_COMPONENT
    if not (component / "vae_cnn.py").is_file():
        raise RuntimeError(f"shipped video_vae bundle not found under {component}")
    # Import diffusers before stubbing optional backends so its backend probes
    # run against the real environment.
    import diffusers.models  # noqa: F401
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import (  # noqa: F401
        AutoencoderKLMiniMaxH3,
    )

    install_import_stubs()
    sys.path.insert(0, str(model_root / "FL2VA"))
    from video_vae.klvae import AutoencoderKL  # noqa: PLC0415
    from video_vae.normalize import get_normalize_transform  # noqa: PLC0415
    from video_vae.vae_cnn import EncoderFCN3D  # noqa: PLC0415

    return component, EncoderFCN3D, AutoencoderKL, get_normalize_transform


def build_encoder(component: Path, EncoderFCN3D, weights: dict[str, torch.Tensor]):
    """Instantiate the released encoder + quant_conv in fp32 from the source config."""
    config = json.loads((component / SOURCE_CONFIG).read_text("utf-8"))
    encoder = EncoderFCN3D(
        ch=config["ch"],
        ch_mult=config["ch_mult"],
        space_down=config["space_down"],
        time_down=config["time_down"],
        num_res_blocks=config["num_res_blocks"],
        in_channels=config["in_channels"],
        z_channels=config["z_channels"],
        double_z=True,  # klvae.py:1158 pins double_z for the released encoder
        zq_ch=config["zq_ch_encoder"],
        padding_mode=config["padding_mode"],
        padding_mode_t=config["padding_mode_t"],
        causal=config["causal_encoder"],
        use_t_isolated_gn=config["use_t_isolated_gn"],
    )
    quant_conv = torch.nn.Conv3d(config["z_channels"] * 2, 2 * config["embed_dim"], 1)
    encoder.load_state_dict(
        {
            name.removeprefix("encoder."): tensor
            for name, tensor in weights.items()
            if name.startswith("encoder.")
        },
        strict=True,
    )
    quant_conv.load_state_dict(
        {
            name.removeprefix("quant_conv."): tensor
            for name, tensor in weights.items()
            if name.startswith("quant_conv.")
        },
        strict=True,
    )
    encoder.eval()
    quant_conv.eval()
    return config, encoder, quant_conv


def read_encoder_weights(path: Path) -> dict[str, torch.Tensor]:
    from safetensors import safe_open  # noqa: PLC0415

    selected: dict[str, torch.Tensor] = {}
    with torch.no_grad(), safe_open(path, framework="pt") as checkpoint:
        for name in checkpoint.keys():
            if name.startswith("encoder.") or name.startswith("quant_conv."):
                selected[name] = checkpoint.get_tensor(name).to(torch.float32)
    if not selected:
        raise RuntimeError(f"no encoder tensors found in {path}")
    return selected


# ---------------------------------------------------------------------------
# Tiling transcription (must match both shipped references exactly)
# ---------------------------------------------------------------------------


def split_tiles(length: int, tile_size: int, minimum_overlap: int):
    """``klvae.py:AutoencoderKL.split_tiles`` / ``_split_tiles`` transcription."""
    if tile_size >= length:
        return [0], [length], []
    count = math.ceil(length / tile_size)
    while tile_size * count - minimum_overlap * (count - 1) - length < 0:
        count += 1
    overlaps = [minimum_overlap] * (count - 1)
    remaining = tile_size * count - sum(overlaps) - length
    for index in range(remaining // SPATIAL_RATIO):
        overlaps[index % (count - 1)] += SPATIAL_RATIO
    starts = [0]
    for index in range(count - 1):
        starts.append(starts[-1] + tile_size - overlaps[index])
    return starts, [tile_size] * count, overlaps


def blend(a: torch.Tensor, b: torch.Tensor, extent: int, dim: int) -> torch.Tensor:
    """``klvae.py:AutoencoderKL.blend`` transcription."""
    extent = min(a.shape[dim], b.shape[dim], extent)
    positions = torch.arange(extent, device=b.device, dtype=b.dtype)
    shape = [1] * a.ndim
    shape[dim] = extent
    weight_a = (1 - positions / extent).view(shape)
    weight_b = (positions / extent).view(shape)
    slice_a = [slice(None)] * a.ndim
    slice_a[dim] = slice(-extent, None)
    slice_b = [slice(None)] * b.ndim
    slice_b[dim] = slice(0, extent)
    blended = a[tuple(slice_a)] * weight_a + b[tuple(slice_b)] * weight_b
    if extent == b.shape[dim]:
        return blended
    rest = [slice(None)] * b.ndim
    rest[dim] = slice(extent, None)
    return torch.cat([blended, b[tuple(rest)]], dim=dim)


def stitch_tiles(rows: list[list[torch.Tensor]], height_overlaps, width_overlaps):
    """``_stitch_tiles`` transcription: row blend, then column blend, then trim."""
    result_rows = []
    for i, row in enumerate(rows):
        result_row = []
        for j, tile in enumerate(row):
            value = tile
            if i > 0:
                value = blend(rows[i - 1][j], value, height_overlaps[i - 1], dim=-2)
            if j > 0:
                value = blend(row[j - 1], value, width_overlaps[j - 1], dim=-1)
            if i < len(rows) - 1:
                value = value[..., : -height_overlaps[i], :]
            if j < len(row) - 1:
                value = value[..., :, : -width_overlaps[j]]
            result_row.append(value)
        result_rows.append(torch.cat(result_row, dim=-1))
    return torch.cat(result_rows, dim=-2)


def diffusers_tiling_reference():
    """Methods of the released diffusers tiler bound to a minimal instance.

    ``AutoencoderKLMiniMaxH3._split_tiles/_blend/_stitch_tiles`` only need
    ``spatial_compression_ratio`` plus each other, so copying them onto a stub
    class exercises the released code without building the whole VAE.
    """
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import (
        AutoencoderKLMiniMaxH3,
    )

    class Reference:
        spatial_compression_ratio = SPATIAL_RATIO
        _split_tiles = AutoencoderKLMiniMaxH3._split_tiles
        _blend = AutoencoderKLMiniMaxH3._blend
        _stitch_tiles = AutoencoderKLMiniMaxH3._stitch_tiles

    return Reference()


def cross_check_tiling(AutoencoderKL) -> dict:
    """Assert the transcription agrees with the shipped implementations."""
    legacy = types.SimpleNamespace(
        tile_size=TILE_PIXELS,
        tile_overlap_min=TILE_OVERLAP_MINIMUM,
        decoder_tile_size=TILE_PIXELS,
        decoder_tile_overlap_min=TILE_OVERLAP_MINIMUM,
        vae_ratio=SPATIAL_RATIO,
    )
    reference = {
        str(length): split_tiles(length, TILE_PIXELS, TILE_OVERLAP_MINIMUM)
        for length in SPLIT_LENGTHS
    }
    diffusers = diffusers_tiling_reference()
    for length, (starts, lengths, overlaps) in reference.items():
        if AutoencoderKL.split_tiles(legacy, int(length), False) != (
            starts,
            lengths,
            overlaps,
        ):
            raise RuntimeError(f"legacy split_tiles differs at length {length}")
        if diffusers._split_tiles(int(length), TILE_PIXELS, TILE_OVERLAP_MINIMUM) != (
            starts,
            lengths,
            overlaps,
        ):
            raise RuntimeError(f"diffusers _split_tiles differs at length {length}")

    # Blend parity on non-trivial tensors: legacy, diffusers, transcription.
    probe = torch.randn(1, 6, 1, 9, 7, generator=torch.Generator().manual_seed(7))
    rows_probe = probe[..., 2:, :]  # (1, 6, 1, 7, 7)
    ours = blend(probe, rows_probe, 3, dim=-2)
    ours_columns = blend(rows_probe, rows_probe.flip(dims=(-1,)), 3, dim=-1)
    if not (
        torch.equal(AutoencoderKL.blend(legacy, probe, rows_probe, 3, dim=-2), ours)
        and torch.equal(diffusers._blend(probe, rows_probe, 3, dim=-2), ours)
        and torch.equal(
            diffusers._blend(rows_probe, rows_probe.flip(dims=(-1,)), 3, dim=-1),
            ours_columns)
    ):
        raise RuntimeError("blend transcription differs from the shipped blend")
    rows = [[probe * ((i + 1) * 10 + (j + 1)) for j in range(3)] for i in range(3)]
    overlaps = [3, 2]
    if not torch.equal(
        stitch_tiles(rows, overlaps, overlaps),
        diffusers._stitch_tiles(rows, overlaps, overlaps),
    ):
        raise RuntimeError("stitch transcription differs from diffusers _stitch_tiles")
    return reference


# ---------------------------------------------------------------------------
# Stage capture
# ---------------------------------------------------------------------------


class StageRecorder:
    """Captures the encoder stage activations of the tile it is armed for."""

    def __init__(self, encoder, quant_conv):
        self.records: dict[str, torch.Tensor] = {}
        self.enabled = False
        self._handles = []
        self._handles.append(
            encoder.conv_in.register_forward_hook(self._hook("conv_in"))
        )
        for level in range(LEVEL_COUNT):
            block = encoder.down[level]
            target = getattr(block, "downsample", None)
            if target is None:
                target = block.block[-1]
            self._handles.append(
                target.register_forward_hook(self._hook(f"down.{level}"))
            )
        # norm_out is GroupNorm; the reference applies SiLU on top of it, so the
        # engine-visible stage is the tensor entering conv_out.
        self._handles.append(
            encoder.conv_out.register_forward_pre_hook(self._pre_hook("norm_out"))
        )
        self._handles.append(encoder.conv_out.register_forward_hook(self._hook("pre_quant")))
        self._handles.append(quant_conv.register_forward_hook(self._hook("moments")))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _hook(self, name: str):
        def callback(module, inputs, output):
            if self.enabled:
                self.records[name] = output.detach().float().cpu().clone()

        return callback

    def _pre_hook(self, name: str):
        def callback(module, inputs):
            if self.enabled:
                self.records[name] = inputs[0].detach().float().cpu().clone()

        return callback


def encode_tile(encoder, quant_conv, tile: torch.Tensor, recorder, capture: bool):
    if recorder is not None:
        recorder.enabled = capture
        recorder.records = {}
    with torch.inference_mode():
        moments = quant_conv(encoder(tile))
    if recorder is not None:
        recorder.enabled = False
    return moments.detach().float()


def tiled_encode(encoder, quant_conv, pixels: torch.Tensor, recorder,
                 tile_size: int = TILE_PIXELS,
                 minimum_overlap: int = TILE_OVERLAP_MINIMUM):
    """Released path: per-tile ``quant_conv(encoder(tile))`` + latent blend."""
    height, width = pixels.shape[-2], pixels.shape[-1]
    y_starts, y_lengths, y_overlaps = split_tiles(height, tile_size, minimum_overlap)
    x_starts, x_lengths, x_overlaps = split_tiles(width, tile_size, minimum_overlap)
    rows: list[list[torch.Tensor]] = []
    first_tile: dict[str, torch.Tensor] | None = None
    for i, (y, length_y) in enumerate(zip(y_starts, y_lengths)):
        row = []
        for j, (x, length_x) in enumerate(zip(x_starts, x_lengths)):
            tile = pixels[..., y : y + length_y, x : x + length_x]
            capture = recorder is not None and i == 0 and j == 0
            moments = encode_tile(encoder, quant_conv, tile, recorder, capture)
            if capture:
                first_tile = dict(recorder.records)
            row.append(moments)
        rows.append(row)
    latent_y = [overlap // SPATIAL_RATIO for overlap in y_overlaps]
    latent_x = [overlap // SPATIAL_RATIO for overlap in x_overlaps]
    return stitch_tiles(rows, latent_y, latent_x), first_tile, rows, latent_y, latent_x


def legacy_model_stub(AutoencoderKL, encoder, quant_conv):
    """A stand-in ``AutoencoderKLLegacy`` carrying the shipped tiling methods."""

    class LegacyStub:
        tile_size = TILE_PIXELS
        tile_overlap_min = TILE_OVERLAP_MINIMUM
        decoder_tile_size = TILE_PIXELS
        decoder_tile_overlap_min = TILE_OVERLAP_MINIMUM
        vae_ratio = SPATIAL_RATIO
        parallel_tiling = False
        stack_tiling = False
        training = False
        encoder_parallel = False
        split_tiles = AutoencoderKL.split_tiles
        blend = AutoencoderKL.blend

        def __init__(self):
            self.encoder = encoder
            self.encode = lambda tile: quant_conv(encoder(tile))

        def freeze_scope(self, _name):
            return contextlib.nullcontext()

        def _local_tile_indices(self, num_tiles, sp_rank, sp_size):
            # klvae.py: interleaved assignment; rank 0 / size 1 keeps all.
            return list(range(sp_rank, num_tiles, sp_size))

        def _run_tile_tasks(self, tiles, tile_indices, forward_fn,
                            stack_tiling, cls_agg=None):
            # The serial path only (stack_tiling False, sp_size 1); the real
            # ClsTokenAggregator state is read exclusively by the
            # sp_size > 1 all-gather branch, so skipping collect() is exact.
            return [forward_fn(tiles[idx]) for idx in tile_indices]

    return LegacyStub()


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    model_root = args.model_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, args.threads))
    torch.use_deterministic_algorithms(True)

    component, EncoderFCN3D, AutoencoderKL, get_normalize_transform = load_bundle(
        model_root
    )
    splits = cross_check_tiling(AutoencoderKL)
    weights = read_encoder_weights(component / SOURCE_WEIGHTS)
    config, encoder, quant_conv = build_encoder(component, EncoderFCN3D, weights)
    parameter_count = sum(p.numel() for p in encoder.parameters()) + sum(
        p.numel() for p in quant_conv.parameters()
    )

    rng = numpy.random.default_rng(args.seed)
    raw = rng.bytes(args.canvas * args.canvas * 3)
    image = numpy.frombuffer(raw, dtype=numpy.uint8).reshape(
        args.canvas, args.canvas, 3
    )
    pixels = torch.from_numpy(image.copy()).permute(2, 0, 1).to(torch.float32).div(255.0)
    normalize = get_normalize_transform(config["pixel_norm_type"])
    normalized = normalize(pixels).unsqueeze(0).unsqueeze(2)  # (1,3,1,H,W)
    if normalized.dtype is not torch.float32:
        raise RuntimeError("normalized keyframe left fp32")

    stage_pixels = normalized[..., : args.stage_canvas, : args.stage_canvas].contiguous()
    recorder = StageRecorder(encoder, quant_conv)
    try:
        stage_moments = encode_tile(
            encoder, quant_conv, stage_pixels, recorder, capture=True
        )
        stage_records = dict(recorder.records)
        canvas = tiled_encode(encoder, quant_conv, normalized.contiguous(), recorder)
        canvas_moments, canvas_records = canvas[0], canvas[1]
    finally:
        recorder.close()

    # Tiling parity on real activations: the shipped diffusers stitcher and the
    # shipped legacy ``tiled_encode`` must agree with the transcription bitwise.
    if not torch.equal(
        diffusers_tiling_reference()._stitch_tiles(canvas[2], canvas[3], canvas[4]),
        canvas_moments,
    ):
        raise RuntimeError("diffusers _stitch_tiles differs on the canvas tiles")
    legacy = legacy_model_stub(AutoencoderKL, encoder, quant_conv)
    probe = normalized[..., :64, :64].contiguous()
    for tile_size, overlap in ((32, 16), (48, 16)):
        legacy.tile_size = tile_size
        legacy.tile_overlap_min = overlap
        transcription = tiled_encode(encoder, quant_conv, probe, None, tile_size, overlap)[0]
        if not torch.equal(AutoencoderKL.tiled_encode(legacy, probe), transcription):
            raise RuntimeError(
                f"legacy tiled_encode differs at tile {tile_size}, overlap {overlap}"
            )
    parity = {
        "real_activations": (
            "diffusers _stitch_tiles and legacy tiled_encode bitwise equal to the "
            "transcription on the canvas tiles and on 64x64 probes at tile 32/48"
        )
    }

    expected_latent = args.canvas // SPATIAL_RATIO
    if tuple(canvas_moments.shape) != (1, 48, 1, expected_latent, expected_latent):
        raise RuntimeError(f"unexpected moments shape {tuple(canvas_moments.shape)}")
    stage_latent = args.stage_canvas // SPATIAL_RATIO
    if tuple(stage_moments.shape) != (1, 48, 1, stage_latent, stage_latent):
        raise RuntimeError(f"unexpected stage moments shape {tuple(stage_moments.shape)}")

    # ------------------------------------------------------------------ files
    output.mkdir(parents=True, exist_ok=True)
    prefix = "video-vae-encoder-"
    files: dict[str, dict] = {}

    keyframe_path = output / f"{prefix}keyframe-{args.canvas}.png"
    try:
        from PIL import Image

        Image.fromarray(image, mode="RGB").save(keyframe_path)
    except Exception as error:  # pillow is optional for the numeric contract
        print(f"warning: PNG export skipped ({error})", file=sys.stderr)
    else:
        files[keyframe_path.name] = {
            "dtype": "uint8-PNG",
            "shape": [args.canvas, args.canvas, 3],
            "bytes": keyframe_path.stat().st_size,
            "sha256": sha256(keyframe_path),
        }

    keyframe_sha256 = hashlib.sha256(raw).hexdigest()
    keyframe_f32 = output / f"{prefix}keyframe-{args.canvas}.f32"
    files[keyframe_f32.name] = write_tensor(keyframe_f32, normalized[0, :, 0])

    moments_path = output / f"{prefix}keyframe-{args.canvas}-moments.f32"
    files[moments_path.name] = write_tensor(moments_path, canvas_moments[0, :, 0])

    tile_files: dict[str, dict] = {}
    if canvas_records:
        for name in ("pre_quant", "moments"):
            path = output / f"{prefix}tile-00-{name}.f32"
            files[path.name] = write_tensor(path, canvas_records[name][0, :, 0])
            tile_files[name] = path.name

    stage_names = ["conv_in"] + [f"down.{i}" for i in range(LEVEL_COUNT)] + [
        "norm_out",
        "pre_quant",
        "moments",
    ]
    stage_entries = {}
    for name in stage_names:
        tensor = stage_records[name][0, :, 0]
        path = output / f"{prefix}stage-{name.replace('.', '-')}.f32"
        files[path.name] = write_tensor(path, tensor)
        stage_entries[name] = {"file": path.name, "shape": list(tensor.shape)}

    checkpoint_path = component / SOURCE_WEIGHTS
    checkpoint_sha = args.source_checkpoint_sha256 or sha256(checkpoint_path)

    contract = {
        "schema": SCHEMA,
        "model_revision": MODEL_REVISION,
        "reference_revision": REFERENCE_REVISION,
        "source_checkpoint": f"{SOURCE_COMPONENT}/{SOURCE_WEIGHTS}",
        "source_checkpoint_sha256": checkpoint_sha,
        # sha256 of the raw canvas*canvas*3 uint8 bytes (not the PNG): lets the
        # sibling FL2VA fixtures prove they condition on the identical keyframe.
        "keyframe_uint8_sha256": keyframe_sha256,
        "reference_implementation": {
            "encoder": f"{SOURCE_COMPONENT}/vae_cnn.py EncoderFCN3D (fp32, eval)",
            "tiling": (
                f"{SOURCE_COMPONENT}/klvae.py split_tiles/blend transcribed and "
                "asserted bitwise equal to the shipped legacy functions and to "
                "diffusers autoencoder_kl_minimax_h3.py _split_tiles/_blend/"
                "_stitch_tiles"
            ),
            "normalization": f"{SOURCE_COMPONENT}/normalize.py imagenet",
        },
        "generator": {
            "algorithm": "numpy.random.default_rng(seed).bytes",
            "seed": args.seed,
            "dtype": "uint8",
            "shape": [args.canvas, args.canvas, 3],
        },
        "normalization": {
            "type": config["pixel_norm_type"],
            "scale": 1.0 / 255.0,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "geometry": {
            "canvas": [args.canvas, args.canvas],
            "moments": [48, expected_latent, expected_latent],
            "stage_canvas": [args.stage_canvas, args.stage_canvas],
            "stage_moments": [48, stage_latent, stage_latent],
            "spatial_ratio": SPATIAL_RATIO,
            "channels": [config["ch"] * factor for factor in config["ch_mult"]],
            "space_down": config["space_down"],
            "time_down": config["time_down"],
        },
        "tiling": {
            "enabled": True,
            "tile_pixels": TILE_PIXELS,
            "minimum_overlap_pixels": TILE_OVERLAP_MINIMUM,
            "round_robin_widening": SPATIAL_RATIO,
            "blend": "linear 1-pos/n then pos/n over latent rows, then columns, then trim",
            "axes": {
                length: {
                    "starts": starts,
                    "lengths": lengths,
                    "overlaps_pixels": overlaps,
                    "overlaps_latent": [value // SPATIAL_RATIO for value in overlaps],
                }
                for length, (starts, lengths, overlaps) in sorted(
                    splits.items(), key=lambda item: int(item[0])
                )
            },
        },
        "arithmetic": {
            "weights": "released F32",
            "activations": "F32",
            "group_norm": "32 groups, eps 1e-6, affine, temporal-isolated (single frame)",
            "causal_temporal": "T=1 collapses every 3x3x3 causal conv to its last temporal tap",
            "padding": "spatial reflect 1; downsample asymmetric reflect (0,1,0,1) then stride",
        },
        "quality_bounds": {"relative_l2": 0.05, "relative_max": 0.05},
        "stages": stage_entries,
        "tile_stages": tile_files,
        "files": files,
        "teacher": {
            "implementation": "shipped checkpoint bundle, fp32 CPU",
            "torch_version": torch.__version__,
            "numpy_version": numpy.__version__,
            "threads": torch.get_num_threads(),
            "deterministic_algorithms": True,
            "encoder_parameters": parameter_count,
            "tiling_parity": parity,
            "elapsed_seconds": time.monotonic() - started,
        },
    }
    fixture = output / "video-vae-encoder-oracle-v1.json"
    fixture.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {fixture} (+{len(files)} blobs) in "
        f"{contract['teacher']['elapsed_seconds']:.2f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
