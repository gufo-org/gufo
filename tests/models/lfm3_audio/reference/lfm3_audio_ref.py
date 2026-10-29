#!/usr/bin/env python3
"""Independent FP32 PyTorch reference for LFM3-Audio-2B.

Written from the Hugging Face remote code of LiquidAI/LFM3-Audio-2B at
revision cddb56b5ec9b4703f9b3b2d0500143d1fff2fee0 (modeling_lfm3_audio.py,
modeling_lfm3.py, modeling_liquid_codec.py, feature_extraction_lfm3_audio.py),
reading the lossless BF16 GGUF pair of LiquidAI/LFM3-Audio-2B-GGUF revision
02355c1a141e464a064f2ab2c276638dbcb7f38a. It shares no code with the native
implementation and runs every operator in float32 on the CPU.

Subcommands write fixtures consumed by the native tests:
  features   WAV -> log-mel (f32 [128, T])
  encode     WAV -> projected audio embeddings (f32 [N, 2048])
  backbone   token/embedding history -> final hidden and logits
  depth      hidden + codes + mask -> depth logits [12, 2017]
  codec      codes [F, 12] -> 24 kHz waveform
  asr        WAV -> greedy transcript (end to end)
  tts        text -> greedy codes and waveform (audio temperature 0)
"""
import argparse
import json
import math
import struct
import sys
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)

# ----------------------------------------------------------------------------
# GGUF


class Gguf:
    TYPES = {0: (np.float32, 4), 1: (np.float16, 2), 30: (None, 2)}

    def __init__(self, path):
        self.path = path
        f = open(path, "rb")
        self.file = f

        def u32():
            return struct.unpack("<I", f.read(4))[0]

        def u64():
            return struct.unpack("<Q", f.read(8))[0]

        def string():
            return f.read(u64()).decode("utf8", "replace")

        scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

        def value(kind):
            if kind in scalar:
                fmt = "<" + scalar[kind]
                return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
            if kind == 8:
                return string()
            if kind == 9:
                inner = u32()
                return [value(inner) for _ in range(u64())]
            raise ValueError(kind)

        assert f.read(4) == b"GGUF"
        u32()
        tensors, keys = u64(), u64()
        self.meta = {}
        for _ in range(keys):
            name = string()
            self.meta[name] = value(u32())
        self.tensors = {}
        for _ in range(tensors):
            name = string()
            dims = [u64() for _ in range(u32())]
            kind = u32()
            offset = u64()
            self.tensors[name] = (dims, kind, offset)
        alignment = self.meta.get("general.alignment", 32)
        position = f.tell()
        self.base = position + (-position % alignment)
        self.map = np.memmap(path, dtype=np.uint8, mode="r")

    def get(self, name):
        dims, kind, offset = self.tensors[name]
        count = int(np.prod(dims))
        start = self.base + offset
        if kind == 0:
            array = np.frombuffer(self.map[start : start + 4 * count], dtype=np.float32)
        elif kind == 1:
            array = np.frombuffer(self.map[start : start + 2 * count], dtype=np.float16).astype(np.float32)
        elif kind == 30:
            raw = np.frombuffer(self.map[start : start + 2 * count], dtype=np.uint16).astype(np.uint32)
            array = (raw << 16).view(np.float32)
        else:
            raise ValueError(f"{name}: unsupported reference type {kind}; use the BF16 GGUFs")
        return torch.from_numpy(array.copy()).reshape(list(reversed(dims)))

    def has(self, name):
        return name in self.tensors


# ----------------------------------------------------------------------------
# Shared blocks


def rms_norm(x, weight, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def linear(x, weight, bias=None):
    return F.linear(x, weight, bias)


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rope(x, positions, inv_freq):
    # x: [T, heads, dim]
    angles = positions[:, None].float() * inv_freq[None, :]
    angles = torch.cat((angles, angles), dim=-1)
    return x * angles.cos()[:, None, :] + rotate_half(x) * angles.sin()[:, None, :]


def inverse_frequencies(dim, theta, factor=1.0):
    angles = int(factor * dim) // 2
    inv = 1.0 / (theta ** (torch.arange(0, 2 * angles, 2, dtype=torch.float64) / dim))
    return torch.cat((inv, torch.zeros(dim // 2 - angles, dtype=torch.float64))).float()


def attention(q, k, v, mask):
    # q: [T, H, D], k/v: [S, KV, D], mask: [T, S] bool (True = visible)
    groups = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    scores = torch.einsum("thd,shd->hts", q, k) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~mask[None], float("-inf"))
    return torch.einsum("hts,shd->thd", scores.softmax(-1), v)


def causal_mask(t, window=None):
    q = torch.arange(t)[:, None]
    k = torch.arange(t)[None, :]
    visible = k <= q
    if window is not None:
        visible &= k > q - window
    return visible


def short_conv(x, in_proj, conv_weight, out_proj):
    # x: [T, hidden]; conv_weight: [hidden, width]
    b, c, h = linear(x, in_proj).chunk(3, dim=-1)
    y = (b * h).T[None]
    width = conv_weight.shape[1]
    y = F.conv1d(F.pad(y, (width - 1, 0)), conv_weight[:, None, :], groups=y.shape[1])[0].T
    return linear(c * y, out_proj)


def swiglu(x, gate, up, down):
    return linear(F.silu(linear(x, gate)) * linear(x, up), down)


class Lfm:
    """LFM2/LFM3 stack over embeddings. Kinds: 'conv', 'attn'."""

    def __init__(self, g, prefix, names, kinds, eps, heads, kv_heads, head_dim, inv_freq, gated, sandwich, final_norm):
        self.layers = []
        for index, kind in enumerate(kinds):
            p = f"{prefix}{index}."
            layer = {"kind": kind}
            for key, name in names.items():
                full = p + name
                if g.has(full):
                    layer[key] = g.get(full)
            self.layers.append(layer)
        self.eps, self.heads, self.kv_heads, self.head_dim = eps, heads, kv_heads, head_dim
        self.inv_freq, self.gated, self.sandwich = inv_freq, gated, sandwich
        self.final_norm = g.get(final_norm) if final_norm else None

    def __call__(self, x, mask, positions):
        for layer in self.layers:
            residual = x
            z = rms_norm(x, layer["norm"], self.eps)
            if layer["kind"] == "conv":
                y = short_conv(z, layer["in_proj"], layer["conv"], layer["out_proj"])
            else:
                t = z.shape[0]
                q = linear(z, layer["q"]).view(t, self.heads, self.head_dim)
                k = linear(z, layer["k"]).view(t, self.kv_heads, self.head_dim)
                v = linear(z, layer["v"]).view(t, self.kv_heads, self.head_dim)
                q = rms_norm(q, layer["q_norm"], self.eps)
                k = rms_norm(k, layer["k_norm"], self.eps)
                inv = self.inv_freq[layer["window_kind"]] if isinstance(self.inv_freq, dict) else self.inv_freq
                q, k = rope(q, positions, inv), rope(k, positions, inv)
                m = mask[layer["window_kind"]] if isinstance(mask, dict) else mask
                o = attention(q, k, v, m)
                if self.gated:
                    o = o * torch.sigmoid(linear(z, layer["gate"]))[..., None]
                y = linear(o.reshape(t, -1), layer["out"])
            if self.sandwich:
                y = rms_norm(y, layer["post_norm"], self.eps)
            x = residual + y
            f = swiglu(rms_norm(x, layer["ffn_norm"], self.eps), layer["ffn_gate"], layer["ffn_up"], layer["ffn_down"])
            if self.sandwich:
                f = rms_norm(f, layer["post_ffn_norm"], self.eps)
            x = x + f
        return rms_norm(x, self.final_norm, self.eps) if self.final_norm is not None else x


# ----------------------------------------------------------------------------
# Model


class Model:
    def __init__(self, backbone_path, mmproj_path):
        self.text = Gguf(backbone_path)
        self.mm = Gguf(mmproj_path)
        self._build_backbone()
        self._build_encoder()
        self._build_depth()
        self._build_codec()

    # ---- backbone (LFM3 1.7B)
    def _build_backbone(self):
        g = self.text
        meta = g.meta
        n = meta["lfm3.block_count"]
        kv = meta["lfm3.attention.head_count_kv"]
        swa = meta["lfm3.attention.sliding_window_pattern"]
        kinds, window_kinds = [], []
        for i in range(n):
            kinds.append("attn" if kv[i] else "conv")
            window_kinds.append("swa" if swa[i] else "full")
        names = {
            "norm": "attn_norm.weight",
            "ffn_norm": "ffn_norm.weight",
            "post_norm": "post_attention_norm.weight",
            "post_ffn_norm": "post_ffw_norm.weight",
            "ffn_gate": "ffn_gate.weight",
            "ffn_up": "ffn_up.weight",
            "ffn_down": "ffn_down.weight",
            "conv": "shortconv.conv.weight",
            "in_proj": "shortconv.in_proj.weight",
            "out_proj": "shortconv.out_proj.weight",
            "q": "attn_q.weight",
            "k": "attn_k.weight",
            "v": "attn_v.weight",
            "q_norm": "attn_q_norm.weight",
            "k_norm": "attn_k_norm.weight",
            "gate": "attn_gate.weight",
            "out": "attn_output.weight",
        }
        head_dim = meta["lfm3.attention.key_length"]
        # official config: partial_rotary_factor 0.5 on full attention, full rotation on sliding layers
        inv = {
            "full": inverse_frequencies(head_dim, meta["lfm3.rope.freq_base"], 0.5),
            "swa": inverse_frequencies(head_dim, meta["lfm3.rope.freq_base_swa"], 1.0),
        }
        self.lm = Lfm(
            g, "blk.", names, kinds, meta["lfm3.attention.layer_norm_rms_epsilon"],
            meta["lfm3.attention.head_count"], max(kv), head_dim, inv, True, True, "output_norm.weight",
        )
        for layer, wk in zip(self.lm.layers, window_kinds):
            layer["window_kind"] = wk
        self.window = meta["lfm3.attention.sliding_window"]
        self.embedding = g.get("token_embd.weight")

    def backbone(self, embeds):
        t = embeds.shape[0]
        masks = {"full": causal_mask(t), "swa": causal_mask(t, self.window)}
        return self.lm(embeds, masks, torch.arange(t))

    def logits(self, hidden):
        return linear(hidden, self.embedding)

    # ---- audio encoder (downsampler + LFM2 + projector)
    def _build_encoder(self):
        g = self.mm
        meta = g.meta
        n = meta["clip.audio.block_count"]
        kinds = ["attn" if g.has(f"a.blk.{i}.attn_q.weight") else "conv" for i in range(n)]
        names = {
            "norm": "ln1.weight",
            "ffn_norm": "ln2.weight",
            "ffn_gate": "ffn_gate.weight",
            "ffn_up": "ffn_up.weight",
            "ffn_down": "ffn_down.weight",
            "conv": "conv_dw.weight",
            "in_proj": "conv_pw1.weight",
            "out_proj": "conv_pw2.weight",
            "q": "attn_q.weight",
            "k": "attn_k.weight",
            "v": "attn_v.weight",
            "q_norm": "attn_q_norm.weight",
            "k_norm": "attn_k_norm.weight",
            "out": "attn_out.weight",
        }
        hidden = meta["clip.audio.embedding_length"]
        heads = meta["clip.audio.attention.head_count"]
        head_dim = hidden // heads
        self.encoder = Lfm(
            g, "a.blk.", names, kinds, meta["clip.audio.attention.layer_norm_epsilon"], heads,
            meta["clip.audio.attention.head_count_kv"], head_dim, inverse_frequencies(head_dim, 1e6), False, False,
            "a.post_ln.weight",
        )
        for layer in self.encoder.layers:
            layer["window_kind"] = "full"
        self.encoder_window = meta["clip.audio.window_size"]
        self.down = {}
        for i in (0, 2, 3, 5, 6):
            w = g.get(f"a.conv1d.{i}.weight")  # [out, in, kh, kw]
            self.down[i] = (w, g.get(f"a.conv1d.{i}.bias").reshape(-1))
        self.down_out = (g.get("a.pre_encode.out.weight"), g.get("a.pre_encode.out.bias"))
        self.proj = [g.get(f"mm.a.mlp.{i}.{p}") for i in (0, 1, 3) for p in ("weight", "bias")]

    def downsample(self, mel):
        # mel: [128, T] -> [ceil(T/8), 768]; (mel, time) plane, stride 2 thrice
        x = mel[None, None]
        w, b = self.down[0]
        x = F.silu(F.conv2d(x, w, b, stride=2, padding=1))
        w, b = self.down[2]
        x = F.conv2d(x, w, b, stride=2, padding=1, groups=x.shape[1])
        w, b = self.down[3]
        x = F.silu(F.conv2d(x, w, b))
        w, b = self.down[5]
        x = F.conv2d(x, w, b, stride=2, padding=1, groups=x.shape[1])
        w, b = self.down[6]
        x = F.silu(F.conv2d(x, w, b))
        x = x.flatten(1, 2)[0]  # [256*16, T/8]
        w, b = self.down_out
        return linear(x.T, w, b)

    def encode(self, mel):
        x = self.downsample(mel)
        t = x.shape[0]
        x = self.encoder(x, causal_mask(t, self.encoder_window), torch.arange(t))
        ln_w, ln_b, w1, b1, w2, b2 = self.proj
        x = F.layer_norm(x, (x.shape[-1],), ln_w, ln_b, 1e-5)
        return linear(F.gelu(linear(x, w1, b1)), w2, b2)

    # ---- depth decoder
    def _build_depth(self):
        g = self.mm
        meta = g.meta
        n = meta["clip.gen.audio.block_count"]
        names = {
            "norm": "ln1.weight",
            "ffn_norm": "ln2.weight",
            "ffn_gate": "ffn_gate.weight",
            "ffn_up": "ffn_up.weight",
            "ffn_down": "ffn_down.weight",
            "q": "attn_q.weight",
            "k": "attn_k.weight",
            "v": "attn_v.weight",
            "q_norm": "attn_q_norm.weight",
            "k_norm": "attn_k_norm.weight",
            "gate": "attn_gate.weight",
            "out": "attn_out.weight",
        }
        hidden = meta["clip.gen.audio.embedding_length"]
        heads = meta["clip.gen.audio.attention.head_count"]
        head_dim = hidden // heads
        self.depth = Lfm(
            g, "a.gen.code.blk.", names, ["attn"] * n, meta["clip.gen.audio.attention.layer_norm_epsilon"], heads,
            meta["clip.gen.audio.attention.head_count_kv"], head_dim, inverse_frequencies(head_dim, 1e6), True, False,
            "a.gen.code.output_norm.weight",
        )
        for layer in self.depth.layers:
            layer["window_kind"] = "full"
        self.depth_in = (g.get("a.gen.code.proj_in.weight"), g.get("a.gen.code.proj_in.bias"))
        self.mask_embedding = g.get("a.gen.code.mask_embd")
        self.code_embeddings = g.get("a.gen.code.embd.weight")  # [12, 2017, 1024]
        self.head_norms = g.get("a.gen.code.head_norm.weight")  # [12, 1024]
        self.audio_out = g.get("a.gen.code.out_embd.weight")  # [12*2017, 2048]
        self.codebooks, self.codebook_size = self.code_embeddings.shape[0], self.code_embeddings.shape[1]
        self.depth_eps = meta["clip.gen.audio.attention.layer_norm_epsilon"]

    def depth_logits(self, hidden, codes, masked):
        # hidden [2048], codes [12] long, masked [12] bool -> [12, 2017]
        x = linear(hidden, *self.depth_in).view(self.codebooks, -1)
        embeds = self.code_embeddings[torch.arange(self.codebooks), codes]
        x = x + torch.where(masked[:, None], self.mask_embedding[None], embeds)
        n = self.codebooks
        x = self.depth(x, torch.ones(n, n, dtype=torch.bool), torch.arange(n))
        out = []
        for i in range(n):
            out.append(linear(rms_norm(x[i], self.head_norms[i], self.depth_eps), self.code_embeddings[i]))
        return torch.stack(out)

    def frame_embedding(self, codes):
        offsets = torch.arange(self.codebooks) * self.codebook_size
        return self.audio_out[codes + offsets].sum(0)

    # ---- codec decoder
    def _build_codec(self):
        g = self.mm
        names = {
            "norm": "ln1.weight",
            "ffn_norm": "ln2.weight",
            "post_norm": "attn_post_norm.weight",
            "post_ffn_norm": "ffn_post_norm.weight",
            "ffn_gate": "ffn_gate.weight",
            "ffn_up": "ffn_up.weight",
            "ffn_down": "ffn_down.weight",
            "conv": "conv_dw.weight",
            "in_proj": "conv_pw1.weight",
            "out_proj": "conv_pw2.weight",
            "q": "attn_q.weight",
            "k": "attn_k.weight",
            "v": "attn_v.weight",
            "q_norm": "attn_q_norm.weight",
            "k_norm": "attn_k_norm.weight",
            "gate": "attn_gate.weight",
            "out": "attn_out.weight",
        }
        kinds = ["attn" if g.has(f"a.gen.wav.tfm.blk.{i}.attn_q.weight") else "conv" for i in range(8)]
        self.codec = Lfm(
            g, "a.gen.wav.tfm.blk.", names, kinds, 1e-5, 8, 2, 64, inverse_frequencies(64, 1e6), True, True,
            "a.gen.wav.tfm.output_norm.weight",
        )
        for layer in self.codec.layers:
            layer["window_kind"] = "full"
        self.fsq = g.get("a.gen.wav.quant.first.codebook.weight")  # [2016, 4]
        self.fsq_out = (g.get("a.gen.wav.quant.first.out_proj.weight"), g.get("a.gen.wav.quant.first.out_proj.bias"))
        self.codec_in = (g.get("a.gen.wav.tfm.in_proj.weight"), g.get("a.gen.wav.tfm.in_proj.bias"))
        self.codec_out = (g.get("a.gen.wav.tfm.out_proj.weight"), g.get("a.gen.wav.tfm.out_proj.bias"))

    def fsq_latents(self, codes):
        # independent closed form of the FSQ decode, checked against the GGUF table
        levels = torch.tensor([8, 7, 6, 6])
        basis = torch.tensor([1, 8, 56, 336])
        half = levels // 2
        latents = ((codes[..., None] // basis) % levels - half).float() / half.float()
        return latents

    def codec_decode(self, codes):
        # codes [F, 12] -> waveform [F*1920]
        table = self.fsq_latents(torch.arange(2016))
        assert torch.allclose(table, self.fsq), "FSQ table differs from the closed form"
        latents = self.fsq[codes].reshape(codes.shape[0], -1)  # [F, 48]
        x = linear(linear(latents, *self.fsq_out), *self.codec_in)  # [F, 512]
        x = x.repeat_interleave(6, dim=0)  # nearest-exact x6
        t = x.shape[0]
        x = self.codec(x, causal_mask(t, 30), torch.arange(t))
        spec = linear(x, *self.codec_out)  # [t, 1282]
        log_magnitude, angle = spec[:, :641], spec[:, 641:]
        complex_spec = torch.polar(log_magnitude.exp(), angle).T  # [641, t]
        window = torch.hann_window(1280, dtype=torch.float32)
        frames = torch.fft.irfft(complex_spec, 1280, dim=0) * window[:, None]  # [1280, t]
        length = (t - 1) * 320 + 1280
        audio = torch.zeros(length)
        envelope = torch.zeros(length)
        for i in range(t):
            audio[i * 320 : i * 320 + 1280] += frames[:, i]
            envelope[i * 320 : i * 320 + 1280] += window.square()
        trim = (1280 - 320) // 2
        return (audio[trim:-trim] / envelope[trim:-trim])


# ----------------------------------------------------------------------------
# Features


def slaney_mel_filters(n_freqs=201, n_mels=128, sample_rate=16000, f_min=0.0, f_max=8000.0):
    def hz_to_mel(f):
        f = np.asarray(f, dtype=np.float64)
        mel = 3.0 * f / 200.0
        log = f >= 1000.0
        mel = np.where(log, 15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) / (np.log(6.4) / 27.0), mel)
        return mel

    def mel_to_hz(m):
        m = np.asarray(m, dtype=np.float64)
        f = 200.0 * m / 3.0
        log = m >= 15.0
        return np.where(log, 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0)), f)

    fft_freqs = np.linspace(0, sample_rate // 2, n_freqs)
    mel_points = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    filter_freqs = mel_to_hz(mel_points)
    diff = np.diff(filter_freqs)
    slopes = filter_freqs[None, :] - fft_freqs[:, None]
    down = -slopes[:, :-2] / diff[:-1]
    up = slopes[:, 2:] / diff[1:]
    filters = np.maximum(0, np.minimum(down, up))
    enorm = 2.0 / (filter_freqs[2 : n_mels + 2] - filter_freqs[:n_mels])
    return (filters * enorm[None, :]).astype(np.float32)  # [n_freqs, n_mels]


def log_mel(samples):
    n_fft, hop = 400, 160
    count = samples.shape[0]
    padded = n_fft if count < n_fft else count + (-(count - n_fft) % hop)
    x = F.pad(samples, (0, padded - count))
    window = torch.hann_window(n_fft, dtype=torch.float32)
    spec = torch.stft(x, n_fft, hop, n_fft, window, center=False, return_complex=True)
    power = spec.abs().square()
    filters = torch.from_numpy(slaney_mel_filters())
    mel = (filters.T @ power).log10().clamp(min=-12.0)
    return (mel + 4.0) / 4.0


def read_wav(path, rate):
    with wave.open(str(path), "rb") as w:
        channels, width, source_rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(w.getnframes())
    if width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError("unsupported WAV width")
    data = data.reshape(-1, channels).mean(1)
    if source_rate != rate:
        from scipy.signal import resample_poly

        divisor = math.gcd(rate, source_rate)
        data = resample_poly(data, rate // divisor, source_rate // divisor).astype(np.float32)
    return torch.from_numpy(data.copy())


def write_wav(path, samples, rate):
    data = (np.clip(samples.numpy(), -1, 1) * 32767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data.tobytes())


# ----------------------------------------------------------------------------
# Prompts and generation


AUDIO, AUDIO_OUT, AUDIO_START, TEXT_END, MIXED_START, IM_END = 125031, 125032, 125012, 125014, 125015, 124900
EOS_CODE = 2016


def tokenizer(path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))


def embed_prompt(model, ids, audio_embeds=None):
    x = model.embedding[torch.tensor(ids)]
    if audio_embeds is not None:
        slots = [i for i, t in enumerate(ids) if t == AUDIO]
        assert len(slots) == audio_embeds.shape[0], (len(slots), audio_embeds.shape)
        x[slots] = audio_embeds
    return x


def save(path, array):
    np.asarray(array, dtype=np.float32).tofile(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command")
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--mmproj", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--wav")
    parser.add_argument("--text")
    parser.add_argument("--system")
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = Model(args.backbone, args.mmproj)
    summary = {}

    if args.command in ("features", "encode", "asr"):
        samples = read_wav(args.wav, 16000)
        mel = log_mel(samples)
        save(out / "mel.f32", mel)
        summary["mel_shape"] = list(mel.shape)
        embeds = model.encode(mel)
        save(out / "audio_embeds.f32", embeds)
        summary["audio_tokens"] = embeds.shape[0]
        if args.command == "asr":
            tok = tokenizer(args.tokenizer)
            system = args.system or "Perform ASR in English."
            head = tok.encode(f"<|startoftext|><|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n", add_special_tokens=False).ids
            tail = tok.encode("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False).ids
            ids = head + [AUDIO] * embeds.shape[0] + tail
            x = embed_prompt(model, ids, embeds)
            generated = []
            for _ in range(96):
                hidden = model.backbone(x)[-1]
                logits = model.logits(hidden)
                logits[[AUDIO, AUDIO_OUT, MIXED_START, AUDIO_START]] = float("-inf")
                token = int(logits.argmax())
                if not generated:
                    save(out / "first_logits.f32", logits.clamp(min=-1e30))
                    save(out / "prompt_hidden.f32", hidden)
                generated.append(token)
                if token in (IM_END, TEXT_END):
                    break
                x = torch.cat((x, model.embedding[token][None]))
            summary["prompt_ids"] = ids
            summary["generated"] = generated
            summary["text"] = tok.decode([t for t in generated if t < 124893])
    elif args.command == "depth":
        torch.manual_seed(0)
        hidden = torch.randn(2048) * 2.0
        codes = torch.randint(0, 2016, (12,))
        masked = torch.tensor([True, False] * 6)
        save(out / "depth_hidden.f32", hidden)
        np.asarray(codes.numpy(), dtype=np.int32).tofile(out / "depth_codes.i32")
        logits = model.depth_logits(hidden, codes, masked)
        save(out / "depth_logits.f32", logits)
        save(out / "frame_embedding.f32", model.frame_embedding(codes))
        summary["masked"] = masked.tolist()
    elif args.command == "codec":
        torch.manual_seed(1)
        codes = torch.randint(0, 2016, (args.frames, 12))
        np.asarray(codes.numpy(), dtype=np.int32).tofile(out / "codec_codes.i32")
        wav = model.codec_decode(codes)
        save(out / "codec_audio.f32", wav)
        summary["samples"] = wav.shape[0]
    elif args.command == "tts":
        tok = tokenizer(args.tokenizer)
        system = args.system or "Perform TTS in American English using a female voice."
        prompt = (
            f"<|startoftext|><|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{args.text}<|im_end|>\n"
            "<|im_start|>assistant\n<|audio_start|>"
        )
        ids = tok.encode(prompt, add_special_tokens=False).ids
        x = embed_prompt(model, ids)
        frames = []
        for step in range(args.frames):
            hidden = model.backbone(x)[-1]
            if step == 0:
                save(out / "tts_prompt_hidden.f32", hidden)
            codes = torch.zeros(12, dtype=torch.long)
            revealed = torch.zeros(12, dtype=torch.bool)
            # deterministic schedule: codebooks in index order, greedy codes
            for pick in (range(0, 4), range(4, 8), range(8, 12)):
                logits = model.depth_logits(hidden, codes, ~revealed)
                for i in pick:
                    codes[i] = int(logits[i].argmax())
                    revealed[i] = True
            if bool((codes == EOS_CODE).any()):
                break
            frames.append(codes.clone())
            x = torch.cat((x, model.frame_embedding(codes)[None]))
        summary["prompt_ids"] = ids
        summary["frames"] = len(frames)
        if frames:
            codes = torch.stack(frames)
            np.asarray(codes.numpy(), dtype=np.int32).tofile(out / "tts_codes.i32")
            wav = model.codec_decode(codes)
            write_wav(out / "tts.wav", wav, 24000)
    else:
        raise SystemExit(f"unknown command {args.command}")
    (out / f"{args.command}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("prompt_ids",)})[:2000])


if __name__ == "__main__":
    main()
