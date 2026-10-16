"""Qwen control tokens, mirrored from ``src/models/qwen/control_tokens.hpp``.

Keep shared names and values identical to the C++ catalogue. Validate changes
against pinned model metadata and independent reference fixtures; centralizing
these strings does not change tokenization or EOS handling.
"""

from __future__ import annotations

# The shared Qwen vocabulary, reused by Qwen3.8, Qwen-Image, Qwen3-TTS,
# Qwen3-ASR and MiniMax H3.
kEndOfText = "<|endoftext|>"
kImStart = "<|im_start|>"
kImEnd = "<|im_end|>"
kObjectRefStart = "<|object_ref_start|>"
kObjectRefEnd = "<|object_ref_end|>"
kBoxStart = "<|box_start|>"
kBoxEnd = "<|box_end|>"
kQuadStart = "<|quad_start|>"
kQuadEnd = "<|quad_end|>"
kVisionStart = "<|vision_start|>"
kVisionEnd = "<|vision_end|>"
kVisionPad = "<|vision_pad|>"
kImagePad = "<|image_pad|>"
kVideoPad = "<|video_pad|>"

# MiniMax H3 audio/video additions on top of the shared vocabulary, used by the
# H3 source-manifest and quality checks. They are not declared in the C++ header
# because no C++ translation unit consumes them.
kLyricsStart = "<|lyrics_start|>"
kLyricsEnd = "<|lyrics_end|>"
kCaptionStart = "<|caption_start|>"
kCaptionEnd = "<|caption_end|>"

# The special tokens the MiniMax H3 tokenizer config must declare, in the order
# the pinned manifest lists them.
H3_SPECIAL_TOKENS = (
    kImStart,
    kImEnd,
    kVisionStart,
    kVisionEnd,
    kImagePad,
    kVideoPad,
    kLyricsStart,
    kLyricsEnd,
    kCaptionStart,
    kCaptionEnd,
)
