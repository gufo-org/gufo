#!/usr/bin/env python3
"""Pinned official video decoder/processor/model oracle; imports no Gufo code.

Run with the pinned transformers source on PYTHONPATH, in nix develop.
Media is encoded once and retained beside the fixture so codec changes cannot
silently replace an already qualified vector. --media-only creates the clips.
"""
import argparse
from fractions import Fraction
import hashlib
import json
import subprocess
from pathlib import Path

import av
import numpy as np
import torch
import transformers
import huggingface_hub
from transformers import AutoConfig, AutoModel, AutoProcessor
from transformers.video_utils import VideoMetadata

REVISION = '50e35cc0ca18ec19b80653690de553e6decf74f5'
CHECKPOINT = '914f7f89142e33e77833254d9c9b90c3cef7303b'
SPECS = [
    ('short', 64, 48, 3, '5', 'libx264', 'mp4'),
    ('fractional', 96, 48, 60, '30000/1001', 'libx264', 'mp4'),
    ('portrait', 32, 96, 7, '2', 'libvpx-vp9', 'webm'),
    ('boundary', 48, 48, 32, '1', 'libx264', 'mp4'),
    ('overflow', 48, 48, 66, '2', 'libx264', 'mp4'),
]


def media(root):
    root.mkdir(parents=True, exist_ok=True)
    for name, width, height, frames, fps, codec, extension in SPECS:
        path = root / f'{name}.{extension}'
        if path.exists():
            continue
        with av.open(str(path), 'w') as container:
            stream = container.add_stream(codec, rate=Fraction(fps))
            stream.width, stream.height, stream.pix_fmt = width, height, 'yuv420p'
            for t in range(frames):
                y, x, c = np.indices((height, width, 3))
                rgb = ((x*17 + y*31 + c*67 + t*29 + x*y % 251) % 256).astype(np.uint8)
                for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format='rgb24')):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

    for name, size, rate, frames in [('oversized', '2048x2050', '1', 1),
                                     ('overduration', '16x16', '1/301', 1),
                                     ('overframes', '16x16', '30', 4097)]:
        path = root / f'{name}.mp4'
        if not path.exists():
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                f'color=c=black:s={size}:r={rate}', '-frames:v', str(frames),
                '-c:v', 'libx264', '-threads', '1', '-pix_fmt', 'yuv420p', str(path)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='models/embeddinggemma-2')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--media-only', action='store_true')
    parser.add_argument('--case', action='append', help='Generate only the named probes')
    args = parser.parse_args()
    root = args.output.parent / 'video'
    media(root)
    audio_track = root / 'with_audio.mp4'
    if not audio_track.exists():
        subprocess.run(['ffmpeg', '-v', 'error', '-i', str(root/'short.mp4'),
            '-f', 'lavfi', '-i', 'sine=frequency=440:duration=0.6', '-map', '0:v:0',
            '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac', '-shortest', str(audio_track)], check=True)
    if args.media_only:
        return
    torch.set_num_threads(8)
    processor = AutoProcessor.from_pretrained(args.model)
    config = AutoConfig.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, config=config, dtype=torch.float32,
                                     attn_implementation={'text_config': 'sdpa', 'vision_config': 'sdpa', 'audio_config': 'eager'}).eval()
    cases = []
    probes = [(s[0], [root / f'{s[0]}.{s[-1]}'], '<|video|>', {}) for s in SPECS]
    probes += [('linspace_rounding', [root/'boundary.mp4'], '<|video|>', {'fps': 0.96875, 'max_frames': 23}),
               ('audio_track', [audio_track], '<|video|>', {}),
               ('upsample', [root/'short.mp4'], '<|video|>', {'fps': 10, 'add_timestamps': True}),
               ('sampling_control', [root/'portrait.webm'], '<|video|>', {'fps': 0.5, 'max_frames': 1}),
               ('timestamps', [root/'portrait.webm'], 'before <|video|> after', {'add_timestamps': True}),
               ('truncate', [root/'overflow.mp4'], '<|video|>', {'max_frames': 3, 'overflow_strategy': 'truncate'}),
               ('repeat', [root/'short.mp4', root/'short.mp4'], '<|video|> then <|video|>', {}),
               ('mixed', [root/'short.mp4', root/'portrait.webm'],
                'before <|video|> middle <|image|> <|audio|> after <|video|>', {})]
    for name, paths, text, options in probes:
        if args.case and name not in args.case:
            continue
        videos, records, metadatas = [], [], []
        for path in paths:
            def sample(metadata, **kwargs):
                return processor.video_processor.sample_frames(metadata, fps=options.get('fps', 1),
                    max_frames=options.get('max_frames', 32),
                    overflow_strategy=options.get('overflow_strategy', 'uniform'))
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                fps = float(stream.average_rate)
                decoded = np.stack([frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)])
            metadata = VideoMetadata(total_num_frames=len(decoded), fps=fps,
                duration=len(decoded)/fps, width=decoded.shape[2], height=decoded.shape[1], video_backend='pyav')
            indices = sample(metadata)
            frames = decoded[indices]
            metadata.frames_indices = indices
            videos.append(decoded)
            metadatas.append(VideoMetadata(total_num_frames=len(decoded), fps=fps,
                duration=len(decoded)/fps, width=decoded.shape[2], height=decoded.shape[1], video_backend='pyav'))
            encoded = processor.video_processor([frames], do_sample_frames=False, return_tensors='pt')
            records.append({'path': 'video/' + path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'indices': metadata.frames_indices.tolist(), 'fps': metadata.fps,
                            'rgb_sha256': [hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames],
                            'patches': []})
            for pixels, positions in zip(encoded['pixel_values_videos'], encoded['video_position_ids']):
                valid = (positions != -1).all(-1)
                raw = (pixels[valid] * 255).round().to(torch.uint8).numpy().tobytes()
                records[-1]['patches'].append({'rows': int(positions[valid, 1].max())+1,
                                             'columns': int(positions[valid, 0].max())+1,
                                             'sha256': hashlib.sha256(raw).hexdigest()})
        # Pass encoded paths to exercise official decoding and its metadata.
        kwargs = {}
        if name == 'mixed':
            from PIL import Image
            kwargs['images'] = [[Image.new('RGB', (32, 32), (255, 0, 0))]]
            kwargs['audio'] = [np.sin(np.arange(3200, dtype=np.float32)*0.07)*0.1]
            kwargs['audio_kwargs'] = {'sampling_rate': 16000}
        inputs = processor(text=[text], videos=videos,
                           videos_kwargs={'video_metadata': metadatas, **options},
                           return_tensors='pt', **kwargs)
        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state
            mask = inputs['attention_mask'].unsqueeze(-1)
            vector = torch.nn.functional.normalize((hidden*mask).sum(1)/mask.sum(1), dim=-1)[0].tolist()
        cases.append({'name': name, 'text': text, 'videos': records, 'options': options,
                      'token_ids': inputs['input_ids'][0].tolist(),
                      'embedding': [float(f'{v:.7g}') for v in vector]})
        print(name, len(cases[-1]['token_ids']), 'tokens', flush=True)
    args.output.write_text(json.dumps({'model_revision': CHECKPOINT,
            'transformers_revision': REVISION, 'transformers': transformers.__version__,
            'torch': torch.__version__, 'av': av.__version__,
            'huggingface_hub': huggingface_hub.__version__,
            'av_libraries': av.library_versions,
            'attention': {'text_config': 'sdpa', 'vision_config': 'sdpa', 'audio_config': 'eager'}, 'dtype': 'float32', 'cases': cases}, separators=(',', ':')))


if __name__ == '__main__':
    main()
