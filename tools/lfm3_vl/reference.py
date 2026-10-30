#!/usr/bin/env python3
"""Compare native VL patches, image embeddings, prompt IDs and full logits.

Loads unmodified pinned LiquidAI source and strictly maps dequantized GGUF
weights into its FP32 CPU classes. Production Gufo does not depend on Python.
Requires transformers 5.21.0, torch, torchvision, pillow, gguf-py.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys
import types

import numpy as np
import torch
from gguf import GGUFReader
from PIL import Image
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.d1.reference import load_weights, text_name, vision_name
sys.path.insert(0, str(ROOT / 'tests/functional'))
from lfm3_logits import compare
from lfm3_vl import png

REVISION = 'ba44bd51db20a2753f28c273a1571ccaa210d954'
PINNED_FILES = {'configuration_lfm3.py': 'dd2f4b18eed69f3a303dbf438a5ecd20d6375abad8d1aa62699b3d5294e6a71a', 'modeling_lfm3.py': '4a676bee2cf184d51a9dacbb9d7c4abae85515a042a920f8e2578b22bba3d6e8', 'lfm3_conv.py': '80d4a985382a22774212759248a6b4c737bfbf9c2a47b3dd7f797dda748e4bb4', 'configuration_lfm3_vl.py': 'ddd68fa327d634db8bbaade1080264dc1206be613db6fbc3521998251a00e2a5', 'modeling_lfm3_vl.py': 'a41cbe7242f379808bcc005207ee0d7b1709528966c1e4da46baa7c41827ac79', 'image_processing_lfm3_vl.py': '57db0cfa668bf89a6ff898874e1321e78c257d200f25c98e0ea52b30b8740088', 'config.json': '18a02e8b5963e38c98683e2c948705ea45220d75f017e9b4bdd9cf54692e6dc3', 'tokenizer.json': 'a39eed58008cbcbc09e5b629b81338a654269c64cd13348aaa6090f4db75c7e7', 'tokenizer_config.json': '0117c195c81c937dbef42e384fa2850aebd066d14b7ea8345cabfbb5c05e584c', 'chat_template.jinja': 'f45005c50cd2a97329722bf3a2dee9f16c04cdf825a589b280fb7e99e6827818', 'processor_config.json': 'd3d5c3f9dc3d3f25c3177ba8bb05938b53c51799244af1ca6406cde78924b84e'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'mmproj', 'reference-dir', 'probe', 'image', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name, expected in PINNED_FILES.items():
        if hashlib.sha256((args.reference_dir / name).read_bytes()).hexdigest() != expected:
            raise ValueError('reference differs from pinned source: ' + name)
    torch.set_num_threads(8)
    sys.path.insert(0, str(args.reference_dir.resolve().parent))
    package = args.reference_dir.name
    mod = lambda name: importlib.import_module(package + '.' + name)
    recorded = json.loads((args.reference_dir / 'config.json').read_text())
    lm, lc = mod('modeling_lfm3'), mod('configuration_lfm3')
    vl, vc = mod('modeling_lfm3_vl'), mod('configuration_lfm3_vl')
    cfg = lc.Lfm3Config(**recorded['text_config'])
    cfg._attn_implementation = 'sdpa'
    vcfg = vc.Lfm3VlVisionConfig(**recorded['vision_config'])
    vcfg._attn_implementation = 'sdpa'
    with torch.device('meta'):
        model = lm.Lfm3ForCausalLM(cfg)
        vision = vl.Lfm3VlVisionModel(vcfg)
        projector = vl.Lfm3VlMultiModalProjector(types.SimpleNamespace(
            downsample_factor=2, vision_config=vcfg, projector_layer_norm_eps=1e-6,
            projector_hidden_size=4096, projector_bias=True, projector_hidden_act='gelu', text_config=cfg))
    model.model.rotary_emb = lm.Lfm3RotaryEmbedding(config=cfg)
    vision.rotary_pos_emb = vl.Lfm3VlVisionRotaryEmbedding(vcfg)
    print('loading strict FP32 reference weights', flush=True)
    model = load_weights(model, GGUFReader(str(args.model)), text_name, 'cpu')
    model.lm_head.weight = model.model.embed_tokens.weight
    mm = GGUFReader(str(args.mmproj))
    vision = load_weights(vision, mm, vision_name, 'cpu')
    names = {'layer_norm.weight': 'mm.input_norm.weight', 'layer_norm.bias': 'mm.input_norm.bias',
             'linear_1.weight': 'mm.0.weight', 'linear_1.bias': 'mm.0.bias',
             'linear_2.weight': 'mm.2.weight', 'linear_2.bias': 'mm.2.bias'}
    projector = load_weights(projector, mm, lambda n: names[n], 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.reference_dir, trust_remote_code=True)
    (output / 'red.png').write_bytes(png((255, 0, 0)))
    (output / 'wide.png').write_bytes(png((0, 0, 255), 1024, 512))
    red, wide, cats = (str(p.resolve()) for p in (output / 'red.png', output / 'wide.png', args.image))
    def message(text, image, offset=0, detail='auto'):
        return {'role': 'user', 'content': text,
                'images': [{'offset': offset, 'path': image, 'detail': detail}]}
    cases = [
        ('red', [message('What color is the image?', red)], False),
        ('cats-thinking', [message('Describe the image.', cats)], True),
        ('wide-auto', [message('Describe the image.', wide)], False),
        ('wide-low', [message('Describe the image.', wide, detail='low')], False),
        ('interleaved-two', [{'role': 'user', 'content': 'First. Second. Describe both.', 'images': [
            {'offset': 7, 'path': red}, {'offset': 15, 'path': wide, 'detail': 'low'}]}], False),
        ('follow-up', [message('What color?', red), {'role': 'assistant', 'content': 'Red.'},
            {'role': 'user', 'content': 'What was the color?'}], False),
    ]
    report = {'source_revision': REVISION, 'teacher': 'unmodified LiquidAI classes; CPU FP32; dequantized GGUF weights',
              'cases': {}, 'status': 'running'}
    for name, messages, thinking in cases:
        print('reference', name, flush=True)
        recipe = {'messages': messages, 'thinking': thinking}
        request = output / (name + '.request.json')
        request.write_text(json.dumps(recipe, indent=2))
        stem = output / name
        with (output / (name + '.native.log')).open('w') as log:
            subprocess.run([str(args.probe.resolve()), str(args.model.resolve()), str(args.mmproj.resolve()),
                            str(request), str(stem)], check=True, stdout=log, stderr=subprocess.STDOUT)
        native = json.loads(stem.with_suffix('.json').read_text())
        hf_messages, counts, features, embeddings, errors = [], [], [], [], []
        for message_value in messages:
            text = message_value['content']
            parts, cursor = [], 0
            for image in message_value.get('images', []):
                parts += [{'type': 'text', 'text': text[cursor:image['offset']]}, {'type': 'image'}]
                cursor = image['offset']
                pixels = Image.open(image['path']).convert('RGB')
                budget = max(65536, 512 * 512 * min(pixels.size) // max(pixels.size)) if image.get('detail') == 'low' else 16777216
                processor = mod('image_processing_lfm3_vl').Lfm3VlImageProcessor(min_pixels=65536, max_pixels=budget)
                feature = processor(images=[pixels], return_tensors='pt')
                features.append(feature)
                with torch.inference_mode():
                    rows = projector(vision(pixel_values=feature['pixel_values'], grid_thw=feature['image_grid_thw']).last_hidden_state)
                embeddings.append(rows)
                counts.append(rows.shape[0])
            parts.append({'type': 'text', 'text': text[cursor:]})
            hf_messages.append({'role': message_value['role'], 'content': parts})
        rendered = tokenizer.apply_chat_template(hf_messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        counts_iter = iter(counts)
        rendered = re.sub(re.escape('<|image|>'), lambda _: '<|image|>' * next(counts_iter), rendered)
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        assert ids == native['tokens'], (name, 'independent prompt IDs differ')
        tokens = torch.tensor([ids])
        with torch.inference_mode():
            inputs = model.model.embed_tokens(tokens)
            for i, (feature, rows, grid) in enumerate(zip(features, embeddings, native['images'])):
                h, w = feature['image_grid_thw'][0, 1:].tolist()
                assert [h, w] == [grid['height'], grid['width']], (name, 'grid')
                patches = feature['pixel_values'].reshape(-1, 3, 2, 16, 16)[:, :, 0].flatten(1).numpy()
                actual_patches = np.fromfile(str(stem) + f'.image{i}.patches.bin', dtype=np.float32).reshape(patches.shape)
                actual_rows = np.fromfile(str(stem) + f'.image{i}.embeddings.bin', dtype=np.float32).reshape(rows.shape)
                patch_error = float(np.max(np.abs(patches - actual_patches)))
                embedding_error = float(np.max(np.abs(rows.numpy() - actual_rows)))
                errors.append({'patch_max_abs': patch_error, 'embedding_max_abs': embedding_error})
                assert patch_error < 1e-6 and embedding_error < .002, (name, errors)
                inputs[0, grid['offset']:grid['offset'] + rows.shape[0]] = rows
            expected = model(inputs_embeds=inputs, logits_to_keep=1).logits[0, -1].float().numpy()
        actual = np.fromfile(str(stem) + '.logits.bin', dtype=np.float32)
        parity = compare(actual.tolist(), expected.tolist())
        expected.tofile(output / (name + '.teacher.logits.bin'))
        report['cases'][name] = {'token_count': len(ids), 'images': errors, 'logits': parity}
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(name, parity, flush=True)
        assert parity['top1_equal'] and parity['kl_teacher_candidate'] < 1e-5 and parity['total_variation'] < .002, (name, parity)
    report['status'] = 'passed'
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
