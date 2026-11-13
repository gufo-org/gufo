"""Real-model image generation through the OpenAI and A1111 (sdapi) APIs.

Both APIs share one generation path, so every sdapi request is checked
pixel-for-pixel against the OpenAI request with the same settings. The OpenAI
side goes through the OpenAI SDK; the sdapi side sends the bodies Open WebUI
("Automatic1111") and SillyTavern ("Stable Diffusion WebUI") send.
"""

import base64
import io
import json
import time

import openai
from PIL import Image

PROMPT = 'A small red cottage beside a lake at sunset, detailed illustration'
NEGATIVE = 'blurry, low quality'
EDIT = 'The same cottage in winter, covered in snow'
# Eight steps at 512 x 512 keep each image to seconds; Turbo accepts only 8.
SIZE, STEPS, SEED = 512, 8, 7


def pixels(encoded):
    image = Image.open(io.BytesIO(base64.b64decode(encoded, validate=True)))
    image.load()
    assert image.format == 'PNG', image.format
    return image.size, image.convert('RGBA').tobytes()


def varied(encoded):
    size, data = pixels(encoded)
    assert len(set(data[i:i + 4] for i in range(0, len(data), 4 * 97))) > 16, (
        'flat image')
    return size


def run(evidence, url, model, loras, lora_default, smoke=False):
    """`loras` maps adapter names to file paths; `lora_default` is a
    (name, scale) pair registered as --lora-default, or None. `smoke` stops
    after discovery, API parity, seeded replay, a reference edit and error
    recovery."""
    sdapi = url + '/sdapi/v1/'
    report = evidence.report
    sdk = openai.OpenAI(base_url=url + '/v1', api_key='unused', max_retries=0,
                        timeout=evidence.timeout)

    def openai_json(label, **fields):
        body = {'model': model, 'prompt': PROMPT, 'size': f'{SIZE}x{SIZE}',
                'steps': STEPS, 'seed': SEED, **fields}
        return [item['b64_json'] for item in evidence.json(
            url + '/v1/images/generations', label=label, json=body)['data']]

    def generate(label, route='txt2img', **fields):
        body = {'prompt': PROMPT, 'width': SIZE, 'height': SIZE,
                'steps': STEPS, 'seed': SEED, **fields}
        result = evidence.json(sdapi + route, label=label, json=body)
        assert set(result) == {'images', 'parameters', 'info'}, result.keys()
        assert result['parameters'] == body, 'parameters echo the request'
        assert isinstance(result['info'], str), 'info is a JSON string'
        info = json.loads(result['info'])
        assert len(info['all_seeds']) == len(result['images']) == len(
            info['infotexts']), info
        return result['images'], info

    def same(a, b, what):
        assert [pixels(x) for x in a] == [pixels(x) for x in b], what

    def error(label, route, body, status, text, method='POST'):
        code, raw, _ = evidence.call(sdapi + route, label=label, method=method,
                                     **({'json': body} if body is not None else {}))
        parsed = json.loads(raw)
        assert code == status and list(parsed) == ['error'] and text in parsed[
            'error'], (code, parsed)

    with evidence.case('discovery'):
        options = evidence.json(sdapi + 'options', label='options', method='GET')
        assert options == {'samples_format': 'png', 'sd_model_checkpoint': model}
        models = evidence.json(sdapi + 'sd-models', label='sd-models', method='GET')
        assert [item['title'] for item in models] == [model], models
        # Open WebUI posts the options back to select the checkpoint.
        evidence.json(sdapi + 'options', label='select-model',
                      json={**options, 'sd_model_checkpoint': models[0]['title']})
        error('select-unknown-model', 'options',
              {'sd_model_checkpoint': 'sd_xl_base_1.0'}, 400, model)
        for route in ('samplers', 'schedulers', 'upscalers',
                      'latent-upscale-modes'):
            assert isinstance(evidence.json(sdapi + route, label=route,
                                            method='GET'), list)
        progress = evidence.json(sdapi + 'progress', label='progress', method='GET')
        assert progress['state']['job_count'] == 0, progress
        listed = evidence.json(sdapi + 'loras', label='loras', method='GET')
        assert [item['name'] for item in listed] == list(loras), listed
        paths = {item['name']: item['path'] for item in listed}
        assert len(set(paths.values())) == len(paths), 'unique selection paths'
        report['lora_paths'] = paths

    with evidence.case('openai-sdk-and-txt2img-parity') as case:
        started = time.perf_counter()
        response = sdk.images.generate(
            model=model, prompt=PROMPT, size=f'{SIZE}x{SIZE}',
            response_format='b64_json',
            extra_body={'steps': STEPS, 'seed': SEED})
        case['sdk_ms'] = (time.perf_counter() - started) * 1000
        reference = [item.b64_json for item in response.data]
        assert varied(reference[0]) == (SIZE, SIZE)
        images, info = generate('txt2img')
        same(images, reference, 'sdapi txt2img equals the OpenAI SDK image')
        assert info['seed'] == SEED and info['steps'] == STEPS, info
        # The runner reports its default; Qwen-Image samples unguided.
        assert info['cfg_scale'] == 1, info
        same(generate('txt2img-replay')[0], reference, 'seeded replay')

    with evidence.case('edit'):
        # An uploaded reference conditions the prompt through the vision path.
        # One edit costs about three plain images; generation covers replay.
        upload = ('cottage.png', base64.b64decode(reference[0]), 'image/png')
        edited = evidence.json(url + '/v1/images/edits', label='edit', data={
            'model': model, 'prompt': EDIT, 'size': f'{SIZE}x{SIZE}',
            'steps': str(STEPS), 'seed': str(SEED)}, files={'image[]': upload})['data']
        assert len(edited) == 1 and varied(edited[0]['b64_json']) == (SIZE, SIZE)
        assert pixels(edited[0]['b64_json']) != pixels(reference[0]), 'the edit changes the image'

    with evidence.case('errors'):
        error('empty-prompt', 'txt2img', {'prompt': ''}, 400, 'prompt required')
        error('bad-size', 'txt2img', {'prompt': PROMPT, 'width': 500}, 400,
              'multiples of 32')
        error('get-txt2img', 'txt2img', None, 405, 'POST', method='GET')
        error('unknown-route', 'interrupt', {}, 404, 'no sdapi route')
        evidence.reject(url + '/v1/images/generations', 'invalid_request',
                        label='openai-bad-size',
                        json={'model': model, 'prompt': PROMPT, 'size': '500x500'})
        # The server still generates after the rejections.
        same(generate('after-errors')[0], reference, 'recovery')

    if smoke:
        report['image_suite'] = 'smoke'
        return

    with evidence.case('guidance'):
        guided = openai_json('openai-cfg', negative_prompt=NEGATIVE,
                             guidance_scale=4)
        images, info = generate('txt2img-cfg', negative_prompt=NEGATIVE,
                                cfg_scale=4)
        same(images, guided, 'cfg_scale equals guidance_scale')
        assert info['cfg_scale'] == 4, info
        assert pixels(guided[0]) != pixels(reference[0]), 'CFG changes the image'
        # Without a scale a negative prompt stays unguided, as in the
        # Qwen-Image-2.1 pipeline, on both APIs.
        images, info = generate('txt2img-negative-only', negative_prompt=NEGATIVE)
        same(images, reference, 'negative prompt alone is unguided')
        same(openai_json('openai-negative-only', negative_prompt=NEGATIVE),
             reference, 'OpenAI negative prompt alone is unguided')
        assert info['cfg_scale'] == 1, info

    if loras:
        names = list(loras)
        with evidence.case('lora'):
            defaults = openai_json('openai-lora-defaults')
            same(generate('txt2img-lora-defaults')[0], defaults,
                 'omitted lora uses --lora-default')
            none = openai_json('openai-lora-none', loras=[])
            same(generate('txt2img-lora-none', lora=[])[0], none,
                 'lora [] clears the defaults')
            if lora_default:
                assert pixels(defaults[0]) != pixels(none[0]), (
                    'the default adapter changes the image')
            pair = [{'name': names[0], 'scale': 0.8},
                    {'name': names[-1], 'scale': 0.5}]
            two = openai_json('openai-lora-two', loras=pair)
            same(generate('txt2img-lora-two', lora=[
                {'path': paths[item['name']], 'multiplier': item['scale']}
                for item in pair])[0], two, 'two adapters')
            one = openai_json('openai-lora-one',
                              loras=[{'name': names[0], 'scale': 0.8}])
            same(generate('txt2img-lora-repeated', lora=[
                {'path': paths[names[0]], 'multiplier': 0.4},
                {'path': paths[names[0]], 'multiplier': 0.4}])[0], one,
                'repeated paths sum their multipliers')
            error('lora-unknown-path', 'txt2img',
                  {'prompt': PROMPT, 'lora': [{'path': 'missing.safetensors'}]},
                  400, 'invalid lora path: missing.safetensors')
            error('lora-missing-path', 'txt2img',
                  {'prompt': PROMPT, 'lora': [{'multiplier': 1}]}, 400,
                  'lora.path required')

    else:
        report['lora_case'] = 'not run: no --lora adapters'

    # The rest selects no adapters so it runs with or without them.
    plain = {'lora': []} if loras else {}
    with evidence.case('img2img'):
        start = reference[0]
        half = openai_json('openai-strength', image=start, strength=0.5,
                           **({'loras': []} if loras else {}))
        images, info = generate('img2img', 'img2img', init_images=[start],
                                denoising_strength=0.5, **plain)
        same(images, half, 'img2img equals OpenAI image + strength')
        assert info['denoising_strength'] == 0.5, info
        # Omitted strength is 0.75; omitted size keeps the init image's.
        default = openai_json('openai-strength-default', image=start,
                              strength=0.75, **({'loras': []} if loras else {}))
        body = {'prompt': PROMPT, 'steps': STEPS, 'seed': SEED,
                'init_images': ['data:image/png;base64,' + start, 'ignored'],
                **plain}
        result = evidence.json(sdapi + 'img2img', label='img2img-defaults',
                               json=body)
        same(result['images'], default, 'default strength and size')
        assert json.loads(result['info'])['denoising_strength'] == 0.75
        error('img2img-without-image', 'img2img', {'prompt': PROMPT}, 400,
              'init_images required')
        error('img2img-mask', 'img2img',
              {'prompt': PROMPT, 'init_images': [start], 'mask': start}, 400,
              'mask')

    with evidence.case('batch'):
        batch = openai_json('openai-n2', n=2, **({'loras': []} if loras else {}))
        images, info = generate('txt2img-batch', batch_size=2, n_iter=3, **plain)
        same(images, batch, 'batch_size equals n; n_iter is ignored')
        assert info['all_seeds'] == [SEED, SEED + 1], info

    with evidence.case('client-payloads'):
        # Open WebUI: no seed, steps or cfg_scale.
        status, raw, _ = evidence.call(sdapi + 'txt2img', label='open-webui', json={
            'prompt': PROMPT, 'batch_size': 1, 'width': SIZE, 'height': SIZE,
            'steps': STEPS, **plain})
        assert status == 200, raw[:500]
        assert varied(json.loads(raw)['images'][0]) == (SIZE, SIZE)
        # SillyTavern: A1111 extras are ignored, cfg_scale 7 is applied.
        images, info = generate(
            'sillytavern', negative_prompt=NEGATIVE, sampler_name='DPM++ 2M',
            scheduler='Karras', cfg_scale=7, restore_faces=False,
            enable_hr=False, hr_upscaler='Latent', hr_scale=2,
            denoising_strength=0.7, override_settings={
                'CLIP_stop_at_last_layers': 1},
            override_settings_restore_afterwards=True, clip_skip=1,
            save_images=True, send_images=True, **plain)
        same(images, openai_json('openai-cfg7', negative_prompt=NEGATIVE,
                                 guidance_scale=7, **({'loras': []} if loras else {})),
             'SillyTavern payload equals OpenAI guidance 7')
        assert info['cfg_scale'] == 7 and info['sampler_name'] == 'Euler', info

    with evidence.case('api-separation'):
        # Each API ignores the other's fields; nothing is translated.
        same(openai_json('openai-with-sdapi-fields', cfg_scale=7, lora=[],
                         batch_size=3, negative_prompt=NEGATIVE), reference,
             'OpenAI ignores cfg_scale, lora and batch_size')
        same(generate('txt2img-with-openai-fields', guidance_scale=7, loras=[],
                      n=3, size='256x256', negative_prompt=NEGATIVE)[0],
             reference, 'sdapi ignores guidance_scale, loras, n and size')
        images, info = generate('lora-tag-literal', prompt=PROMPT + ' <lora:x:1>')
        assert info['prompt'].endswith('<lora:x:1>'), info
