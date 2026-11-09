"""Sampling-defaults expectations for models without an in-tree preset.

Run in nix develop for the OpenAI SDK.
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import contextlib
import io
import json
import threading
import unittest
from unittest.mock import patch

import openai_sdk

# What /props reports for LFM3, whose provider declares its own defaults.
PARAMS = {"temperature": 0.10000000149011612, "top_k": 50, "top_p": 1.0, "min_p": 0.0,
          "seed": -1, "repeat_penalty": 1.100000023841858, "repeat_last_n": 64,
          "frequency_penalty": 0.0, "presence_penalty": 0.0, "max_tokens": -1,
          "n_predict": -1}


class Stop(Exception):
    """Stop after the first thinking mode's explicit replay."""


class Checks(dict):
    def __setitem__(self, name, value):
        super().__setitem__(name, value)
        if name == "sampling_preset_thinkingFalse":
            raise Stop


class SamplingDefaultsTest(unittest.TestCase):
    def test_provider_defaults_come_from_props(self):
        class Props(BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps({"default_generation_settings": {
                    "n_ctx": 8192, "params": PARAMS}}).encode()
                self.send_response(200 if self.path == "/props" else 404)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Props)
        worker = threading.Thread(target=server.serve_forever)
        worker.start()
        try:
            port = server.server_address[1]
            defaults = openai_sdk.provider_sampling(f"http://127.0.0.1:{port}/v1")
        finally:
            server.shutdown()
            worker.join(timeout=5)
            server.server_close()
        self.assertEqual(defaults, {key: PARAMS[key] for key in openai_sdk.PROVIDER_SAMPLING})

    def test_provider_preset_writes_out_the_reported_defaults(self):
        requests = []

        def chat_result(client, request, streaming=False, on_chunk=None):
            requests.append(request)
            reasoning = request.get("extra_body", {}).get("chat_template_kwargs", {}).get(
                "enable_thinking", True)
            return {"text": "Four.", "reasoning": "Think." if reasoning else "",
                    "finish": "length", "tools": [], "usage": {}}

        defaults = {key: PARAMS[key] for key in openai_sdk.PROVIDER_SAMPLING}
        with patch.object(openai_sdk, "chat_result", chat_result), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(Stop):
            openai_sdk.check_sampling_defaults(None, "lfm3", Checks(), "provider", {},
                                               provider=defaults)
        default, explicit_default, inherited, supplied = requests[:4]
        # Default thinking is compared with an explicit thinking mode, not
        # with an effort that only the in-tree presets define.
        self.assertNotIn("reasoning_effort", explicit_default)
        self.assertEqual(explicit_default["extra_body"]["chat_template_kwargs"],
                         {"enable_thinking": True})
        self.assertNotIn("temperature", inherited)
        self.assertEqual({key: supplied[key] for key in
                          ("temperature", "top_p", "presence_penalty", "frequency_penalty")},
                         {"temperature": PARAMS["temperature"], "top_p": 1.0,
                          "presence_penalty": 0.0, "frequency_penalty": 0.0})
        self.assertEqual({key: supplied["extra_body"][key] for key in
                          ("top_k", "min_p", "repeat_penalty", "repeat_last_n")},
                         {"top_k": 50, "min_p": 0.0, "repeat_penalty": PARAMS["repeat_penalty"],
                          "repeat_last_n": 64})
        self.assertNotIn("min_keep", supplied["extra_body"])

    def test_reported_defaults_need_the_provider_preset(self):
        for preset, provider in (("provider", None), ("qwen38", {"temperature": 1.0})):
            with self.subTest(preset=preset), self.assertRaises(ValueError):
                openai_sdk.check_sampling_defaults(None, "model", {}, preset, {},
                                                   provider=provider)


if __name__ == "__main__":
    unittest.main()
