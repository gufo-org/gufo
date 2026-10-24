"""Coverage refusal checks for audio correctness and terminal streaming events."""

import base64
import math
from pathlib import Path
import tempfile
import unittest

import httpx

from service_http import Evidence, events, audio_check, pcm_wav, speech_events, transcript_events, word_error_rate


class ServiceContracts(unittest.TestCase):
    def test_word_error_rate_counts_insertions_deletions_and_unicode(self):
        self.assertEqual(word_error_rate('HELLO, world!', 'hello world'), 0)
        self.assertEqual(word_error_rate('è una prova', 'È una prova.'), 0)
        self.assertEqual(word_error_rate('one extra two', 'one two'), .5)
        self.assertEqual(word_error_rate('one', 'one two'), .5)
        self.assertEqual(word_error_rate('', 'one two'), 1)
        with self.assertRaises(ValueError):
            word_error_rate('one', '')

    def test_streams_require_completion_and_matching_payload(self):
        delta = {'type': 'transcript.text.delta', 'delta': 'hello'}
        done = {'type': 'transcript.text.done', 'text': 'hello'}
        self.assertEqual(transcript_events([delta, done]), 'hello')
        for items in ([], [delta], [delta, {**done, 'text': 'other'}], [done, done]):
            with self.assertRaises(AssertionError):
                transcript_events(items)
        audio = {'type': 'speech.audio.delta', 'audio': base64.b64encode(b'pcm').decode()}
        end = {'type': 'speech.audio.done'}
        self.assertEqual(speech_events([audio, end]), b'pcm')
        for items in ([], [audio], [end], [end, end]):
            with self.assertRaises(AssertionError):
                speech_events(items)

    def test_malformed_stream_retains_wire_evidence(self):
        wire = 'event: speech.audio.delta\ndata: {broken json}\n\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'partial.sse'
            with self.assertRaises(ValueError):
                list(events(httpx.Response(200, text=wire), path=path))
            self.assertEqual(path.read_text(), wire)
        wire = 'event: transcript.text.delta\ndata: {"type":"other"}\n\n'
        with self.assertRaises(AssertionError):
            list(events(httpx.Response(200, text=wire)))

    def test_failed_case_is_retained(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = Evidence(root, {'status': 'running'})
            with self.assertRaisesRegex(AssertionError, 'incorrect transcript'):
                with evidence.case('quality'):
                    raise AssertionError('incorrect transcript')
            report = json.loads((root / 'report.json').read_text())
            self.assertEqual(report['cases'][0]['status'], 'failed')
            self.assertIn('incorrect transcript', report['cases'][0]['error'])

    def test_audio_rejects_silence_truncation_and_partial_samples(self):
        for data in (b'', bytes(24000), b'x'):
            with self.assertRaises(AssertionError):
                audio_check(data, wav=False)
        pcm = b'\x00\x10' * 24000
        decoded, metrics = audio_check(pcm_wav(pcm))
        self.assertEqual(decoded, pcm)
        self.assertEqual(metrics['duration_seconds'], 1)
        self.assertTrue(math.isfinite(metrics['rms']))
        with self.assertRaises(AssertionError):
            audio_check(pcm_wav(pcm)[:-2])


if __name__ == '__main__':
    unittest.main()
