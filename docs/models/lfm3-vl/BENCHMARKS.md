# LFM3-VL measurement scope

No matched end-to-end comparison or performance headline is qualified yet.
Focused HTTP evidence retains individual prefill/decode timings in
[qualification metrics](artifacts/qualification.json); those are observations.
The full native context and large-image workloads remain unmeasured.

The standard backbone smoke benchmark is retained locally as
`artifacts/lfm3-vl/backbone-bench.log`:

```sh
gufo bench --model models/lfm3-vl/LFM3-VL-2B-Q8_0.gguf \
  -p 128 -n 128 -r 1
```

This measures text-backbone work only. It excludes image decoding, resize,
vision encoding and projector work, so it is not a multimodal throughput
claim. See [quality](QUALITY.md) for the independent numerical checks and
scope of the HTTP histories.
