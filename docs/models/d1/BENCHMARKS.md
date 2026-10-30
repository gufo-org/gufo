# d1-Omni-2B measurements

October 30, 2026, Strix Halo gfx1151, production release build, Q8_0 backbone
and BF16 projector. Both servers use the same weights and toolchain. The
reference is the pinned llama.cpp HIP server identified in [QUALITY.md](QUALITY.md).
Each row is one complete warm HTTP request, including media decoding/encoding,
all of its question prompts and result readout. No tokens are generated.
Input tokens sum the separate question prompts. Model loading is excluded.
These are individual observations, with no averaging or speedup claim.

| Request | Input tokens | Gufo ms | llama.cpp ms |
| --- | ---: | ---: | ---: |
| text | 156 | 85.356 | 55.213 |
| image | 665 | 355.684 | 265.031 |
| audio | 352 | 128.569 | 131.668 |
| json-unicode | 57 | 29.447 | 20.741 |
| literal-controls | 51 | 29.878 | 18.729 |
| long-state | 734 | 163.448 | 140.732 |
| mixed-media | 954 | 421.406 | 365.050 |
| spoken-instructions | 171 | 71.609 | 68.715 |

Gufo is currently slower on most of these fixtures. The implementation runs
each question independently and allocates complete-prompt workspace for its
forward pass. These measurements do not qualify concurrent throughput,
maximum context or high-resolution images. The concurrent fixture in the
functional runner tests isolation, rather than performance.

Every observed duration and probability error is retained in the
[qualification record](artifacts/qualification.json). The initial failed
mixed-media run and the passing final run are both retained locally; no
failed result was discarded or a timing margin widened.
