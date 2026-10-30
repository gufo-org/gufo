# d1-Omni-2B measurements

## Packed sets and boxes

October 30, 2026, production release binary, Strix Halo gfx1151, GCC 16.2 /
ROCm 10.1 HIP Clang 24. Same pinned Q8_0 backbone, BF16 projector and 8192-token
context as commit `431efcc7`. Every cell below is an individual warm HTTP
request, including serialization/transport, media processing, forward passes
and readout. Loading is excluded; output tokens are always zero. Warmups and
all earlier attempts remain in the local raw reports. No requests are averaged.

Matched `set_size=1` controls preserve their exact responses and actual input
work. None exceeds both the 5% and 10 ms timing regression thresholds:

| Request | Input tokens | `431efcc7` ms | Candidate ms |
| --- | ---: | ---: | ---: |
| text | 150 | 90.447 | 90.183 |
| image | 665 | 344.966 | 351.777 |
| audio | 352 | 125.633 | 130.762 |
| long | 734 | 165.666 | 166.775 |

Default packing reads the same questions in sets of 16. Grouping changes the
trained function as well as repeated state work, so this table is not an
identical-logit speed comparison with `set_size=1`:

| Request | Packed input tokens | Candidate ms |
| --- | ---: | ---: |
| text | 122 | 38.961 |
| image | 359 | 270.840 |
| audio | 208 | 81.534 |

One-picture `boxes` requests, with the requested number of answer blocks:

| Request | Input tokens | Candidate ms |
| --- | ---: | ---: |
| boxes-1 | 422 | 281.282 |
| boxes-4 | 482 | 284.496 |
| boxes-16 | 722 | 341.991 |

[Retained compact evidence](artifacts/packed-qualification.json). Raw requests,
responses and prior attempts are under local ignored `artifacts/d1/packed/`.
The independent Python reference is a CPU FP32 quality teacher; its execution
time is not used as a performance comparison. No throughput, high-resolution
or maximum-context claim is made.

## Original single-question implementation

These earlier llama.cpp observations cover the original single-question path
and its own fixtures; they are retained as historical evidence, not current
packed-path measurements.


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
