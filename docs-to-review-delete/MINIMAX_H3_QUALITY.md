# MiniMax H3 Quality Oracles

Status: audited foundation contract, 2026-08-22

## Rule

MiniMax H3 work is promoted in this order:

1. an independent analytic or pinned teacher oracle;
2. correctness and quality gates;
3. profiling;
4. optimization or quantization.

A successful kernel launch, finite latent, recognizable frame, or playable MP4
is not a quality result. The released exact route—BF16/F32 for the denoiser and
released F32 for both VAEs—is the reference for every later fast or quantized
route.

The dependency-light oracle and corruption tests run separately from the
pinned PyTorch teacher and LPIPS construction test:

```sh
nix build .#checks.x86_64-linux.h3-quality
nix build .#checks.x86_64-linux.h3-ml-quality
```

The generic host CTest suite does not import PyTorch. The dependency-light
`h3-quality` derivation is part of the canonical hosted PR gate. The
multi-gigabyte ROCm PyTorch/LPIPS closure is an explicit offline gate so a
GitHub-hosted runner does not exhaust its disk while realizing dependencies.

## Pinned Contract

The committed contract is
`tests/fixtures/minimax_h3/quality-contract-v1.json`. It pins:

- MiniMax H3 FL2VA revision
  `42ed227ee7df40d41602854ae760620d6eb651fe`;
- `h3.c` revision `8974cc055ea9c02fcd14cc27dfda3e1027c05153`;
- the source-manifest SHA-256;
- exact tokenizer IDs and special-token IDs from the operator-supplied
  checkpoint;
- a 256x256x22 fox artifact whose first complete 50-block forward is the
  routine end-to-end denoiser gate;
- release-only 512x512x22 fox and independent ceramic-mug cases configured for
  the audited 49-evaluation exact route;
- component parity ceilings inherited from the pinned upstream tests.

Changing a prompt, seed, canvas, frame count, schedule, block count, reuse
interval, metric, or ceiling creates a new contract version. A threshold is
never loosened to admit a candidate.

The pre-audit multi-step fox artifacts are historical only. They used reversed
AdaLN shift/scale interpretation, restarted video/audio RNG streams, BF16
Euler velocity boundaries, and the old preset counts. They are not promotion
evidence and must not be compared with the audited native route.

## Oracle Pyramid

Small host fixtures cover token IDs, special tokens, temporal alignment,
packed-row maps, schedules, patching, seeded noise, reuse selection, and Euler
updates. Those values are hand-checkable and committed.

Component artifacts then retain:

- Qwen layer-50 hidden states;
- one full DiT block and AdaLN modulation;
- complete video and audio velocity outputs;
- VisualVAE latent-to-frame output;
- AudioVAE latent-to-stereo-waveform output.

End-to-end artifacts retain initial and final latents, selected or complete
frames, stereo audio, metrics, and human review. The rapid case decodes only
frames 0, 11, and 21. H3 is still run with its legal 22-frame latent geometry;
the selected-frame path is a development diagnostic, not a one-frame model.
Generating that selected-frame artifact is reserved for a one-time visual
smoke test after a cross-boundary change passes its operator, block, and
single-forward parity gates; ordinary iterations do not advance a denoising
schedule.

The large artifacts are deliberately absent from Git. CI validates the schema,
metrics, corruption behavior, and committed small fixtures but never
regenerates teacher data.

## Content-Addressed Artifact Format

Every external artifact lives at:

```text
<store>/<sha256>/
  manifest.json
  ... payloads named by manifest entries ...
```

The directory name is the SHA-256 of the canonical manifest after omitting its
self-referential `artifact_id`. The manifest records:

- contract ID and file hash;
- model/source manifest and immutable revisions;
- oracle implementation and revision;
- runtime version, platform, hardware, command, storage/compute/accumulation
  dtypes;
- prompt and prompt hash, seed, canvas, frames, evaluations, blocks, and reuse;
- every payload's relative path, role, size, SHA-256, dtype, and shape;
- measured metrics and their frozen threshold source;
- the capture count, with byte identity claimed only when at least two
  inexpensive captures were deliberately made.

Absolute paths, parent traversal, symbolic links, unreferenced files,
non-finite metrics, wrong hashes, and incomplete role sets fail closed.

Seal a completed staging directory atomically:

```sh
nix develop -c python3 tools/h3/gufo-h3-quality.py seal \
  --staging /var/llms/h3-oracles/staging/fox-layer50 \
  --template /var/llms/h3-oracles/specs/fox-layer50.json \
  --store /var/llms/h3-oracles/sha256
```

Verify without executing the model:

```sh
nix develop -c python3 tools/h3/gufo-h3-quality.py verify \
  --artifact /var/llms/h3-oracles/sha256/<sha256>
```

The production runtime never imports model-provided Python. A separately
isolated offline teacher capture may use a pinned official runtime, but that
runtime, its command, and its dependency versions become part of the artifact.

## Prompt-Encoder Frozen Evidence

The text-only port was checked against isolated Transformers 5.14.1 and ROCm
PyTorch 2.12.0/HIP 7.2.53211 on the supported Radeon 8060S. The teacher
instantiates only the Qwen3-VL text model on the meta device, loads the exact
embedding and requested layers from the pinned H3 shards, uses eager
attention, and captures the BF16 tensor entering the final RMSNorm.

For `A red fox walking through snow`, token IDs are
`32 2518 38835 11435 1526 11794`. External raw-BF16 payloads are:

| Boundary | Shape | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| after layer 1 | 6x5120 | 61,440 | `72c28520b592a79bf278fcced10f215b44e12864e803a7ca2a18e7528305ef08` |
| after layer 50 | 6x5120 | 61,440 | `8015af1d2a551a0509bccc84372a30087f66721613f909c3f99e36ad7e269692` |

The gfx1151 HIP implementation matched both payloads byte-for-byte: measured
relative max and relative L2 were zero. The retained layer-50 invocation took
21.38 seconds, issued 850 dispatches, and left zero registered host bytes.
Routine quality work invokes this boundary once; it does not repeat the
complete 50-layer encoder for statistical evidence.

## Host Sampler Frozen Evidence

The text-only sampler tests freeze the exact released linear base grid after
independent video shift 12 and audio shift 3. Concatenated little-endian F32
video/audio arrays have these SHA-256 values:

| Sigma points / evaluations | Schedule SHA-256 |
| ---: | --- |
| 5 / 4 | `dee27052a1aa4aa508a7b81bbe39e6bb4b880dd2d61dd3953380c5d0da3eb686` |
| 8 / 7 | `51488c2e9779388e9b62aa3e0981bd5d654c38ac1ce5358be5e317fb63213ffc` |
| 20 / 19 | `0f71f36f5f3066f0c1aece22ffb9dd2150cc85c3b52d3f15e6a0cec2bd593e9b` |
| 50 / 49 | `ddb451d3edad496ef895ce468774632a570434e23be3d7b05a3b326c9090099c` |

For six text rows and the 22-frame temporal shape, canonical packed-layout and
step-7 modulation-map hashes are:

| Canvas | Rows | Layout SHA-256 | Row-map SHA-256 |
| --- | ---: | --- | --- |
| 256x256 | 528 | `61be3a596b2670766a7e86b5a526e13438e89a68523d99eb5bdd9f2c8ca7f61a` | `8574df4af41e95bb6ec35f3ec9459ceb88a319f68ad07e9ec70af5a84cb6a6e3` |
| 512x512 | 1,872 | `c3481ff47fabdf534823889a95df5655bd19ac11ff1755f5daa7b6e1655d00b0` | `c95bbf9938ac6b5c2d901af8466d374f6e173953954872e19ec12ed82fec98a1` |

The 256x256x22 seed-42 initial video and audio noise payloads hash together to
`6318dbfea74c61415d470c12c019cda9df6a8491f86b075e403d1c2fc2403b4d`.
One request stream produces the video payload first and then packed
channel-major audio rows. The rows are unpacked to `[32,2,T]`; audio is not a
restarted copy of the video prefix.

The gfx1151 analytic tests keep the sample in device F32 and match both a
single transition and complete four-transition video/audio trajectories
captured from the pinned `MiniMaxH3Scheduler`. They freeze Diffusers'
two-stage float32 operation order: sigma for the velocity term is reconstructed
as `1 - (1 - sigma)`, `denoised` is materialized, and the final blend uses the
original sigma-grid ratio. This deliberately differs by a few final bits from
direct `sigma - sigma_next` or a contracted update. Zero-scale and
out-of-range updates fail before launch.

## DiT Block Frozen Evidence

The block-0 teacher uses direct ROCm PyTorch formulas rather than the native
runtime. It loads only the eight released BF16 block tensors, constructs the
canonical six-text-row 256x256x22 layout, executes explicit FP32 full
attention, and retains every required BF16 boundary. The payloads remain
operator-owned; their immutable sizes and hashes are frozen in
`tests/fixtures/minimax_h3/dit-block0-oracle-v1.json`.

On the supported gfx1151 route, the complete 528-row block measured:

| Retained boundary | Relative L2 | Relative max |
| --- | ---: | ---: |
| attention AdaLN | `1.07122e-5` | `0.00170068` |
| attention output | `0.00174641` | `0.00429185` |
| MLP AdaLN | `0.00206138` | `0.00887574` |
| block output | `0.00461029` | `0.00483092` |

Every value is below the frozen `1e-2` limits. The session rejects a
pre-cancelled block, out-of-range row maps, changed tensor shapes, non-finite
retained values, and a rocBLAS solution that fails its construction-time
validation.

The issue #175 retained performance route uses CK BF16 WMMA fused attention
for the supported production shape and does not materialize full attention
score or probability matrices. Its one 1,872-row profile measured 88.279 ms
GPU time and 92.494 ms wall time with output SHA-256
`b049f1cebca31e5416bba9252606746900fc7874a838d3a9ad0f98d5732e4b16`.
That profile is performance evidence only; the independent 528-row teacher
above remains the correctness gate.

Operator-local rocprof/ISA records identify the actual full-attention,
grouped-QKV/RoPE, AdaLN, gate, SwiGLU, and gfx1151 rocBLAS kernels. Generated
profiles remain outside Git; only their reviewed conclusions and immutable
hashes belong in documentation or issue comments.

## Full Denoiser Frozen Evidence

The full-denoiser teacher is a direct ROCm PyTorch program. It does not call
the native runtime. It independently executes condition projection, two text
refiner blocks, F32 timestep embedding, all 50 released AdaLN projections,
all 50 dense transformer blocks, and modality-specific final heads. The
routine teacher stops after the first complete forward and never advances the
denoising schedule.

The raw 256x256x22 payloads remain operator-owned. Their shapes, hashes,
arithmetic contract, and gates are frozen in
`tests/fixtures/minimax_h3/denoiser-oracle-v1.json`.

Before admission as the native oracle, the direct teacher was checked against
the pinned official Diffusers transformer converted by its pinned conversion
script. On the identical frozen conditioning and F32 video/audio inputs, the
teacher differed from Diffusers by video relative L2 `0.0201796`, video
relative max `0.0479220`, audio relative L2 `0.0112314`, and audio relative max
`0.0273933`. All values are below the native oracle ceilings, and the converted
checkpoint retained the released mixed-precision contract: F32 video/audio
input projections, timestep MLP, and output heads; BF16 text projection and
transformer blocks.

Measured gfx1151 parity was:

| Boundary | Relative L2 | Relative max | Frozen ceiling |
| --- | ---: | ---: | ---: |
| refined text | `0.00477293` | `0.00167411` | `0.02` |
| block-0 AdaLN projection | `0.00227907` | `0.00478469` | `0.02` |
| step-0 video velocity | `0.0334148` | `0.0426847` | L2 `0.04`, max `0.05` |
| step-0 audio velocity | `0.0111732` | `0.0259974` | `0.05` |

The refreshed seed-42 artifact retains one forward only:

| Payload | SHA-256 |
| --- | --- |
| video initial | `fce1ea483ab56dc3de354d0c056f05020ae80aba70450046b08d8ac12d9bd463` |
| audio initial | `d1c2934f2556c98e12aa0d9de550b0801037b8fea293946112d608132ca80b5f` |
| block-0 modulation | `38b0e3db26f24730a21304d5fa13fa049b6c5196c925cf5e87114505a1e87d39` |
| video velocity | `2ae88a7d019cf64b6fc0d3a8d17e685e510304cf0e1a373a01acbf27db3d995b` |
| audio velocity | `b17813b33cd76383b4614f9e025b4ecdfdad7ba60a204508d1a3ded25ada9079` |

Sampler host/HIP tests validate the full sigma trajectory and Euler update
without executing another 50-block model forward. Cancellation and resident
session reuse remain targeted lifecycle tests rather than reasons to repeat
an expensive quality run.

### Production-shape single-forward evidence

The current 512x512x22 gate uses the real 1,872-row layout and one complete
50-block forward. It does not advance the sampler or invoke either VAE:

| Boundary | Relative L2 | Relative max |
| --- | ---: | ---: |
| refined text | `0.00469088` | `0.00167411` |
| block-0 AdaLN projection | `0.0022895` | `0.00478469` |
| video velocity | `0.0132707` | `0.0154553` |
| audio velocity | `0.00731056` | `0.0149873` |

The issue #184 retained row-parallel forward took 5.49537 seconds after
10.870 seconds of core loading and 12.744 seconds of AdaLN precompute.
Accounted peak live memory was 58.880 GiB and process swap was zero. The fresh
#183 baseline for the same one-forward workload was 9.02751 seconds, so the
device-resident block chain and split no-spill QKV/AdaLN path reduce forward
latency by 39.13% without changing the oracle boundaries.

The refreshed seed-42 production-shape teacher hashes are:

| Payload | SHA-256 |
| --- | --- |
| video initial | `6788a83c4b5aa596cde87c901156b6ad5982da2a6ce674d7067f16699924f293` |
| audio initial | `2edc6221f9abca6c3b9a8b07cd419bf8b61649499b92706c8987633a6a41d027` |
| block-0 modulation | `d7acea1b164b1c93ef3681d451ea49da6c51deb1adc98c2ee6882fbb0f027045` |
| video velocity | `528728732d4e99d19c3f47ab0a815f93025c5970ffcbaee629ff3844c44d6ff9` |
| audio velocity | `895b1b8b839b37af06f8afea0577b882c8166038dccd649509a6b19febecb052` |

### Historical 512x512 exact denoiser

The pre-audit production-shape run completed all 50 blocks and 50 fresh
evaluations for 512x512x22. It measured 8.630 seconds of AdaLN precompute,
18.696 seconds of core load, 12,496.600 seconds of denoising, a 59.67 GiB
accounted peak, and zero swap:

| Payload | SHA-256 |
| --- | --- |
| final video latent | `0ebaf61b03ba48a5107ffce4ec04f7955e5c4966c9b85792d50acb9c09f8614b` |
| final audio latent | `e1da6b038d0f664a250d600e6aca48e3b227dc2f1006256adf8bcac784e75921` |

Those final latents predate the Diffusers audit and are retained only as a
performance-history record. They are not current parity or quality evidence
and will not be regenerated during ordinary development. The exact route now
uses 49 evaluations over 50 sigma points.

## VisualVAE Frozen Evidence

The isolated F32 teacher retained the normalized latent, all 22 RGB frames,
and frames 0, 5, 11, 16, and 21:

| Payload | Bytes | SHA-256 |
| --- | ---: | --- |
| normalized latent | 172,032 | `23411f438f351f96632884476def1f17513a6d2105b42ebf698a6d69d79d52` |
| full frames | 17,301,504 | `36dd863ffcdf34c2f882f57f401d2c66d7fbc441f5185d32aa1b03db46ca24c1` |
| selected frames | 3,932,160 | `a6cc2b34da6b86f70d91dee8e152051ddc0bac0dc7cd5e1f8e605e7f37696ec7` |

The bounded native gate decodes frames 0, 5, 11, 16, and 21 once. The retained
gfx1151 route uses strided-batched F32 rocBLAS QK and PV GEMMs around an
explicit stable in-place F32 softmax, plus exact-shape gfx1151 solution indices
for the dense projections. The issue #184 pass computes Q/K norm reductions
once per head, preserves GEMM-round-then-bias behavior in fused epilogues, and
vectorizes SwiGLU without changing the softmax reduction order. Its final
quality pass measured selected-frame relative L2 `8.14968e-7`, relative max
`2.24925e-6`, and maximum absolute error `1.6503e-6`. The output remains
effectively identical to the frozen teacher and far inside the declared
`0.05` limits.

The same fixed tile fell from `46.6852` seconds with scalar full attention to
`4.79063` seconds with batched attention, then to `2.82558` seconds after dense
projection tuning. The final route is `16.522x` faster than the scalar
baseline and `1.695x` faster than the first batched route. The explicit
score/probability workspace raises the phase peak from 9.38 GiB to 9.77 GiB;
the process used zero swap.

Those issue #175 measurements predate the later parity and setup changes.
Against the fresh issue #184 revision, the same five-selected-frame gate fell
from 8.60171 seconds to 6.12212 seconds, a further 28.83% reduction, while the
9.766 GiB peak and zero-swap result remained unchanged.

A register-cached softmax and faster QK/PV solution indices were measured once
and rejected because their changed FP32 reduction order exceeded the quality
gate. They are not present in the retained implementation. The complete-frame
teacher payload remains available for offline analysis, but routine native
validation does not execute a second full or repeated decode.

## AudioVAE Frozen Evidence

The isolated teacher retained the normalized latent, stereo waveform, and
deterministic STFT magnitudes:

| Payload | Shape | SHA-256 |
| --- | --- | --- |
| normalized latent | `[32,2,37]` | `f742e12e3f55d91886a1b4218e6e5a3e64efcd65bce63de6a6d42976f3b2e04c` |
| waveform | `[2,29600]` | `ea122eb789d73424ff1c3bbe89719963dd407377165259a4f861f5dba55f837c` |
| spectrogram | `[2,513,116]` | `20f3d1ca41c3b2a005654aec1c25313de6f849b7dbb7bf7b1ae5ffb92e100424` |

The issue #184 native decoder measured waveform maximum absolute error
`6.03162e-5`, waveform relative L2 `1.08936e-5`, spectrogram relative L2
`3.62714e-6`, and spectrogram relative max `5.8276e-6`. The single decode took
2.42326 seconds with a 0.286 GiB phase peak and zero major page faults, versus
59.4383 seconds and 0.251 GiB for the fresh scalar-convolution baseline. The
24.53x speedup uses one bounded reusable im2col workspace and F32 rocBLAS GEMM;
inputs, weights, accumulation, outputs, released transposed-convolution
arithmetic, and alias-free SnakeBeta order are unchanged. Channel independence,
clipping, cancellation, and zero-swap gates passed; routine validation does
not execute a second waveform decode.

## Rapid End-to-End Evidence

The original two development-preset runs used the real 256x256x22 latent
window, four fresh evaluations, all 50 blocks, and selected-frame decode. They
completed in 159.4 and 163.7 seconds with zero swap, but they predate the
Diffusers audit and their checkerboard-like output is invalid quality
evidence.

After correcting AdaLN slot order, one intentionally low-cost 256x256,
two-evaluation visual smoke produced a coherent subject without the full-frame
grid. Selected frame 11 hashes to
`5db8440d5cd2b71323c38c83eedaa612c2151f3619ea9f6b31ad4f19b8d37fea`.
At only two evaluations it still has a localized striped/noisy edge and is a
visual debugging artifact, not a semantic-quality baseline. Current
development uses one invocation of each needed short oracle and does not rerun
a passing full route.

## Numerical Gates

The initial upstream ceilings are strict inequalities:

| Boundary | Required ceiling |
| --- | --- |
| BF16 DiT block | relative max and relative L2 below `1e-2` |
| Qwen layer-50 prompt output | relative max below `0.1`, relative L2 below `0.05` |
| 256x256x22 video velocity | relative max below `0.05`, relative L2 below `0.04` |
| 256x256x22 audio velocity | relative max and relative L2 below `0.05` |
| VisualVAE | relative max and relative L2 below `0.05` |
| AudioVAE | max absolute below `1e-3`, relative L2 below `0.05` |
| AudioVAE selected STFT magnitudes | relative max below `0.1`, relative L2 below `0.08` |

All numerical comparisons also require equal shapes and zero non-finite values.
An implementation issue may freeze a tighter measured boundary, but not a
looser one.

Teacher and native manifests first verify provenance rather than relying on
filenames or directory placement. They must agree on the immutable
model/reference revisions, seed, noise mode, internal geometry, evaluation
count, active block count, reuse interval, and SHA-256 of the exact layer-50
BF16 conditioning payload. Attention-local iterations use the frozen
single-block teacher plus one 1,872-row block profile; they do not run 50
blocks or a denoising trajectory. Cross-block work uses one full forward.
Sampler and schedule changes use analytic host/HIP trajectory tests. There is
no routine full-schedule development gate.

For thinned presets, block count alone is not interpreted as a prefix.
Candidate blocks 2 through 48 are ranked by mean absolute AdaLN gate magnitude
over all schedule rows, modalities, hidden columns, and gate slots 2 and 5.
Blocks 0, 1, and 49 are always retained, h3.c's strict comparison and
selection-sort tie behavior are frozen by a host test, and the retained blocks
execute in ascending model order.

Complete `exact` generation belongs to end-user operation or rare,
explicit release validation. The checkerboard MP4 captured from the older
scalar binary is invalid quality evidence: its container is valid, but its
frames are a 32x32 grid of 16-pixel VAE cells produced from a noise-like
latent. It must not be used as a baseline or promotion artifact.

## Audiovisual Report

End-to-end reports include:

- video/audio latent relative max, relative L2, and non-finite counts;
- first/middle/last PSNR, pinned windowed SSIM, and pinned LPIPS;
- full-clip SSIM/LPIPS aggregates and adjacent-frame-delta error;
- waveform maximum/relative-L2 error and deterministic spectrogram error;
- independent left/right channel results, 32 kHz duration, 24 fps duration,
  and A/V sync;
- subject, count, anatomy, motion, composition, color, and prompt-adherence
  review.

`tools/gufo/h3_quality.py` includes dependency-light numerical, global-SSIM,
Gaussian-window SSIM, temporal, and spectrogram diagnostics used by CI
corruption tests. `tools/gufo/h3_preset_quality.py` additionally evaluates
AlexNet LPIPS from the pinned Nix toolchain. Every promoted report records the
NumPy/LPIPS/Torch/Torchvision versions and a canonical SHA-256 of the loaded
LPIPS module state. The Nix shell also pins the official Torchvision AlexNet
weights and LPIPS v0.1 calibration weights by complete SHA-256; the reporter
verifies both files before model construction, so evaluation cannot silently
download or substitute weights. Promotion reports require both sibling `parameters.json`
documents, validate their immutable model/reference revisions and frozen
preset values, bind those values to the delivered media geometry, and check
both decoded and FFprobe-reported stream timing. The
`--allow-missing-parameters` and `--skip-lpips` switches are diagnostic only.
The delivery spectrogram uses the same 1,024-point, 256-hop, centered
reflect-padded periodic-Hann STFT as the pinned AudioVAE oracle, including the
last samples of the waveform.

## Ownership by Implementation Boundary

Issues that add H3 runtime boundaries also add their retained artifact:

- loader/runtime: schema and lifecycle failure tests;
- tokenizer/Qwen: exact IDs and layer-50 artifact;
- host sampler: schedules/layout/RNG/Euler fixtures;
- DiT kernels: complete block artifact;
- full denoiser: one-forward refined-text, modulation, and F32 video/audio
  velocity artifacts, invoked only for changes crossing the denoiser boundary;
- VisualVAE: one selected-frame native gate, with selected and archival full
  teacher artifacts;
- AudioVAE: waveform, spectrogram, channel, duration, and sync artifacts;
- presets/optimization/quantization: component and short production-shape
  reports against BF16.

Fast and quantized routes are never compared only with each other.
