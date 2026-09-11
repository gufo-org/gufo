# Licensing

Status: design policy, 2026-08-11

## Purpose

This document records the project license and the licensing boundaries
of the Linux XRT and `amdxdna` integration.

It is an engineering policy, not legal advice. License files and source-level
notices must be checked again against the exact dependency revisions used for a
release.

## Project License

Original gufo source code is licensed under:

```text
MIT License
SPDX-License-Identifier: MIT
```

MIT is selected to match the permissive licensing approach used by llama.cpp
and DwarfStar. It permits commercial use, modification, redistribution, and
use in proprietary products while imposing only the requirement to preserve
the copyright and license notice.

Unlike Apache-2.0, MIT does not contain an explicit contributor patent grant or
patent-retaliation clause. This is an intentional project tradeoff and does not
change the patent grants supplied by Apache-2.0 dependencies for their own
code.

The complete MIT license text is stored in the top-level `LICENSE` file. This
document does not replace that file.

## XRT and amdxdna Components

The relevant components have different licenses:

| Component | License | Effect on gufo |
| --- | --- | --- |
| XRT userspace runtime | Apache-2.0 | Compatible with an MIT server; retain XRT's Apache notices when distributing it |
| AMD XDNA XRT userspace shim | Apache-2.0 | Compatible with an MIT server; retained or adapted shim code remains Apache-2.0 |
| Upstream Linux `amdxdna` kernel driver | GPL-2.0-only | Does not relicense a userspace process that uses the driver interface |
| `amdxdna` UAPI headers | GPL-2.0 with Linux syscall note | Intended for use by non-GPL userspace applications |
| AMD NPU firmware and selected binary artifacts | Separate binary redistribution terms | Must be handled independently from the server source license |

The XDNA userspace shim currently identifies its source and build files with:

```text
SPDX-License-Identifier: Apache-2.0
```

The upstream kernel driver identifies its kernel source with:

```text
SPDX-License-Identifier: GPL-2.0-only
```

The public userspace API header identifies itself with:

```text
SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note
```

## Kernel Boundary

Running gufo against the installed GPL-licensed `amdxdna` kernel driver
does not require gufo to use the GPL. The server remains an independent
userspace program communicating through the documented kernel UAPI.

Do not:

- Copy GPL kernel-driver implementation code into `gufo`.
- Compile kernel implementation files into a userspace library.
- Treat an internal kernel function as a stable userspace interface.
- Remove SPDX identifiers or copyright notices from copied UAPI material.

Changes made to the kernel driver and distributed as a kernel module must be
handled under the applicable GPL terms, including corresponding-source
obligations.

## Linking XRT

gufo may dynamically or statically link Apache-2.0 XRT userspace
components without changing the MIT license of original gufo code. The
combined distribution contains components under both licenses; it is not
accurate to relabel XRT itself as MIT.

When distributing XRT or a modified XRT component:

- Include a copy of the Apache-2.0 license.
- Preserve copyright, patent, trademark, and attribution notices.
- Mark modified files.
- Include relevant upstream `NOTICE` content when present.

Prefer using distribution-provided XRT and XDNA packages initially. This keeps
driver and firmware updates independent from the server release and reduces
the number of third-party binary artifacts distributed by the project.

## Firmware

NPU firmware is not covered by the gufo MIT license.

The AMD XDNA repository includes a `LICENSE.amdnpu` file that permits
redistribution of covered software only in unmodified binary form and includes
additional restrictions. Do not assume that every firmware, overlay, compiler
output, or binary found in an AMD repository may be republished under
Apache-2.0.

The initial distribution policy is:

- Do not bundle AMD NPU firmware with gufo.
- Require firmware supplied by the Linux distribution or an AMD package.
- Detect missing or incompatible firmware at startup.
- Document the tested firmware version without copying the firmware.

If future installation packages bundle firmware, review the exact artifact and
its accompanying license before every release.

Gufo-owned AIE overlays and `ctrlcode` compiled from original MIT
project source may be distributed with the server, subject to the licenses of
the compiler, headers, and libraries used to produce them.

## Reference Projects

Reference projects may be studied without copying their implementation.
Copied or adapted code retains its original license obligations.

In particular:

- Do not copy AGPL-licensed `hipEngine` implementation code into an
  MIT server unless the project deliberately accepts the resulting
  obligations.
- Preserve MIT and other permissive notices for copied code from compatible
  projects.
- Record the source repository, revision, original path, license, and local
  modifications for every imported implementation.
- Do not rely on repository-level license detection when files contain their
  own SPDX identifiers or third-party notices.

Clean-room reimplementation of an architectural idea should cite the reference
project in design documentation but should not copy source expression.

## Models and Quantized Artifacts

The engine license does not grant rights to model weights, tokenizers,
calibration corpora, benchmark datasets, or generated quantized artifacts.

gufo follows a bring-your-own-weights policy:

- The engine never downloads a gated model at build or runtime.
- Model weights and derived quantized artifacts are excluded from source and
  binary packages.
- An operator supplies a local path after independently obtaining access and
  accepting or obtaining the terms applicable to that operator.
- A missing, incomplete, or unsupported checkpoint fails closed; the runtime
  does not fetch a replacement or fall back to an unreviewed model.
- Repository documentation of a model integration is not a grant of model
  rights and is not a substitute for the publisher's terms.

The MiniMax H3-specific boundary is recorded in
[MINIMAX_H3.md](MINIMAX_H3.md). Its official community agreement contains
territorial, hosted-service, acceptable-use, safeguards, user-terms,
attribution, and redistribution conditions. Operators in a territory or use
case requiring separate authorization must obtain it directly from MiniMax.
The repository records only an operator attestation and public artifact
metadata; it does not store private authorization or credentials.

Every published model artifact must record:

- Source model and revision.
- Source model license.
- Whether redistribution and modification are permitted.
- Required attribution or acceptable-use terms.
- Calibration-data provenance.
- License or terms applied to the converted artifact.

Hugging Face publication must include the model license independently from the
gufo engine license.

MiniMax H3 weights and derived H3 quantized artifacts are not publication
targets for this repository. Changing that boundary requires a separate
license review and an explicit release decision.

## Release Files

Before the first source or binary release, add:

```text
LICENSE
THIRD_PARTY_NOTICES.md
```

When an included Apache-2.0 dependency has a `NOTICE` file, preserve the
applicable contents in the binary distribution. A top-level gufo
`NOTICE` file is not required by MIT itself.

`THIRD_PARTY_NOTICES.md` records at least:

- Dependency name and pinned revision.
- Component or files used.
- SPDX license identifier.
- Copyright notice.
- Whether it is linked, copied, modified, or merely required from the system.
- Location of the complete corresponding source when distribution requires it.

Source files should use SPDX headers. Original project files use:

```cpp
// SPDX-License-Identifier: MIT
```

Generated files record both their generator and applicable license.

## Release Gate

A release fails its licensing gate when:

- A source or binary artifact has no identified license.
- A required copyright or notice file is absent.
- GPL kernel implementation code has entered the userspace server.
- Restricted firmware is bundled without confirmed redistribution rights.
- Copied code cannot be traced to a source revision and license.
- Model redistribution terms are missing or incompatible.
- The software bill of materials disagrees with the shipped files.

The dependency and artifact license inventory is regenerated for every release
candidate rather than treated as a one-time review.
