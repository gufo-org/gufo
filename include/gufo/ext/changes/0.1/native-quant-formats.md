Publish the GGUF quant geometry needed by the Flash-Next routed expert kernels:
- `GGUFTensorKind::kQ2_0` joins the extension-visible quant enum.
- The public quant header now exposes the Q2_0 block (`block_q2_0`, 18-byte
  blocks over 256 values), the IQ3/IQ2/IQ4_XS block structs and their inline
  codebook tables, and scalar accessors (`Iq4XsValue`, `Iq3sValue`, `Iq2sValue`,
  `Iq3xxsValue`, `Iq2sBlock`, `Iq3sValue`) for model-side decode kernels.
- `QuantizedBlockElements` / `QuantizedRowBytes` cover the added formats so
  loaders can size rows without private internals.
- GGUF tensor-shape validation checks quant block alignment per expert slab
  for fused 3-D+ expert tensors (rows x experts x block) instead of requiring
  the first dimension alone to be a multiple of the block size.
