# LFM3-VL integration decisions

- Share the identical LFM3 vision architecture with d1; preserve d1's own
  pixel budget. The existing d1 pixel/operator goldens and HTTP suite remain.
- Keep native static-image preprocessing and its full pixel range. Low
  detail uses a smaller aspect-ratio-aware budget with the native minimum.
- Feed image rows through causal backbone embedding prefill. The decision
  model's bidirectional forward path is not used for VL generation.
- Bind supplemental cache identity to processed image inputs and the full
  projector identity. Persistent compatibility includes the image ABI;
  identical token placeholders alone never identify a photo.
- Keep the Python teacher and raw tensor probes outside production. Source
  hashes and strict tensor mapping make their provenance reviewable.
