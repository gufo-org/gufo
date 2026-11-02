# GGML block format support

Accept every GGML block format (`gguf_reader.hpp`, `experimental/ggml_dequant.hpp`):
- `GgmlType` gains `kIQ2_XS`, `kIQ3_XXS`, `kIQ1_S`, `kIQ2_S` and `kIQ1_M`, so
  a GGUF with these tensors opens and the model binder decides; exhaustive
  switches over `GgmlType` in extensions must handle them.
- CPU dequantizers for Q4_0, Q4_1, Q5_0, Q5_1, Q2_K and the IQ1/IQ2/IQ3
  formats, and `DequantizeRow`, which dispatches every block format plus
  F32/F16/BF16.
