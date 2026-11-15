# Image decode memory planning

`core::InspectImage()` inspects bounded encoded input without allocating decoder
or image storage. `ImageDecodePlan` reports format, original and display
dimensions, orientation and a conservative decoder memory bound. The caller
must keep the encoded input unchanged until decoding finishes and separately
account for resizing and retained images. Header inspection does not replace
full compressed-stream validation by `DecodeImage()`.

These declarations are additive through the image extension's forwarded core
header. Rebuild source extensions; no binary ABI compatibility is promised.
