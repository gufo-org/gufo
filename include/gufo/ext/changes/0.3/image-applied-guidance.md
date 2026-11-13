# Applied image guidance scale

Image (`src/ext/image.hpp`):
- `ImageMetrics::guidance_scale` (`std::optional<float>`) reports the
  classifier-free guidance scale a runner applied after resolving an omitted
  `ImageRequest::guidance_scale`; nullopt when the runner does not report it.
  The A1111 API shows it as `cfg_scale` in its `info`.

The member defaults to nullopt, so existing runners keep compiling and report
nothing. Runners must be rebuilt against the new header.
