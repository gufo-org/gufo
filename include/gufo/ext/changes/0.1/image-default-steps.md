# Image checkpoint default steps

Image (`src/ext/image.hpp`): `ImageRequest::steps` defaults to zero, which
selects the runner's checkpoint default; runners reject step counts their
checkpoint does not support. The HTTP image API forwards omitted `steps` as
zero, so Qwen-Image-2.1 keeps 40 steps and Qwen-Image-2.1-Turbo uses its
eight-step schedule. Image runners that read `steps` must resolve zero.
`ImageMetrics::steps` reports the denoising steps a runner executed.
