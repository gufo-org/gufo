# Image guidance, image to image and resource errors

Image (`src/ext/image.hpp`):
- `ImageRequest::negative_prompt` and `ImageRequest::guidance_scale`
  (`std::optional<float>` in [0, 100], nullopt selects the runner's default,
  at most 1 is unguided) carry classifier-free guidance;
  `ImageCapabilities::guidance` declares support.
- `ImageRequest::init_image` (`std::optional<ImageBuffer>`) and
  `ImageRequest::strength` (default 1, Diffusers img2img semantics: the last
  `steps * strength` steps run) carry image to image;
  `ImageCapabilities::image_to_image` declares support for strength below 1.
  `init_image` may come together with reference `images`.
  The host refuses guidance or strength below 1 with HTTP 400 when the runner
  does not declare the capability; legacy runners declare neither.
- `ImageResourceError`: thrown by `ImageRunner::Generate` when a request does
  not fit into the device memory free now; the host answers HTTP 503.
- `ImageRunner::DeviceLost()` (default false) reports an unusable device
  context to the server's device-lost handling.

The new members have defaults that keep earlier requests unchanged. Runners
must be rebuilt against the new header.
