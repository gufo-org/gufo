# Keyframe conditioning anchors

Video (`src/ext/video.hpp`): `VideoRequest` gains optional
`first_frame_path` and `last_frame_path` keyframe conditioning anchors
(issue #177). Both fields default to empty, which selects the unchanged
text-only contract; exactly three conditioning modes exist (first only,
last only, first and last together). Callers stage the referenced
PNG/JPEG/WebP file (already in display orientation) at a path that
outlives the request; runners must report an error rather than silently
ignore anchors when the selected preset cannot condition. With an anchor,
`VideoParameters` geometry is a size budget: a runner may generate at the
geometry anchor's aspect ratio within it (MiniMax H3 follows the released
canvas rule) and reports the generated canvas in its telemetry. No existing
declaration changed, so previously built extensions keep compiling and
observe empty paths.
