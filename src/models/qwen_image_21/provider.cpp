#include <gufo/ext/generated/qwen_image_21_constants.hpp>
#include <gufo/ext/image.hpp>
#include <gufo/ext/registry.hpp>

#include "model.hpp"
#include "probe.hpp"

namespace gufo::models::qwen_image_21 {

#if defined(ENGINE_ENABLE_HIP)
class QwenImage21RunnerImpl : public ext::ImageRunner {
public:
  explicit QwenImage21RunnerImpl(std::unique_ptr<Model> model)
      : model_(std::move(model)) {}

  ext::ImageResult Generate(
      const ext::ImageRequest& request,
      const ext::CancellationCheck& is_cancelled) override {
    Request req{
        .prompt = request.prompt,
        .width = request.width,
        .height = request.height,
        .steps = request.steps,
        .seed = request.seed,
    };
    req.images.reserve(request.images.size());
    for (const auto& img : request.images) {
      req.images.push_back(Image{img.width, img.height, img.rgba});
    }
    auto result = model_->Generate(req, is_cancelled);
    return ext::ImageResult{
        .image =
            ext::ImageBuffer{
                .width = result.image.width,
                .height = result.image.height,
                .rgba = std::move(result.image.rgba),
            },
        .metrics =
            ext::ImageMetrics{
                .prompt_ms = result.metrics.prompt_ms,
                .denoise_ms = result.metrics.denoise_ms,
                .vae_ms = result.metrics.vae_ms,
                .total_ms = result.metrics.total_ms,
                .resident_weight_bytes = result.metrics.resident_weight_bytes,
            },
    };
  }

private:
  std::unique_ptr<Model> model_;
};

std::unique_ptr<ext::ImageRunner> CreateQwenImage21Runner(
    const std::string& model_root, std::string* error) {
  (void)error;
  auto model = std::make_unique<Model>(model_root);
  return std::make_unique<QwenImage21RunnerImpl>(std::move(model));
}
#endif

}  // namespace gufo::models::qwen_image_21

void qwen_image_21_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(
      gufo::ext::generated::extension_qwen_image_21::kProvider_qwen_image_21);
#if defined(ENGINE_ENABLE_HIP)
  using namespace gufo::models::qwen_image_21;
  registry.AddImageProvider({
      .id = "qwen-image-21",
      .probe = &ProbeQwenImage21,
      .resolve = &ResolveQwenImage21Artifacts,
      .create = &CreateQwenImage21Runner,
  });
#endif
}
