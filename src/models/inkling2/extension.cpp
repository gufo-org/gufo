#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include "src/models/inkling2/stream_dialect.hpp"

void inkling2_register(gufo::ext::Registry&) {
  // The model's tool-call dialect as the runner streams it (grammar + parser);
  // runners select it by id.
  gufo::ext::RegisterToolDialect(
      gufo::models::inkling2::MakeStreamToolDialect());
}
