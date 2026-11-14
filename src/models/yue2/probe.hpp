#pragma once

#include <string>
#include <string_view>

namespace gufo::models::yue2 {

[[nodiscard]] bool ProbeYue2(std::string_view model_root);
[[nodiscard]] bool ResolveYue2Artifacts(std::string_view model_root,
                                        std::string* error);

}  // namespace gufo::models::yue2
