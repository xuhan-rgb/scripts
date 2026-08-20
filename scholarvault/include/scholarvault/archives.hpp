#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace scholarvault {

[[nodiscard]] bool archiveEntryPathIsSafe(std::string_view path);
[[nodiscard]] bool archiveListingIsSafe(const std::vector<std::string>& paths);

} // namespace scholarvault
