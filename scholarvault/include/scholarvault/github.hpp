#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace scholarvault {

struct GitHubRepositoryReference {
    std::string owner;
    std::string name;
    std::string normalizedUrl;

    [[nodiscard]] std::string directoryName() const;
    [[nodiscard]] std::string identityKey() const;
};

[[nodiscard]] std::optional<GitHubRepositoryReference>
parseGitHubRepository(std::string_view input);

} // namespace scholarvault
