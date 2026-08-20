#include "scholarvault/github.hpp"

#include <algorithm>
#include <cctype>
#include <regex>

namespace scholarvault {
namespace {

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

} // namespace

std::string GitHubRepositoryReference::directoryName() const {
    return owner + "-" + name;
}

std::string GitHubRepositoryReference::identityKey() const {
    return lower(owner + "/" + name);
}

std::optional<GitHubRepositoryReference> parseGitHubRepository(std::string_view rawInput) {
    std::string input(rawInput);
    input.erase(input.begin(), std::find_if_not(input.begin(), input.end(), [](unsigned char c) {
        return std::isspace(c) != 0;
    }));
    input.erase(std::find_if_not(input.rbegin(), input.rend(), [](unsigned char c) {
        return std::isspace(c) != 0;
    }).base(), input.end());

    static const std::regex pattern(
        R"(^https://github\.com/([A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/([A-Za-z0-9._-]+?)(?:\.git)?/?$)",
        std::regex::icase);
    std::smatch match;
    if (!std::regex_match(input, match, pattern)) return std::nullopt;
    if (match[2].str() == "." || match[2].str() == "..") return std::nullopt;

    GitHubRepositoryReference result;
    result.owner = match[1].str();
    result.name = match[2].str();
    result.normalizedUrl = "https://github.com/" + result.owner + "/" + result.name;
    return result;
}

} // namespace scholarvault
