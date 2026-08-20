#include "scholarvault/archives.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>

namespace scholarvault {

bool archiveEntryPathIsSafe(std::string_view rawPath) {
    std::string path(rawPath);
    std::replace(path.begin(), path.end(), '\\', '/');
    while (path.starts_with("./")) path.erase(0, 2);
    if (path.empty() || path.front() == '/') return false;
    if (path.size() >= 2 && std::isalpha(static_cast<unsigned char>(path[0])) != 0 &&
        path[1] == ':') {
        return false;
    }
    std::istringstream components(path);
    std::string component;
    while (std::getline(components, component, '/')) {
        if (component == "..") return false;
    }
    return true;
}

bool archiveListingIsSafe(const std::vector<std::string>& paths) {
    return !paths.empty() &&
           std::all_of(paths.begin(), paths.end(), archiveEntryPathIsSafe);
}

} // namespace scholarvault
