#include "scholarvault/domain.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <ctime>
#include <iomanip>
#include <random>
#include <sstream>

namespace scholarvault {

std::string projectOriginName(ProjectOrigin origin) {
    switch (origin) {
    case ProjectOrigin::Arxiv: return "arxiv";
    case ProjectOrigin::LocalPdf: return "local-pdf";
    case ProjectOrigin::Zotero: return "zotero";
    }
    throw std::invalid_argument("Unknown project origin");
}

ProjectOrigin projectOriginFromName(const std::string& name) {
    if (name == "arxiv") return ProjectOrigin::Arxiv;
    if (name == "local-pdf") return ProjectOrigin::LocalPdf;
    if (name == "zotero") return ProjectOrigin::Zotero;
    throw std::invalid_argument("Unknown project origin: " + name);
}

std::string makeStableId() {
    std::array<unsigned char, 16> bytes{};
    std::random_device random;
    for (auto& byte : bytes) byte = static_cast<unsigned char>(random());
    bytes[6] = static_cast<unsigned char>((bytes[6] & 0x0fU) | 0x40U);
    bytes[8] = static_cast<unsigned char>((bytes[8] & 0x3fU) | 0x80U);

    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        if (index == 4 || index == 6 || index == 8 || index == 10) out << '-';
        out << std::setw(2) << static_cast<unsigned int>(bytes[index]);
    }
    return out.str();
}

std::string utcTimestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm value{};
    gmtime_r(&time, &value);
    std::ostringstream out;
    out << std::put_time(&value, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

std::string sanitizeDirectoryName(const std::string& name, const std::string& fallback) {
    std::string result;
    result.reserve(name.size());
    bool previousSpace = false;
    for (const unsigned char character : name) {
        const bool forbidden = character < 32 || character == '/' || character == '\\' ||
                               character == ':' || character == '*' || character == '?' ||
                               character == '"' || character == '<' || character == '>' ||
                               character == '|';
        if (forbidden) {
            if (!result.empty() && !previousSpace) result += '-';
            previousSpace = true;
            continue;
        }
        if (std::isspace(character) != 0) {
            if (!result.empty() && !previousSpace) result += ' ';
            previousSpace = true;
            continue;
        }
        result += static_cast<char>(character);
        previousSpace = false;
    }
    while (!result.empty() && (result.back() == ' ' || result.back() == '.')) result.pop_back();
    while (!result.empty() && (result.front() == ' ' || result.front() == '.')) result.erase(0, 1);
    if (result.empty() || result == "." || result == "..") result = fallback;

    constexpr std::size_t maximumBytes = 120;
    if (result.size() > maximumBytes) {
        std::size_t end = maximumBytes;
        while (end > 0 && (static_cast<unsigned char>(result[end]) & 0xc0U) == 0x80U) --end;
        result.resize(end);
        while (!result.empty() && result.back() == ' ') result.pop_back();
    }
    return result;
}

} // namespace scholarvault
