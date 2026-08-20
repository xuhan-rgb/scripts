#include "scholarvault/arxiv.hpp"

#include <algorithm>
#include <cctype>
#include <regex>

namespace scholarvault {
namespace {

std::string trim(std::string value) {
    const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char c) {
        return std::isspace(c) != 0;
    });
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c) {
        return std::isspace(c) != 0;
    }).base();
    return first < last ? std::string(first, last) : std::string{};
}

} // namespace

std::string ArxivReference::abstractUrl() const {
    return "https://arxiv.org/abs/" + id;
}

std::string ArxivReference::pdfUrl() const {
    return "https://arxiv.org/pdf/" + id + ".pdf";
}

std::string ArxivReference::sourceUrl() const {
    return "https://arxiv.org/src/" + id;
}

std::optional<ArxivReference> parseArxivReference(std::string_view rawInput) {
    std::string candidate = trim(std::string(rawInput));
    if (candidate.empty()) return std::nullopt;

    const auto schemePosition = candidate.find("://");
    if (schemePosition != std::string::npos) {
        const std::string scheme = candidate.substr(0, schemePosition);
        if (scheme != "https" && scheme != "http") return std::nullopt;

        const auto authorityStart = schemePosition + 3;
        const auto pathStart = candidate.find('/', authorityStart);
        const std::string authority = candidate.substr(
            authorityStart,
            pathStart == std::string::npos ? std::string::npos : pathStart - authorityStart);
        if (authority != "arxiv.org" && authority != "www.arxiv.org" &&
            authority != "export.arxiv.org") {
            return std::nullopt;
        }
        if (pathStart == std::string::npos) return std::nullopt;
        candidate = candidate.substr(pathStart + 1);
        if (candidate.starts_with("abs/")) {
            candidate.erase(0, 4);
        } else if (candidate.starts_with("pdf/")) {
            candidate.erase(0, 4);
        } else {
            return std::nullopt;
        }
    }

    const auto suffixStart = candidate.find_first_of("?#");
    if (suffixStart != std::string::npos) candidate.erase(suffixStart);
    if (candidate.ends_with(".pdf")) candidate.erase(candidate.size() - 4);
    while (!candidate.empty() && candidate.back() == '/') candidate.pop_back();

    static const std::regex modern(R"(^([0-9]{4}\.[0-9]{4,5})(?:v([1-9][0-9]*))?$)");
    static const std::regex legacy(
        R"(^([A-Za-z0-9][A-Za-z0-9.-]*/[0-9]{7})(?:v([1-9][0-9]*))?$)");
    std::smatch match;
    if (!std::regex_match(candidate, match, modern) &&
        !std::regex_match(candidate, match, legacy)) {
        return std::nullopt;
    }

    ArxivReference result;
    result.baseId = match[1].str();
    result.version = match[2].matched ? std::stoi(match[2].str()) : 0;
    result.id = result.baseId;
    if (result.version > 0) result.id += "v" + std::to_string(result.version);
    return result;
}

std::optional<ArxivReference> findArxivReference(std::string_view metadata) {
    if (const auto direct = parseArxivReference(metadata)) return direct;
    const std::string text(metadata);
    static const std::regex modern(
        R"((?:arxiv(?:\.org/(?:abs|pdf)/|[:.\s/]+))([0-9]{4}\.[0-9]{4,5}(?:v[1-9][0-9]*)?))",
        std::regex::icase);
    static const std::regex legacy(
        R"((?:arxiv(?:\.org/(?:abs|pdf)/|[:.\s/]+))([A-Za-z0-9][A-Za-z0-9.-]*/[0-9]{7}(?:v[1-9][0-9]*)?))",
        std::regex::icase);
    std::smatch match;
    if ((std::regex_search(text, match, modern) ||
         std::regex_search(text, match, legacy)) && match.size() >= 2) {
        return parseArxivReference(match[1].str());
    }
    return std::nullopt;
}

} // namespace scholarvault
