#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace scholarvault {

struct ArxivReference {
    std::string id;
    std::string baseId;
    int version{0};

    [[nodiscard]] std::string abstractUrl() const;
    [[nodiscard]] std::string pdfUrl() const;
    [[nodiscard]] std::string sourceUrl() const;
};

[[nodiscard]] std::optional<ArxivReference> parseArxivReference(std::string_view input);
[[nodiscard]] std::optional<ArxivReference> findArxivReference(std::string_view metadata);

} // namespace scholarvault
