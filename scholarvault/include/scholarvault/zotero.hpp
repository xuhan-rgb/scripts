#pragma once

#ifndef Q_MOC_RUN
#include <filesystem>
#endif
#include <stdexcept>
#include <string>
#include <vector>

namespace scholarvault {

struct ZoteroCollection {
    std::string key;
    std::string name;
};

struct ZoteroPaper {
    std::string title;
    std::string authors;
    std::string year;
    std::string itemKey;
    std::string attachmentKey;
    std::filesystem::path dataDirectory;
    std::filesystem::path pdfPath;
    std::vector<ZoteroCollection> collectionPath;
    std::string arxivId;
};

class ZoteroError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

[[nodiscard]] std::filesystem::path resolveZoteroAttachmentPath(
    const std::filesystem::path& dataDirectory,
    const std::string& attachmentKey,
    const std::string& storedPath);

[[nodiscard]] std::vector<ZoteroPaper> readZoteroSqliteLibrary(
    const std::filesystem::path& dataDirectory);

} // namespace scholarvault
