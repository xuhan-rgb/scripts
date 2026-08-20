#pragma once

#ifndef Q_MOC_RUN
#include <filesystem>
#endif
#include <optional>
#include <string>
#include <vector>

namespace scholarvault {

enum class ProjectOrigin {
    Arxiv,
    LocalPdf,
    Zotero,
};

struct ZoteroProvenance {
    std::string itemKey;
    std::string attachmentKey;
    std::string dataDirectory;
    std::string attachmentPath;
    std::string syncedTitle;
    std::string authors;
    std::string year;
    std::string lastSyncedAt;
};

struct RelatedRepository {
    std::string url;
    std::string owner;
    std::string name;
    std::string defaultBranch;
    std::string commitSha;
    std::string relativePath;
    std::string status{"pending"};
};

struct Topic {
    std::string id;
    std::string name;
    std::filesystem::path path;
    std::string zoteroCollectionKey;
};

struct Project {
    std::string id;
    std::string topicId;
    std::string title;
    ProjectOrigin origin{ProjectOrigin::LocalPdf};
    std::string arxivId;
    std::optional<ZoteroProvenance> zotero;
    std::string sourceStatus{"not-requested"};
    std::filesystem::path path;
    std::filesystem::path pdfRelativePath{"paper/paper.pdf"};
    std::vector<RelatedRepository> repositories;
};

[[nodiscard]] std::string projectOriginName(ProjectOrigin origin);
[[nodiscard]] ProjectOrigin projectOriginFromName(const std::string& name);
[[nodiscard]] std::string makeStableId();
[[nodiscard]] std::string utcTimestamp();
[[nodiscard]] std::string sanitizeDirectoryName(const std::string& name,
                                                const std::string& fallback);

} // namespace scholarvault
