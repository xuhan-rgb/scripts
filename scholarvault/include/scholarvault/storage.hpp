#pragma once

#include "scholarvault/arxiv.hpp"
#include "scholarvault/domain.hpp"
#include "scholarvault/zotero.hpp"

#ifndef Q_MOC_RUN
#include <filesystem>
#endif
#include <stdexcept>
#include <string>
#include <vector>

namespace scholarvault {

class StorageError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class VaultStorage {
public:
    explicit VaultStorage(std::filesystem::path root);

    void initialize(const std::string& displayName);

    [[nodiscard]] const std::filesystem::path& rootPath() const { return root_; }
    [[nodiscard]] std::filesystem::path topicsPath() const;

    [[nodiscard]] Topic createTopic(const std::string& name) const;
    [[nodiscard]] Project createProjectFromPdf(
        const std::filesystem::path& topicPath,
        const std::filesystem::path& sourcePdf,
        const std::string& title) const;
    [[nodiscard]] Project createProjectFromArxivFiles(
        const std::filesystem::path& topicPath,
        const ArxivReference& arxiv,
        const std::filesystem::path& sourcePdf,
        const std::filesystem::path& sourceArchive,
        const std::filesystem::path& extractedSource,
        const std::string& title,
        const std::string& sourceStatus) const;
    [[nodiscard]] Project installArxivSource(
        const Project& project,
        const ArxivReference& arxiv,
        const std::filesystem::path& sourceArchive,
        const std::filesystem::path& extractedSource,
        const std::string& sourceStatus) const;
    [[nodiscard]] Project createProjectFromZotero(
        const std::filesystem::path& topicPath,
        const ZoteroPaper& paper,
        const std::string& title) const;
    struct ZoteroSyncResult {
        Project project;
        bool pdfUpdated{false};
        bool metadataUpdated{false};
    };
    struct ZoteroLibrarySyncResult {
        int created{0};
        int updated{0};
        int unchanged{0};
        int moved{0};
        int skipped{0};
        std::vector<std::string> errors;
        std::vector<Project> changedProjects;
    };
    [[nodiscard]] ZoteroSyncResult syncProjectFromZotero(
        const Project& project, const ZoteroPaper& paper) const;
    [[nodiscard]] std::filesystem::path moveTopicToTrash(
        const std::filesystem::path& topicPath) const;
    [[nodiscard]] ZoteroLibrarySyncResult syncZoteroLibrary(
        const std::vector<ZoteroPaper>& papers) const;

    [[nodiscard]] std::vector<Topic> discoverTopics() const;
    [[nodiscard]] std::vector<Topic> discoverChildTopics(const Topic& topic) const;
    [[nodiscard]] std::vector<Project> discoverAllProjects() const;
    [[nodiscard]] std::vector<Project> discoverProjects(const Topic& topic) const;
    [[nodiscard]] Topic loadTopic(const std::filesystem::path& topicPath) const;
    [[nodiscard]] Project loadProject(const std::filesystem::path& projectPath) const;
    void saveProject(const Project& project) const;

private:
    std::filesystem::path root_;

    [[nodiscard]] std::filesystem::path uniqueChildPath(
        const std::filesystem::path& parent,
        const std::string& preferredName) const;
    [[nodiscard]] Topic requireOwnedTopic(const std::filesystem::path& topicPath) const;
    [[nodiscard]] Topic ensureZoteroTopic(
        const std::vector<ZoteroCollection>& collectionPath) const;
    [[nodiscard]] Project createPdfProject(
        const std::filesystem::path& topicPath,
        const std::filesystem::path& sourcePdf,
        const std::string& title,
        ProjectOrigin origin,
        std::optional<ZoteroProvenance> zotero) const;
};

} // namespace scholarvault
