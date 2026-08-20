#include "scholarvault/storage.hpp"

#define BOOST_BIND_GLOBAL_PLACEHOLDERS
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <system_error>
#include <unordered_map>

namespace scholarvault {
namespace fs = std::filesystem;
namespace pt = boost::property_tree;

namespace {

std::string jsonString(const std::string& value) {
    std::ostringstream output;
    output << '"';
    for (const unsigned char character : value) {
        switch (character) {
        case '"': output << "\\\""; break;
        case '\\': output << "\\\\"; break;
        case '\b': output << "\\b"; break;
        case '\f': output << "\\f"; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default:
            if (character < 0x20U) {
                output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                       << static_cast<unsigned int>(character) << std::dec;
            } else {
                output << static_cast<char>(character);
            }
        }
    }
    output << '"';
    return output.str();
}

void writeJsonAtomically(const fs::path& destination, const std::string& json) {
    const fs::path temporary = destination.string() + ".tmp-" + makeStableId();
    try {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        output.exceptions(std::ios::failbit | std::ios::badbit);
        output << json;
        output.close();
        fs::rename(temporary, destination);
    } catch (...) {
        std::error_code error;
        fs::remove(temporary, error);
        throw;
    }
}

pt::ptree readJson(const fs::path& path) {
    pt::ptree value;
    try {
        pt::read_json(path.string(), value);
    } catch (const std::exception& error) {
        throw StorageError("Cannot read metadata " + path.string() + ": " + error.what());
    }
    return value;
}

bool isPdf(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    char magic[5]{};
    input.read(magic, sizeof(magic));
    return input.gcount() == 5 && std::string(magic, sizeof(magic)) == "%PDF-";
}

bool filesEqual(const fs::path& left, const fs::path& right) {
    std::error_code error;
    if (fs::file_size(left, error) != fs::file_size(right, error) || error) return false;
    std::ifstream leftInput(left, std::ios::binary);
    std::ifstream rightInput(right, std::ios::binary);
    std::array<char, 64 * 1024> leftBuffer{};
    std::array<char, 64 * 1024> rightBuffer{};
    while (leftInput && rightInput) {
        leftInput.read(leftBuffer.data(), static_cast<std::streamsize>(leftBuffer.size()));
        rightInput.read(rightBuffer.data(), static_cast<std::streamsize>(rightBuffer.size()));
        if (leftInput.gcount() != rightInput.gcount() ||
            !std::equal(leftBuffer.begin(),
                        leftBuffer.begin() + leftInput.gcount(), rightBuffer.begin())) {
            return false;
        }
    }
    return leftInput.eof() && rightInput.eof();
}

bool isContainedBy(const fs::path& rootPath, const fs::path& candidatePath) {
    std::error_code error;
    const fs::path root = fs::weakly_canonical(rootPath, error);
    if (error) return false;
    const fs::path candidate = fs::weakly_canonical(candidatePath, error);
    if (error || candidate == root) return false;
    auto rootPart = root.begin();
    auto candidatePart = candidate.begin();
    for (; rootPart != root.end(); ++rootPart, ++candidatePart) {
        if (candidatePart == candidate.end() || *rootPart != *candidatePart) return false;
    }
    return true;
}

std::string topicJson(const Topic& topic) {
    std::ostringstream metadata;
    metadata << "{\n"
             << "  \"schemaVersion\": 1,\n"
             << "  \"id\": " << jsonString(topic.id) << ",\n"
             << "  \"name\": " << jsonString(topic.name);
    if (!topic.zoteroCollectionKey.empty()) {
        metadata << ",\n  \"zoteroCollectionKey\": "
                 << jsonString(topic.zoteroCollectionKey);
    }
    metadata << ",\n  \"createdAt\": " << jsonString(utcTimestamp()) << "\n"
             << "}\n";
    return metadata.str();
}

std::string projectJson(const Project& project) {
    std::ostringstream output;
    output << "{\n"
           << "  \"schemaVersion\": 1,\n"
           << "  \"id\": " << jsonString(project.id) << ",\n"
           << "  \"topicId\": " << jsonString(project.topicId) << ",\n"
           << "  \"title\": " << jsonString(project.title) << ",\n"
           << "  \"origin\": {\n"
           << "    \"type\": " << jsonString(projectOriginName(project.origin));
    if (!project.arxivId.empty()) {
        output << ",\n    \"arxivId\": " << jsonString(project.arxivId);
    }
    if (project.zotero) {
        output << ",\n"
               << "    \"zotero\": {\n"
               << "      \"itemKey\": " << jsonString(project.zotero->itemKey) << ",\n"
               << "      \"attachmentKey\": "
               << jsonString(project.zotero->attachmentKey) << ",\n"
               << "      \"dataDirectory\": "
               << jsonString(project.zotero->dataDirectory) << ",\n"
               << "      \"attachmentPath\": "
               << jsonString(project.zotero->attachmentPath) << ",\n"
               << "      \"syncedTitle\": "
               << jsonString(project.zotero->syncedTitle) << ",\n"
               << "      \"authors\": " << jsonString(project.zotero->authors) << ",\n"
               << "      \"year\": " << jsonString(project.zotero->year) << ",\n"
               << "      \"lastSyncedAt\": "
               << jsonString(project.zotero->lastSyncedAt) << "\n"
               << "    }";
    }
    output << "\n  },\n"
           << "  \"files\": {\"pdf\": "
           << jsonString(project.pdfRelativePath.generic_string()) << "},\n"
           << "  \"source\": {\"status\": " << jsonString(project.sourceStatus) << "},\n"
           << "  \"repositories\": [";
    for (std::size_t index = 0; index < project.repositories.size(); ++index) {
        const auto& repository = project.repositories[index];
        output << (index == 0 ? "\n" : ",\n")
               << "    {\n"
               << "      \"url\": " << jsonString(repository.url) << ",\n"
               << "      \"owner\": " << jsonString(repository.owner) << ",\n"
               << "      \"name\": " << jsonString(repository.name) << ",\n"
               << "      \"defaultBranch\": " << jsonString(repository.defaultBranch) << ",\n"
               << "      \"commitSha\": " << jsonString(repository.commitSha) << ",\n"
               << "      \"path\": " << jsonString(repository.relativePath) << ",\n"
               << "      \"status\": " << jsonString(repository.status) << "\n"
               << "    }";
    }
    if (!project.repositories.empty()) output << '\n';
    output << "  ],\n"
           << "  \"updatedAt\": " << jsonString(utcTimestamp()) << "\n"
           << "}\n";
    return output.str();
}

Project projectFromTree(const fs::path& path, const pt::ptree& value) {
    Project project;
    project.id = value.get<std::string>("id");
    project.topicId = value.get<std::string>("topicId");
    project.title = value.get<std::string>("title");
    project.origin = projectOriginFromName(value.get<std::string>("origin.type"));
    project.arxivId = value.get<std::string>("origin.arxivId", "");
    if (project.origin == ProjectOrigin::Zotero) {
        ZoteroProvenance zotero;
        zotero.itemKey = value.get<std::string>("origin.zotero.itemKey", "");
        zotero.attachmentKey =
            value.get<std::string>("origin.zotero.attachmentKey", "");
        zotero.dataDirectory =
            value.get<std::string>("origin.zotero.dataDirectory", "");
        zotero.attachmentPath =
            value.get<std::string>("origin.zotero.attachmentPath", "");
        zotero.syncedTitle =
            value.get<std::string>("origin.zotero.syncedTitle", "");
        zotero.authors = value.get<std::string>("origin.zotero.authors", "");
        zotero.year = value.get<std::string>("origin.zotero.year", "");
        zotero.lastSyncedAt =
            value.get<std::string>("origin.zotero.lastSyncedAt", "");
        project.zotero = std::move(zotero);
    }
    project.pdfRelativePath = value.get<std::string>("files.pdf", "paper/paper.pdf");
    project.sourceStatus = value.get<std::string>("source.status", "not-requested");
    project.path = path;
    if (const auto repositories = value.get_child_optional("repositories")) {
        for (const auto& entry : *repositories) {
            RelatedRepository repository;
            repository.url = entry.second.get<std::string>("url", "");
            repository.owner = entry.second.get<std::string>("owner", "");
            repository.name = entry.second.get<std::string>("name", "");
            repository.defaultBranch = entry.second.get<std::string>("defaultBranch", "");
            repository.commitSha = entry.second.get<std::string>("commitSha", "");
            repository.relativePath = entry.second.get<std::string>("path", "");
            repository.status = entry.second.get<std::string>("status", "pending");
            project.repositories.push_back(std::move(repository));
        }
    }
    return project;
}

void createProjectDirectories(const fs::path& root) {
    fs::create_directories(root / "paper");
    fs::create_directories(root / "source" / "extracted");
    fs::create_directories(root / "code");
    fs::create_directories(root / "notes");
    fs::create_directories(root / "annotations");
    std::ofstream(root / "notes" / "notes.md") << "# Notes\n";
    std::ofstream(root / "annotations" / "annotations.json")
        << "{\n  \"schemaVersion\": 1,\n  \"annotations\": []\n}\n";
}

} // namespace

VaultStorage::VaultStorage(fs::path root) : root_(std::move(root)) {
    if (root_.empty()) throw StorageError("Vault path cannot be empty");
    root_ = fs::absolute(root_).lexically_normal();
}

void VaultStorage::initialize(const std::string& displayName) {
    fs::create_directories(topicsPath());
    const fs::path metadata = root_ / "vault.json";
    if (!fs::exists(metadata)) {
        const std::string name = displayName.empty() ? root_.filename().string() : displayName;
        std::ostringstream value;
        value << "{\n"
              << "  \"schemaVersion\": 1,\n"
              << "  \"id\": " << jsonString(makeStableId()) << ",\n"
              << "  \"name\": " << jsonString(name) << ",\n"
              << "  \"createdAt\": " << jsonString(utcTimestamp()) << "\n"
              << "}\n";
        writeJsonAtomically(metadata, value.str());
    }
}

fs::path VaultStorage::topicsPath() const {
    return root_ / "topics";
}

fs::path VaultStorage::uniqueChildPath(const fs::path& parent,
                                       const std::string& preferredName) const {
    fs::path candidate = parent / preferredName;
    for (int suffix = 2; fs::exists(candidate); ++suffix) {
        candidate = parent / (preferredName + " (" + std::to_string(suffix) + ")");
    }
    return candidate;
}

Topic VaultStorage::createTopic(const std::string& name) const {
    if (!fs::exists(root_ / "vault.json")) throw StorageError("Vault is not initialized");
    const std::string safeName = sanitizeDirectoryName(name, "Untitled Topic");
    const fs::path destination = uniqueChildPath(topicsPath(), safeName);
    const fs::path temporary = topicsPath() / (".scholarvault-tmp-" + makeStableId());
    Topic topic{makeStableId(), name.empty() ? safeName : name, destination, {}};
    try {
        fs::create_directories(temporary);
        writeJsonAtomically(temporary / "topic.json", topicJson(topic));
        fs::rename(temporary, destination);
    } catch (const std::exception& error) {
        std::error_code cleanupError;
        fs::remove_all(temporary, cleanupError);
        throw StorageError("Cannot create topic: " + std::string(error.what()));
    }
    return topic;
}

Topic VaultStorage::loadTopic(const fs::path& topicPath) const {
    if (!isContainedBy(topicsPath(), topicPath)) {
        throw StorageError("Topic is not contained by this Vault");
    }
    const auto value = readJson(topicPath / "topic.json");
    return Topic{value.get<std::string>("id"), value.get<std::string>("name"),
                 fs::weakly_canonical(topicPath),
                 value.get<std::string>("zoteroCollectionKey", "")};
}

Topic VaultStorage::requireOwnedTopic(const fs::path& topicPath) const {
    if (!fs::exists(topicPath / "topic.json")) throw StorageError("Topic metadata is missing");
    return loadTopic(topicPath);
}

Topic VaultStorage::ensureZoteroTopic(
    const std::vector<ZoteroCollection>& requestedPath) const {
    std::vector<ZoteroCollection> path = requestedPath;
    if (path.empty()) path.push_back(ZoteroCollection{"__unfiled__", "未分类"});

    fs::path parent = topicsPath();
    Topic current;
    for (const auto& collection : path) {
        std::optional<Topic> found;
        if (fs::is_directory(parent)) {
            for (const auto& entry : fs::directory_iterator(parent)) {
                if (!entry.is_directory() || !fs::exists(entry.path() / "topic.json")) continue;
                try {
                    Topic candidate = loadTopic(entry.path());
                    if (candidate.zoteroCollectionKey == collection.key) {
                        found = std::move(candidate);
                        break;
                    }
                } catch (const StorageError&) {}
            }
        }
        if (!found) {
            const std::string displayName = collection.name.empty()
                ? "未分类" : collection.name;
            const fs::path destination = uniqueChildPath(
                parent, sanitizeDirectoryName(displayName, "Zotero Collection"));
            const fs::path temporary = parent / (".scholarvault-tmp-" + makeStableId());
            Topic created{makeStableId(), displayName, destination, collection.key};
            try {
                fs::create_directories(temporary);
                writeJsonAtomically(temporary / "topic.json", topicJson(created));
                fs::rename(temporary, destination);
                found = std::move(created);
            } catch (const std::exception& error) {
                std::error_code cleanupError;
                fs::remove_all(temporary, cleanupError);
                throw StorageError("Cannot create Zotero topic: " +
                                   std::string(error.what()));
            }
        }
        current = *found;
        parent = current.path;
    }
    return current;
}

Project VaultStorage::createProjectFromPdf(const fs::path& topicPath,
                                           const fs::path& sourcePdf,
                                           const std::string& title) const {
    return createPdfProject(topicPath, sourcePdf, title, ProjectOrigin::LocalPdf,
                            std::nullopt);
}

Project VaultStorage::createProjectFromZotero(const fs::path& topicPath,
                                              const ZoteroPaper& paper,
                                              const std::string& title) const {
    ZoteroProvenance provenance;
    provenance.itemKey = paper.itemKey;
    provenance.attachmentKey = paper.attachmentKey;
    provenance.dataDirectory = paper.dataDirectory.string();
    provenance.attachmentPath = paper.pdfPath.string();
    provenance.syncedTitle = paper.title;
    provenance.authors = paper.authors;
    provenance.year = paper.year;
    provenance.lastSyncedAt = utcTimestamp();
    Project project = createPdfProject(topicPath, paper.pdfPath,
                                       title.empty() ? paper.title : title,
                                       ProjectOrigin::Zotero, std::move(provenance));
    if (!paper.arxivId.empty()) {
        project.arxivId = paper.arxivId;
        saveProject(project);
    }
    return project;
}

VaultStorage::ZoteroSyncResult VaultStorage::syncProjectFromZotero(
    const Project& project, const ZoteroPaper& paper) const {
    Project updated = loadProject(project.path);
    if (updated.id != project.id || updated.origin != ProjectOrigin::Zotero ||
        !updated.zotero) {
        throw StorageError("Project is not linked to Zotero");
    }
    if (paper.itemKey.empty() || updated.zotero->itemKey != paper.itemKey) {
        throw StorageError("Zotero item identity does not match this project");
    }
    if (!fs::is_regular_file(paper.pdfPath) || !isPdf(paper.pdfPath)) {
        throw StorageError("Zotero PDF is not readable");
    }

    const fs::path destination = updated.path / updated.pdfRelativePath;
    const bool pdfUpdated = !filesEqual(destination, paper.pdfPath);
    const bool metadataUpdated = updated.title != paper.title ||
        updated.zotero->attachmentKey != paper.attachmentKey ||
        updated.zotero->dataDirectory != paper.dataDirectory.string() ||
        updated.zotero->attachmentPath != paper.pdfPath.string() ||
        updated.zotero->syncedTitle != paper.title ||
        updated.zotero->authors != paper.authors ||
        updated.zotero->year != paper.year ||
        (!paper.arxivId.empty() && updated.arxivId != paper.arxivId);

    if (pdfUpdated) {
        const fs::path temporary = destination.string() + ".sync-" + makeStableId();
        try {
            fs::copy_file(paper.pdfPath, temporary, fs::copy_options::none);
            if (!isPdf(temporary)) throw StorageError("Copied Zotero PDF is invalid");
            fs::rename(temporary, destination);
        } catch (...) {
            std::error_code cleanupError;
            fs::remove(temporary, cleanupError);
            throw;
        }
    }
    if (pdfUpdated || metadataUpdated) {
        updated.title = paper.title;
        updated.zotero->attachmentKey = paper.attachmentKey;
        updated.zotero->dataDirectory = paper.dataDirectory.string();
        updated.zotero->attachmentPath = paper.pdfPath.string();
        updated.zotero->syncedTitle = paper.title;
        updated.zotero->authors = paper.authors;
        updated.zotero->year = paper.year;
        if (!paper.arxivId.empty()) updated.arxivId = paper.arxivId;
        updated.zotero->lastSyncedAt = utcTimestamp();
        saveProject(updated);
    }
    return ZoteroSyncResult{std::move(updated), pdfUpdated, metadataUpdated};
}

VaultStorage::ZoteroLibrarySyncResult VaultStorage::syncZoteroLibrary(
    const std::vector<ZoteroPaper>& papers) const {
    ZoteroLibrarySyncResult result;
    std::unordered_map<std::string, Project> existingByItem;
    for (auto project : discoverAllProjects()) {
        if (project.origin != ProjectOrigin::Zotero || !project.zotero ||
            project.zotero->itemKey.empty()) {
            continue;
        }
        existingByItem.try_emplace(project.zotero->itemKey, std::move(project));
    }

    for (const auto& paper : papers) {
        if (paper.itemKey.empty()) {
            ++result.skipped;
            result.errors.push_back(paper.title + ": missing Zotero item key");
            continue;
        }
        try {
            const Topic targetTopic = ensureZoteroTopic(paper.collectionPath);
            const auto found = existingByItem.find(paper.itemKey);
            if (found == existingByItem.end()) {
                Project created = createProjectFromZotero(targetTopic.path, paper, {});
                existingByItem.emplace(paper.itemKey, created);
                result.changedProjects.push_back(std::move(created));
                ++result.created;
                continue;
            }

            Project project = found->second;
            bool projectMoved = false;
            if (project.path.parent_path() != targetTopic.path) {
                const fs::path original = project.path;
                const fs::path destination = uniqueChildPath(
                    targetTopic.path, project.path.filename().string());
                fs::rename(original, destination);
                project.path = destination;
                project.topicId = targetTopic.id;
                try {
                    saveProject(project);
                } catch (...) {
                    std::error_code rollbackError;
                    fs::rename(destination, original, rollbackError);
                    throw;
                }
                ++result.moved;
                projectMoved = true;
            }

            ZoteroSyncResult synchronized = syncProjectFromZotero(project, paper);
            found->second = synchronized.project;
            if (synchronized.pdfUpdated || synchronized.metadataUpdated) {
                ++result.updated;
                result.changedProjects.push_back(std::move(synchronized.project));
            } else {
                ++result.unchanged;
                if (projectMoved) {
                    result.changedProjects.push_back(found->second);
                }
            }
        } catch (const std::exception& error) {
            ++result.skipped;
            result.errors.push_back(paper.title + ": " + error.what());
        }
    }
    return result;
}

fs::path VaultStorage::moveTopicToTrash(const fs::path& topicPath) const {
    const Topic topic = requireOwnedTopic(topicPath);
    const fs::path trash = root_ / ".trash" / "topics";
    fs::create_directories(trash);
    const std::string preferred =
        sanitizeDirectoryName(topic.path.filename().string() + "-" + topic.id,
                              "Deleted Topic");
    const fs::path destination = uniqueChildPath(trash, preferred);
    try {
        fs::rename(topic.path, destination);
    } catch (const std::exception& error) {
        throw StorageError("Cannot move topic to trash: " + std::string(error.what()));
    }
    return destination;
}

Project VaultStorage::createPdfProject(const fs::path& topicPath,
                                       const fs::path& sourcePdf,
                                       const std::string& title,
                                       ProjectOrigin origin,
                                       std::optional<ZoteroProvenance> zotero) const {
    const Topic topic = requireOwnedTopic(topicPath);
    if (!fs::is_regular_file(sourcePdf) || !isPdf(sourcePdf)) {
        throw StorageError("Selected file is not a readable PDF");
    }

    const std::string displayTitle = title.empty() ? sourcePdf.stem().string() : title;
    const std::string safeName = sanitizeDirectoryName(displayTitle, "Untitled Paper");
    const fs::path destination = uniqueChildPath(topic.path, safeName);
    const fs::path temporary = topic.path / (".scholarvault-tmp-" + makeStableId());
    Project project;
    project.id = makeStableId();
    project.topicId = topic.id;
    project.title = displayTitle;
    project.origin = origin;
    project.zotero = std::move(zotero);
    project.path = destination;

    try {
        createProjectDirectories(temporary);
        fs::copy_file(sourcePdf, temporary / project.pdfRelativePath,
                      fs::copy_options::none);
        writeJsonAtomically(temporary / "project.json", projectJson(project));
        fs::rename(temporary, destination);
    } catch (const std::exception& error) {
        std::error_code cleanupError;
        fs::remove_all(temporary, cleanupError);
        throw StorageError("Cannot import PDF: " + std::string(error.what()));
    }
    return project;
}

Project VaultStorage::createProjectFromArxivFiles(
    const fs::path& topicPath,
    const ArxivReference& arxiv,
    const fs::path& sourcePdf,
    const fs::path& sourceArchive,
    const fs::path& extractedSource,
    const std::string& title,
    const std::string& sourceStatus) const {
    const Topic topic = requireOwnedTopic(topicPath);
    if (!fs::is_regular_file(sourcePdf) || !isPdf(sourcePdf)) {
        throw StorageError("Downloaded arXiv PDF is invalid");
    }
    const std::string displayTitle = title.empty() ? arxiv.id : title;
    const std::string preferred = sanitizeDirectoryName(displayTitle + "-" + arxiv.baseId,
                                                        arxiv.baseId);
    const fs::path destination = uniqueChildPath(topic.path, preferred);
    const fs::path temporary = topic.path / (".scholarvault-tmp-" + makeStableId());
    Project project;
    project.id = makeStableId();
    project.topicId = topic.id;
    project.title = displayTitle;
    project.origin = ProjectOrigin::Arxiv;
    project.arxivId = arxiv.id;
    project.sourceStatus = sourceStatus;
    project.path = destination;

    try {
        createProjectDirectories(temporary);
        fs::copy_file(sourcePdf, temporary / project.pdfRelativePath);
        if (!sourceArchive.empty() && fs::is_regular_file(sourceArchive)) {
            fs::copy_file(sourceArchive, temporary / "source" / "original");
        }
        if (!extractedSource.empty() && fs::is_directory(extractedSource)) {
            fs::copy(extractedSource, temporary / "source" / "extracted",
                     fs::copy_options::recursive | fs::copy_options::overwrite_existing);
        }
        writeJsonAtomically(temporary / "project.json", projectJson(project));
        fs::rename(temporary, destination);
    } catch (const std::exception& error) {
        std::error_code cleanupError;
        fs::remove_all(temporary, cleanupError);
        throw StorageError("Cannot create arXiv project: " + std::string(error.what()));
    }
    return project;
}

Project VaultStorage::installArxivSource(
    const Project& project,
    const ArxivReference& arxiv,
    const fs::path& sourceArchive,
    const fs::path& extractedSource,
    const std::string& sourceStatus) const {
    Project updated = loadProject(project.path);
    if (updated.id != project.id) throw StorageError("Project identity changed");
    const Topic owner = requireOwnedTopic(updated.path.parent_path());
    if (owner.id != updated.topicId) throw StorageError("Project topic ownership changed");

    if (sourceStatus == "ready") {
        if (!fs::is_regular_file(sourceArchive) || !fs::is_directory(extractedSource)) {
            throw StorageError("Downloaded arXiv source is incomplete");
        }
        const fs::path source = updated.path / "source";
        if (fs::exists(source)) {
            for (const auto& entry : fs::recursive_directory_iterator(source)) {
                if (entry.is_regular_file() || entry.is_symlink()) {
                    throw StorageError("Project source directory already contains files");
                }
            }
        }
        const fs::path temporary = updated.path / (".source-tmp-" + makeStableId());
        const fs::path backup = updated.path / (".source-backup-" + makeStableId());
        try {
            fs::create_directories(temporary / "extracted");
            fs::copy_file(sourceArchive, temporary / "original");
            fs::copy(extractedSource, temporary / "extracted",
                     fs::copy_options::recursive | fs::copy_options::overwrite_existing);
            if (fs::exists(source)) fs::rename(source, backup);
            fs::rename(temporary, source);
            std::error_code cleanupError;
            fs::remove_all(backup, cleanupError);
        } catch (const std::exception& error) {
            std::error_code cleanupError;
            fs::remove_all(temporary, cleanupError);
            if (!fs::exists(source) && fs::exists(backup)) {
                fs::rename(backup, source, cleanupError);
            }
            throw StorageError("Cannot install arXiv source: " +
                               std::string(error.what()));
        }
    }
    updated.arxivId = arxiv.id;
    updated.sourceStatus = sourceStatus;
    saveProject(updated);
    return updated;
}

std::vector<Topic> VaultStorage::discoverTopics() const {
    std::vector<Topic> topics;
    if (!fs::is_directory(topicsPath())) return topics;
    for (const auto& entry : fs::directory_iterator(topicsPath())) {
        if (!entry.is_directory() || entry.path().filename().string().starts_with('.')) continue;
        if (!fs::exists(entry.path() / "topic.json")) continue;
        try { topics.push_back(loadTopic(entry.path())); } catch (const StorageError&) {}
    }
    std::sort(topics.begin(), topics.end(), [](const Topic& left, const Topic& right) {
        return left.name < right.name;
    });
    return topics;
}

std::vector<Topic> VaultStorage::discoverChildTopics(const Topic& topic) const {
    const Topic owned = requireOwnedTopic(topic.path);
    if (owned.id != topic.id) throw StorageError("Topic identity does not match metadata");
    std::vector<Topic> topics;
    for (const auto& entry : fs::directory_iterator(owned.path)) {
        if (!entry.is_directory() || entry.path().filename().string().starts_with('.')) continue;
        if (!fs::exists(entry.path() / "topic.json")) continue;
        try { topics.push_back(loadTopic(entry.path())); } catch (const StorageError&) {}
    }
    std::sort(topics.begin(), topics.end(), [](const Topic& left, const Topic& right) {
        return left.name < right.name;
    });
    return topics;
}

std::vector<Project> VaultStorage::discoverAllProjects() const {
    std::vector<Project> projects;
    std::vector<Topic> pending = discoverTopics();
    while (!pending.empty()) {
        Topic topic = std::move(pending.back());
        pending.pop_back();
        auto children = discoverChildTopics(topic);
        pending.insert(pending.end(),
                       std::make_move_iterator(children.begin()),
                       std::make_move_iterator(children.end()));
        auto topicProjects = discoverProjects(topic);
        projects.insert(projects.end(),
                        std::make_move_iterator(topicProjects.begin()),
                        std::make_move_iterator(topicProjects.end()));
    }
    return projects;
}

Project VaultStorage::loadProject(const fs::path& projectPath) const {
    return projectFromTree(projectPath, readJson(projectPath / "project.json"));
}

void VaultStorage::saveProject(const Project& project) const {
    if (!fs::is_directory(project.path) || !fs::exists(project.path / "project.json")) {
        throw StorageError("Project directory is missing");
    }
    const Topic topic = requireOwnedTopic(project.path.parent_path());
    if (topic.id != project.topicId) throw StorageError("Project topic ownership changed");
    writeJsonAtomically(project.path / "project.json", projectJson(project));
}

std::vector<Project> VaultStorage::discoverProjects(const Topic& topic) const {
    const Topic owned = requireOwnedTopic(topic.path);
    if (owned.id != topic.id) throw StorageError("Topic identity does not match metadata");
    std::vector<Project> projects;
    for (const auto& entry : fs::directory_iterator(topic.path)) {
        if (!entry.is_directory() || entry.path().filename().string().starts_with('.')) continue;
        if (!fs::exists(entry.path() / "project.json")) continue;
        try {
            Project project = loadProject(entry.path());
            if (project.topicId == topic.id) projects.push_back(std::move(project));
        } catch (const StorageError&) {}
    }
    std::sort(projects.begin(), projects.end(), [](const Project& left, const Project& right) {
        return left.title < right.title;
    });
    return projects;
}

} // namespace scholarvault
