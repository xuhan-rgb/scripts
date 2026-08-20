#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "scholarvault/arxiv.hpp"
#include "scholarvault/archives.hpp"
#include "scholarvault/github.hpp"
#include "scholarvault/storage.hpp"
#include "scholarvault/zotero.hpp"

#include <sqlite3.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;
using namespace scholarvault;

namespace {

class TemporaryDirectory {
public:
    TemporaryDirectory()
        : path_(fs::temp_directory_path() /
                ("scholarvault-test-" + std::to_string(
                    std::chrono::steady_clock::now().time_since_epoch().count()))) {
        fs::create_directories(path_);
    }

    ~TemporaryDirectory() { std::error_code error; fs::remove_all(path_, error); }

    const fs::path& path() const { return path_; }

private:
    fs::path path_;
};

void writeFakePdf(const fs::path& path) {
    std::ofstream output(path, std::ios::binary);
    output << "%PDF-1.7\n% ScholarVault test fixture\n";
}

void runSql(sqlite3* database, const std::string& sql) {
    char* message = nullptr;
    const int result = sqlite3_exec(database, sql.c_str(), nullptr, nullptr, &message);
    const std::string error = message == nullptr ? "" : message;
    sqlite3_free(message);
    INFO(error);
    REQUIRE(result == SQLITE_OK);
}

std::string readText(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

} // namespace

TEST_CASE("arXiv links and identifiers are normalized") {
    const auto modern = parseArxivReference("https://arxiv.org/pdf/2504.16054v2.pdf?download=1");
    REQUIRE(modern.has_value());
    CHECK(modern->id == "2504.16054v2");
    CHECK(modern->baseId == "2504.16054");
    CHECK(modern->version == 2);
    CHECK(modern->sourceUrl() == "https://arxiv.org/src/2504.16054v2");

    const auto legacy = parseArxivReference("https://arxiv.org/abs/hep-th/9901001v3");
    REQUIRE(legacy.has_value());
    CHECK(legacy->id == "hep-th/9901001v3");
    CHECK(legacy->baseId == "hep-th/9901001");
    CHECK(legacy->version == 3);

    CHECK_FALSE(parseArxivReference("https://example.com/abs/2504.16054").has_value());
    CHECK_FALSE(parseArxivReference("2504.160").has_value());

    const auto zoteroExtra = findArxivReference("Citation Key: law2025\narXiv: 2504.16054v3");
    REQUIRE(zoteroExtra.has_value());
    CHECK(zoteroExtra->id == "2504.16054v3");
    const auto arxivDoi = findArxivReference("10.48550/arXiv.2504.16054");
    REQUIRE(arxivDoi.has_value());
    CHECK(arxivDoi->id == "2504.16054");
    CHECK_FALSE(findArxivReference("doi:10.1234/2504.16054").has_value());
}

TEST_CASE("public GitHub repository links reject credentials and subpaths") {
    const auto repository = parseGitHubRepository("https://github.com/openai/codex.git/");
    REQUIRE(repository.has_value());
    CHECK(repository->owner == "openai");
    CHECK(repository->name == "codex");
    CHECK(repository->normalizedUrl == "https://github.com/openai/codex");

    CHECK_FALSE(parseGitHubRepository("https://token@github.com/openai/codex").has_value());
    CHECK_FALSE(parseGitHubRepository("https://github.com/openai/codex/tree/main").has_value());
    CHECK_FALSE(parseGitHubRepository("git@github.com:openai/codex.git").has_value());
}

TEST_CASE("arXiv source archive entries cannot escape extraction root") {
    CHECK(archiveEntryPathIsSafe("paper/main.tex"));
    CHECK(archiveEntryPathIsSafe("./figures/chart.pdf"));
    CHECK_FALSE(archiveEntryPathIsSafe("../outside.tex"));
    CHECK_FALSE(archiveEntryPathIsSafe("paper/../../outside.tex"));
    CHECK_FALSE(archiveEntryPathIsSafe("/etc/passwd"));
    CHECK_FALSE(archiveEntryPathIsSafe("C:\\Windows\\system.ini"));
    CHECK(archiveListingIsSafe({"main.tex", "figures/a.png"}));
    CHECK_FALSE(archiveListingIsSafe({"main.tex", "../escape"}));
    CHECK_FALSE(archiveListingIsSafe({}));
}

TEST_CASE("one project is physically contained by one topic") {
    TemporaryDirectory temporary;
    VaultStorage storage(temporary.path() / "vault");
    storage.initialize("Research");
    CHECK(readText(storage.rootPath() / "vault.json").find("\"schemaVersion\": 1") !=
          std::string::npos);

    const Topic topic = storage.createTopic("World Models");
    CHECK(topic.name == "World Models");
    CHECK(topic.path.parent_path() == storage.topicsPath());
    CHECK(fs::exists(topic.path / "topic.json"));
    CHECK(readText(topic.path / "topic.json").find("\"schemaVersion\": 1") !=
          std::string::npos);

    const fs::path sourcePdf = temporary.path() / "LAW.pdf";
    writeFakePdf(sourcePdf);
    const Project project = storage.createProjectFromPdf(topic.path, sourcePdf, "LAW: World Model");

    CHECK(project.topicId == topic.id);
    CHECK(project.path.parent_path() == topic.path);
    CHECK(fs::exists(project.path / "paper" / "paper.pdf"));
    CHECK(fs::exists(project.path / "notes" / "notes.md"));
    CHECK(fs::exists(project.path / "annotations" / "annotations.json"));
    CHECK(readText(project.path / "project.json").find("\"schemaVersion\": 1") !=
          std::string::npos);

    const auto topics = storage.discoverTopics();
    REQUIRE(topics.size() == 1);
    const auto projects = storage.discoverProjects(topics.front());
    REQUIRE(projects.size() == 1);
    CHECK(projects.front().id == project.id);
    CHECK(projects.front().topicId == topic.id);

    Project updated = projects.front();
    updated.repositories.push_back(RelatedRepository{
        "https://github.com/openai/codex", "openai", "codex", "main",
        "0123456789abcdef", "code/openai-codex", "ready"});
    storage.saveProject(updated);
    const Project reloaded = storage.loadProject(updated.path);
    REQUIRE(reloaded.repositories.size() == 1);
    CHECK(reloaded.repositories.front().commitSha == "0123456789abcdef");
    CHECK(reloaded.repositories.front().relativePath == "code/openai-codex");
}

TEST_CASE("PDF import copies input and rejects non-PDF files") {
    TemporaryDirectory temporary;
    VaultStorage storage(temporary.path() / "vault");
    storage.initialize("Research");
    const Topic topic = storage.createTopic("UWB");

    const fs::path sourcePdf = temporary.path() / "sensor.pdf";
    writeFakePdf(sourcePdf);
    const Project project = storage.createProjectFromPdf(topic.path, sourcePdf, "Sensor / Fusion");
    CHECK_FALSE(fs::is_symlink(project.path / "paper" / "paper.pdf"));
    CHECK(fs::exists(sourcePdf));
    CHECK(project.path.filename().string().find('/') == std::string::npos);

    const fs::path textFile = temporary.path() / "not-a-pdf.txt";
    std::ofstream(textFile) << "plain text";
    CHECK_THROWS_AS(storage.createProjectFromPdf(topic.path, textFile, "Invalid"), StorageError);
}

TEST_CASE("Zotero catalog resolves stored PDFs and preserves read-only provenance") {
    TemporaryDirectory temporary;
    const fs::path zotero = temporary.path() / "Zotero";
    const fs::path attachmentDirectory = zotero / "storage" / "ATTACH01";
    fs::create_directories(attachmentDirectory);
    const fs::path sourcePdf = attachmentDirectory / "paper.pdf";
    writeFakePdf(sourcePdf);

    sqlite3* database = nullptr;
    REQUIRE(sqlite3_open((zotero / "zotero.sqlite").c_str(), &database) == SQLITE_OK);
    runSql(database, R"SQL(
        CREATE TABLE items(itemID INTEGER PRIMARY KEY, key TEXT);
        CREATE TABLE itemAttachments(itemID INTEGER PRIMARY KEY, parentItemID INT,
            linkMode INT, contentType TEXT, path TEXT);
        CREATE TABLE fields(fieldID INTEGER PRIMARY KEY, fieldName TEXT);
        CREATE TABLE itemData(itemID INT, fieldID INT, valueID INT);
        CREATE TABLE itemDataValues(valueID INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE itemCreators(itemID INT, creatorID INT, creatorTypeID INT,
            orderIndex INT);
        CREATE TABLE creators(creatorID INTEGER PRIMARY KEY, firstName TEXT,
            lastName TEXT, fieldMode INT);
        CREATE TABLE creatorTypes(creatorTypeID INTEGER PRIMARY KEY, creatorType TEXT);
        CREATE TABLE deletedItems(itemID INTEGER PRIMARY KEY);
        CREATE TABLE collections(collectionID INTEGER PRIMARY KEY,
            collectionName TEXT, parentCollectionID INT, key TEXT);
        CREATE TABLE collectionItems(collectionID INT, itemID INT, orderIndex INT);
        INSERT INTO items VALUES(1, 'PARENT01'), (2, 'ATTACH01');
        INSERT INTO itemAttachments VALUES(2, 1, 0, 'application/pdf',
            'storage:paper.pdf');
        INSERT INTO fields VALUES(1, 'title'), (2, 'date'), (3, 'extra');
        INSERT INTO itemData VALUES(1, 1, 1), (1, 2, 2), (1, 3, 3);
        INSERT INTO itemDataValues VALUES(1, 'A Zotero Paper'), (2, '2025-07-01'),
                                         (3, 'Citation Key: paper2025\narXiv: 2504.16054v2');
        INSERT INTO creatorTypes VALUES(1, 'author');
        INSERT INTO creators VALUES(1, 'Ada', 'Lovelace', 0),
                                   (2, 'Alan', 'Turing', 0);
        INSERT INTO itemCreators VALUES(1, 1, 1, 0), (1, 2, 1, 1);
        INSERT INTO collections VALUES(1, 'Engineering', NULL, 'COLLROOT'),
                                      (2, 'World Models', 1, 'COLLCHILD');
        INSERT INTO collectionItems VALUES(2, 1, 0);
    )SQL");
    sqlite3_close(database);

    const auto papers = readZoteroSqliteLibrary(zotero);
    REQUIRE(papers.size() == 1);
    CHECK(papers.front().title == "A Zotero Paper");
    CHECK(papers.front().authors == "Ada Lovelace, Alan Turing");
    CHECK(papers.front().year == "2025");
    CHECK(papers.front().arxivId == "2504.16054v2");
    CHECK(papers.front().itemKey == "PARENT01");
    CHECK(papers.front().attachmentKey == "ATTACH01");
    CHECK(papers.front().pdfPath == sourcePdf);
    REQUIRE(papers.front().collectionPath.size() == 2);
    CHECK(papers.front().collectionPath[0].name == "Engineering");
    CHECK(papers.front().collectionPath[1].name == "World Models");

    VaultStorage storage(temporary.path() / "vault");
    storage.initialize("Research");
    const Topic topic = storage.createTopic("Imported");
    const Project project = storage.createProjectFromZotero(topic.path, papers.front(), "");
    REQUIRE(project.zotero.has_value());
    CHECK(project.origin == ProjectOrigin::Zotero);
    CHECK(project.title == "A Zotero Paper");
    CHECK(project.arxivId == "2504.16054v2");
    CHECK(project.zotero->itemKey == "PARENT01");
    CHECK(fs::exists(sourcePdf));

    const Project reloaded = storage.loadProject(project.path);
    REQUIRE(reloaded.zotero.has_value());
    CHECK(reloaded.arxivId == "2504.16054v2");
    CHECK(reloaded.zotero->attachmentKey == "ATTACH01");
    CHECK(reloaded.zotero->attachmentPath == sourcePdf.string());

    const fs::path sourceArchive = temporary.path() / "source.tar.gz";
    std::ofstream(sourceArchive, std::ios::binary) << "archive";
    const fs::path extractedSource = temporary.path() / "extracted";
    fs::create_directories(extractedSource);
    std::ofstream(extractedSource / "main.tex") << "\\documentclass{article}";
    const Project withSource = storage.installArxivSource(
        reloaded, *parseArxivReference(reloaded.arxivId), sourceArchive,
        extractedSource, "ready");
    CHECK(withSource.sourceStatus == "ready");
    CHECK(fs::exists(withSource.path / "source" / "original"));
    CHECK(fs::exists(withSource.path / "source" / "extracted" / "main.tex"));
    const Project sourceReloaded = storage.loadProject(withSource.path);
    REQUIRE(sourceReloaded.zotero.has_value());
    CHECK(sourceReloaded.zotero->itemKey == "PARENT01");
    CHECK(sourceReloaded.arxivId == "2504.16054v2");

    const fs::path updatedPdf = attachmentDirectory / "updated.pdf";
    {
        std::ofstream output(updatedPdf, std::ios::binary);
        output << "%PDF-1.7\n% updated Zotero attachment\n";
    }
    ZoteroPaper updatedPaper = papers.front();
    updatedPaper.title = "A Renamed Zotero Paper";
    updatedPaper.authors = "Grace Hopper";
    updatedPaper.year = "2026";
    updatedPaper.pdfPath = updatedPdf;
    const auto synchronized = storage.syncProjectFromZotero(sourceReloaded, updatedPaper);
    CHECK(synchronized.pdfUpdated);
    CHECK(synchronized.metadataUpdated);
    CHECK(synchronized.project.title == "A Renamed Zotero Paper");
    REQUIRE(synchronized.project.zotero.has_value());
    CHECK(synchronized.project.zotero->authors == "Grace Hopper");
    CHECK(synchronized.project.zotero->year == "2026");
    CHECK(readText(synchronized.project.path / "paper" / "paper.pdf") ==
          readText(updatedPdf));
    CHECK(fs::exists(updatedPdf));

    const auto unchanged = storage.syncProjectFromZotero(synchronized.project,
                                                          updatedPaper);
    CHECK_FALSE(unchanged.pdfUpdated);
    CHECK_FALSE(unchanged.metadataUpdated);
}

TEST_CASE("full Zotero synchronization mirrors nested collections and imports PDFs") {
    TemporaryDirectory temporary;
    VaultStorage storage(temporary.path() / "vault");
    storage.initialize("Research");

    const fs::path firstPdf = temporary.path() / "first.pdf";
    const fs::path unfiledPdf = temporary.path() / "unfiled.pdf";
    writeFakePdf(firstPdf);
    writeFakePdf(unfiledPdf);
    ZoteroPaper first{"First Paper", "Ada", "2026", "ITEM1", "ATT1",
                      temporary.path(), firstPdf,
                      {{"ROOT", "Engineering"}, {"CHILD", "World Models"}}};
    ZoteroPaper unfiled{"Unfiled Paper", "Alan", "2025", "ITEM2", "ATT2",
                        temporary.path(), unfiledPdf, {}};

    const auto initial = storage.syncZoteroLibrary({first, unfiled});
    CHECK(initial.created == 2);
    CHECK(initial.updated == 0);
    const auto roots = storage.discoverTopics();
    REQUIRE(roots.size() == 2);
    const auto engineering = std::find_if(roots.begin(), roots.end(), [](const Topic& topic) {
        return topic.name == "Engineering";
    });
    REQUIRE(engineering != roots.end());
    const auto children = storage.discoverChildTopics(*engineering);
    REQUIRE(children.size() == 1);
    CHECK(children.front().name == "World Models");
    const auto projects = storage.discoverProjects(children.front());
    REQUIRE(projects.size() == 1);
    CHECK_FALSE(fs::is_symlink(projects.front().path / "paper" / "paper.pdf"));
    CHECK(readText(projects.front().path / "paper" / "paper.pdf") == readText(firstPdf));

    const auto repeated = storage.syncZoteroLibrary({first, unfiled});
    CHECK(repeated.created == 0);
    CHECK(repeated.unchanged == 2);

    first.collectionPath = {{"ROOT", "Engineering"}, {"NEW", "Planning"}};
    const auto moved = storage.syncZoteroLibrary({first, unfiled});
    CHECK(moved.moved == 1);
    const auto all = storage.discoverAllProjects();
    REQUIRE(all.size() == 2);
    const auto movedProject = std::find_if(all.begin(), all.end(), [](const Project& project) {
        return project.zotero && project.zotero->itemKey == "ITEM1";
    });
    REQUIRE(movedProject != all.end());
    CHECK(movedProject->path.parent_path().filename() == "Planning");
}

TEST_CASE("deleting a topic moves the complete directory to Vault trash") {
    TemporaryDirectory temporary;
    VaultStorage storage(temporary.path() / "vault");
    storage.initialize("Research");
    const Topic topic = storage.createTopic("Recoverable Topic");
    const fs::path sourcePdf = temporary.path() / "paper.pdf";
    writeFakePdf(sourcePdf);
    const Project project = storage.createProjectFromPdf(topic.path, sourcePdf, "Paper");

    const fs::path trashed = storage.moveTopicToTrash(topic.path);
    CHECK_FALSE(fs::exists(topic.path));
    CHECK(trashed.parent_path() == storage.rootPath() / ".trash" / "topics");
    CHECK(fs::exists(trashed / "topic.json"));
    CHECK(fs::exists(trashed / project.path.filename() / "project.json"));
    CHECK(storage.discoverTopics().empty());
    CHECK_THROWS_AS(storage.moveTopicToTrash(temporary.path()), StorageError);
}
