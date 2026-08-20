#include "scholarvault/zotero.hpp"

#include "scholarvault/arxiv.hpp"

#include <sqlite3.h>

#include <algorithm>
#include <cctype>
#include <memory>
#include <system_error>
#include <unordered_map>

namespace scholarvault {
namespace fs = std::filesystem;
namespace {

using Database = std::unique_ptr<sqlite3, decltype(&sqlite3_close)>;
using Statement = std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)>;

std::string columnText(sqlite3_stmt* statement, int column) {
    const auto* value = sqlite3_column_text(statement, column);
    return value == nullptr ? std::string{} :
                              reinterpret_cast<const char*>(value);
}

std::string extractYear(const std::string& date) {
    for (std::size_t index = 0; index + 4 <= date.size(); ++index) {
        if (!std::all_of(date.begin() + static_cast<std::ptrdiff_t>(index),
                         date.begin() + static_cast<std::ptrdiff_t>(index + 4),
                         [](unsigned char value) { return std::isdigit(value) != 0; })) {
            continue;
        }
        const int year = std::stoi(date.substr(index, 4));
        if (year >= 1000 && year <= 2999) return date.substr(index, 4);
    }
    return {};
}

bool isWithin(const fs::path& root, const fs::path& candidate) {
    auto rootPart = root.begin();
    auto candidatePart = candidate.begin();
    for (; rootPart != root.end(); ++rootPart, ++candidatePart) {
        if (candidatePart == candidate.end() || *rootPart != *candidatePart) return false;
    }
    return true;
}

} // namespace

fs::path resolveZoteroAttachmentPath(const fs::path& dataDirectory,
                                     const std::string& attachmentKey,
                                     const std::string& storedPath) {
    if (storedPath.starts_with("storage:")) {
        const fs::path root = fs::absolute(dataDirectory / "storage" / attachmentKey)
                                  .lexically_normal();
        const fs::path candidate = (root / storedPath.substr(8)).lexically_normal();
        if (!isWithin(root, candidate)) {
            throw ZoteroError("Zotero attachment path escapes its storage directory");
        }
        return candidate;
    }
    const fs::path path(storedPath);
    return path.is_absolute() ? path.lexically_normal()
                              : (dataDirectory / path).lexically_normal();
}

std::vector<ZoteroPaper> readZoteroSqliteLibrary(const fs::path& dataDirectory) {
    const fs::path databasePath = dataDirectory / "zotero.sqlite";
    sqlite3* rawDatabase = nullptr;
    const int openResult = sqlite3_open_v2(databasePath.c_str(), &rawDatabase,
                                           SQLITE_OPEN_READONLY, nullptr);
    Database database(rawDatabase, sqlite3_close);
    if (openResult != SQLITE_OK) {
        const std::string error = rawDatabase == nullptr ? "unknown SQLite error"
                                                         : sqlite3_errmsg(rawDatabase);
        throw ZoteroError("Cannot read Zotero library: " + error);
    }
    sqlite3_busy_timeout(database.get(), 800);

    struct CollectionRow {
        std::string key;
        std::string name;
        sqlite3_int64 parentId{0};
    };
    std::unordered_map<sqlite3_int64, CollectionRow> collections;
    sqlite3_stmt* rawCollections = nullptr;
    constexpr const char* collectionQuery =
        "SELECT collectionID, key, collectionName, "
        "COALESCE(parentCollectionID, 0) FROM collections";
    if (sqlite3_prepare_v2(database.get(), collectionQuery, -1, &rawCollections,
                           nullptr) == SQLITE_OK) {
        Statement collectionStatement(rawCollections, sqlite3_finalize);
        while (sqlite3_step(collectionStatement.get()) == SQLITE_ROW) {
            collections.emplace(
                sqlite3_column_int64(collectionStatement.get(), 0),
                CollectionRow{columnText(collectionStatement.get(), 1),
                              columnText(collectionStatement.get(), 2),
                              sqlite3_column_int64(collectionStatement.get(), 3)});
        }
    }

    std::unordered_map<sqlite3_int64, std::vector<sqlite3_int64>> itemCollections;
    sqlite3_stmt* rawMembership = nullptr;
    if (sqlite3_prepare_v2(database.get(),
                           "SELECT itemID, collectionID FROM collectionItems", -1,
                           &rawMembership, nullptr) == SQLITE_OK) {
        Statement membership(rawMembership, sqlite3_finalize);
        while (sqlite3_step(membership.get()) == SQLITE_ROW) {
            itemCollections[sqlite3_column_int64(membership.get(), 0)].push_back(
                sqlite3_column_int64(membership.get(), 1));
        }
    }

    const auto collectionPath = [&collections](sqlite3_int64 collectionId) {
        std::vector<ZoteroCollection> reversed;
        std::size_t guard = 0;
        while (collectionId != 0 && guard++ <= collections.size()) {
            const auto found = collections.find(collectionId);
            if (found == collections.end()) break;
            reversed.push_back({found->second.key, found->second.name});
            collectionId = found->second.parentId;
        }
        std::reverse(reversed.begin(), reversed.end());
        return reversed;
    };

    constexpr const char* query = R"SQL(
        SELECT ia.itemID,
               COALESCE(ia.parentItemID, 0),
               attachment.key,
               COALESCE(parent.key, ''),
               ia.path,
               COALESCE(
                   (SELECT value.value
                      FROM itemData data
                      JOIN fields field ON field.fieldID=data.fieldID
                      JOIN itemDataValues value ON value.valueID=data.valueID
                     WHERE data.itemID=ia.parentItemID AND field.fieldName='title'
                     LIMIT 1),
                   ''),
               COALESCE(
                   (SELECT value.value
                      FROM itemData data
                      JOIN fields field ON field.fieldID=data.fieldID
                      JOIN itemDataValues value ON value.valueID=data.valueID
                     WHERE data.itemID=ia.parentItemID AND field.fieldName='date'
                     LIMIT 1),
                   ''),
               COALESCE(
                   (SELECT group_concat(displayName, ', ')
                      FROM (
                          SELECT trim(CASE WHEN creator.fieldMode=1
                                           THEN creator.lastName
                                           ELSE creator.firstName || ' ' || creator.lastName
                                      END) AS displayName
                            FROM itemCreators itemCreator
                            JOIN creators creator USING(creatorID)
                            JOIN creatorTypes creatorType USING(creatorTypeID)
                           WHERE itemCreator.itemID=ia.parentItemID
                             AND creatorType.creatorType IN ('author', 'editor')
                           ORDER BY CASE creatorType.creatorType WHEN 'author' THEN 0 ELSE 1 END,
                                    itemCreator.orderIndex
                      )),
                   ''),
               COALESCE(
                   (SELECT value.value
                      FROM itemData data
                      JOIN fields field ON field.fieldID=data.fieldID
                      JOIN itemDataValues value ON value.valueID=data.valueID
                     WHERE data.itemID=COALESCE(ia.parentItemID, ia.itemID)
                       AND field.fieldName='extra'
                     LIMIT 1), ''),
               COALESCE(
                   (SELECT value.value
                      FROM itemData data
                      JOIN fields field ON field.fieldID=data.fieldID
                      JOIN itemDataValues value ON value.valueID=data.valueID
                     WHERE data.itemID=COALESCE(ia.parentItemID, ia.itemID)
                       AND field.fieldName='archiveID'
                     LIMIT 1), ''),
               COALESCE(
                   (SELECT value.value
                      FROM itemData data
                      JOIN fields field ON field.fieldID=data.fieldID
                      JOIN itemDataValues value ON value.valueID=data.valueID
                     WHERE data.itemID=COALESCE(ia.parentItemID, ia.itemID)
                       AND field.fieldName='url'
                     LIMIT 1), ''),
               COALESCE(
                   (SELECT value.value
                      FROM itemData data
                      JOIN fields field ON field.fieldID=data.fieldID
                      JOIN itemDataValues value ON value.valueID=data.valueID
                     WHERE data.itemID=COALESCE(ia.parentItemID, ia.itemID)
                       AND field.fieldName='DOI'
                     LIMIT 1), '')
          FROM itemAttachments ia
          JOIN items attachment ON attachment.itemID=ia.itemID
          LEFT JOIN items parent ON parent.itemID=ia.parentItemID
          LEFT JOIN deletedItems deletedAttachment ON deletedAttachment.itemID=ia.itemID
          LEFT JOIN deletedItems deletedParent ON deletedParent.itemID=ia.parentItemID
         WHERE ia.contentType='application/pdf'
           AND deletedAttachment.itemID IS NULL
           AND deletedParent.itemID IS NULL
         ORDER BY 6 COLLATE NOCASE
    )SQL";

    sqlite3_stmt* rawStatement = nullptr;
    if (sqlite3_prepare_v2(database.get(), query, -1, &rawStatement, nullptr) != SQLITE_OK) {
        throw ZoteroError("Cannot query Zotero library: " +
                          std::string(sqlite3_errmsg(database.get())));
    }
    Statement statement(rawStatement, sqlite3_finalize);
    std::vector<ZoteroPaper> papers;
    int result = SQLITE_ROW;
    while ((result = sqlite3_step(statement.get())) == SQLITE_ROW) {
        ZoteroPaper paper;
        const sqlite3_int64 attachmentItemId = sqlite3_column_int64(statement.get(), 0);
        const sqlite3_int64 parentItemId = sqlite3_column_int64(statement.get(), 1);
        paper.attachmentKey = columnText(statement.get(), 2);
        paper.itemKey = columnText(statement.get(), 3);
        if (paper.itemKey.empty()) paper.itemKey = paper.attachmentKey;
        const std::string storedPath = columnText(statement.get(), 4);
        paper.title = columnText(statement.get(), 5);
        paper.year = extractYear(columnText(statement.get(), 6));
        paper.authors = columnText(statement.get(), 7);
        for (int column = 8; column <= 11; ++column) {
            const auto reference = findArxivReference(columnText(statement.get(), column));
            if (reference) {
                paper.arxivId = reference->id;
                break;
            }
        }
        paper.dataDirectory = fs::absolute(dataDirectory).lexically_normal();
        try {
            paper.pdfPath = resolveZoteroAttachmentPath(
                paper.dataDirectory, paper.attachmentKey, storedPath);
        } catch (const ZoteroError&) {
            continue;
        }
        if (!fs::is_regular_file(paper.pdfPath)) continue;
        if (paper.title.empty()) paper.title = paper.pdfPath.stem().string();
        const sqlite3_int64 logicalItemId = parentItemId == 0
            ? attachmentItemId : parentItemId;
        std::vector<std::vector<ZoteroCollection>> paths;
        for (const sqlite3_int64 collectionId : itemCollections[logicalItemId]) {
            auto path = collectionPath(collectionId);
            if (!path.empty()) paths.push_back(std::move(path));
        }
        if (!paths.empty()) {
            std::sort(paths.begin(), paths.end(), [](const auto& left, const auto& right) {
                const auto names = [](const auto& path) {
                    std::string value;
                    for (const auto& item : path) value += "/" + item.name;
                    return value;
                };
                return names(left) < names(right);
            });
            paper.collectionPath = std::move(paths.front());
        }
        papers.push_back(std::move(paper));
    }
    if (result != SQLITE_DONE) {
        throw ZoteroError("Cannot finish Zotero query: " +
                          std::string(sqlite3_errmsg(database.get())));
    }
    return papers;
}

} // namespace scholarvault
