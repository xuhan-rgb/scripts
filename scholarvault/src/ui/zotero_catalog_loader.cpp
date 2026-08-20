#include "ui/zotero_catalog_loader.hpp"

#include "scholarvault/arxiv.hpp"

#include <QFileInfo>
#include <QFutureWatcher>
#include <QHash>
#include <QHostAddress>
#include <QJsonDocument>
#include <QJsonObject>
#include <QRegularExpression>
#include <QTcpSocket>
#include <QTimer>
#include <QUrl>
#include <QtConcurrent>

#include <algorithm>

namespace scholarvault::ui {
namespace {

struct CatalogLoadResult {
    std::vector<ZoteroPaper> papers;
    QString error;
};

QString creatorName(const QJsonObject& creator) {
    const QString singleName = creator.value("name").toString().trimmed();
    if (!singleName.isEmpty()) return singleName;
    return (creator.value("firstName").toString() + " " +
            creator.value("lastName").toString()).trimmed();
}

QString yearFromDate(const QString& date) {
    static const QRegularExpression expression(R"((?:^|\D)(\d{4})(?:\D|$))");
    const auto match = expression.match(date);
    return match.hasMatch() ? match.captured(1) : QString{};
}

} // namespace

std::vector<ZoteroPaper> parseZoteroApiItems(const QJsonArray& items,
                                             const QString& dataDirectory,
                                             const QJsonArray& collections) {
    QHash<QString, QJsonObject> parents;
    for (const auto& value : items) {
        const QJsonObject item = value.toObject();
        const QJsonObject data = item.value("data").toObject();
        if (data.value("itemType").toString() != "attachment") {
            parents.insert(item.value("key").toString(), data);
        }
    }
    QHash<QString, QJsonObject> collectionByKey;
    for (const auto& value : collections) {
        const QJsonObject collection = value.toObject();
        collectionByKey.insert(collection.value("key").toString(),
                               collection.value("data").toObject());
    }
    const auto collectionPath = [&collectionByKey](QString key) {
        std::vector<ZoteroCollection> reversed;
        int guard = 0;
        while (!key.isEmpty() && guard++ <= collectionByKey.size()) {
            const QJsonObject collection = collectionByKey.value(key);
            if (collection.isEmpty()) break;
            reversed.push_back({key.toStdString(),
                                collection.value("name").toString().toStdString()});
            key = collection.value("parentCollection").toString();
        }
        std::reverse(reversed.begin(), reversed.end());
        return reversed;
    };

    std::vector<ZoteroPaper> papers;
    for (const auto& value : items) {
        const QJsonObject item = value.toObject();
        const QJsonObject data = item.value("data").toObject();
        if (data.value("itemType").toString() != "attachment" ||
            data.value("contentType").toString() != "application/pdf") {
            continue;
        }
        const QString attachmentKey = item.value("key").toString();
        QString itemKey = data.value("parentItem").toString();
        const QJsonObject parent = parents.value(itemKey);
        const QJsonObject enclosure = item.value("links").toObject()
                                          .value("enclosure").toObject();
        const QUrl fileUrl(enclosure.value("href").toString());
        const QString pdfPath = fileUrl.isLocalFile() ? fileUrl.toLocalFile() : QString{};
        if (pdfPath.isEmpty() || !QFileInfo(pdfPath).isFile()) continue;

        QStringList authors;
        for (const auto& creatorValue : parent.value("creators").toArray()) {
            const QJsonObject creator = creatorValue.toObject();
            const QString type = creator.value("creatorType").toString();
            if (type != "author" && type != "editor") continue;
            const QString name = creatorName(creator);
            if (!name.isEmpty()) authors.push_back(name);
        }

        ZoteroPaper paper;
        paper.title = parent.value("title").toString().trimmed().toStdString();
        if (paper.title.empty()) {
            paper.title = QFileInfo(pdfPath).completeBaseName().toStdString();
        }
        paper.authors = authors.join(", ").toStdString();
        paper.year = yearFromDate(parent.value("date").toString()).toStdString();
        paper.itemKey = itemKey.toStdString();
        if (paper.itemKey.empty()) paper.itemKey = attachmentKey.toStdString();
        paper.attachmentKey = attachmentKey.toStdString();
        paper.dataDirectory = dataDirectory.toStdString();
        paper.pdfPath = pdfPath.toStdString();
        const QJsonObject metadata = parent.isEmpty() ? data : parent;
        for (const char* field : {"extra", "archiveID", "url", "DOI"}) {
            const auto reference = findArxivReference(
                metadata.value(field).toString().toStdString());
            if (reference) {
                paper.arxivId = reference->id;
                break;
            }
        }
        const QJsonArray membership = parent.isEmpty()
            ? data.value("collections").toArray()
            : parent.value("collections").toArray();
        std::vector<std::vector<ZoteroCollection>> paths;
        for (const auto& key : membership) {
            auto path = collectionPath(key.toString());
            if (!path.empty()) paths.push_back(std::move(path));
        }
        if (!paths.empty()) {
            std::sort(paths.begin(), paths.end(), [](const auto& left, const auto& right) {
                const auto value = [](const auto& path) {
                    std::string result;
                    for (const auto& entry : path) result += "/" + entry.name;
                    return result;
                };
                return value(left) < value(right);
            });
            paper.collectionPath = std::move(paths.front());
        }
        papers.push_back(std::move(paper));
    }
    std::sort(papers.begin(), papers.end(), [](const ZoteroPaper& left,
                                                const ZoteroPaper& right) {
        return left.title < right.title;
    });
    return papers;
}

ZoteroCatalogLoader::ZoteroCatalogLoader(QString dataDirectory, QObject* parent,
                                         quint16 apiPort)
    : QObject(parent),
      dataDirectory_(QFileInfo(dataDirectory).absoluteFilePath()),
      apiTimeout_(new QTimer(this)),
      apiPort_(apiPort) {
    apiTimeout_->setSingleShot(true);
    apiTimeout_->setInterval(2500);
    connect(apiTimeout_, &QTimer::timeout, this, [this] {
        if (apiSocket_ == nullptr) return;
        apiError_ = tr("连接本机 Zotero 超时");
        apiSocket_->abort();
        finishApiPage();
    });
}

void ZoteroCatalogLoader::start() {
    loadingCollections_ = false;
    apiItems_ = {};
    apiCollections_ = {};
    emit progress(tr("正在连接本机 Zotero…"));
    requestApiPage(0);
}

void ZoteroCatalogLoader::requestApiPage(int start) {
    apiStart_ = start;
    apiResponse_.clear();
    apiError_.clear();
    apiSocket_ = new QTcpSocket(this);
    connect(apiSocket_, &QTcpSocket::connected, this, [this, start] {
        const QByteArray request =
            "GET /api/users/0/" +
            QByteArray(loadingCollections_ ? "collections" : "items") +
            "?limit=100&start=" + QByteArray::number(start) +
            " HTTP/1.0\r\nHost: 127.0.0.1:" + QByteArray::number(apiPort_) +
            "\r\nConnection: close\r\n\r\n";
        apiSocket_->write(request);
    });
    connect(apiSocket_, &QTcpSocket::readyRead, this,
            [this] { apiResponse_.append(apiSocket_->readAll()); });
    connect(apiSocket_, &QTcpSocket::disconnected, this,
            &ZoteroCatalogLoader::finishApiPage);
    connect(apiSocket_, &QTcpSocket::errorOccurred, this,
            [this](QAbstractSocket::SocketError error) {
                if (apiSocket_ == nullptr || error == QAbstractSocket::RemoteHostClosedError) {
                    return;
                }
                apiError_ = apiSocket_->errorString();
            });
    apiTimeout_->start();
    apiSocket_->connectToHost(QHostAddress::LocalHost, apiPort_);
}

void ZoteroCatalogLoader::finishApiPage() {
    if (apiSocket_ == nullptr) return;
    apiTimeout_->stop();
    apiResponse_.append(apiSocket_->readAll());
    apiSocket_->deleteLater();
    apiSocket_ = nullptr;

    const qsizetype headerEnd = apiResponse_.indexOf("\r\n\r\n");
    const QByteArray headers = headerEnd >= 0 ? apiResponse_.left(headerEnd) : QByteArray{};
    const QByteArray body = headerEnd >= 0 ? apiResponse_.mid(headerEnd + 4) : QByteArray{};
    int total = 0;
    for (const QByteArray& line : headers.split('\n')) {
        if (line.trimmed().toLower().startsWith("total-results:")) {
            total = line.mid(line.indexOf(':') + 1).trimmed().toInt();
            break;
        }
    }
    const bool successfulStatus = headers.startsWith("HTTP/1.0 200 ") ||
                                  headers.startsWith("HTTP/1.1 200 ");
    const QJsonDocument document = QJsonDocument::fromJson(body);
    if (!successfulStatus || !document.isArray()) {
        const QString error = apiError_.isEmpty() ? tr("响应格式无效") : apiError_;
        loadSqliteFallback(tr("本机 API 不可用：%1").arg(error));
        return;
    }
    const QJsonArray page = document.array();
    QJsonArray& destination = loadingCollections_ ? apiCollections_ : apiItems_;
    for (const auto& value : page) destination.append(value);
    emit progress(loadingCollections_
        ? tr("正在读取 Zotero 分类：%1 / %2").arg(destination.size()).arg(total)
        : tr("正在读取 Zotero：%1 / %2 条目").arg(destination.size()).arg(total));
    if (!page.isEmpty() && apiStart_ + page.size() < total) {
        requestApiPage(apiStart_ + page.size());
        return;
    }
    if (!loadingCollections_) {
        loadingCollections_ = true;
        requestApiPage(0);
        return;
    }
    emit loaded(parseZoteroApiItems(apiItems_, dataDirectory_, apiCollections_),
                tr("Zotero 本机 API"));
}

void ZoteroCatalogLoader::loadSqliteFallback(const QString& reason) {
    if (fallbackStarted_) return;
    fallbackStarted_ = true;
    emit progress(reason + tr("；正在只读打开 zotero.sqlite…"));
    auto* watcher = new QFutureWatcher<CatalogLoadResult>(this);
    connect(watcher, &QFutureWatcher<CatalogLoadResult>::finished, this,
            [this, watcher] {
                const CatalogLoadResult result = watcher->result();
                if (result.error.isEmpty()) {
                    emit loaded(result.papers, tr("只读 SQLite"));
                } else {
                    emit failed(tr("无法读取 Zotero：%1").arg(result.error));
                }
                watcher->deleteLater();
            });
    const std::filesystem::path directory = dataDirectory_.toStdString();
    watcher->setFuture(QtConcurrent::run([directory] {
        CatalogLoadResult result;
        try {
            result.papers = readZoteroSqliteLibrary(directory);
        } catch (const std::exception& error) {
            result.error = QString::fromUtf8(error.what());
        }
        return result;
    }));
}

} // namespace scholarvault::ui
