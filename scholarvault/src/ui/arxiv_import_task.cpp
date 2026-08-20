#include "ui/arxiv_import_task.hpp"

#include "scholarvault/archives.hpp"

#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QProcess>
#include <QUrlQuery>
#include <QXmlStreamReader>

namespace scholarvault::ui {
namespace {

constexpr qint64 MaxDownloadBytes = 512LL * 1024LL * 1024LL;

QString networkError(QNetworkReply* reply, const QString& stage) {
    return QObject::tr("%1失败：%2").arg(stage, reply->errorString());
}

} // namespace

ArxivImportTask::ArxivImportTask(std::shared_ptr<VaultStorage> storage,
                                 std::filesystem::path topicPath,
                                 ArxivReference reference,
                                 QString requestedTitle,
                                 QObject* parent)
    : QObject(parent),
      storage_(std::move(storage)),
      topicPath_(std::move(topicPath)),
      reference_(std::move(reference)),
      title_(std::move(requestedTitle)),
      temporary_("scholarvault-arxiv-XXXXXX") {
    temporary_.setAutoRemove(true);
}

ArxivImportTask::ArxivImportTask(std::shared_ptr<VaultStorage> storage,
                                 Project project,
                                 ArxivReference reference,
                                 QObject* parent)
    : QObject(parent),
      storage_(std::move(storage)),
      reference_(std::move(reference)),
      existingProject_(std::move(project)),
      temporary_("scholarvault-arxiv-XXXXXX") {
    temporary_.setAutoRemove(true);
}

void ArxivImportTask::start() {
    if (!temporary_.isValid()) {
        fail(tr("无法创建 arXiv 临时下载目录"));
        return;
    }
    if (existingProject_) {
        requestFile(QUrl(QString::fromStdString(reference_.sourceUrl())),
                    temporary_.filePath("source.original"), Stage::Source);
    } else if (title_.isEmpty()) requestMetadata();
    else requestFile(QUrl(QString::fromStdString(reference_.pdfUrl())),
                     temporary_.filePath("paper.pdf"), Stage::Pdf);
}

void ArxivImportTask::requestMetadata() {
    stage_ = Stage::Metadata;
    emit progress(3, tr("正在读取 arXiv 元数据"));
    QUrl url("https://export.arxiv.org/api/query");
    QUrlQuery query;
    query.addQueryItem("id_list", QString::fromStdString(reference_.id));
    url.setQuery(query);
    QNetworkRequest request(url);
    request.setAttribute(QNetworkRequest::RedirectPolicyAttribute,
                         QNetworkRequest::NoLessSafeRedirectPolicy);
    request.setHeader(QNetworkRequest::UserAgentHeader, "ScholarVault/0.1");
    auto* reply = network_.get(request);
    connect(reply, &QNetworkReply::finished, this,
            [this, reply] { handleMetadataFinished(reply); });
}

void ArxivImportTask::requestFile(const QUrl& url, const QString& destination, Stage stage) {
    stage_ = stage;
    output_ = std::make_unique<QFile>(destination);
    if (!output_->open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        fail(tr("无法创建临时文件：%1").arg(destination));
        return;
    }
    bytesWritten_ = 0;
    QNetworkRequest request(url);
    request.setAttribute(QNetworkRequest::RedirectPolicyAttribute,
                         QNetworkRequest::NoLessSafeRedirectPolicy);
    request.setHeader(QNetworkRequest::UserAgentHeader, "ScholarVault/0.1");
    auto* reply = network_.get(request);
    connect(reply, &QNetworkReply::readyRead, this, [this, reply] {
        const QByteArray bytes = reply->readAll();
        bytesWritten_ += bytes.size();
        if (bytesWritten_ > MaxDownloadBytes) {
            reply->abort();
            return;
        }
        if (output_) output_->write(bytes);
    });
    connect(reply, &QNetworkReply::downloadProgress, this,
            [this, stage](qint64 received, qint64 total) {
                const int stageStart = stage == Stage::Pdf ? 10 : 50;
                const int stageWidth = stage == Stage::Pdf ? 35 : 25;
                const int amount = total > 0
                    ? stageStart + static_cast<int>(stageWidth * received / total)
                    : stageStart;
                emit progress(amount, stage == Stage::Pdf
                                          ? tr("正在下载论文 PDF")
                                          : tr("正在下载 arXiv 原始源码"));
            });
    connect(reply, &QNetworkReply::finished, this,
            [this, reply] { handleFileFinished(reply); });
}

void ArxivImportTask::handleMetadataFinished(QNetworkReply* reply) {
    const QByteArray body = reply->readAll();
    const bool successful = reply->error() == QNetworkReply::NoError;
    reply->deleteLater();
    if (successful) {
        QXmlStreamReader xml(body);
        bool insideEntry = false;
        while (!xml.atEnd()) {
            xml.readNext();
            if (xml.isStartElement() && xml.name() == u"entry") insideEntry = true;
            else if (insideEntry && xml.isStartElement() && xml.name() == u"title") {
                title_ = xml.readElementText().simplified();
                break;
            }
        }
    }
    if (title_.isEmpty()) title_ = QString::fromStdString(reference_.id);
    requestFile(QUrl(QString::fromStdString(reference_.pdfUrl())),
                temporary_.filePath("paper.pdf"), Stage::Pdf);
}

void ArxivImportTask::handleFileFinished(QNetworkReply* reply) {
    if (output_) {
        const QByteArray remaining = reply->readAll();
        bytesWritten_ += remaining.size();
        output_->write(remaining);
        output_->close();
        output_.reset();
    }
    const bool tooLarge = bytesWritten_ > MaxDownloadBytes;
    const bool successful = reply->error() == QNetworkReply::NoError && !tooLarge;
    const Stage completedStage = stage_;
    const QString error = tooLarge ? tr("下载超过 512 MB 安全上限")
                                   : networkError(reply, completedStage == Stage::Pdf
                                                              ? tr("PDF 下载")
                                                              : tr("源码下载"));
    reply->deleteLater();
    if (!successful) {
        fail(error);
        return;
    }
    if (completedStage == Stage::Pdf) {
        finishProject("not-requested");
    } else {
        prepareSource();
    }
}

void ArxivImportTask::prepareSource() {
    QFile source(temporary_.filePath("source.original"));
    if (!source.open(QIODevice::ReadOnly)) {
        fail(tr("无法读取已下载的 arXiv 源码文件"));
        return;
    }
    const QByteArray head = source.peek(2048);
    if (head.startsWith("%PDF-")) {
        finishProject("no-source");
        return;
    }
    const bool looksLikeTex = head.contains("\\documentclass") ||
                              head.contains("\\begin{document}");
    if (looksLikeTex) {
        QDir().mkpath(temporary_.filePath("extracted"));
        QFile::copy(temporary_.filePath("source.original"),
                    temporary_.filePath("extracted/main.tex"));
        finishProject("ready");
        return;
    }
    listArchive();
}

void ArxivImportTask::listArchive() {
    stage_ = Stage::ListingArchive;
    emit progress(78, tr("正在检查源码压缩包"));
    archiveProcess_ = new QProcess(this);
    archiveProcess_->setProcessChannelMode(QProcess::MergedChannels);
    connect(archiveProcess_, qOverload<int, QProcess::ExitStatus>(&QProcess::finished),
            this, [this](int exitCode, QProcess::ExitStatus status) {
                const QByteArray listing = archiveProcess_->readAll();
                archiveProcess_->deleteLater();
                archiveProcess_ = nullptr;
                if (status != QProcess::NormalExit || exitCode != 0 ||
                    !archivePathsAreSafe(listing)) {
                    fail(tr("arXiv 源码压缩包无效或包含不安全路径"));
                    return;
                }
                extractArchive();
            });
    archiveProcess_->start("tar", {"-tf", temporary_.filePath("source.original")});
}

bool ArxivImportTask::archivePathsAreSafe(const QByteArray& listing) const {
    const QList<QByteArray> paths = listing.split('\n');
    std::vector<std::string> entries;
    for (QByteArray path : paths) {
        path = path.trimmed();
        if (path.isEmpty()) continue;
        entries.push_back(path.toStdString());
    }
    return archiveListingIsSafe(entries);
}

void ArxivImportTask::extractArchive() {
    stage_ = Stage::ExtractingArchive;
    emit progress(84, tr("正在解压原始 TeX"));
    QDir().mkpath(temporary_.filePath("extracted"));
    archiveProcess_ = new QProcess(this);
    archiveProcess_->setProcessChannelMode(QProcess::MergedChannels);
    connect(archiveProcess_, qOverload<int, QProcess::ExitStatus>(&QProcess::finished),
            this, [this](int exitCode, QProcess::ExitStatus status) {
                const bool ok = status == QProcess::NormalExit && exitCode == 0;
                archiveProcess_->deleteLater();
                archiveProcess_ = nullptr;
                if (ok) {
                    QDirIterator iterator(temporary_.filePath("extracted"),
                                          QDir::AllEntries | QDir::NoDotAndDotDot,
                                          QDirIterator::Subdirectories);
                    while (iterator.hasNext()) {
                        const QFileInfo info(iterator.next());
                        if (info.isSymLink()) {
                            QFile::remove(info.absoluteFilePath());
                        }
                    }
                }
                if (ok) finishProject("ready");
                else fail(tr("无法解压 arXiv 源码压缩包"));
            });
    archiveProcess_->start("tar", {"-xf", temporary_.filePath("source.original"),
                                    "-C", temporary_.filePath("extracted"),
                                    "--no-same-owner", "--no-same-permissions"});
}

void ArxivImportTask::finishProject(const QString& sourceStatus) {
    emit progress(94, tr("正在写入论文项目"));
    try {
        const std::filesystem::path archive = sourceStatus == "ready"
            ? std::filesystem::path(temporary_.filePath("source.original").toStdString())
            : std::filesystem::path{};
        const std::filesystem::path extracted = sourceStatus == "ready"
            ? std::filesystem::path(temporary_.filePath("extracted").toStdString())
            : std::filesystem::path{};
        Project project;
        if (existingProject_) {
            project = storage_->installArxivSource(
                *existingProject_, reference_, archive, extracted,
                sourceStatus.toStdString());
            emit progress(100, sourceStatus == "ready"
                                   ? tr("arXiv 原始 TeX 已下载")
                                   : tr("该 arXiv 提交没有可用 TeX 源码"));
        } else {
            project = storage_->createProjectFromArxivFiles(
                topicPath_, reference_, temporary_.filePath("paper.pdf").toStdString(),
                archive, extracted, title_.toStdString(), sourceStatus.toStdString());
            emit progress(100, tr("arXiv 项目创建完成"));
        }
        emit succeeded(project);
    } catch (const std::exception& error) {
        fail(QString::fromUtf8(error.what()));
    }
}

void ArxivImportTask::fail(const QString& message) {
    if (output_) {
        output_->close();
        output_.reset();
    }
    emit failed(message);
}

} // namespace scholarvault::ui
