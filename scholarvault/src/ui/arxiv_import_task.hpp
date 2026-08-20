#pragma once

#include "scholarvault/arxiv.hpp"
#include "scholarvault/storage.hpp"

#include <QNetworkAccessManager>
#include <QObject>
#include <QTemporaryDir>

#include <memory>

class QFile;
class QNetworkReply;
class QProcess;

namespace scholarvault::ui {

class ArxivImportTask final : public QObject {
    Q_OBJECT

public:
    ArxivImportTask(std::shared_ptr<VaultStorage> storage,
                    std::filesystem::path topicPath,
                    ArxivReference reference,
                    QString requestedTitle,
                    QObject* parent = nullptr);
    ArxivImportTask(std::shared_ptr<VaultStorage> storage,
                    Project project,
                    ArxivReference reference,
                    QObject* parent = nullptr);

    void start();

signals:
    void progress(int value, const QString& status);
    void succeeded(const scholarvault::Project& project);
    void failed(const QString& error);

private:
    enum class Stage { Metadata, Pdf, Source, ListingArchive, ExtractingArchive };

    void requestMetadata();
    void requestFile(const QUrl& url, const QString& destination, Stage stage);
    void handleMetadataFinished(QNetworkReply* reply);
    void handleFileFinished(QNetworkReply* reply);
    void prepareSource();
    void listArchive();
    void extractArchive();
    void finishProject(const QString& sourceStatus);
    void fail(const QString& message);
    [[nodiscard]] bool archivePathsAreSafe(const QByteArray& listing) const;

    std::shared_ptr<VaultStorage> storage_;
    std::filesystem::path topicPath_;
    ArxivReference reference_;
    QString title_;
    std::optional<Project> existingProject_;
    QNetworkAccessManager network_;
    QTemporaryDir temporary_;
    Stage stage_{Stage::Metadata};
    std::unique_ptr<QFile> output_;
    QProcess* archiveProcess_{nullptr};
    qint64 bytesWritten_{0};
};

} // namespace scholarvault::ui
