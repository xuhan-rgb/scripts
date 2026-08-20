#pragma once

#include "scholarvault/zotero.hpp"

#include <QJsonArray>
#include <QObject>

#include <vector>

class QTcpSocket;
class QTimer;

namespace scholarvault::ui {

[[nodiscard]] std::vector<ZoteroPaper> parseZoteroApiItems(
    const QJsonArray& items, const QString& dataDirectory,
    const QJsonArray& collections = {});

class ZoteroCatalogLoader final : public QObject {
    Q_OBJECT

public:
    explicit ZoteroCatalogLoader(QString dataDirectory, QObject* parent = nullptr,
                                 quint16 apiPort = 23119);
    void start();

signals:
    void progress(const QString& status);
    void loaded(const std::vector<ZoteroPaper>& papers, const QString& source);
    void failed(const QString& error);

private:
    void requestApiPage(int start);
    void finishApiPage();
    void loadSqliteFallback(const QString& reason);

    QString dataDirectory_;
    QTcpSocket* apiSocket_{nullptr};
    QTimer* apiTimeout_;
    QByteArray apiResponse_;
    int apiStart_{0};
    quint16 apiPort_{23119};
    QString apiError_;
    QJsonArray apiItems_;
    QJsonArray apiCollections_;
    bool loadingCollections_{false};
    bool fallbackStarted_{false};
};

} // namespace scholarvault::ui
