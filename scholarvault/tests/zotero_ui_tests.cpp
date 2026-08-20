#include "scholarvault/storage.hpp"
#include "ui/project_dialogs.hpp"
#include "ui/zotero_catalog_loader.hpp"
#include "ui/zotero_import_dialog.hpp"

#include <QComboBox>
#include <QFile>
#include <QEventLoop>
#include <QHostAddress>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLineEdit>
#include <QLabel>
#include <QPushButton>
#include <QTableWidget>
#include <QTcpServer>
#include <QTcpSocket>
#include <QTemporaryDir>
#include <QTest>
#include <QTimer>
#include <QUrl>

#include <algorithm>
#include <filesystem>

using namespace scholarvault;
using namespace scholarvault::ui;

class ZoteroUiTests final : public QObject {
    Q_OBJECT

private slots:
    void parsesLocalApiItems();
    void acceptsPastedPdfPathButRejectsDirectory();
    void liveLibraryLoadsWhenRequested();
    void readsZoteroHttp10Response();
    void liveFullLibrarySyncsToTemporaryVault();
};

void ZoteroUiTests::parsesLocalApiItems() {
    QTemporaryDir temporary;
    QVERIFY(temporary.isValid());
    const QString pdfPath = temporary.filePath("paper.pdf");
    QFile pdf(pdfPath);
    QVERIFY(pdf.open(QIODevice::WriteOnly));
    pdf.write("%PDF-1.7\n");
    pdf.close();

    QJsonArray creators{
        QJsonObject{{"creatorType", "author"}, {"firstName", "Ada"},
                    {"lastName", "Lovelace"}},
        QJsonObject{{"creatorType", "author"}, {"name", "Research Group"}}
    };
    QJsonObject parent{
        {"key", "PARENT01"},
        {"data", QJsonObject{{"itemType", "journalArticle"},
                              {"title", "API Paper"},
                              {"date", "2026-08-13"},
                              {"extra", "Citation Key: api2026\narXiv: 2504.16054"},
                              {"collections", QJsonArray{"COLLCHILD"}},
                              {"creators", creators}}}
    };
    QJsonObject attachment{
        {"key", "ATTACH01"},
        {"data", QJsonObject{{"itemType", "attachment"},
                              {"parentItem", "PARENT01"},
                              {"contentType", "application/pdf"}}},
        {"links", QJsonObject{{"enclosure", QJsonObject{
            {"href", QUrl::fromLocalFile(pdfPath).toString()}}}}}
    };

    const QJsonArray collections{
        QJsonObject{{"key", "COLLROOT"},
                    {"data", QJsonObject{{"name", "Engineering"},
                                         {"parentCollection", false}}}},
        QJsonObject{{"key", "COLLCHILD"},
                    {"data", QJsonObject{{"name", "World Models"},
                                         {"parentCollection", "COLLROOT"}}}}
    };
    const auto papers = parseZoteroApiItems({parent, attachment}, temporary.path(),
                                            collections);
    QCOMPARE(papers.size(), static_cast<std::size_t>(1));
    QCOMPARE(QString::fromStdString(papers.front().title), QString("API Paper"));
    QCOMPARE(QString::fromStdString(papers.front().authors),
             QString("Ada Lovelace, Research Group"));
    QCOMPARE(QString::fromStdString(papers.front().year), QString("2026"));
    QCOMPARE(QString::fromStdString(papers.front().arxivId), QString("2504.16054"));
    QCOMPARE(QString::fromStdString(papers.front().itemKey), QString("PARENT01"));
    QCOMPARE(QString::fromStdString(papers.front().pdfPath.string()), pdfPath);
    QCOMPARE(papers.front().collectionPath.size(), static_cast<std::size_t>(2));
    QCOMPARE(QString::fromStdString(papers.front().collectionPath[0].name),
             QString("Engineering"));
    QCOMPARE(QString::fromStdString(papers.front().collectionPath[1].name),
             QString("World Models"));
}

void ZoteroUiTests::acceptsPastedPdfPathButRejectsDirectory() {
    QTemporaryDir temporary;
    QVERIFY(temporary.isValid());
    const QString pdfPath = temporary.filePath("pasted.pdf");
    QFile pdf(pdfPath);
    QVERIFY(pdf.open(QIODevice::WriteOnly));
    pdf.write("%PDF-1.7\n");
    pdf.close();

    NewProjectDialog dialog;
    auto* source = dialog.findChild<QComboBox*>();
    auto* input = dialog.findChild<QLineEdit*>("pdfPathInput");
    auto* create = dialog.findChild<QPushButton*>("createProjectButton");
    QVERIFY(source != nullptr);
    QVERIFY(input != nullptr);
    QVERIFY(create != nullptr);
    QCOMPARE(source->count(), 3);
    QVERIFY(!input->isReadOnly());

    source->setCurrentIndex(1);
    input->setText(temporary.path());
    QVERIFY(!create->isEnabled());
    input->setText(pdfPath);
    QVERIFY(create->isEnabled());
    QCOMPARE(dialog.request().input, pdfPath);
}

void ZoteroUiTests::liveLibraryLoadsWhenRequested() {
    if (!qEnvironmentVariableIsSet("SCHOLARVAULT_TEST_LIVE_ZOTERO")) {
        QSKIP("set SCHOLARVAULT_TEST_LIVE_ZOTERO=1 for the local integration test");
    }
    ZoteroImportDialog dialog("/home/qwer/Zotero");
    auto* table = dialog.findChild<QTableWidget*>("zoteroPaperTable");
    QVERIFY(table != nullptr);
    QTRY_VERIFY_WITH_TIMEOUT(table->rowCount() > 0, 12000);
}

void ZoteroUiTests::readsZoteroHttp10Response() {
    QTemporaryDir temporary;
    QVERIFY(temporary.isValid());
    const QString pdfPath = temporary.filePath("server-paper.pdf");
    QFile pdf(pdfPath);
    QVERIFY(pdf.open(QIODevice::WriteOnly));
    pdf.write("%PDF-1.7\n");
    pdf.close();

    const QJsonArray items{
        QJsonObject{{"key", "PARENT02"},
                    {"data", QJsonObject{{"itemType", "journalArticle"},
                                         {"title", "HTTP 1.0 Paper"},
                                         {"date", "2026"},
                                         {"creators", QJsonArray{}}}}},
        QJsonObject{{"key", "ATTACH02"},
                    {"data", QJsonObject{{"itemType", "attachment"},
                                         {"parentItem", "PARENT02"},
                                         {"contentType", "application/pdf"}}},
                    {"links", QJsonObject{{"enclosure", QJsonObject{
                        {"href", QUrl::fromLocalFile(pdfPath).toString()}}}}}}
    };
    const QByteArray body = QJsonDocument(items).toJson(QJsonDocument::Compact);

    QTcpServer server;
    QVERIFY(server.listen(QHostAddress::LocalHost, 0));
    connect(&server, &QTcpServer::newConnection, &server, [&server, body] {
        auto* socket = server.nextPendingConnection();
        QObject::connect(socket, &QTcpSocket::readyRead, socket, [socket, body] {
            socket->readAll();
            socket->write("HTTP/1.0 200 OK\r\nTotal-Results: 2\r\nContent-Type: "
                          "application/json\r\nContent-Length: " +
                          QByteArray::number(body.size()) + "\r\n\r\n" + body);
            socket->disconnectFromHost();
        });
    });

    ZoteroImportDialog dialog(temporary.path(), nullptr, server.serverPort());
    auto* table = dialog.findChild<QTableWidget*>("zoteroPaperTable");
    QVERIFY(table != nullptr);
    QTRY_COMPARE_WITH_TIMEOUT(table->rowCount(), 1, 3000);
    QCOMPARE(table->item(0, 0)->text(), QString("HTTP 1.0 Paper"));
}

void ZoteroUiTests::liveFullLibrarySyncsToTemporaryVault() {
    if (!qEnvironmentVariableIsSet("SCHOLARVAULT_TEST_LIVE_ZOTERO_FULL")) {
        QSKIP("set SCHOLARVAULT_TEST_LIVE_ZOTERO_FULL=1 for the full local import test");
    }
    QTemporaryDir temporary;
    QVERIFY(temporary.isValid());
    std::vector<ZoteroPaper> papers;
    QString error;
    QEventLoop loop;
    QTimer timeout;
    timeout.setSingleShot(true);
    timeout.setInterval(15000);
    ZoteroCatalogLoader loader("/home/qwer/Zotero");
    connect(&loader, &ZoteroCatalogLoader::loaded, &loop,
            [&papers, &loop](const std::vector<ZoteroPaper>& loaded, const QString&) {
                papers = loaded;
                loop.quit();
            });
    connect(&loader, &ZoteroCatalogLoader::failed, &loop,
            [&error, &loop](const QString& message) {
                error = message;
                loop.quit();
            });
    connect(&timeout, &QTimer::timeout, &loop, &QEventLoop::quit);
    loader.start();
    timeout.start();
    loop.exec();
    QVERIFY2(error.isEmpty(), qPrintable(error));
    QVERIFY(papers.size() >= 80);

    VaultStorage storage(std::filesystem::path(temporary.path().toStdString()) / "Vault");
    storage.initialize("Live Zotero Test");
    const auto result = storage.syncZoteroLibrary(papers);
    QCOMPARE(result.created, static_cast<int>(papers.size()));
    QCOMPARE(result.skipped, 0);
    QCOMPARE(storage.discoverAllProjects().size(), papers.size());
    const auto roots = storage.discoverTopics();
    QVERIFY(std::any_of(roots.begin(), roots.end(), [](const Topic& topic) {
        return topic.name == "工程类";
    }));
    for (const auto& project : storage.discoverAllProjects()) {
        QVERIFY(!std::filesystem::is_symlink(project.path / project.pdfRelativePath));
    }
}

QTEST_MAIN(ZoteroUiTests)

#include "zotero_ui_tests.moc"
