#include "scholarvault/storage.hpp"
#include "ui/arxiv_import_task.hpp"

#include <QEventLoop>
#include <QTemporaryDir>
#include <QTest>
#include <QTimer>

#include <filesystem>
#include <fstream>

using namespace scholarvault;
using namespace scholarvault::ui;

class ArxivImportTaskTest final : public QObject {
    Q_OBJECT

private slots:
    void downloadsKnownSourceWhenRequested();
};

void ArxivImportTaskTest::downloadsKnownSourceWhenRequested() {
    if (!qEnvironmentVariableIsSet("SCHOLARVAULT_TEST_LIVE_ARXIV")) {
        QSKIP("set SCHOLARVAULT_TEST_LIVE_ARXIV=1 for the live arXiv integration test");
    }
    QTemporaryDir temporary;
    QVERIFY(temporary.isValid());
    auto storage = std::make_shared<VaultStorage>(
        std::filesystem::path(temporary.path().toStdString()) / "Vault");
    storage->initialize("arXiv test");
    const Topic topic = storage->createTopic("Papers");
    const std::filesystem::path pdf =
        std::filesystem::path(temporary.path().toStdString()) / "paper.pdf";
    std::ofstream(pdf, std::ios::binary) << "%PDF-1.7\n";
    const Project project = storage->createProjectFromPdf(topic.path, pdf, "Paper");
    const auto reference = parseArxivReference("1906.08332");
    QVERIFY(reference.has_value());

    QEventLoop loop;
    QTimer timeout;
    timeout.setSingleShot(true);
    timeout.setInterval(60000);
    QString failure;
    QString status;
    ArxivImportTask task(storage, project, *reference);
    connect(&task, &ArxivImportTask::succeeded, &loop,
            [&loop, &status](const Project& updated) {
                status = QString::fromStdString(updated.sourceStatus);
                loop.quit();
            });
    connect(&task, &ArxivImportTask::failed, &loop,
            [&loop, &failure](const QString& error) {
                failure = error;
                loop.quit();
            });
    connect(&timeout, &QTimer::timeout, &loop, &QEventLoop::quit);
    task.start();
    timeout.start();
    loop.exec();

    QVERIFY2(timeout.isActive(), "arXiv source download timed out");
    QVERIFY2(failure.isEmpty(), qPrintable(failure));
    QCOMPARE(status, QString("ready"));
    QVERIFY(std::filesystem::is_regular_file(project.path / "source" /
                                             "extracted" / "bare_jrnl.tex"));
}

QTEST_MAIN(ArxivImportTaskTest)

#include "arxiv_import_task_test.moc"
