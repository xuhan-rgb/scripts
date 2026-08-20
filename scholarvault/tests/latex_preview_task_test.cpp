#include "ui/latex_preview_task.hpp"

#include <QFile>
#include <QFileInfo>
#include <QSignalSpy>
#include <QStandardPaths>
#include <QTemporaryDir>
#include <QTest>
#include <QUuid>

using scholarvault::ui::LatexPreviewTask;

class LatexPreviewTaskTest final : public QObject {
    Q_OBJECT

private slots:
    void compilesAndReusesCache();
};

void LatexPreviewTaskTest::compilesAndReusesCache() {
    if (QStandardPaths::findExecutable("xelatex").isEmpty()) {
        QSKIP("xelatex is not installed");
    }

    QTemporaryDir sourceDirectory;
    QVERIFY(sourceDirectory.isValid());
    const QString sourcePath = sourceDirectory.filePath("paper.tex");
    QFile source(sourcePath);
    QVERIFY(source.open(QIODevice::WriteOnly | QIODevice::Text));
    source.write(
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "ScholarVault lazy TeX preview.\n"
        "\\end{document}\n");
    source.close();

    const QString projectId = QUuid::createUuid().toString(QUuid::WithoutBraces);
    LatexPreviewTask first(sourcePath, projectId);
    QSignalSpy firstReady(&first, &LatexPreviewTask::ready);
    QSignalSpy firstFailed(&first, &LatexPreviewTask::failed);
    first.start();
    QVERIFY2(firstReady.wait(30000),
             firstFailed.isEmpty()
                 ? "TeX preview timed out"
                 : qPrintable(firstFailed.constFirst().constFirst().toString()));
    QCOMPARE(firstFailed.count(), 0);
    const QString pdfPath = firstReady.constFirst().constFirst().toString();
    QVERIFY(QFileInfo(pdfPath).isFile());

    const QByteArray originalPath = qgetenv("PATH");
    qputenv("PATH", QByteArray());
    LatexPreviewTask cached(sourcePath, projectId);
    QSignalSpy cachedReady(&cached, &LatexPreviewTask::ready);
    QSignalSpy cachedFailed(&cached, &LatexPreviewTask::failed);
    cached.start();
    const bool reused = cachedReady.wait(5000);
    qputenv("PATH", originalPath);

    QVERIFY2(reused, "cached TeX preview was not reused without xelatex on PATH");
    QCOMPARE(cachedFailed.count(), 0);
    QCOMPARE(cachedReady.constFirst().constFirst().toString(), pdfPath);
}

QTEST_GUILESS_MAIN(LatexPreviewTaskTest)

#include "latex_preview_task_test.moc"
