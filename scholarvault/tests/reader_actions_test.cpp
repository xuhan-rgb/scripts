#include "ui/reader.hpp"

#include <QSignalSpy>
#include <QPainter>
#include <QPdfDocument>
#include <QPdfWriter>
#include <QTemporaryDir>
#include <QTest>
#include <QToolButton>

using scholarvault::Project;
using scholarvault::ui::ProjectReader;

class ReaderActionsTest final : public QObject {
    Q_OBJECT

private slots:
    void downloadActionsFollowCurrentProject();
    void switchingProjectsReplacesPdfDocument();
};

namespace {

void writePdf(const QString& path, int pages) {
    QPdfWriter writer(path);
    QPainter painter(&writer);
    for (int page = 0; page < pages; ++page) {
        if (page > 0) writer.newPage();
        painter.drawText(QPointF(100, 100), QString("Page %1").arg(page + 1));
    }
}

} // namespace

void ReaderActionsTest::downloadActionsFollowCurrentProject() {
    QTemporaryDir temporary;
    QVERIFY(temporary.isValid());

    ProjectReader reader;
    auto* latex = reader.findChild<QToolButton*>("downloadLatexButton");
    auto* github = reader.findChild<QToolButton*>("downloadGitHubButton");
    QVERIFY(latex != nullptr);
    QVERIFY(github != nullptr);
    QVERIFY(!latex->isEnabled());
    QVERIFY(!github->isEnabled());

    QSignalSpy latexRequested(&reader, SIGNAL(arxivSourceDownloadRequested()));
    QSignalSpy githubRequested(&reader, SIGNAL(gitHubDownloadRequested()));
    QVERIFY(latexRequested.isValid());
    QVERIFY(githubRequested.isValid());

    Project project;
    project.id = "paper-1";
    project.title = "Paper";
    project.path = temporary.path().toStdString();
    project.sourceStatus = "not-requested";
    reader.openProject(project);
    QVERIFY(latex->isEnabled());
    QVERIFY(github->isEnabled());
    latex->click();
    github->click();
    QCOMPARE(latexRequested.count(), 1);
    QCOMPARE(githubRequested.count(), 1);

    project.sourceStatus = "ready";
    reader.refreshProject(project);
    QVERIFY(!latex->isEnabled());
    QVERIFY(latex->text().contains(QString::fromUtf8("已下载")));
    QVERIFY(github->isEnabled());

    reader.clearProject();
    QVERIFY(!latex->isEnabled());
    QVERIFY(!github->isEnabled());
}

void ReaderActionsTest::switchingProjectsReplacesPdfDocument() {
    QTemporaryDir temporary;
    QVERIFY(temporary.isValid());
    writePdf(temporary.filePath("first.pdf"), 1);
    writePdf(temporary.filePath("second.pdf"), 2);

    Project first;
    first.id = "first";
    first.title = "First";
    first.path = temporary.path().toStdString();
    first.pdfRelativePath = "first.pdf";
    Project second = first;
    second.id = "second";
    second.title = "Second";
    second.pdfRelativePath = "second.pdf";

    ProjectReader reader;
    auto* document = reader.findChild<QPdfDocument*>();
    QVERIFY(document != nullptr);
    reader.openProject(first);
    QTRY_COMPARE(document->status(), QPdfDocument::Status::Ready);
    QCOMPARE(document->pageCount(), 1);

    reader.openProject(second);
    QTRY_COMPARE(document->status(), QPdfDocument::Status::Ready);
    QCOMPARE(document->pageCount(), 2);
}

QTEST_MAIN(ReaderActionsTest)

#include "reader_actions_test.moc"
