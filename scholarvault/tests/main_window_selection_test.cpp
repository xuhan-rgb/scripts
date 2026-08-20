#include "scholarvault/storage.hpp"
#include "ui/library_model.hpp"
#include "ui/main_window.hpp"
#include "ui/reader.hpp"

#include <QItemSelectionModel>
#include <QPainter>
#include <QPdfWriter>
#include <QSettings>
#include <QSplitter>
#include <QTemporaryDir>
#include <QTest>
#include <QToolBar>
#include <QTreeView>

using scholarvault::VaultStorage;
using scholarvault::ui::LibraryModel;
using scholarvault::ui::MainWindow;
using scholarvault::ui::ProjectReader;

namespace {

void writePdf(const QString& path, const QString& label) {
    QPdfWriter writer(path);
    QPainter painter(&writer);
    painter.drawText(QPointF(100, 100), label);
}

} // namespace

class MainWindowSelectionTest final : public QObject {
    Q_OBJECT

private slots:
    void currentSelectionAlwaysOpensProject();
};

void MainWindowSelectionTest::currentSelectionAlwaysOpensProject() {
    QTemporaryDir temporary;
    QVERIFY(temporary.isValid());
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope,
                       temporary.filePath("settings"));
    QSettings::setPath(QSettings::NativeFormat, QSettings::UserScope,
                       temporary.filePath("settings"));

    const QString vaultPath = temporary.filePath("Vault");
    auto storage = std::make_shared<VaultStorage>(vaultPath.toStdString());
    storage->initialize("Test Vault");
    const auto topic = storage->createTopic("Topic");
    const QString firstPdf = temporary.filePath("first.pdf");
    const QString secondPdf = temporary.filePath("second.pdf");
    writePdf(firstPdf, "First");
    writePdf(secondPdf, "Second");
    const auto first = storage->createProjectFromPdf(
        topic.path, firstPdf.toStdString(), "First paper");
    const auto second = storage->createProjectFromPdf(
        topic.path, secondPdf.toStdString(), "Second paper");
    QSettings("ScholarVault", "ScholarVault").setValue("vaultPath", vaultPath);

    MainWindow window;
    auto* tree = window.findChild<QTreeView*>("libraryTree");
    auto* reader = window.findChild<ProjectReader*>();
    QVERIFY(tree != nullptr);
    QVERIFY(reader != nullptr);
    auto* model = qobject_cast<LibraryModel*>(tree->model());
    QVERIFY(model != nullptr);

    const QModelIndex topicIndex = model->index(0, 0);
    QVERIFY(topicIndex.isValid());
    if (model->canFetchMore(topicIndex)) model->fetchMore(topicIndex);

    QModelIndex firstIndex;
    QModelIndex secondIndex;
    for (int row = 0; row < model->rowCount(topicIndex); ++row) {
        const QModelIndex index = model->index(row, 0, topicIndex);
        const QString id = index.data(LibraryModel::IdentifierRole).toString();
        if (id == QString::fromStdString(first.id)) firstIndex = index;
        if (id == QString::fromStdString(second.id)) secondIndex = index;
    }
    QVERIFY(firstIndex.isValid());
    QVERIFY(secondIndex.isValid());

    QStringList toolbarTexts;
    for (auto* toolbar : window.findChildren<QToolBar*>()) {
        for (const QAction* action : toolbar->actions()) {
            if (!action->isSeparator()) toolbarTexts.push_back(action->text());
        }
    }
    QVERIFY(toolbarTexts.contains(QString::fromUtf8("选择 Vault")));
    QVERIFY(toolbarTexts.contains(QString::fromUtf8("同步 Zotero")));
    QVERIFY(!toolbarTexts.contains(QString::fromUtf8("新建话题")));
    QVERIFY(!toolbarTexts.contains(QString::fromUtf8("刷新目录")));
    QVERIFY(window.libraryContextActionTexts(topicIndex).isEmpty());
    QCOMPARE(window.libraryContextActionTexts(firstIndex),
             QStringList{QString::fromUtf8("发送 PDF 给 ChatGPT")});

    tree->selectionModel()->setCurrentIndex(
        firstIndex, QItemSelectionModel::ClearAndSelect | QItemSelectionModel::Rows);
    QTRY_VERIFY(reader->currentProject().has_value());
    QCOMPARE(QString::fromStdString(reader->currentProject()->id),
             QString::fromStdString(first.id));

    tree->selectionModel()->setCurrentIndex(
        secondIndex, QItemSelectionModel::ClearAndSelect | QItemSelectionModel::Rows);
    QTRY_COMPARE(QString::fromStdString(reader->currentProject()->id),
                 QString::fromStdString(second.id));

    tree->setExpanded(topicIndex, true);
    window.resize(1580, 980);
    window.show();
    QTest::qWait(20);
    window.splitter_->setSizes({280, 800, 500});
    const QList<int> savedSplitterSizes = window.splitter_->sizes();
    window.saveWorkspaceState(true);

    MainWindow restored;
    restored.show();
    QTest::qWait(20);
    auto* restoredTree = restored.findChild<QTreeView*>("libraryTree");
    auto* restoredReader = restored.findChild<ProjectReader*>();
    QVERIFY(restoredTree != nullptr);
    QVERIFY(restoredReader != nullptr);
    QTRY_VERIFY(restoredReader->currentProject().has_value());
    QCOMPARE(QString::fromStdString(restoredReader->currentProject()->id),
             QString::fromStdString(second.id));
    QCOMPARE(restored.splitter_->sizes(), savedSplitterSizes);
}

QTEST_MAIN(MainWindowSelectionTest)

#include "main_window_selection_test.moc"
