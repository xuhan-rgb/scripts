#include "ui/main_window.hpp"

#include "scholarvault/arxiv.hpp"
#include "scholarvault/github.hpp"
#include "ui/arxiv_import_task.hpp"
#include "ui/assistant_panel.hpp"
#include "ui/chatgpt_upload_task.hpp"
#include "ui/github_clone_task.hpp"
#include "ui/library_model.hpp"
#include "ui/latex_preview_task.hpp"
#include "ui/project_dialogs.hpp"
#include "ui/reader.hpp"
#include "ui/task_panel.hpp"
#include "ui/zotero_catalog_loader.hpp"

#include <QAction>
#include <QCloseEvent>
#include <QDockWidget>
#include <QFileDialog>
#include <QFutureWatcher>
#include <QInputDialog>
#include <QItemSelectionModel>
#include <QMessageBox>
#include <QMenu>
#include <QSettings>
#include <QSplitter>
#include <QStandardPaths>
#include <QStatusBar>
#include <QStyle>
#include <QToolBar>
#include <QTreeView>
#include <QTimer>
#include <QtConcurrent>

#include <map>

namespace scholarvault::ui {
namespace {

struct ZoteroGroupSyncResult {
    std::vector<Project> updatedProjects;
    int updated{0};
    int unchanged{0};
    int skipped{0};
    QStringList errors;
};

std::optional<ZoteroPaper> matchZoteroPaper(
    const Project& project, const std::vector<ZoteroPaper>& papers,
    QString* error) {
    if (!project.zotero) {
        *error = QObject::tr("项目没有 Zotero 关联信息");
        return std::nullopt;
    }
    for (const auto& paper : papers) {
        if (paper.attachmentKey == project.zotero->attachmentKey &&
            paper.itemKey == project.zotero->itemKey) {
            return paper;
        }
    }
    std::vector<ZoteroPaper> itemPapers;
    for (const auto& paper : papers) {
        if (paper.itemKey == project.zotero->itemKey) itemPapers.push_back(paper);
    }
    if (itemPapers.size() == 1) return itemPapers.front();
    *error = itemPapers.empty()
        ? QObject::tr("Zotero 条目或本地 PDF 已不存在")
        : QObject::tr("Zotero 条目有多个 PDF，无法确定替换附件");
    return std::nullopt;
}

} // namespace

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle("ScholarVault");
    resize(1580, 980);
    setMinimumSize(1100, 700);

    libraryModel_ = new LibraryModel(this);
    libraryTree_ = new QTreeView(this);
    libraryTree_->setObjectName("libraryTree");
    libraryTree_->setModel(libraryModel_);
    libraryTree_->setHeaderHidden(true);
    libraryTree_->setUniformRowHeights(true);
    libraryTree_->setAnimated(false);
    libraryTree_->setMinimumWidth(230);
    libraryTree_->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(libraryTree_, &QTreeView::customContextMenuRequested, this,
            &MainWindow::showLibraryContextMenu);
    connect(libraryTree_->selectionModel(), &QItemSelectionModel::currentChanged, this,
            [this](const QModelIndex& current) {
                activateIndex(current);
                updateContextActions();
                scheduleWorkspaceSave();
            });
    connect(libraryTree_, &QTreeView::expanded, this,
            [this] { scheduleWorkspaceSave(); });
    connect(libraryTree_, &QTreeView::collapsed, this,
            [this] { scheduleWorkspaceSave(); });

    reader_ = new ProjectReader(this);
    connect(reader_, &ProjectReader::statusMessage, this,
            [this](const QString& message) { statusBar()->showMessage(message, 4000); });
    connect(reader_, &ProjectReader::latexPreviewRequested, this,
            [this](const QString& sourcePath, const QString& projectId) {
                const QString taskId = tasks_->beginTask(
                    tr("渲染 TeX：%1").arg(QFileInfo(sourcePath).fileName()));
                taskDock_->show();
                auto* task = new LatexPreviewTask(sourcePath, projectId, this);
                connect(task, &LatexPreviewTask::progress, this,
                        [this, taskId](int value, const QString& status) {
                            tasks_->updateTask(taskId, value, status);
                        });
                connect(task, &LatexPreviewTask::ready, this,
                        [this, task, taskId, projectId](const QString& pdf,
                                                       const QString& source) {
                            tasks_->finishTask(taskId, tr("TeX 预览已就绪"));
                            const auto current = reader_->currentProject();
                            if (current && QString::fromStdString(current->id) == projectId) {
                                reader_->showLatexPreview(pdf, source);
                            }
                            task->deleteLater();
                        });
                connect(task, &LatexPreviewTask::failed, this,
                        [this, task, taskId](const QString& error) {
                            tasks_->failTask(taskId, error);
                            task->deleteLater();
                        });
                task->start();
            });
    connect(reader_, &ProjectReader::arxivSourceDownloadRequested, this,
            &MainWindow::requestArxivSourceDownload);
    connect(reader_, &ProjectReader::gitHubDownloadRequested, this,
            &MainWindow::addGitHubRepository);
    connect(reader_, &ProjectReader::viewStateChanged, this,
            &MainWindow::scheduleWorkspaceSave);
    assistant_ = new AssistantPanel(this);
    assistant_->setMinimumWidth(390);
    connect(assistant_, &AssistantPanel::analysisRequested, this,
            &MainWindow::prepareAssistantAnalysis);
    connect(assistant_, &AssistantPanel::viewStateChanged, this,
            &MainWindow::scheduleWorkspaceSave);
    connect(assistant_, &AssistantPanel::chatGptReady, this,
            &MainWindow::startPendingChatGptUpload);

    splitter_ = new QSplitter(Qt::Horizontal, this);
    splitter_->setObjectName("workspaceSplitter");
    splitter_->addWidget(libraryTree_);
    splitter_->addWidget(reader_);
    splitter_->addWidget(assistant_);
    splitter_->setStretchFactor(0, 0);
    splitter_->setStretchFactor(1, 1);
    splitter_->setStretchFactor(2, 0);
    splitter_->setSizes({260, 900, 420});
    splitter_->setChildrenCollapsible(false);
    connect(splitter_, &QSplitter::splitterMoved, this,
            [this] { scheduleWorkspaceSave(); });
    setCentralWidget(splitter_);

    tasks_ = new TaskPanel(this);
    taskDock_ = new QDockWidget(tr("后台任务"), this);
    taskDock_->setObjectName("taskDock");
    taskDock_->setAllowedAreas(Qt::BottomDockWidgetArea);
    taskDock_->setWidget(tasks_);
    taskDock_->setMinimumHeight(150);
    addDockWidget(Qt::BottomDockWidgetArea, taskDock_);
    taskDock_->hide();

    auto* toolbar = addToolBar(tr("工作区"));
    toolbar->setObjectName("mainToolbar");
    toolbar->setMovable(false);
    toolbar->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    auto* vaultAction = toolbar->addAction(
        style()->standardIcon(QStyle::SP_DirOpenIcon), tr("选择 Vault"));
    connect(vaultAction, &QAction::triggered, this, &MainWindow::chooseVault);
    syncZoteroAction_ = toolbar->addAction(tr("同步 Zotero"));
    syncZoteroAction_->setObjectName("syncZoteroAction");
    syncZoteroAction_->setEnabled(false);
    connect(syncZoteroAction_, &QAction::triggered, this,
            &MainWindow::syncAllZoteroProjects);
    sendPdfToChatGptAction_ = new QAction(tr("发送 PDF 给 ChatGPT"), this);
    sendPdfToChatGptAction_->setObjectName("sendPdfToChatGptAction");
    connect(sendPdfToChatGptAction_, &QAction::triggered, this,
            &MainWindow::sendCurrentPdfToChatGpt);
    updateContextActions();

    workspaceSaveTimer_ = new QTimer(this);
    workspaceSaveTimer_->setSingleShot(true);
    workspaceSaveTimer_->setInterval(400);
    connect(workspaceSaveTimer_, &QTimer::timeout, this,
            [this] { saveWorkspaceState(); });

    QSettings settings("ScholarVault", "ScholarVault");
    QString vault = settings.value("vaultPath").toString();
    const QString hiddenDefault = QDir(
        QStandardPaths::writableLocation(QStandardPaths::GenericDataLocation))
        .filePath("ScholarVault/Vault");
    const QString legacyDefault = QDir(
        QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation))
        .filePath("ScholarVault");
    if (vault.isEmpty()) {
        vault = hiddenDefault;
    } else if (QDir::cleanPath(vault) == QDir::cleanPath(legacyDefault) &&
               QFileInfo(QDir(legacyDefault).filePath("vault.json")).isFile() &&
               !QFileInfo::exists(hiddenDefault)) {
        try {
            QDir().mkpath(QFileInfo(hiddenDefault).absolutePath());
            std::filesystem::rename(legacyDefault.toStdString(),
                                    hiddenDefault.toStdString());
            vault = hiddenDefault;
            settings.setValue("vaultPath", vault);
        } catch (const std::exception& error) {
            statusBar()->showMessage(
                tr("无法迁移旧默认 Vault，继续使用原目录：%1")
                    .arg(QString::fromUtf8(error.what())), 10000);
        }
    }
    openVault(vault);
    restoreWorkspaceState();
}

void MainWindow::closeEvent(QCloseEvent* event) {
    saveWorkspaceState(true);
    QMainWindow::closeEvent(event);
}

void MainWindow::scheduleWorkspaceSave() {
    if (!restoringWorkspace_ && workspaceSaveTimer_ != nullptr) {
        workspaceSaveTimer_->start();
    }
}

QStringList MainWindow::expandedTopicIds() const {
    QStringList result;
    std::function<void(const QModelIndex&)> collect =
        [this, &result, &collect](const QModelIndex& parent) {
            for (int row = 0; row < libraryModel_->rowCount(parent); ++row) {
                const QModelIndex index = libraryModel_->index(row, 0, parent);
                if (index.data(LibraryModel::NodeTypeRole).toInt() !=
                    static_cast<int>(LibraryModel::NodeType::Topic)) {
                    continue;
                }
                if (libraryTree_->isExpanded(index)) {
                    result.push_back(index.data(LibraryModel::IdentifierRole).toString());
                    collect(index);
                }
            }
        };
    collect({});
    return result;
}

void MainWindow::saveWorkspaceState(bool forceSync) {
    if (restoringWorkspace_) return;
    QSettings settings("ScholarVault", "ScholarVault");
    settings.beginGroup("workspace");
    settings.setValue("geometry", saveGeometry());
    settings.setValue("windowState", saveState());
    QVariantList splitterSizes;
    for (int size : splitter_->sizes()) splitterSizes.push_back(size);
    settings.setValue("splitterSizes", splitterSizes);
    settings.setValue("expandedTopicIds", expandedTopicIds());
    settings.setValue("selectedProjectId",
                      libraryTree_->currentIndex().data(
                          LibraryModel::IdentifierRole).toString());
    const auto readerState = reader_->viewState();
    settings.setValue("readerTab", readerState.tabIndex);
    settings.setValue("pdfZoomMode",
                      static_cast<int>(readerState.pdf.zoomMode));
    settings.setValue("pdfZoomFactor", readerState.pdf.zoomFactor);
    settings.setValue("pdfPage", readerState.pdf.page);
    settings.setValue("pdfPageOffset", readerState.pdf.pageOffset);
    settings.setValue("assistantTab", assistant_->currentTabIndex());
    settings.setValue("chatGptOpen", assistant_->chatGptOpen());
    settings.setValue("codexAttached", assistant_->codexAttached());
    settings.endGroup();
    if (forceSync) settings.sync();
}

void MainWindow::restoreWorkspaceState() {
    restoringWorkspace_ = true;
    QSettings settings("ScholarVault", "ScholarVault");
    settings.beginGroup("workspace");
    const QByteArray geometry = settings.value("geometry").toByteArray();
    if (!geometry.isEmpty()) restoreGeometry(geometry);
    const QByteArray windowState = settings.value("windowState").toByteArray();
    if (!windowState.isEmpty()) restoreState(windowState);
    const QVariantList savedSizes = settings.value("splitterSizes").toList();
    if (savedSizes.size() == 3) {
        splitter_->setSizes({savedSizes[0].toInt(), savedSizes[1].toInt(),
                             savedSizes[2].toInt()});
    }
    for (const QString& id : settings.value("expandedTopicIds").toStringList()) {
        const QModelIndex index = libraryModel_->indexForIdentifier(id);
        if (index.isValid()) libraryTree_->setExpanded(index, true);
    }
    const QString projectId = settings.value("selectedProjectId").toString();
    const QModelIndex projectIndex = libraryModel_->indexForIdentifier(projectId);
    if (projectIndex.isValid() && libraryModel_->projectForIndex(projectIndex)) {
        libraryTree_->selectionModel()->setCurrentIndex(
            projectIndex,
            QItemSelectionModel::ClearAndSelect | QItemSelectionModel::Rows);
        libraryTree_->scrollTo(projectIndex);
        ProjectReader::ViewState readerState;
        readerState.tabIndex = settings.value("readerTab", 0).toInt();
        readerState.pdf.zoomMode = static_cast<PdfDocumentView::ZoomMode>(
            settings.value("pdfZoomMode", 1).toInt());
        readerState.pdf.zoomFactor = settings.value("pdfZoomFactor", 1.0).toDouble();
        readerState.pdf.page = settings.value("pdfPage", 0).toInt();
        readerState.pdf.pageOffset = settings.value("pdfPageOffset", 0.0).toDouble();
        reader_->restoreViewState(readerState);
    }
    assistant_->setCurrentTabIndex(settings.value("assistantTab", 0).toInt());
    if (settings.value("codexAttached", false).toBool() &&
        reader_->currentProject()) {
        assistant_->startCodex();
    }
    settings.endGroup();
    restoringWorkspace_ = false;
}

void MainWindow::chooseVault() {
    const QString current = storage_ ? QString::fromStdString(storage_->rootPath().string())
                                     : QDir::homePath();
    const QString path = QFileDialog::getExistingDirectory(this, tr("选择 ScholarVault 目录"),
                                                            current);
    if (!path.isEmpty()) openVault(path);
}

void MainWindow::openVault(const QString& path) {
    try {
        auto storage = std::make_shared<VaultStorage>(path.toStdString());
        storage->initialize(QFileInfo(path).fileName().toStdString());
        storage_ = std::move(storage);
        libraryModel_->setStorage(storage_);
        reader_->clearProject();
        assistant_->setProjectPath({});
        QSettings("ScholarVault", "ScholarVault").setValue("vaultPath", path);
        statusBar()->showMessage(tr("Vault：%1").arg(path), 5000);
        syncZoteroAction_->setEnabled(true);
        updateContextActions();
    } catch (const std::exception& error) {
        QMessageBox::critical(this, tr("无法打开 Vault"), QString::fromUtf8(error.what()));
    }
}

void MainWindow::createTopic() {
    if (!storage_) return;
    bool accepted = false;
    const QString name = QInputDialog::getText(this, tr("新建话题"), tr("话题名称"),
                                               QLineEdit::Normal, {}, &accepted).trimmed();
    if (!accepted || name.isEmpty()) return;
    try {
        const Topic created = storage_->createTopic(name.toStdString());
        refreshLibrary();
        statusBar()->showMessage(
            tr("话题已创建：%1").arg(QString::fromStdString(created.name)), 5000);
    } catch (const std::exception& error) {
        QMessageBox::warning(this, tr("无法创建话题"), QString::fromUtf8(error.what()));
    }
}

std::optional<Topic> MainWindow::selectedTopic() const {
    return libraryModel_->topicForIndex(libraryTree_->currentIndex());
}

void MainWindow::createProject() {
    const auto topic = selectedTopic();
    if (!topic) {
        QMessageBox::information(this, tr("请选择话题"),
                                 tr("论文必须属于一个话题。请先选择或新建话题。"));
        return;
    }
    NewProjectDialog dialog(this);
    if (dialog.exec() != QDialog::Accepted) return;
    const auto request = dialog.request();
    if (request.source == NewProjectRequest::Source::Pdf) {
        importPdf(*topic, request.input, request.title);
    } else if (request.source == NewProjectRequest::Source::Zotero) {
        if (request.zoteroPaper) importZotero(*request.zoteroPaper, request.title);
    } else {
        importArxiv(*topic, request.input, request.title);
    }
}

void MainWindow::importPdf(const Topic& topic, const QString& path, const QString& title) {
    const QString taskId = tasks_->beginTask(tr("导入 PDF：%1").arg(QFileInfo(path).fileName()), true);
    taskDock_->show();
    auto* watcher = new QFutureWatcher<Project>(this);
    connect(watcher, &QFutureWatcher<Project>::finished, this,
            [this, watcher, taskId] {
                try {
                    const Project project = watcher->result();
                    tasks_->finishTask(taskId, tr("PDF 项目创建完成"));
                    refreshLibrary();
                    reader_->openProject(project);
                    assistant_->setProjectPath(project.path);
                } catch (const std::exception& error) {
                    tasks_->failTask(taskId, QString::fromUtf8(error.what()));
                }
                watcher->deleteLater();
            });
    auto storage = storage_;
    watcher->setFuture(QtConcurrent::run([storage, topic, path, title] {
        return storage->createProjectFromPdf(topic.path, path.toStdString(),
                                             title.toStdString());
    }));
}

void MainWindow::importZotero(const ZoteroPaper& requestedPaper,
                              const QString& title) {
    ZoteroPaper paper = requestedPaper;
    if (!title.trimmed().isEmpty()) paper.title = title.trimmed().toStdString();
    QSettings("ScholarVault", "ScholarVault").setValue(
        "zoteroDataDirectory", QString::fromStdString(paper.dataDirectory.string()));
    const QString taskId = tasks_->beginTask(
        tr("导入 Zotero：%1").arg(QString::fromStdString(paper.title)), true);
    taskDock_->show();
    auto* watcher = new QFutureWatcher<VaultStorage::ZoteroLibrarySyncResult>(this);
    connect(watcher, &QFutureWatcher<VaultStorage::ZoteroLibrarySyncResult>::finished, this,
            [this, watcher, taskId] {
                try {
                    const auto result = watcher->result();
                    tasks_->finishTask(taskId, tr("Zotero 论文导入完成"));
                    refreshLibrary();
                    if (!result.changedProjects.empty()) {
                        const Project& project = result.changedProjects.front();
                        reader_->openProject(project);
                        assistant_->setProjectPath(project.path);
                    }
                } catch (const std::exception& error) {
                    tasks_->failTask(taskId, QString::fromUtf8(error.what()));
                }
                watcher->deleteLater();
            });
    const auto storage = storage_;
    watcher->setFuture(QtConcurrent::run([storage, paper] {
        return storage->syncZoteroLibrary({paper});
    }));
}

void MainWindow::importArxiv(const Topic& topic, const QString& input, const QString& title) {
    const auto reference = parseArxivReference(input.toStdString());
    if (!reference) {
        QMessageBox::warning(this, tr("arXiv 链接无效"), tr("请输入完整 arXiv 链接或有效论文 ID。"));
        return;
    }
    const QString taskId = tasks_->beginTask(
        tr("下载 arXiv：%1").arg(QString::fromStdString(reference->id)));
    taskDock_->show();
    auto* task = new ArxivImportTask(storage_, topic.path, *reference, title, this);
    connect(task, &ArxivImportTask::progress, this,
            [this, taskId](int value, const QString& status) {
                tasks_->updateTask(taskId, value, status);
            });
    connect(task, &ArxivImportTask::succeeded, this,
            [this, task, taskId](const Project& project) {
                tasks_->finishTask(taskId, tr("arXiv 项目创建完成"));
                refreshLibrary();
                reader_->openProject(project);
                assistant_->setProjectPath(project.path);
                task->deleteLater();
            });
    connect(task, &ArxivImportTask::failed, this,
            [this, task, taskId](const QString& error) {
                tasks_->failTask(taskId, error);
                task->deleteLater();
            });
    task->start();
}

void MainWindow::addGitHubRepository() {
    auto project = libraryModel_->projectForIndex(libraryTree_->currentIndex());
    if (!project) project = reader_->currentProject();
    if (!project) {
        QMessageBox::information(this, tr("请选择论文"), tr("请先打开需要关联代码的论文项目。"));
        return;
    }
    if (gitHubDownloadRunning_) {
        statusBar()->showMessage(tr("已有 GitHub 下载任务正在运行"), 4000);
        return;
    }
    bool accepted = false;
    const QString input = QInputDialog::getText(
        this, tr("添加相关代码"), tr("公开 GitHub 仓库链接"), QLineEdit::Normal,
        "https://github.com/", &accepted).trimmed();
    if (!accepted || input.isEmpty()) return;
    const auto repository = parseGitHubRepository(input.toStdString());
    if (!repository) {
        QMessageBox::warning(this, tr("GitHub 链接无效"),
                             tr("请输入 https://github.com/owner/repository 形式的公开仓库链接。"));
        return;
    }
    const QString taskId = tasks_->beginTask(
        tr("克隆代码：%1/%2").arg(QString::fromStdString(repository->owner),
                                    QString::fromStdString(repository->name)));
    taskDock_->show();
    gitHubDownloadRunning_ = true;
    reader_->setGitHubDownloadActive(true);
    auto* task = new GitHubCloneTask(storage_, *project, *repository, this);
    connect(task, &GitHubCloneTask::progress, this,
            [this, taskId](int value, const QString& status) {
                tasks_->updateTask(taskId, value, status);
            });
    connect(task, &GitHubCloneTask::succeeded, this,
            [this, task, taskId](const Project& updated) {
                tasks_->finishTask(taskId, tr("GitHub 代码已下载"));
                gitHubDownloadRunning_ = false;
                reader_->setGitHubDownloadActive(false);
                const auto current = reader_->currentProject();
                if (current && current->id == updated.id) reader_->openProject(updated);
                refreshLibrary();
                task->deleteLater();
            });
    connect(task, &GitHubCloneTask::failed, this,
            [this, task, taskId](const QString& error) {
                tasks_->failTask(taskId, error);
                gitHubDownloadRunning_ = false;
                reader_->setGitHubDownloadActive(false);
                task->deleteLater();
            });
    task->start();
}

void MainWindow::requestArxivSourceDownload() {
    const auto project = reader_->currentProject();
    if (!storage_ || !project) return;
    if (project->sourceStatus == "ready") {
        statusBar()->showMessage(tr("当前论文的 LaTeX 已经下载"), 4000);
        return;
    }
    if (arxivSourceRunning_) {
        statusBar()->showMessage(tr("已有 arXiv 源码下载任务正在运行"), 4000);
        return;
    }

    QString initial = "https://arxiv.org/abs/";
    if (const auto known = parseArxivReference(project->arxivId)) {
        initial = QString::fromStdString(known->abstractUrl());
    }
    bool accepted = false;
    const QString input = QInputDialog::getText(
        this, tr("下载 arXiv LaTeX"), tr("arXiv 链接或论文 ID"),
        QLineEdit::Normal, initial, &accepted).trimmed();
    if (!accepted || input.isEmpty()) return;
    const auto reference = parseArxivReference(input.toStdString());
    if (!reference) {
        QMessageBox::warning(this, tr("arXiv 链接无效"),
                             tr("请输入 arxiv.org 的 abs/PDF 链接或有效论文 ID。"));
        return;
    }
    startArxivSourceDownload(*project, *reference);
}

void MainWindow::startArxivSourceDownload(
    const Project& project,
    const ArxivReference& reference,
    std::function<void()> completion) {
    if (arxivSourceRunning_) return;
    arxivSourceRunning_ = true;
    reader_->setSourceDownloadActive(true);
    assistant_->setSourcePreparationActive(true);
    const QString taskId = tasks_->beginTask(
        tr("按需下载 arXiv 原始 TeX：%1").arg(QString::fromStdString(reference.id)));
    taskDock_->show();
    auto* task = new ArxivImportTask(storage_, project, reference, this);
    connect(task, &ArxivImportTask::progress, this,
            [this, taskId](int value, const QString& status) {
                tasks_->updateTask(taskId, value, status);
            });
    connect(task, &ArxivImportTask::succeeded, this,
            [this, task, taskId, completion](const Project& updated) {
                tasks_->finishTask(taskId, updated.sourceStatus == "ready"
                    ? tr("arXiv 原始 TeX 已保存")
                    : tr("该 arXiv 条目没有可用的 LaTeX 源码"));
                arxivSourceRunning_ = false;
                reader_->setSourceDownloadActive(false);
                assistant_->setSourcePreparationActive(false);
                const auto current = reader_->currentProject();
                if (current && current->id == updated.id) reader_->openProject(updated);
                refreshLibrary();
                task->deleteLater();
                if (completion) completion();
            });
    connect(task, &ArxivImportTask::failed, this,
            [this, task, taskId, completion](const QString& error) {
                tasks_->failTask(taskId, error);
                arxivSourceRunning_ = false;
                reader_->setSourceDownloadActive(false);
                assistant_->setSourcePreparationActive(false);
                task->deleteLater();
                if (completion) completion();
            });
    task->start();
}

void MainWindow::prepareAssistantAnalysis(bool codex) {
    const auto launch = [this, codex] {
        if (codex) assistant_->startCodex();
        else assistant_->startChatGpt();
    };
    const auto project = reader_->currentProject();
    if (!storage_ || !project || project->arxivId.empty() ||
        project->sourceStatus == "ready" || project->sourceStatus == "no-source" ||
        project->sourceStatus == "invalid-source") {
        launch();
        return;
    }
    if (arxivSourceRunning_) return;
    const auto reference = parseArxivReference(project->arxivId);
    if (!reference) {
        launch();
        return;
    }

    startArxivSourceDownload(*project, *reference, launch);
}

void MainWindow::activateIndex(const QModelIndex& index) {
    const auto project = libraryModel_->projectForIndex(index);
    if (!project) return;
    reader_->openProject(*project);
    assistant_->setProjectPath(project->path);
}

QStringList MainWindow::libraryContextActionTexts(const QModelIndex& index) const {
    if (!libraryModel_->projectForIndex(index)) return {};
    return {sendPdfToChatGptAction_->text()};
}

void MainWindow::showLibraryContextMenu(const QPoint& position) {
    const QModelIndex index = libraryTree_->indexAt(position);
    if (!libraryModel_->projectForIndex(index)) return;
    libraryTree_->selectionModel()->setCurrentIndex(
        index, QItemSelectionModel::ClearAndSelect | QItemSelectionModel::Rows);
    sendPdfToChatGptAction_->setEnabled(!chatGptUploadRunning_);
    QMenu menu(libraryTree_);
    menu.addAction(sendPdfToChatGptAction_);
    menu.exec(libraryTree_->viewport()->mapToGlobal(position));
}

void MainWindow::sendCurrentPdfToChatGpt() {
    const auto project = libraryModel_->projectForIndex(libraryTree_->currentIndex());
    if (!project) return;
    if (chatGptUploadRunning_ || pendingChatGptUpload_) {
        statusBar()->showMessage(tr("已有 PDF 正在发送给 ChatGPT"), 4000);
        return;
    }
    const QString validation = ChatGptUploadTask::validationError(*project);
    if (!validation.isEmpty()) {
        QMessageBox::warning(this, tr("无法发送 PDF"), validation);
        return;
    }
    reader_->openProject(*project);
    assistant_->setProjectPath(project->path);
    pendingChatGptUpload_ = *project;
    assistant_->setCurrentTabIndex(0);
    if (assistant_->chatGptNeedsDebuggingRestart()) {
        if (QMessageBox::question(
                this, tr("重启 ChatGPT 会话"),
                tr("首次发送 PDF 需要重启 ScholarVault 管理的 ChatGPT 窗口。登录状态会保留，是否继续？"))
            != QMessageBox::Yes) {
            pendingChatGptUpload_.reset();
            return;
        }
        assistant_->restartChatGptWithDebugging();
        return;
    }
    if (assistant_->chatGptOpen() && assistant_->chatGptDevToolsReady()) {
        startPendingChatGptUpload();
    } else {
        assistant_->startChatGpt();
    }
}

void MainWindow::startPendingChatGptUpload() {
    if (!pendingChatGptUpload_ || chatGptUploadRunning_) return;
    if (!assistant_->chatGptDevToolsReady()) {
        statusBar()->showMessage(tr("等待 ChatGPT 本地连接就绪…"), 3000);
        return;
    }
    const Project project = *pendingChatGptUpload_;
    pendingChatGptUpload_.reset();
    chatGptUploadRunning_ = true;
    updateContextActions();
    const QString taskId = tasks_->beginTask(
        tr("发送 PDF 给 ChatGPT：%1").arg(QString::fromStdString(project.title)));
    taskDock_->show();
    auto* task = new ChatGptUploadTask(project, assistant_->chatGptProfilePath(), this);
    connect(task, &ChatGptUploadTask::progress, this,
            [this, taskId](int value, const QString& status) {
                tasks_->updateTask(taskId, value, status);
            });
    connect(task, &ChatGptUploadTask::succeeded, this,
            [this, task, taskId] {
                tasks_->finishTask(taskId, tr("PDF 已附加，可以开始提问"));
                statusBar()->showMessage(tr("PDF 已附加，可以开始提问"), 6000);
                chatGptUploadRunning_ = false;
                updateContextActions();
                task->deleteLater();
            });
    connect(task, &ChatGptUploadTask::failed, this,
            [this, task, taskId](const QString& error) {
                tasks_->failTask(taskId, error);
                chatGptUploadRunning_ = false;
                updateContextActions();
                task->deleteLater();
            });
    task->start();
}

void MainWindow::syncAllZoteroProjects() {
    if (!storage_ || zoteroSyncRunning_ || topicMoveRunning_) return;
    zoteroSyncRunning_ = true;
    syncZoteroAction_->setEnabled(false);
    updateContextActions();
    QSettings settings("ScholarVault", "ScholarVault");
    QString directory = settings.value("zoteroDataDirectory").toString();
    if (directory.isEmpty()) directory = QDir::home().filePath("Zotero");
    startFullZoteroSync(directory);
}

void MainWindow::startFullZoteroSync(const QString& dataDirectory) {
    const QString taskId = tasks_->beginTask(tr("扫描 Zotero 全部论文"), true);
    taskDock_->show();
    auto* loader = new ZoteroCatalogLoader(dataDirectory, this);
    connect(loader, &ZoteroCatalogLoader::progress, this,
            [this, taskId](const QString& status) {
                tasks_->updateTask(taskId, 0, status);
            });
    connect(loader, &ZoteroCatalogLoader::failed, this,
            [this, loader, taskId](const QString& error) {
                tasks_->failTask(taskId, error);
                loader->deleteLater();
                zoteroSyncRunning_ = false;
                syncZoteroAction_->setEnabled(storage_ != nullptr);
                updateContextActions();
            });
    connect(loader, &ZoteroCatalogLoader::loaded, this,
            [this, loader, taskId, dataDirectory](
                const std::vector<ZoteroPaper>& papers, const QString&) {
                loader->deleteLater();
                QSettings("ScholarVault", "ScholarVault").setValue(
                    "zoteroDataDirectory", dataDirectory);
                tasks_->updateTask(taskId, 0,
                    tr("正在复制或更新 %1 篇 Zotero 论文").arg(papers.size()));
                auto* watcher = new QFutureWatcher<VaultStorage::ZoteroLibrarySyncResult>(this);
                connect(watcher,
                        &QFutureWatcher<VaultStorage::ZoteroLibrarySyncResult>::finished,
                        this, [this, watcher, taskId] {
                            const auto result = watcher->result();
                            watcher->deleteLater();
                            const QString summary =
                                tr("新增 %1，更新 %2，移动 %3，未变化 %4，跳过 %5")
                                    .arg(result.created).arg(result.updated)
                                    .arg(result.moved).arg(result.unchanged)
                                    .arg(result.skipped);
                            if (result.created == 0 && result.updated == 0 &&
                                result.moved == 0 && result.skipped > 0) {
                                tasks_->failTask(taskId, summary + "：" +
                                    QString::fromStdString(result.errors.front()));
                            } else {
                                tasks_->finishTask(taskId, summary);
                            }
                            const auto current = reader_->currentProject();
                            if (current) {
                                for (const auto& project : result.changedProjects) {
                                    if (project.id == current->id) {
                                        reader_->refreshProject(project);
                                        break;
                                    }
                                }
                            }
                            refreshLibrary();
                            zoteroSyncRunning_ = false;
                            syncZoteroAction_->setEnabled(storage_ != nullptr);
                            updateContextActions();
                        });
                const auto storage = storage_;
                watcher->setFuture(QtConcurrent::run(
                    [storage, papers] { return storage->syncZoteroLibrary(papers); }));
            });
    loader->start();
}

void MainWindow::syncCurrentZoteroProject() {
    if (zoteroSyncRunning_ || topicMoveRunning_) return;
    const auto project = libraryModel_->projectForIndex(libraryTree_->currentIndex());
    if (!project || project->origin != ProjectOrigin::Zotero || !project->zotero) return;
    zoteroSyncRunning_ = true;
    syncZoteroAction_->setEnabled(false);
    updateContextActions();
    syncZoteroProjects({*project});
}

void MainWindow::syncZoteroProjects(std::vector<Project> projects) {
    std::map<QString, std::vector<Project>> groups;
    for (auto& project : projects) {
        const QString directory = project.zotero
            ? QString::fromStdString(project.zotero->dataDirectory)
            : QString{};
        groups[directory].push_back(std::move(project));
    }
    zoteroSyncJobs_ = static_cast<int>(groups.size());
    for (auto& [directory, group] : groups) {
        startZoteroSyncGroup(std::move(group), directory);
    }
}

void MainWindow::startZoteroSyncGroup(std::vector<Project> projects,
                                      const QString& dataDirectory) {
    const QString taskId = tasks_->beginTask(
        tr("同步 Zotero：%1 篇").arg(projects.size()), true);
    taskDock_->show();
    auto* loader = new ZoteroCatalogLoader(dataDirectory, this);
    connect(loader, &ZoteroCatalogLoader::progress, this,
            [this, taskId](const QString& status) {
                tasks_->updateTask(taskId, 0, status);
            });
    connect(loader, &ZoteroCatalogLoader::failed, this,
            [this, loader, taskId](const QString& error) {
                tasks_->failTask(taskId, error);
                loader->deleteLater();
                finishZoteroSyncJob();
            });
    connect(loader, &ZoteroCatalogLoader::loaded, this,
            [this, loader, taskId, projects = std::move(projects)](
                const std::vector<ZoteroPaper>& papers, const QString&) mutable {
                loader->deleteLater();
                auto* watcher = new QFutureWatcher<ZoteroGroupSyncResult>(this);
                connect(watcher, &QFutureWatcher<ZoteroGroupSyncResult>::finished, this,
                        [this, watcher, taskId] {
                            const ZoteroGroupSyncResult result = watcher->result();
                            watcher->deleteLater();
                            if (result.updated == 0 && result.unchanged == 0 &&
                                result.skipped > 0) {
                                tasks_->failTask(taskId, result.errors.join("；"));
                            } else {
                                tasks_->finishTask(
                                    taskId,
                                    tr("已更新 %1，未变化 %2，跳过 %3")
                                        .arg(result.updated)
                                        .arg(result.unchanged)
                                        .arg(result.skipped));
                            }
                            const auto current = reader_->currentProject();
                            if (current) {
                                for (const auto& project : result.updatedProjects) {
                                    if (project.id == current->id) {
                                        reader_->refreshProject(project);
                                        break;
                                    }
                                }
                            }
                            refreshLibrary();
                            finishZoteroSyncJob();
                        });
                const auto storage = storage_;
                watcher->setFuture(QtConcurrent::run(
                    [storage, projects = std::move(projects), papers] {
                        ZoteroGroupSyncResult result;
                        for (const auto& project : projects) {
                            QString matchError;
                            const auto paper = matchZoteroPaper(project, papers, &matchError);
                            if (!paper) {
                                ++result.skipped;
                                result.errors.push_back(
                                    QString::fromStdString(project.title) + "：" + matchError);
                                continue;
                            }
                            try {
                                auto sync = storage->syncProjectFromZotero(project, *paper);
                                if (sync.pdfUpdated || sync.metadataUpdated) {
                                    ++result.updated;
                                    result.updatedProjects.push_back(std::move(sync.project));
                                } else {
                                    ++result.unchanged;
                                }
                            } catch (const std::exception& error) {
                                ++result.skipped;
                                result.errors.push_back(
                                    QString::fromStdString(project.title) + "：" +
                                    QString::fromUtf8(error.what()));
                            }
                        }
                        return result;
                    }));
            });
    loader->start();
}

void MainWindow::finishZoteroSyncJob() {
    if (--zoteroSyncJobs_ > 0) return;
    zoteroSyncJobs_ = 0;
    zoteroSyncRunning_ = false;
    syncZoteroAction_->setEnabled(storage_ != nullptr);
    updateContextActions();
}

void MainWindow::moveSelectedTopicToTrash() {
    if (!storage_ || zoteroSyncRunning_ || topicMoveRunning_) return;
    const QModelIndex index = libraryTree_->currentIndex();
    if (libraryModel_->projectForIndex(index)) return;
    const auto topic = libraryModel_->topicForIndex(index);
    if (!topic) return;
    const auto answer = QMessageBox::question(
        this, tr("删除话题"),
        tr("确定将话题“%1”及其中全部论文移入 Vault 回收站吗？\n\n"
           "目录会移动到 .trash/topics，可手动恢复；不会永久删除。")
            .arg(QString::fromStdString(topic->name)),
        QMessageBox::Yes | QMessageBox::Cancel, QMessageBox::Cancel);
    if (answer != QMessageBox::Yes) return;
    topicMoveRunning_ = true;
    syncZoteroAction_->setEnabled(false);
    updateContextActions();

    const auto current = reader_->currentProject();
    const bool closesCurrent = current && current->path.parent_path() == topic->path;
    const QString taskId = tasks_->beginTask(
        tr("删除话题：%1").arg(QString::fromStdString(topic->name)), true);
    taskDock_->show();
    auto* watcher = new QFutureWatcher<QString>(this);
    connect(watcher, &QFutureWatcher<QString>::finished, this,
            [this, watcher, taskId, closesCurrent] {
                const QString result = watcher->result();
                watcher->deleteLater();
                if (result.startsWith("ERROR:")) {
                    tasks_->failTask(taskId, result.mid(6));
                    topicMoveRunning_ = false;
                    syncZoteroAction_->setEnabled(storage_ != nullptr);
                    updateContextActions();
                    return;
                }
                tasks_->finishTask(taskId, tr("话题已移入回收站：%1").arg(result));
                if (closesCurrent) {
                    reader_->clearProject();
                    assistant_->setProjectPath({});
                }
                refreshLibrary();
                topicMoveRunning_ = false;
                syncZoteroAction_->setEnabled(storage_ != nullptr);
                updateContextActions();
            });
    const auto storage = storage_;
    const auto path = topic->path;
    watcher->setFuture(QtConcurrent::run([storage, path] {
        try {
            return QString::fromStdString(storage->moveTopicToTrash(path).string());
        } catch (const std::exception& error) {
            return QString("ERROR:") + QString::fromUtf8(error.what());
        }
    }));
}

void MainWindow::updateContextActions() {
    if (sendPdfToChatGptAction_ != nullptr) {
        sendPdfToChatGptAction_->setEnabled(
            libraryModel_->projectForIndex(libraryTree_->currentIndex()).has_value() &&
            !chatGptUploadRunning_);
    }
}

void MainWindow::refreshLibrary() {
    libraryModel_->reload();
}

void MainWindow::setBusyActions(bool busy) {
    if (syncZoteroAction_ != nullptr) {
        syncZoteroAction_->setEnabled(!busy && storage_ != nullptr);
    }
}

} // namespace scholarvault::ui
