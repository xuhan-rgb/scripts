#pragma once

#include "scholarvault/storage.hpp"

#include <QMainWindow>
#include <QStringList>

#include <functional>
#include <memory>

class QAction;
class QCloseEvent;
class QDockWidget;
class QModelIndex;
class QPoint;
class QSplitter;
class QTimer;
class QTreeView;
class MainWindowSelectionTest;

namespace scholarvault::ui {

class AssistantPanel;
class LibraryModel;
class ProjectReader;
class TaskPanel;

class MainWindow final : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);

protected:
    void closeEvent(QCloseEvent* event) override;

private:
    friend class ::MainWindowSelectionTest;

    void chooseVault();
    void openVault(const QString& path);
    void createTopic();
    void createProject();
    void importPdf(const Topic& topic, const QString& path, const QString& title);
    void importZotero(const ZoteroPaper& paper, const QString& title);
    void importArxiv(const Topic& topic, const QString& input, const QString& title);
    void requestArxivSourceDownload();
    void startArxivSourceDownload(const Project& project,
                                  const ArxivReference& reference,
                                  std::function<void()> completion = {});
    void prepareAssistantAnalysis(bool codex);
    void sendCurrentPdfToChatGpt();
    void startPendingChatGptUpload();
    void showLibraryContextMenu(const QPoint& position);
    [[nodiscard]] QStringList libraryContextActionTexts(
        const QModelIndex& index) const;
    void addGitHubRepository();
    void syncAllZoteroProjects();
    void startFullZoteroSync(const QString& dataDirectory);
    void syncCurrentZoteroProject();
    void syncZoteroProjects(std::vector<Project> projects);
    void startZoteroSyncGroup(std::vector<Project> projects,
                              const QString& dataDirectory);
    void finishZoteroSyncJob();
    void moveSelectedTopicToTrash();
    void updateContextActions();
    void activateIndex(const QModelIndex& index);
    [[nodiscard]] std::optional<Topic> selectedTopic() const;
    void refreshLibrary();
    void setBusyActions(bool busy);
    void scheduleWorkspaceSave();
    void saveWorkspaceState(bool forceSync = false);
    void restoreWorkspaceState();
    [[nodiscard]] QStringList expandedTopicIds() const;

    std::shared_ptr<VaultStorage> storage_;
    LibraryModel* libraryModel_;
    QTreeView* libraryTree_;
    ProjectReader* reader_;
    AssistantPanel* assistant_;
    TaskPanel* tasks_;
    QDockWidget* taskDock_;
    QSplitter* splitter_;
    QTimer* workspaceSaveTimer_;
    QAction* syncZoteroAction_{nullptr};
    QAction* sendPdfToChatGptAction_{nullptr};
    bool zoteroSyncRunning_{false};
    int zoteroSyncJobs_{0};
    bool topicMoveRunning_{false};
    bool arxivSourceRunning_{false};
    bool gitHubDownloadRunning_{false};
    bool chatGptUploadRunning_{false};
    std::optional<Project> pendingChatGptUpload_;
    bool restoringWorkspace_{false};
};

} // namespace scholarvault::ui
