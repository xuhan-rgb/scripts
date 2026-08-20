#pragma once

#include <QSet>
#include <QWidget>

#include <memory>
#include <optional>

#ifndef Q_MOC_RUN
#include <filesystem>
#endif

class QLabel;
class QProcess;
class QPushButton;
class QSize;
class QTabWidget;
class QTimer;
class AssistantPanelTest;

namespace scholarvault::ui {

class X11WindowController;

class AssistantPanel final : public QWidget {
    Q_OBJECT

public:
    explicit AssistantPanel(QWidget* parent = nullptr);
    ~AssistantPanel() override;

    void setProjectPath(const std::filesystem::path& path);
    void setSourcePreparationActive(bool active);
    void startChatGpt();
    void startCodex();
    [[nodiscard]] int currentTabIndex() const;
    void setCurrentTabIndex(int index);
    [[nodiscard]] bool chatGptOpen() const { return chatWindowId_ != 0; }
    [[nodiscard]] bool codexAttached() const;
    [[nodiscard]] QString chatGptProfilePath() const;
    [[nodiscard]] bool chatGptDevToolsReady() const;
    [[nodiscard]] bool chatGptNeedsDebuggingRestart() const;
    void restartChatGptWithDebugging();
    [[nodiscard]] static QString codexSessionName(
        const std::filesystem::path& projectPath);

signals:
    void analysisRequested(bool codex);
    void viewStateChanged();
    void chatGptReady();

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

private:
    friend class ::AssistantPanelTest;

    [[nodiscard]] static QSize nativePixelSize(const QWidget* widget);
    void pollChatGptWindow();
    void pollCodexWindow();
    void embedChatGptWindow(qulonglong windowId);
    void resizeEmbeddedWindows();
    void verifyChatGptWindow();
    void endCodexSession();
    [[nodiscard]] bool codexSessionExists() const;
    [[nodiscard]] QSet<qulonglong> x11ClientWindows() const;

    QTabWidget* tabs_;
    QWidget* chatHost_;
    QLabel* chatStatus_;
    QPushButton* chatButton_;
    QTimer* chatPollTimer_;
    QTimer* chatGuardTimer_;
    QTimer* resizeTimer_;
    QSet<qulonglong> knownWindowIds_;
    qulonglong chatWindowId_{0};
    qulonglong chatCandidateId_{0};
    qint64 remoteChromePid_{0};
    int chatPollCount_{0};
    int chatCandidateSeenCount_{0};

    QWidget* codexHost_;
    QLabel* codexStatus_;
    QPushButton* codexButton_;
    QPushButton* endCodexButton_;
    QProcess* codexProcess_;
    QTimer* codexPollTimer_;
    qulonglong codexWindowId_{0};
    int codexPollCount_{0};
    QString codexSessionName_;
    std::unique_ptr<X11WindowController> x11_;
    std::filesystem::path projectPath_;
};

} // namespace scholarvault::ui
