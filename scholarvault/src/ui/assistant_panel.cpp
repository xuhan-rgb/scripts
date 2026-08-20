#include "ui/assistant_panel.hpp"

#include <QDir>
#include <QEvent>
#include <QFile>
#include <QFileInfo>
#include <QGuiApplication>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QProcess>
#include <QPushButton>
#include <QMessageBox>
#include <QRegularExpression>
#include <QResizeEvent>
#include <QStandardPaths>
#include <QTabWidget>
#include <QTimer>
#include <QVBoxLayout>
#include <QtGui/qguiapplication_platform.h>

#include <X11/Xlib.h>

#include <algorithm>

namespace scholarvault::ui {
namespace {

struct RemoteChromeSession {
    QString executable;
    QString userDataDirectory;
    qint64 pid{0};
};

struct X11WindowInfo {
    QString windowClass;
    qint64 pid{0};
    int width{0};
    int height{0};
    bool viewable{false};
};

QStringList processArguments(qint64 pid) {
    QFile file(QString("/proc/%1/cmdline").arg(pid));
    if (!file.open(QIODevice::ReadOnly)) return {};
    QStringList result;
    for (const QByteArray& value : file.readAll().split('\0')) {
        if (!value.isEmpty()) result.push_back(QString::fromUtf8(value));
    }
    return result;
}

std::optional<RemoteChromeSession> findManagedChrome(const QString& userDataDirectory) {
    const QString profileArgument = "--user-data-dir=" + userDataDirectory;
    const auto entries = QDir("/proc").entryList(QDir::Dirs | QDir::NoDotAndDotDot,
                                                  QDir::Name);
    for (const QString& entry : entries) {
        bool validPid = false;
        const qint64 pid = entry.toLongLong(&validPid);
        if (!validPid) continue;
        const QStringList arguments = processArguments(pid);
        if (arguments.isEmpty() || !arguments.contains(profileArgument)) continue;
        if (std::any_of(arguments.begin(), arguments.end(), [](const QString& value) {
                return value.startsWith("--type=");
            })) {
            continue;
        }
        const QString executableName = QFileInfo(arguments.front()).fileName().toLower();
        if (!executableName.contains("chrome") && !executableName.contains("chromium")) {
            continue;
        }
        RemoteChromeSession session{arguments.front(), userDataDirectory, pid};
        for (const QString& argument : arguments) {
            if (argument.startsWith("--user-data-dir=")) {
                session.userDataDirectory = argument.mid(16);
            }
        }
        return session;
    }
    return std::nullopt;
}

QString findChromeExecutable() {
    for (const QString& name : {QString("google-chrome"),
                                QString("google-chrome-stable"),
                                QString("chromium"),
                                QString("chromium-browser")}) {
        const QString executable = QStandardPaths::findExecutable(name);
        if (!executable.isEmpty()) return executable;
    }
    return {};
}

QSet<qulonglong> readX11ClientWindows() {
    QProcess process;
    process.start("xprop", {"-root", "_NET_CLIENT_LIST"});
    if (!process.waitForFinished(600) || process.exitCode() != 0) return {};
    const QString output = QString::fromUtf8(process.readAllStandardOutput());
    static const QRegularExpression idExpression("0x[0-9a-fA-F]+");
    QSet<qulonglong> windows;
    auto match = idExpression.globalMatch(output);
    while (match.hasNext()) {
        bool ok = false;
        const qulonglong id = match.next().captured().mid(2).toULongLong(&ok, 16);
        if (ok) windows.insert(id);
    }
    return windows;
}

std::optional<X11WindowInfo> readX11WindowInfo(qulonglong id) {
    QProcess properties;
    properties.start("xprop", {"-id", QString("0x%1").arg(id, 0, 16),
                                "WM_CLASS", "_NET_WM_PID"});
    if (!properties.waitForFinished(400) || properties.exitCode() != 0) {
        return std::nullopt;
    }
    QProcess geometry;
    geometry.start("xwininfo", {"-id", QString("0x%1").arg(id, 0, 16)});
    if (!geometry.waitForFinished(400) || geometry.exitCode() != 0) {
        return std::nullopt;
    }
    const QString propertyText = QString::fromUtf8(properties.readAllStandardOutput());
    const QString geometryText = QString::fromUtf8(geometry.readAllStandardOutput());
    static const QRegularExpression classExpression(
        "^WM_CLASS.*?=\\s*(.+)$", QRegularExpression::MultilineOption);
    static const QRegularExpression pidExpression(
        "^_NET_WM_PID.*?=\\s*(\\d+)$", QRegularExpression::MultilineOption);
    static const QRegularExpression widthExpression(
        "^\\s*Width:\\s*(\\d+)$", QRegularExpression::MultilineOption);
    static const QRegularExpression heightExpression(
        "^\\s*Height:\\s*(\\d+)$", QRegularExpression::MultilineOption);
    const auto width = widthExpression.match(geometryText);
    const auto height = heightExpression.match(geometryText);
    if (!width.hasMatch() || !height.hasMatch()) return std::nullopt;
    const auto windowClass = classExpression.match(propertyText);
    const auto pid = pidExpression.match(propertyText);
    return X11WindowInfo{windowClass.hasMatch() ? windowClass.captured(1) : QString{},
                         pid.hasMatch() ? pid.captured(1).toLongLong() : 0,
                         width.captured(1).toInt(), height.captured(1).toInt(),
                         geometryText.contains("Map State: IsViewable")};
}

bool isChatGptWindow(const X11WindowInfo& window, qint64 remoteChromePid) {
    const QString windowClass = window.windowClass.toLower();
    return (remoteChromePid == 0 || window.pid == remoteChromePid) && window.viewable &&
           window.width >= 320 && window.height >= 320 &&
           windowClass.contains("chatgpt.com") &&
           (windowClass.contains("chrome") || windowClass.contains("chromium"));
}

} // namespace

class X11WindowController {
public:
    X11WindowController() {
        auto* native = qGuiApp->nativeInterface<QNativeInterface::QX11Application>();
        if (native != nullptr) display_ = native->display();
    }

    [[nodiscard]] bool available() const { return display_ != nullptr; }

    [[nodiscard]] qulonglong parentWindow(qulonglong window) const {
        if (!display_ || window == 0) return 0;
        Window root = 0;
        Window parent = 0;
        Window* children = nullptr;
        unsigned int childCount = 0;
        const bool ok = XQueryTree(display_, window, &root, &parent,
                                   &children, &childCount) != 0;
        if (children != nullptr) XFree(children);
        return ok ? parent : 0;
    }

    [[nodiscard]] qulonglong largestChild(qulonglong parent) const {
        if (!display_ || parent == 0) return 0;
        Window root = 0;
        Window actualParent = 0;
        Window* children = nullptr;
        unsigned int childCount = 0;
        if (XQueryTree(display_, parent, &root, &actualParent,
                       &children, &childCount) == 0) {
            return 0;
        }
        qulonglong result = 0;
        int largestArea = 0;
        for (unsigned int index = 0; index < childCount; ++index) {
            XWindowAttributes attributes{};
            if (XGetWindowAttributes(display_, children[index], &attributes) == 0) continue;
            const int area = attributes.width * attributes.height;
            if (attributes.width >= 100 && attributes.height >= 100 && area > largestArea) {
                largestArea = area;
                result = children[index];
            }
        }
        if (children != nullptr) XFree(children);
        return result;
    }

    bool reparent(qulonglong child, qulonglong parent, int width, int height) {
        if (!display_ || child == 0 || parent == 0) return false;
        XSetWindowAttributes attributes{};
        attributes.override_redirect = True;
        XUnmapWindow(display_, child);
        XSync(display_, False);
        XChangeWindowAttributes(display_, child, CWOverrideRedirect, &attributes);
        XReparentWindow(display_, child, parent, 0, 0);
        XMoveResizeWindow(display_, child, 0, 0,
                          std::max(width, 1), std::max(height, 1));
        XMapWindow(display_, child);
        XSync(display_, False);
        return parentWindow(child) == parent;
    }

    void resize(qulonglong window, int width, int height) {
        if (!display_ || window == 0) return;
        XMoveResizeWindow(display_, window, 0, 0,
                          std::max(width, 1), std::max(height, 1));
        XFlush(display_);
    }

    void destroy(qulonglong window) {
        if (!display_ || window == 0) return;
        XDestroyWindow(display_, window);
        XSync(display_, False);
    }

private:
    Display* display_{nullptr};
};

AssistantPanel::AssistantPanel(QWidget* parent) : QWidget(parent) {
    x11_ = std::make_unique<X11WindowController>();
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    tabs_ = new QTabWidget(this);
    connect(tabs_, &QTabWidget::currentChanged, this,
            &AssistantPanel::viewStateChanged);
    layout->addWidget(tabs_);

    auto* chatPage = new QWidget(tabs_);
    auto* chatLayout = new QVBoxLayout(chatPage);
    chatStatus_ = new QLabel(tr("使用 ScholarVault 的持久 Chrome 会话；首次使用需要登录 ChatGPT。"), chatPage);
    chatStatus_->setWordWrap(true);
    chatButton_ = new QPushButton(tr("打开并嵌入 ChatGPT"), chatPage);
    chatHost_ = new QWidget(chatPage);
    chatHost_->setObjectName("chatGptNativeHost");
    chatHost_->setAttribute(Qt::WA_NativeWindow);
    chatHost_->setStyleSheet("background: #ffffff; border: 1px solid #d0d5dd;");
    chatLayout->addWidget(chatStatus_);
    chatLayout->addWidget(chatButton_);
    chatLayout->addWidget(chatHost_, 1);
    tabs_->addTab(chatPage, tr("ChatGPT"));

    auto* codexPage = new QWidget(tabs_);
    auto* codexLayout = new QVBoxLayout(codexPage);
    codexStatus_ = new QLabel(tr("选择论文后，在该项目目录启动完整 Codex 终端。"), codexPage);
    codexStatus_->setWordWrap(true);
    codexButton_ = new QPushButton(tr("启动 Codex 终端"), codexPage);
    codexButton_->setEnabled(false);
    endCodexButton_ = new QPushButton(tr("结束后台 Codex 会话"), codexPage);
    endCodexButton_->setEnabled(false);
    codexHost_ = new QWidget(codexPage);
    codexHost_->setObjectName("codexTerminalHost");
    codexHost_->setAttribute(Qt::WA_NativeWindow);
    codexHost_->setStyleSheet("background: #101828; border: 1px solid #344054;");
    codexLayout->addWidget(codexStatus_);
    codexLayout->addWidget(codexButton_);
    codexLayout->addWidget(endCodexButton_);
    codexLayout->addWidget(codexHost_, 1);
    tabs_->addTab(codexPage, tr("Codex 终端"));

    codexProcess_ = new QProcess(this);
    connect(codexProcess_, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this,
            [this](int, QProcess::ExitStatus) {
                codexButton_->setEnabled(!projectPath_.empty());
                codexButton_->setText(tr("重新启动 Codex 终端"));
                codexStatus_->setText(codexSessionExists()
                    ? tr("Codex 仍在后台运行，可重新连接")
                    : tr("Codex 会话已结束"));
                endCodexButton_->setEnabled(codexSessionExists());
                codexWindowId_ = 0;
                codexPollTimer_->stop();
            });
    connect(codexProcess_, &QProcess::errorOccurred, this,
            [this](QProcess::ProcessError) {
                codexStatus_->setText(tr("无法启动 xterm/Codex，请检查命令是否已安装。"));
                codexButton_->setEnabled(!projectPath_.empty());
            });

    chatPollTimer_ = new QTimer(this);
    chatPollTimer_->setInterval(150);
    connect(chatPollTimer_, &QTimer::timeout, this, &AssistantPanel::pollChatGptWindow);
    chatGuardTimer_ = new QTimer(this);
    chatGuardTimer_->setInterval(1000);
    connect(chatGuardTimer_, &QTimer::timeout, this, &AssistantPanel::verifyChatGptWindow);
    resizeTimer_ = new QTimer(this);
    resizeTimer_->setSingleShot(true);
    resizeTimer_->setInterval(60);
    connect(resizeTimer_, &QTimer::timeout, this, &AssistantPanel::resizeEmbeddedWindows);
    chatHost_->installEventFilter(this);
    codexHost_->installEventFilter(this);
    codexPollTimer_ = new QTimer(this);
    codexPollTimer_->setInterval(100);
    connect(codexPollTimer_, &QTimer::timeout, this, &AssistantPanel::pollCodexWindow);
    connect(chatButton_, &QPushButton::clicked, this,
            [this] { emit analysisRequested(false); });
    connect(codexButton_, &QPushButton::clicked, this,
            [this] { emit analysisRequested(true); });
    connect(endCodexButton_, &QPushButton::clicked, this,
            &AssistantPanel::endCodexSession);
}

AssistantPanel::~AssistantPanel() {
    if (chatWindowId_ != 0 && x11_->parentWindow(chatWindowId_) != 0) {
        x11_->destroy(chatWindowId_);
    }
    if (codexProcess_->state() != QProcess::NotRunning) {
        codexProcess_->terminate();
        if (!codexProcess_->waitForFinished(1000)) codexProcess_->kill();
    }
}

QSize AssistantPanel::nativePixelSize(const QWidget* widget) {
    const qreal scale = widget->devicePixelRatioF();
    return {qRound(widget->width() * scale),
            qRound(widget->height() * scale)};
}

int AssistantPanel::currentTabIndex() const { return tabs_->currentIndex(); }

void AssistantPanel::setCurrentTabIndex(int index) {
    if (index >= 0 && index < tabs_->count()) tabs_->setCurrentIndex(index);
}

bool AssistantPanel::codexAttached() const {
    return codexProcess_->state() != QProcess::NotRunning;
}

QString AssistantPanel::chatGptProfilePath() const {
    return QDir(QStandardPaths::writableLocation(QStandardPaths::GenericDataLocation))
        .filePath("ScholarVault/chrome-profile");
}

bool AssistantPanel::chatGptDevToolsReady() const {
    QFile file(QDir(chatGptProfilePath()).filePath("DevToolsActivePort"));
    if (!file.open(QIODevice::ReadOnly)) return false;
    bool valid = false;
    const int port = file.readLine().trimmed().toInt(&valid);
    return valid && port > 0 && port <= 65535;
}

bool AssistantPanel::chatGptNeedsDebuggingRestart() const {
    return findManagedChrome(chatGptProfilePath()).has_value() &&
           !chatGptDevToolsReady();
}

void AssistantPanel::restartChatGptWithDebugging() {
    chatPollTimer_->stop();
    chatGuardTimer_->stop();
    if (chatWindowId_ != 0 && x11_->parentWindow(chatWindowId_) != 0) {
        x11_->destroy(chatWindowId_);
    }
    chatWindowId_ = 0;
    const auto session = findManagedChrome(chatGptProfilePath());
    if (session) {
        QProcess::execute("kill", {"-TERM", QString::number(session->pid)});
    }
    chatStatus_->show();
    chatButton_->show();
    chatButton_->setEnabled(false);
    chatStatus_->setText(tr("正在重启 ScholarVault ChatGPT 会话…"));
    QTimer::singleShot(1200, this, &AssistantPanel::startChatGpt);
}

QString AssistantPanel::codexSessionName(const std::filesystem::path& projectPath) {
    QFile project(QString::fromStdString((projectPath / "project.json").string()));
    QString identifier;
    if (project.open(QIODevice::ReadOnly)) {
        const QJsonDocument document = QJsonDocument::fromJson(project.readAll());
        identifier = document.object().value("id").toString();
    }
    if (identifier.isEmpty()) {
        identifier = QString::number(qHash(QString::fromStdString(projectPath.string())), 16);
    }
    identifier.replace(QRegularExpression("[^A-Za-z0-9_-]"), "-");
    return "scholarvault-" + identifier.left(80);
}

bool AssistantPanel::codexSessionExists() const {
    const QString tmux = QStandardPaths::findExecutable("tmux");
    if (tmux.isEmpty() || codexSessionName_.isEmpty()) return false;
    return QProcess::execute(tmux, {"has-session", "-t", codexSessionName_}) == 0;
}

void AssistantPanel::setProjectPath(const std::filesystem::path& path) {
    if (projectPath_ == path) return;
    if (codexProcess_->state() != QProcess::NotRunning) {
        codexProcess_->terminate();
        if (!codexProcess_->waitForFinished(800)) codexProcess_->kill();
    }
    projectPath_ = path;
    codexSessionName_ = projectPath_.empty() ? QString{} : codexSessionName(projectPath_);
    codexButton_->setEnabled(!projectPath_.empty());
    endCodexButton_->setEnabled(codexSessionExists());
    codexStatus_->setText(projectPath_.empty()
        ? tr("选择论文后，在该项目目录启动完整 Codex 终端。")
        : tr("工作目录：%1").arg(QString::fromStdString(projectPath_.string())));
}

void AssistantPanel::setSourcePreparationActive(bool active) {
    if (active) {
        chatButton_->setEnabled(false);
        codexButton_->setEnabled(false);
        chatStatus_->setText(tr("正在按需准备当前论文的 arXiv 原始 TeX…"));
        codexStatus_->setText(tr("正在按需准备当前论文的 arXiv 原始 TeX…"));
        return;
    }
    chatButton_->setEnabled(chatWindowId_ == 0);
    if (chatWindowId_ == 0) {
        chatStatus_->setText(
            tr("使用 ScholarVault 的持久 Chrome 会话；首次使用需要登录 ChatGPT。"));
    }
    codexButton_->setEnabled(!projectPath_.empty() &&
                             codexProcess_->state() == QProcess::NotRunning);
    if (codexProcess_->state() == QProcess::NotRunning) {
        codexStatus_->setText(projectPath_.empty()
            ? tr("选择论文后，在该项目目录启动完整 Codex 终端。")
            : tr("工作目录：%1").arg(QString::fromStdString(projectPath_.string())));
    }
}

void AssistantPanel::startCodex() {
    tabs_->setCurrentIndex(1);
    if (projectPath_.empty() || codexProcess_->state() != QProcess::NotRunning) return;
    const QString xterm = QStandardPaths::findExecutable("xterm");
    const QString codex = QStandardPaths::findExecutable("codex");
    const QString tmux = QStandardPaths::findExecutable("tmux");
    if (xterm.isEmpty() || codex.isEmpty() || tmux.isEmpty()) {
        codexStatus_->setText(tr("需要安装 xterm、tmux，并确保 codex 命令在 PATH 中。"));
        return;
    }
    if (!codexSessionExists()) {
        const int result = QProcess::execute(
            tmux, {"new-session", "-d", "-s", codexSessionName_, "-c",
                   QString::fromStdString(projectPath_.string()), codex,
                   "--dangerously-bypass-approvals-and-sandbox", "-p", "yolo"});
        if (result != 0) {
            codexStatus_->setText(tr("无法创建后台 Codex tmux 会话。"));
            return;
        }
    }
    codexHost_->winId();
    codexProcess_->setWorkingDirectory(QString::fromStdString(projectPath_.string()));
    const QString hostId = QString::number(static_cast<qulonglong>(codexHost_->winId()));
    codexProcess_->start(xterm, {"-into", hostId, "-xrm", "XTerm*faceName: DejaVu Sans Mono",
                                 "-xrm", "XTerm*faceSize: 11", "-e", tmux,
                                 "attach-session", "-t", codexSessionName_});
    codexWindowId_ = 0;
    codexPollCount_ = 0;
    codexPollTimer_->start();
    codexButton_->setEnabled(false);
    endCodexButton_->setEnabled(true);
    codexStatus_->setText(tr("Codex 正在当前论文项目中运行"));
    emit viewStateChanged();
}

void AssistantPanel::endCodexSession() {
    if (!codexSessionExists()) return;
    if (QMessageBox::question(this, tr("结束 Codex 会话"),
                              tr("确定结束当前论文的后台 Codex 会话吗？")) !=
        QMessageBox::Yes) {
        return;
    }
    if (codexProcess_->state() != QProcess::NotRunning) {
        codexProcess_->terminate();
        if (!codexProcess_->waitForFinished(800)) codexProcess_->kill();
    }
    const QString tmux = QStandardPaths::findExecutable("tmux");
    if (!tmux.isEmpty()) {
        QProcess::execute(tmux, {"kill-session", "-t", codexSessionName_});
    }
    endCodexButton_->setEnabled(false);
    codexButton_->setEnabled(!projectPath_.empty());
    codexStatus_->setText(tr("Codex 后台会话已结束"));
    emit viewStateChanged();
}

QSet<qulonglong> AssistantPanel::x11ClientWindows() const {
    return readX11ClientWindows();
}

void AssistantPanel::startChatGpt() {
    tabs_->setCurrentIndex(0);
    if (chatWindowId_ != 0) return;
    if (!x11_->available() || QStandardPaths::findExecutable("xprop").isEmpty() ||
        QStandardPaths::findExecutable("xwininfo").isEmpty()) {
        chatStatus_->setText(tr("ChatGPT 窗口嵌入需要 X11、xprop 与 xwininfo。"));
        return;
    }
    const QString profile = chatGptProfilePath();
    if (!QDir().mkpath(profile)) {
        chatStatus_->setText(tr("无法创建 ScholarVault 的 Chrome 登录目录。"));
        return;
    }
    const auto session = findManagedChrome(profile);
    const QString executable = session ? session->executable : findChromeExecutable();
    if (executable.isEmpty()) {
        chatStatus_->setText(tr("没有找到 Google Chrome 或 Chromium。"));
        return;
    }
    knownWindowIds_ = x11ClientWindows();
    chatHost_->winId();
    QStringList arguments{"--user-data-dir=" + profile,
                          "--no-first-run",
                          "--no-default-browser-check",
                          "--remote-debugging-address=127.0.0.1",
                          "--remote-debugging-port=0",
                          "--remote-allow-origins=*",
                          "--ozone-platform=x11",
                          "--new-window",
                          "--app=https://chatgpt.com/"};
    if (!QProcess::startDetached(executable, arguments)) {
        chatStatus_->setText(tr("无法请求 Chrome 打开 ChatGPT 窗口。"));
        return;
    }
    remoteChromePid_ = session ? session->pid : 0;
    chatPollCount_ = 0;
    chatCandidateId_ = 0;
    chatCandidateSeenCount_ = 0;
    chatButton_->setEnabled(false);
    chatStatus_->setText(tr("正在启动并嵌入 ChatGPT；首次使用请完成登录…"));
    chatPollTimer_->start();
}

void AssistantPanel::pollChatGptWindow() {
    ++chatPollCount_;
    const QSet<qulonglong> current = x11ClientWindows();
    const QSet<qulonglong> candidates = current - knownWindowIds_;
    qulonglong matched = 0;
    for (const qulonglong id : candidates) {
        const auto info = readX11WindowInfo(id);
        if (info && isChatGptWindow(*info, remoteChromePid_)) {
            matched = id;
            break;
        }
    }
    if (matched != 0) {
        if (matched == chatCandidateId_) ++chatCandidateSeenCount_;
        else {
            chatCandidateId_ = matched;
            chatCandidateSeenCount_ = 1;
        }
        if (chatCandidateSeenCount_ >= 3) {
            embedChatGptWindow(matched);
            return;
        }
    } else {
        chatCandidateId_ = 0;
        chatCandidateSeenCount_ = 0;
    }
    if (chatPollCount_ >= 100) {
        chatPollTimer_->stop();
        chatButton_->setEnabled(true);
        chatStatus_->setText(tr("15 秒内没有找到 ChatGPT 窗口，请确认 Chrome 可以正常启动。"));
    }
}

void AssistantPanel::embedChatGptWindow(qulonglong windowId) {
    chatPollTimer_->stop();
    const qulonglong host = static_cast<qulonglong>(chatHost_->winId());
    const QSize size = nativePixelSize(chatHost_);
    if (!x11_->reparent(windowId, host, size.width(), size.height())) {
        chatButton_->setEnabled(true);
        chatStatus_->setText(tr("Chrome 窗口父节点校验失败，未能嵌入右侧面板。"));
        return;
    }
    chatWindowId_ = windowId;
    chatStatus_->hide();
    chatButton_->hide();
    chatGuardTimer_->start();
    emit chatGptReady();
    emit viewStateChanged();
}

void AssistantPanel::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
    resizeTimer_->start();
}

bool AssistantPanel::eventFilter(QObject* watched, QEvent* event) {
    if ((watched == chatHost_ || watched == codexHost_) &&
        event->type() == QEvent::Resize) {
        resizeTimer_->start();
    }
    return QWidget::eventFilter(watched, event);
}

void AssistantPanel::pollCodexWindow() {
    ++codexPollCount_;
    codexWindowId_ = x11_->largestChild(
        static_cast<qulonglong>(codexHost_->winId()));
    if (codexWindowId_ != 0) {
        codexPollTimer_->stop();
        resizeEmbeddedWindows();
    } else if (codexPollCount_ >= 50) {
        codexPollTimer_->stop();
    }
}

void AssistantPanel::resizeEmbeddedWindows() {
    if (chatWindowId_ != 0) {
        const QSize size = nativePixelSize(chatHost_);
        x11_->resize(chatWindowId_, size.width(), size.height());
    }
    if (codexWindowId_ != 0) {
        const QSize size = nativePixelSize(codexHost_);
        x11_->resize(codexWindowId_, size.width(), size.height());
    }
}

void AssistantPanel::verifyChatGptWindow() {
    if (chatWindowId_ == 0) return;
    const qulonglong host = static_cast<qulonglong>(chatHost_->winId());
    const qulonglong parent = x11_->parentWindow(chatWindowId_);
    if (parent == host) return;
    if (parent == 0) {
        chatGuardTimer_->stop();
        chatWindowId_ = 0;
        chatStatus_->show();
        chatButton_->show();
        chatButton_->setEnabled(true);
        chatStatus_->setText(tr("ChatGPT 窗口已关闭，点击可重新连接。"));
        return;
    }
    const QSize size = nativePixelSize(chatHost_);
    if (!x11_->reparent(chatWindowId_, host, size.width(), size.height())) {
        chatGuardTimer_->stop();
        chatWindowId_ = 0;
        chatStatus_->show();
        chatButton_->show();
        chatButton_->setEnabled(true);
        chatStatus_->setText(tr("ChatGPT 窗口已脱离面板，点击可重新连接。"));
    }
}

} // namespace scholarvault::ui
