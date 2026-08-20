#include "ui/github_clone_task.hpp"

#include <QProcess>
#include <QRegularExpression>

#include <algorithm>

namespace scholarvault::ui {

GitHubCloneTask::GitHubCloneTask(std::shared_ptr<VaultStorage> storage,
                                 Project project,
                                 GitHubRepositoryReference repository,
                                 QObject* parent)
    : QObject(parent),
      storage_(std::move(storage)),
      project_(std::move(project)),
      repository_(std::move(repository)),
      process_(new QProcess(this)) {
    destinationPath_ = project_.path / "code" / repository_.directoryName();
    temporaryPath_ = project_.path / "code" /
        (".clone-" + repository_.directoryName() + "-" + makeStableId());
    process_->setProcessChannelMode(QProcess::MergedChannels);
    connect(process_, &QProcess::readyReadStandardOutput, this,
            &GitHubCloneTask::handleOutput);
    connect(process_, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this,
            &GitHubCloneTask::handleFinished);
}

GitHubCloneTask::~GitHubCloneTask() {
    if (process_->state() != QProcess::NotRunning) {
        process_->kill();
        process_->waitForFinished(500);
    }
}

void GitHubCloneTask::start() {
    const auto duplicate = std::find_if(project_.repositories.begin(), project_.repositories.end(),
                                        [this](const RelatedRepository& item) {
        const auto parsed = parseGitHubRepository(item.url);
        return parsed && parsed->identityKey() == repository_.identityKey();
    });
    if (duplicate != project_.repositories.end()) {
        fail(tr("该 GitHub 仓库已经关联到当前论文"));
        return;
    }
    if (std::filesystem::exists(destinationPath_) || std::filesystem::exists(temporaryPath_)) {
        fail(tr("代码目标目录已经存在，未执行覆盖"));
        return;
    }
    emit progress(2, tr("正在克隆 %1/%2").arg(QString::fromStdString(repository_.owner),
                                               QString::fromStdString(repository_.name)));
    startProcess("git", {"clone", "--progress", "--filter=blob:none",
                          QString::fromStdString(repository_.normalizedUrl),
                          QString::fromStdString(temporaryPath_.string())}, Stage::Clone);
}

void GitHubCloneTask::startProcess(const QString& program,
                                   const QStringList& arguments,
                                   Stage stage) {
    stage_ = stage;
    output_.clear();
    process_->start(program, arguments);
}

void GitHubCloneTask::handleOutput() {
    const QString chunk = QString::fromUtf8(process_->readAllStandardOutput());
    output_ += chunk;
    if (stage_ != Stage::Clone) return;
    static const QRegularExpression percentage(R"((\d{1,3})%)");
    const auto match = percentage.match(chunk);
    if (match.hasMatch()) {
        const int gitProgress = match.captured(1).toInt();
        emit progress(qBound(2, gitProgress * 8 / 10, 80), tr("正在克隆代码"));
    }
}

void GitHubCloneTask::handleFinished(int exitCode, QProcess::ExitStatus status) {
    handleOutput();
    if (status != QProcess::NormalExit || exitCode != 0) {
        const QString lastLine = output_.trimmed().section('\n', -1);
        fail(lastLine.isEmpty() ? tr("Git 命令执行失败") : lastLine);
        return;
    }
    if (stage_ == Stage::Clone) {
        emit progress(84, tr("正在读取默认分支"));
        startProcess("git", {"-C", QString::fromStdString(temporaryPath_.string()),
                              "rev-parse", "--abbrev-ref", "HEAD"}, Stage::Branch);
        return;
    }
    if (stage_ == Stage::Branch) {
        defaultBranch_ = output_.trimmed();
        emit progress(90, tr("正在固定代码版本"));
        startProcess("git", {"-C", QString::fromStdString(temporaryPath_.string()),
                              "rev-parse", "HEAD"}, Stage::Commit);
        return;
    }
    commitSha_ = output_.trimmed();
    finishClone();
}

void GitHubCloneTask::finishClone() {
    try {
        std::filesystem::rename(temporaryPath_, destinationPath_);
        RelatedRepository related;
        related.url = repository_.normalizedUrl;
        related.owner = repository_.owner;
        related.name = repository_.name;
        related.defaultBranch = defaultBranch_.toStdString();
        related.commitSha = commitSha_.toStdString();
        related.relativePath = std::filesystem::relative(destinationPath_, project_.path).generic_string();
        related.status = "ready";
        project_.repositories.push_back(std::move(related));
        storage_->saveProject(project_);
        emit progress(100, tr("相关代码已固定到 commit %1").arg(commitSha_.left(10)));
        emit succeeded(project_);
    } catch (const std::exception& error) {
        fail(QString::fromUtf8(error.what()));
    }
}

void GitHubCloneTask::fail(const QString& error) {
    std::error_code cleanupError;
    std::filesystem::remove_all(temporaryPath_, cleanupError);
    emit failed(error);
}

} // namespace scholarvault::ui
