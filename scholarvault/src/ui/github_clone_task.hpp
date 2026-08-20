#pragma once

#include "scholarvault/domain.hpp"
#include "scholarvault/github.hpp"
#include "scholarvault/storage.hpp"

#include <QObject>
#include <QProcess>

#include <memory>

namespace scholarvault::ui {

class GitHubCloneTask final : public QObject {
    Q_OBJECT

public:
    GitHubCloneTask(std::shared_ptr<VaultStorage> storage,
                    Project project,
                    GitHubRepositoryReference repository,
                    QObject* parent = nullptr);
    ~GitHubCloneTask() override;

    void start();

signals:
    void progress(int value, const QString& status);
    void succeeded(const scholarvault::Project& project);
    void failed(const QString& error);

private:
    enum class Stage { Clone, Branch, Commit };

    void startProcess(const QString& program, const QStringList& arguments, Stage stage);
    void handleOutput();
    void handleFinished(int exitCode, QProcess::ExitStatus status);
    void finishClone();
    void fail(const QString& error);

    std::shared_ptr<VaultStorage> storage_;
    Project project_;
    GitHubRepositoryReference repository_;
    QProcess* process_;
    Stage stage_{Stage::Clone};
    std::filesystem::path temporaryPath_;
    std::filesystem::path destinationPath_;
    QString output_;
    QString defaultBranch_;
    QString commitSha_;
};

} // namespace scholarvault::ui
