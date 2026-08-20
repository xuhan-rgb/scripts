#include "ui/chatgpt_upload_task.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QProcess>

namespace scholarvault::ui {

ChatGptUploadTask::ChatGptUploadTask(Project project, QString profilePath,
                                     QObject* parent)
    : QObject(parent), project_(std::move(project)),
      profilePath_(std::move(profilePath)), process_(new QProcess(this)) {
    process_->setProcessChannelMode(QProcess::MergedChannels);
    connect(process_, &QProcess::readyReadStandardOutput, this,
            &ChatGptUploadTask::readOutput);
    connect(process_, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this,
            [this](int exitCode, QProcess::ExitStatus status) {
                readOutput();
                if (successSeen_ && status == QProcess::NormalExit && exitCode == 0) {
                    emit succeeded();
                } else {
                    emit failed(finalError_.isEmpty()
                        ? tr("ChatGPT PDF 上传程序异常结束") : finalError_);
                }
            });
    connect(process_, &QProcess::errorOccurred, this,
            [this](QProcess::ProcessError) {
                if (process_->state() == QProcess::NotRunning && !successSeen_) {
                    emit failed(tr("无法启动 ChatGPT PDF 上传程序"));
                }
            });
}

QString ChatGptUploadTask::validationError(const Project& project) {
    std::error_code error;
    const auto projectPath = std::filesystem::weakly_canonical(project.path, error);
    if (error || projectPath.empty()) return tr("论文项目目录不存在");
    const auto pdfPath = std::filesystem::weakly_canonical(
        project.path / project.pdfRelativePath, error);
    if (error || !std::filesystem::is_regular_file(pdfPath)) {
        return tr("当前论文 PDF 不存在");
    }
    auto projectIterator = projectPath.begin();
    auto pdfIterator = pdfPath.begin();
    for (; projectIterator != projectPath.end(); ++projectIterator, ++pdfIterator) {
        if (pdfIterator == pdfPath.end() || *projectIterator != *pdfIterator) {
            return tr("PDF 路径超出当前论文项目目录");
        }
    }
    if (pdfPath.extension() != ".pdf") return tr("当前论文文件不是 PDF");
    QFile file(QString::fromStdString(pdfPath.string()));
    if (!file.open(QIODevice::ReadOnly) || !file.read(5).startsWith("%PDF-")) {
        return tr("当前论文文件没有有效的 PDF 签名");
    }
    return {};
}

QString ChatGptUploadTask::helperPath() const {
    const QString override = qEnvironmentVariable("SCHOLARVAULT_CHATGPT_UPLOAD_HELPER");
    if (!override.isEmpty()) return override;
    return QDir(QCoreApplication::applicationDirPath())
        .filePath("../libexec/scholarvault-chatgpt-upload");
}

void ChatGptUploadTask::start() {
    const QString error = validationError(project_);
    if (!error.isEmpty()) {
        emit failed(error);
        return;
    }
    const QString helper = helperPath();
    if (!QFileInfo(helper).isExecutable()) {
        emit failed(tr("没有找到 ChatGPT PDF 上传组件"));
        return;
    }
    const auto pdf = std::filesystem::weakly_canonical(
        project_.path / project_.pdfRelativePath);
    process_->start(helper, {"--profile", profilePath_, "--pdf",
                             QString::fromStdString(pdf.string())});
}

void ChatGptUploadTask::readOutput() {
    outputBuffer_ += process_->readAllStandardOutput();
    while (true) {
        const qsizetype newline = outputBuffer_.indexOf('\n');
        if (newline < 0) break;
        const QByteArray line = outputBuffer_.left(newline).trimmed();
        outputBuffer_.remove(0, newline + 1);
        if (line.isEmpty()) continue;
        const QJsonDocument document = QJsonDocument::fromJson(line);
        if (!document.isObject()) {
            finalError_ = tr("ChatGPT 上传组件返回了无效进度数据");
            continue;
        }
        const QJsonObject object = document.object();
        const QString event = object.value("event").toString();
        const QString status = object.value("status").toString();
        const int value = object.value("progress").toInt();
        emit progress(value, status);
        if (event == "succeeded") successSeen_ = true;
        if (event == "failed") finalError_ = status;
    }
}

} // namespace scholarvault::ui
