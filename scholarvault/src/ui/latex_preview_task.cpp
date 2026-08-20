#include "ui/latex_preview_task.hpp"

#include <QCryptographicHash>
#include <QDir>
#include <QDirIterator>
#include <QFileInfo>
#include <QFutureWatcher>
#include <QProcess>
#include <QStandardPaths>
#include <QtConcurrent>

#include <algorithm>

namespace scholarvault::ui {

LatexPreviewTask::LatexPreviewTask(QString sourcePath, QString projectId, QObject* parent)
    : QObject(parent),
      sourcePath_(std::move(sourcePath)),
      projectId_(std::move(projectId)),
      process_(new QProcess(this)) {
    process_->setProcessChannelMode(QProcess::MergedChannels);
    connect(process_, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this,
            &LatexPreviewTask::handlePassFinished);
}

LatexPreviewTask::~LatexPreviewTask() {
    if (process_->state() != QProcess::NotRunning) {
        process_->kill();
        process_->waitForFinished(500);
    }
}

void LatexPreviewTask::start() {
    emit progress(3, tr("正在计算 TeX 源码指纹"));
    auto* watcher = new QFutureWatcher<QString>(this);
    connect(watcher, &QFutureWatcher<QString>::finished, this, [this, watcher] {
        const QString directory = watcher->result();
        watcher->deleteLater();
        beginCompile(directory);
    });
    const QString source = sourcePath_;
    const QString project = projectId_;
    watcher->setFuture(QtConcurrent::run([source, project] {
        const QFileInfo selected(source);
        const QString root = selected.absolutePath();
        QStringList files;
        QDirIterator iterator(root, QDir::Files | QDir::NoDotAndDotDot,
                              QDirIterator::Subdirectories);
        while (iterator.hasNext()) files.push_back(iterator.next());
        std::sort(files.begin(), files.end());

        QCryptographicHash hash(QCryptographicHash::Sha256);
        hash.addData("scholarvault-xelatex-cache-v1");
        const QDir rootDirectory(root);
        for (const QString& path : files) {
            const QFileInfo info(path);
            hash.addData(rootDirectory.relativeFilePath(path).toUtf8());
            hash.addData(QByteArray::number(info.size()));
            hash.addData(QByteArray::number(info.lastModified().toMSecsSinceEpoch()));
        }
        hash.addData(selected.fileName().toUtf8());
        const QString cacheRoot = QDir(
            QStandardPaths::writableLocation(QStandardPaths::CacheLocation))
            .filePath("tex/" + project + "/" + QString::fromLatin1(hash.result().toHex()));
        return cacheRoot;
    }));
}

void LatexPreviewTask::beginCompile(const QString& outputDirectory) {
    outputDirectory_ = outputDirectory;
    const QFileInfo source(sourcePath_);
    expectedPdfPath_ = QDir(outputDirectory_).filePath(source.completeBaseName() + ".pdf");
    if (QFileInfo(expectedPdfPath_).isFile()) {
        emit progress(100, tr("已复用 TeX 预览缓存"));
        emit ready(expectedPdfPath_, sourcePath_);
        return;
    }
    if (QStandardPaths::findExecutable("xelatex").isEmpty()) {
        emit failed(tr("未找到 xelatex，无法生成 TeX 预览"));
        return;
    }
    if (!QDir().mkpath(outputDirectory_)) {
        emit failed(tr("无法创建 TeX 预览缓存目录"));
        return;
    }
    pass_ = 0;
    startPass();
}

void LatexPreviewTask::startPass() {
    ++pass_;
    emit progress(pass_ == 1 ? 20 : 65,
                  tr("正在运行 XeLaTeX（%1/2）").arg(pass_));
    const QFileInfo source(sourcePath_);
    process_->setWorkingDirectory(source.absolutePath());
    process_->start("xelatex",
                    {"-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
                     "-synctex=1", "-output-directory=" + outputDirectory_,
                     source.fileName()});
}

void LatexPreviewTask::handlePassFinished(int exitCode, QProcess::ExitStatus status) {
    const QString output = QString::fromUtf8(process_->readAllStandardOutput());
    if (status != QProcess::NormalExit || exitCode != 0) {
        const QStringList lines = output.split('\n');
        const qsizetype firstLine =
            std::max<qsizetype>(0, lines.size() - static_cast<qsizetype>(12));
        emit failed(tr("XeLaTeX 编译失败：\n%1")
                        .arg(lines.sliced(firstLine).join('\n').trimmed()));
        return;
    }
    if (pass_ < 2) {
        startPass();
        return;
    }
    if (!QFileInfo(expectedPdfPath_).isFile()) {
        emit failed(tr("XeLaTeX 已结束，但没有生成 PDF"));
        return;
    }
    emit progress(100, tr("TeX 预览已生成"));
    emit ready(expectedPdfPath_, sourcePath_);
}

} // namespace scholarvault::ui
