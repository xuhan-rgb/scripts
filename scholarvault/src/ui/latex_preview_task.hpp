#pragma once

#include <QObject>
#include <QProcess>

namespace scholarvault::ui {

class LatexPreviewTask final : public QObject {
    Q_OBJECT

public:
    explicit LatexPreviewTask(QString sourcePath, QString projectId,
                              QObject* parent = nullptr);
    ~LatexPreviewTask() override;

    void start();

signals:
    void progress(int value, const QString& status);
    void ready(const QString& pdfPath, const QString& sourcePath);
    void failed(const QString& error);

private:
    void beginCompile(const QString& outputDirectory);
    void startPass();
    void handlePassFinished(int exitCode, QProcess::ExitStatus status);

    QString sourcePath_;
    QString projectId_;
    QString outputDirectory_;
    QString expectedPdfPath_;
    QProcess* process_;
    int pass_{0};
};

} // namespace scholarvault::ui
