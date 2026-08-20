#pragma once

#include "scholarvault/domain.hpp"

#include <QObject>

class QProcess;

namespace scholarvault::ui {

class ChatGptUploadTask final : public QObject {
    Q_OBJECT

public:
    ChatGptUploadTask(Project project, QString profilePath,
                      QObject* parent = nullptr);

    [[nodiscard]] static QString validationError(const Project& project);
    void start();

signals:
    void progress(int value, const QString& status);
    void succeeded();
    void failed(const QString& error);

private:
    void readOutput();
    [[nodiscard]] QString helperPath() const;

    Project project_;
    QString profilePath_;
    QProcess* process_;
    QByteArray outputBuffer_;
    QString finalError_;
    bool successSeen_{false};
};

} // namespace scholarvault::ui
