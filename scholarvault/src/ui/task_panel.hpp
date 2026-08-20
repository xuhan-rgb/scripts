#pragma once

#include <QHash>
#include <QWidget>

class QLabel;
class QProgressBar;
class QTreeWidget;
class QTreeWidgetItem;

namespace scholarvault::ui {

class TaskPanel final : public QWidget {
    Q_OBJECT

public:
    explicit TaskPanel(QWidget* parent = nullptr);

    QString beginTask(const QString& title, bool indeterminate = false);
    void updateTask(const QString& id, int progress, const QString& status);
    void finishTask(const QString& id, const QString& status);
    void failTask(const QString& id, const QString& error);

private:
    struct Row {
        QTreeWidgetItem* item{};
        QProgressBar* progress{};
        QLabel* status{};
    };

    QTreeWidget* tree_;
    QHash<QString, Row> rows_;
};

} // namespace scholarvault::ui
