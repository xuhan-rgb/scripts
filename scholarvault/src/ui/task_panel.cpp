#include "ui/task_panel.hpp"

#include <QHeaderView>
#include <QLabel>
#include <QProgressBar>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QUuid>
#include <QVBoxLayout>

namespace scholarvault::ui {

TaskPanel::TaskPanel(QWidget* parent) : QWidget(parent) {
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(4, 4, 4, 4);
    tree_ = new QTreeWidget(this);
    tree_->setHeaderLabels({tr("任务"), tr("进度"), tr("状态")});
    tree_->header()->setSectionResizeMode(0, QHeaderView::Stretch);
    tree_->header()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    tree_->header()->setSectionResizeMode(2, QHeaderView::Stretch);
    layout->addWidget(tree_);
}

QString TaskPanel::beginTask(const QString& title, bool indeterminate) {
    const QString id = QUuid::createUuid().toString(QUuid::WithoutBraces);
    auto* item = new QTreeWidgetItem(tree_);
    item->setText(0, title);
    auto* progress = new QProgressBar(tree_);
    progress->setFixedWidth(190);
    progress->setRange(0, indeterminate ? 0 : 100);
    auto* status = new QLabel(tr("准备中"), tree_);
    tree_->setItemWidget(item, 1, progress);
    tree_->setItemWidget(item, 2, status);
    rows_.insert(id, Row{item, progress, status});
    return id;
}

void TaskPanel::updateTask(const QString& id, int progress, const QString& status) {
    if (!rows_.contains(id)) return;
    Row row = rows_.value(id);
    if (row.progress->maximum() != 0) row.progress->setValue(qBound(0, progress, 100));
    row.status->setText(status);
}

void TaskPanel::finishTask(const QString& id, const QString& status) {
    if (!rows_.contains(id)) return;
    Row row = rows_.value(id);
    row.progress->setRange(0, 100);
    row.progress->setValue(100);
    row.status->setText(status);
    row.status->setStyleSheet("color: #15803d; font-weight: 600;");
}

void TaskPanel::failTask(const QString& id, const QString& error) {
    if (!rows_.contains(id)) return;
    Row row = rows_.value(id);
    row.progress->setRange(0, 100);
    row.progress->setValue(0);
    row.status->setText(error);
    row.status->setStyleSheet("color: #b42318; font-weight: 600;");
}

} // namespace scholarvault::ui
