#include "ui/zotero_import_dialog.hpp"
#include "ui/zotero_catalog_loader.hpp"

#include <QDialogButtonBox>
#include <QFileInfo>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QTableWidget>
#include <QVBoxLayout>

namespace scholarvault::ui {

ZoteroImportDialog::ZoteroImportDialog(QString dataDirectory, QWidget* parent,
                                       quint16 apiPort)
    : QDialog(parent),
      dataDirectory_(QFileInfo(dataDirectory).absoluteFilePath()),
      loader_(new ZoteroCatalogLoader(dataDirectory_, this, apiPort)) {
    setWindowTitle(tr("从 Zotero 导入论文"));
    resize(980, 620);
    auto* layout = new QVBoxLayout(this);

    auto* directory = new QLabel(
        tr("Zotero 数据目录：%1").arg(dataDirectory_), this);
    directory->setTextInteractionFlags(Qt::TextSelectableByMouse);
    layout->addWidget(directory);

    search_ = new QLineEdit(this);
    search_->setObjectName("zoteroSearchInput");
    search_->setPlaceholderText(tr("搜索标题、作者、年份或 PDF 文件名"));
    search_->setClearButtonEnabled(true);
    layout->addWidget(search_);

    table_ = new QTableWidget(this);
    table_->setObjectName("zoteroPaperTable");
    table_->setColumnCount(4);
    table_->setHorizontalHeaderLabels({tr("标题"), tr("作者"), tr("年份"),
                                       tr("PDF 文件")});
    table_->setSelectionBehavior(QAbstractItemView::SelectRows);
    table_->setSelectionMode(QAbstractItemView::SingleSelection);
    table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    table_->setAlternatingRowColors(true);
    table_->verticalHeader()->hide();
    table_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    table_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    table_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    table_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    layout->addWidget(table_, 1);

    status_ = new QLabel(tr("正在连接本机 Zotero…"), this);
    status_->setObjectName("zoteroStatusLabel");
    status_->setStyleSheet("color: #667085;");
    layout->addWidget(status_);

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Cancel | QDialogButtonBox::Ok,
                                         this);
    acceptButton_ = buttons->button(QDialogButtonBox::Ok);
    acceptButton_->setText(tr("导入所选论文"));
    acceptButton_->setEnabled(false);
    layout->addWidget(buttons);

    connect(search_, &QLineEdit::textChanged, this, &ZoteroImportDialog::filterRows);
    connect(table_, &QTableWidget::itemSelectionChanged, this,
            &ZoteroImportDialog::updateSelection);
    connect(table_, &QTableWidget::cellDoubleClicked, this,
            [this](int, int) { acceptSelection(); });
    connect(buttons, &QDialogButtonBox::accepted, this,
            &ZoteroImportDialog::acceptSelection);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    connect(loader_, &ZoteroCatalogLoader::progress, status_, &QLabel::setText);
    connect(loader_, &ZoteroCatalogLoader::loaded, this,
            [this](const std::vector<ZoteroPaper>& papers, const QString& source) {
                setPapers(papers, source);
            });
    connect(loader_, &ZoteroCatalogLoader::failed, status_, &QLabel::setText);
    loader_->start();
}

void ZoteroImportDialog::setPapers(std::vector<ZoteroPaper> papers,
                                   const QString& source) {
    papers_ = std::move(papers);
    table_->setRowCount(static_cast<int>(papers_.size()));
    for (std::size_t index = 0; index < papers_.size(); ++index) {
        const auto& paper = papers_[index];
        const int row = static_cast<int>(index);
        auto* title = new QTableWidgetItem(QString::fromStdString(paper.title));
        title->setData(Qt::UserRole, static_cast<qulonglong>(index));
        table_->setItem(row, 0, title);
        table_->setItem(row, 1, new QTableWidgetItem(QString::fromStdString(paper.authors)));
        table_->setItem(row, 2, new QTableWidgetItem(QString::fromStdString(paper.year)));
        table_->setItem(row, 3, new QTableWidgetItem(
            QString::fromStdString(paper.pdfPath.filename().string())));
    }
    status_->setText(tr("通过%1读取到 %2 篇带本地 PDF 的论文")
                         .arg(source).arg(papers_.size()));
    filterRows(search_->text());
}

void ZoteroImportDialog::filterRows(const QString& text) {
    const QString needle = text.trimmed();
    int visible = 0;
    for (int row = 0; row < table_->rowCount(); ++row) {
        bool match = needle.isEmpty();
        for (int column = 0; !match && column < table_->columnCount(); ++column) {
            const auto* item = table_->item(row, column);
            match = item != nullptr && item->text().contains(needle, Qt::CaseInsensitive);
        }
        table_->setRowHidden(row, !match);
        if (match) ++visible;
    }
    if (!papers_.empty()) {
        status_->setToolTip(tr("当前筛选显示 %1 篇").arg(visible));
    }
    updateSelection();
}

void ZoteroImportDialog::updateSelection() {
    selectedIndex_.reset();
    const auto rows = table_->selectionModel()->selectedRows();
    if (!rows.isEmpty() && !table_->isRowHidden(rows.front().row())) {
        selectedIndex_ = table_->item(rows.front().row(), 0)
                             ->data(Qt::UserRole).toULongLong();
    }
    acceptButton_->setEnabled(selectedIndex_.has_value());
}

void ZoteroImportDialog::acceptSelection() {
    updateSelection();
    if (selectedIndex_ && *selectedIndex_ < papers_.size()) accept();
}

std::optional<ZoteroPaper> ZoteroImportDialog::selectedPaper() const {
    if (!selectedIndex_ || *selectedIndex_ >= papers_.size()) return std::nullopt;
    return papers_[*selectedIndex_];
}

} // namespace scholarvault::ui
