#include "ui/project_dialogs.hpp"

#include "scholarvault/arxiv.hpp"
#include "ui/zotero_import_dialog.hpp"

#include <QComboBox>
#include <QDir>
#include <QDialogButtonBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QStackedWidget>
#include <QVBoxLayout>

namespace scholarvault::ui {

NewProjectDialog::NewProjectDialog(QWidget* parent) : QDialog(parent) {
    setWindowTitle(tr("新建论文项目"));
    resize(620, 250);
    auto* layout = new QVBoxLayout(this);
    auto* form = new QFormLayout();

    source_ = new QComboBox(this);
    source_->addItem(tr("从 arXiv 链接创建"));
    source_->addItem(tr("从本地 PDF 创建"));
    source_->addItem(tr("从 Zotero 导入"));
    form->addRow(tr("项目来源"), source_);

    sourcePages_ = new QStackedWidget(this);
    auto* arxivPage = new QWidget(sourcePages_);
    auto* arxivLayout = new QVBoxLayout(arxivPage);
    arxivLayout->setContentsMargins(0, 0, 0, 0);
    arxivInput_ = new QLineEdit(arxivPage);
    arxivInput_->setPlaceholderText("https://arxiv.org/abs/2504.16054");
    arxivLayout->addWidget(arxivInput_);
    sourcePages_->addWidget(arxivPage);

    auto* pdfPage = new QWidget(sourcePages_);
    auto* pdfLayout = new QHBoxLayout(pdfPage);
    pdfLayout->setContentsMargins(0, 0, 0, 0);
    pdfInput_ = new QLineEdit(pdfPage);
    pdfInput_->setObjectName("pdfPathInput");
    pdfInput_->setPlaceholderText(tr("可直接粘贴 PDF 完整路径"));
    pdfInput_->setClearButtonEnabled(true);
    auto* browse = new QPushButton(tr("选择 PDF…"), pdfPage);
    pdfLayout->addWidget(pdfInput_, 1);
    pdfLayout->addWidget(browse);
    sourcePages_->addWidget(pdfPage);

    auto* zoteroPage = new QWidget(sourcePages_);
    auto* zoteroLayout = new QVBoxLayout(zoteroPage);
    zoteroLayout->setContentsMargins(0, 0, 0, 0);
    auto* directoryLayout = new QHBoxLayout();
    directoryLayout->setContentsMargins(0, 0, 0, 0);
    zoteroDirectory_ = new QLineEdit(zoteroPage);
    zoteroDirectory_->setObjectName("zoteroDirectoryInput");
    zoteroDirectory_->setPlaceholderText(tr("Zotero 数据目录，例如 /home/qwer/Zotero"));
    const QString defaultZotero = QDir::home().filePath("Zotero");
    if (QFileInfo::exists(QDir(defaultZotero).filePath("zotero.sqlite"))) {
        zoteroDirectory_->setText(defaultZotero);
    }
    auto* browseZotero = new QPushButton(tr("选择目录…"), zoteroPage);
    directoryLayout->addWidget(zoteroDirectory_, 1);
    directoryLayout->addWidget(browseZotero);
    zoteroLayout->addLayout(directoryLayout);
    zoteroPaperButton_ = new QPushButton(tr("选择 Zotero 论文…"), zoteroPage);
    zoteroPaperButton_->setObjectName("chooseZoteroPaperButton");
    zoteroSelection_ = new QLabel(tr("尚未选择论文"), zoteroPage);
    zoteroSelection_->setWordWrap(true);
    zoteroSelection_->setStyleSheet("color: #667085;");
    zoteroLayout->addWidget(zoteroPaperButton_);
    zoteroLayout->addWidget(zoteroSelection_);
    sourcePages_->addWidget(zoteroPage);
    form->addRow(tr("论文"), sourcePages_);

    titleInput_ = new QLineEdit(this);
    titleInput_->setPlaceholderText(tr("可选；arXiv 会自动读取标题，PDF 默认使用文件名"));
    form->addRow(tr("项目标题"), titleInput_);
    layout->addLayout(form);

    auto* hint = new QLabel(tr("项目将复制到当前话题的真实磁盘目录；原始 PDF 不会被移动或修改。"), this);
    hint->setWordWrap(true);
    hint->setStyleSheet("color: #667085;");
    layout->addWidget(hint);

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Cancel | QDialogButtonBox::Ok,
                                         this);
    acceptButton_ = buttons->button(QDialogButtonBox::Ok);
    acceptButton_->setObjectName("createProjectButton");
    acceptButton_->setText(tr("创建项目"));
    layout->addWidget(buttons);

    connect(source_, qOverload<int>(&QComboBox::currentIndexChanged), sourcePages_,
            &QStackedWidget::setCurrentIndex);
    connect(source_, qOverload<int>(&QComboBox::currentIndexChanged), this,
            &NewProjectDialog::validate);
    connect(arxivInput_, &QLineEdit::textChanged, this, &NewProjectDialog::validate);
    connect(pdfInput_, &QLineEdit::textChanged, this, &NewProjectDialog::validate);
    connect(browse, &QPushButton::clicked, this, &NewProjectDialog::choosePdf);
    connect(zoteroDirectory_, &QLineEdit::textChanged, this, [this] {
        if (zoteroPaper_ &&
            QFileInfo(QString::fromStdString(zoteroPaper_->dataDirectory.string())) !=
                QFileInfo(zoteroDirectory_->text().trimmed())) {
            zoteroPaper_.reset();
            zoteroSelection_->setText(tr("目录已改变，请重新选择论文"));
        }
        validate();
    });
    connect(browseZotero, &QPushButton::clicked, this,
            &NewProjectDialog::chooseZoteroDirectory);
    connect(zoteroPaperButton_, &QPushButton::clicked, this,
            &NewProjectDialog::chooseZoteroPaper);
    connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    validate();
}

void NewProjectDialog::choosePdf() {
    const QString path = QFileDialog::getOpenFileName(this, tr("选择论文 PDF"), {},
                                                      tr("PDF 文件 (*.pdf)"));
    if (path.isEmpty()) return;
    pdfInput_->setText(path);
    if (titleInput_->text().trimmed().isEmpty()) {
        titleInput_->setText(QFileInfo(path).completeBaseName());
    }
}

void NewProjectDialog::chooseZoteroDirectory() {
    const QString path = QFileDialog::getExistingDirectory(
        this, tr("选择 Zotero 数据目录"), zoteroDirectory_->text().trimmed());
    if (!path.isEmpty()) zoteroDirectory_->setText(path);
}

void NewProjectDialog::chooseZoteroPaper() {
    const QString directory = zoteroDirectory_->text().trimmed();
    if (!QFileInfo(QDir(directory).filePath("zotero.sqlite")).isFile()) {
        QMessageBox::warning(this, tr("Zotero 目录无效"),
                             tr("所选目录中没有 zotero.sqlite。"));
        return;
    }
    ZoteroImportDialog dialog(directory, this);
    if (dialog.exec() != QDialog::Accepted) return;
    zoteroPaper_ = dialog.selectedPaper();
    if (!zoteroPaper_) return;
    zoteroSelection_->setText(
        tr("已选择：%1\n%2  %3")
            .arg(QString::fromStdString(zoteroPaper_->title),
                 QString::fromStdString(zoteroPaper_->authors),
                 QString::fromStdString(zoteroPaper_->year)));
    if (titleInput_->text().trimmed().isEmpty()) {
        titleInput_->setText(QString::fromStdString(zoteroPaper_->title));
    }
    validate();
}

void NewProjectDialog::validate() {
    bool valid = false;
    if (source_->currentIndex() == 0) {
        valid = parseArxivReference(arxivInput_->text().trimmed().toStdString()).has_value();
    } else {
        if (source_->currentIndex() == 1) {
            const QFileInfo pdf(pdfInput_->text().trimmed());
            valid = pdf.isFile() && pdf.suffix().compare("pdf", Qt::CaseInsensitive) == 0;
        } else {
            valid = zoteroPaper_.has_value() &&
                    QFileInfo(QString::fromStdString(zoteroPaper_->pdfPath.string())).isFile();
        }
    }
    acceptButton_->setEnabled(valid);
}

NewProjectRequest NewProjectDialog::request() const {
    NewProjectRequest result;
    if (source_->currentIndex() == 0) result.source = NewProjectRequest::Source::Arxiv;
    else if (source_->currentIndex() == 1) result.source = NewProjectRequest::Source::Pdf;
    else result.source = NewProjectRequest::Source::Zotero;
    if (result.source == NewProjectRequest::Source::Arxiv) {
        result.input = arxivInput_->text().trimmed();
    } else if (result.source == NewProjectRequest::Source::Pdf) {
        result.input = pdfInput_->text().trimmed();
    } else {
        result.zoteroPaper = zoteroPaper_;
    }
    result.title = titleInput_->text().trimmed();
    return result;
}

} // namespace scholarvault::ui
