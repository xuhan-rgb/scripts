#include "ui/reader.hpp"

#include "ui/pdf_document_view.hpp"

#include <QFile>
#include <QFileInfo>
#include <QFileSystemModel>
#include <QHBoxLayout>
#include <QLabel>
#include <QPdfDocument>
#include <QPlainTextEdit>
#include <QTabWidget>
#include <QTextStream>
#include <QTimer>
#include <QToolButton>
#include <QTreeView>
#include <QVBoxLayout>

namespace scholarvault::ui {

ProjectReader::ProjectReader(QWidget* parent) : QWidget(parent) {
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    titleLabel_ = new QLabel(tr("选择一个论文项目"), this);
    titleLabel_->setObjectName("projectTitle");
    titleLabel_->setStyleSheet("font-size: 18px; font-weight: 650; padding: 10px 14px;");
    layout->addWidget(titleLabel_);

    tabs_ = new QTabWidget(this);
    connect(tabs_, &QTabWidget::currentChanged, this,
            &ProjectReader::viewStateChanged);
    auto* downloadActions = new QWidget(tabs_);
    auto* downloadLayout = new QHBoxLayout(downloadActions);
    downloadLayout->setContentsMargins(0, 0, 4, 0);
    downloadLayout->setSpacing(4);
    downloadLatexButton_ = new QToolButton(downloadActions);
    downloadLatexButton_->setObjectName("downloadLatexButton");
    downloadLatexButton_->setText(tr("下载 LaTeX"));
    downloadLatexButton_->setToolTip(tr("手动输入 arXiv 链接并按需下载原始 LaTeX"));
    downloadLatexButton_->setAutoRaise(true);
    downloadGitHubButton_ = new QToolButton(downloadActions);
    downloadGitHubButton_->setObjectName("downloadGitHubButton");
    downloadGitHubButton_->setText(tr("下载代码"));
    downloadGitHubButton_->setToolTip(tr("手动输入公开 GitHub 仓库链接并按需下载"));
    downloadGitHubButton_->setAutoRaise(true);
    downloadLayout->addWidget(downloadLatexButton_);
    downloadLayout->addWidget(downloadGitHubButton_);
    tabs_->setCornerWidget(downloadActions, Qt::TopRightCorner);
    connect(downloadLatexButton_, &QToolButton::clicked, this,
            &ProjectReader::arxivSourceDownloadRequested);
    connect(downloadGitHubButton_, &QToolButton::clicked, this,
            &ProjectReader::gitHubDownloadRequested);
    updateDownloadActions();

    pdfDocument_ = new QPdfDocument(this);
    auto* pdfPage = new QWidget(tabs_);
    auto* pdfLayout = new QVBoxLayout(pdfPage);
    pdfLayout->setContentsMargins(0, 0, 0, 0);
    pdfLayout->setSpacing(0);
    auto* controls = new QWidget(pdfPage);
    auto* controlLayout = new QHBoxLayout(controls);
    controlLayout->setContentsMargins(8, 4, 8, 4);
    auto* zoomOut = new QToolButton(controls);
    zoomOut->setObjectName("pdfZoomOutButton");
    zoomOut->setText(QString::fromUtf8("−"));
    zoomOut->setToolTip(tr("缩小（Ctrl+-）"));
    auto* zoomLabel = new QLabel("100%", controls);
    zoomLabel->setObjectName("pdfZoomLabel");
    zoomLabel->setMinimumWidth(54);
    zoomLabel->setAlignment(Qt::AlignCenter);
    auto* zoomIn = new QToolButton(controls);
    zoomIn->setObjectName("pdfZoomInButton");
    zoomIn->setText("+");
    zoomIn->setToolTip(tr("放大（Ctrl++）"));
    auto* fitWidth = new QToolButton(controls);
    fitWidth->setObjectName("pdfFitWidthButton");
    fitWidth->setText(tr("适合宽度"));
    fitWidth->setToolTip(tr("恢复整页宽度；按住 Ctrl 滚轮也可缩放"));
    controlLayout->addStretch();
    controlLayout->addWidget(zoomOut);
    controlLayout->addWidget(zoomLabel);
    controlLayout->addWidget(zoomIn);
    controlLayout->addWidget(fitWidth);
    controlLayout->addStretch();

    pdfView_ = new PdfDocumentView(pdfPage);
    pdfView_->setDocument(pdfDocument_);
    connect(zoomOut, &QToolButton::clicked, pdfView_, &PdfDocumentView::zoomOut);
    connect(zoomIn, &QToolButton::clicked, pdfView_, &PdfDocumentView::zoomIn);
    connect(fitWidth, &QToolButton::clicked, pdfView_, &PdfDocumentView::fitToWidth);
    connect(pdfView_, &PdfDocumentView::zoomFactorChanged, zoomLabel,
            [zoomLabel](qreal factor) {
                zoomLabel->setText(QString::number(qRound(factor * 100)) + "%");
            });
    connect(pdfView_, &PdfDocumentView::viewStateChanged, this,
            &ProjectReader::viewStateChanged);
    pdfLayout->addWidget(controls);
    pdfLayout->addWidget(pdfView_, 1);
    tabs_->addTab(pdfPage, tr("PDF"));

    fileModel_ = new QFileSystemModel(this);
    sourceTree_ = new QTreeView(tabs_);
    sourceTree_->setModel(fileModel_);
    sourceTree_->setAlternatingRowColors(true);
    sourceTree_->setHeaderHidden(false);
    tabs_->addTab(sourceTree_, tr("项目文件"));
    connect(sourceTree_, &QTreeView::doubleClicked, this, [this](const QModelIndex& index) {
        if (!project_) return;
        const QString path = fileModel_->filePath(index);
        const QFileInfo info(path);
        if (!info.isFile() || info.suffix().compare("tex", Qt::CaseInsensitive) != 0) return;
        emit latexPreviewRequested(path, QString::fromStdString(project_->id));
    });

    notes_ = new QPlainTextEdit(tabs_);
    notes_->setPlaceholderText(tr("当前论文的 Markdown 笔记"));
    tabs_->addTab(notes_, tr("笔记"));
    layout->addWidget(tabs_, 1);

    noteSaveTimer_ = new QTimer(this);
    noteSaveTimer_->setSingleShot(true);
    noteSaveTimer_->setInterval(700);
    connect(notes_, &QPlainTextEdit::textChanged, noteSaveTimer_,
            qOverload<>(&QTimer::start));
    connect(noteSaveTimer_, &QTimer::timeout, this, &ProjectReader::saveNotes);
}

ProjectReader::ViewState ProjectReader::viewState() const {
    return {tabs_->currentIndex(), pdfView_->viewState()};
}

void ProjectReader::restoreViewState(const ViewState& state) {
    if (state.tabIndex >= 0 && state.tabIndex < tabs_->count()) {
        tabs_->setCurrentIndex(state.tabIndex);
    }
    pdfView_->restoreViewState(state.pdf);
}

void ProjectReader::showLatexPreview(const QString& pdfPath, const QString& sourcePath) {
    if (latexView_ == nullptr) {
        latexDocument_ = new QPdfDocument(this);
        latexView_ = new PdfDocumentView(tabs_);
        latexView_->setDocument(latexDocument_);
        tabs_->addTab(latexView_, tr("TeX 预览"));

        sourceText_ = new QPlainTextEdit(tabs_);
        sourceText_->setReadOnly(true);
        sourceText_->setLineWrapMode(QPlainTextEdit::NoWrap);
        tabs_->addTab(sourceText_, tr("TeX 源码"));
    }
    QFile source(sourcePath);
    if (source.open(QIODevice::ReadOnly | QIODevice::Text)) {
        sourceText_->setPlainText(QString::fromUtf8(source.readAll()));
    }
    latexDocument_->close();
    latexDocument_->load(pdfPath);
    tabs_->setCurrentWidget(latexView_);
}

void ProjectReader::clearProject() {
    noteSaveTimer_->stop();
    project_.reset();
    pdfDocument_->close();
    notes_->blockSignals(true);
    notes_->clear();
    notes_->blockSignals(false);
    titleLabel_->setText(tr("选择一个论文项目"));
    sourceDownloadActive_ = false;
    gitHubDownloadActive_ = false;
    updateDownloadActions();
}

void ProjectReader::openProject(const Project& project) {
    if (project_ && project_->id == project.id) {
        project_ = project;
        updateDownloadActions();
        return;
    }
    if (project_) saveNotes();
    project_ = project;
    titleLabel_->setText(QString::fromStdString(project.title));

    const auto pdfPath = project.path / project.pdfRelativePath;
    pdfDocument_->close();
    if (std::filesystem::is_regular_file(pdfPath)) {
        pdfDocument_->load(QString::fromStdString(pdfPath.string()));
    } else {
        emit statusMessage(tr("项目PDF不存在：%1").arg(QString::fromStdString(pdfPath.string())));
    }

    fileModel_->setRootPath(QString::fromStdString(project.path.string()));
    sourceTree_->setRootIndex(fileModel_->index(QString::fromStdString(project.path.string())));
    sourceTree_->setColumnWidth(0, 330);

    QFile notesFile(QString::fromStdString((project.path / "notes" / "notes.md").string()));
    notes_->blockSignals(true);
    if (notesFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
        notes_->setPlainText(QString::fromUtf8(notesFile.readAll()));
    } else {
        notes_->clear();
    }
    notes_->blockSignals(false);
    updateDownloadActions();
}

void ProjectReader::refreshProject(const Project& project) {
    if (!project_ || project_->id != project.id) return;
    project_ = project;
    titleLabel_->setText(QString::fromStdString(project.title));
    const auto pdfPath = project.path / project.pdfRelativePath;
    pdfDocument_->close();
    if (std::filesystem::is_regular_file(pdfPath)) {
        pdfDocument_->load(QString::fromStdString(pdfPath.string()));
    } else {
        emit statusMessage(tr("项目PDF不存在：%1")
                               .arg(QString::fromStdString(pdfPath.string())));
    }
    updateDownloadActions();
}

void ProjectReader::setSourceDownloadActive(bool active) {
    sourceDownloadActive_ = active;
    updateDownloadActions();
}

void ProjectReader::setGitHubDownloadActive(bool active) {
    gitHubDownloadActive_ = active;
    updateDownloadActions();
}

void ProjectReader::updateDownloadActions() {
    const bool hasProject = project_.has_value();
    const bool sourceReady = hasProject && project_->sourceStatus == "ready";
    downloadLatexButton_->setText(sourceReady ? tr("LaTeX 已下载")
                                               : tr("下载 LaTeX"));
    downloadLatexButton_->setEnabled(hasProject && !sourceReady &&
                                     !sourceDownloadActive_);
    downloadGitHubButton_->setText(gitHubDownloadActive_ ? tr("代码下载中…")
                                                         : tr("下载代码"));
    downloadGitHubButton_->setEnabled(hasProject && !gitHubDownloadActive_);
}

void ProjectReader::saveNotes() {
    if (!project_) return;
    QFile file(QString::fromStdString((project_->path / "notes" / "notes.md").string()));
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text)) {
        emit statusMessage(tr("无法保存论文笔记"));
        return;
    }
    file.write(notes_->toPlainText().toUtf8());
    emit statusMessage(tr("笔记已保存"));
}

} // namespace scholarvault::ui
