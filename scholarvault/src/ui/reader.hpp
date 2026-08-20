#pragma once

#include "scholarvault/domain.hpp"
#include "ui/pdf_document_view.hpp"

#include <QWidget>

class QFileSystemModel;
class QLabel;
class QPdfDocument;
class QPlainTextEdit;
class QTabWidget;
class QTimer;
class QToolButton;
class QTreeView;

namespace scholarvault::ui {

class ProjectReader final : public QWidget {
    Q_OBJECT

public:
    struct ViewState {
        int tabIndex{0};
        PdfDocumentView::ViewState pdf;
    };
    explicit ProjectReader(QWidget* parent = nullptr);
    void clearProject();
    void openProject(const Project& project);
    void refreshProject(const Project& project);
    void setSourceDownloadActive(bool active);
    void setGitHubDownloadActive(bool active);
    [[nodiscard]] std::optional<Project> currentProject() const { return project_; }
    [[nodiscard]] ViewState viewState() const;
    void restoreViewState(const ViewState& state);

signals:
    void statusMessage(const QString& message);
    void latexPreviewRequested(const QString& sourcePath, const QString& projectId);
    void arxivSourceDownloadRequested();
    void gitHubDownloadRequested();
    void viewStateChanged();

public slots:
    void showLatexPreview(const QString& pdfPath, const QString& sourcePath);

private:
    void saveNotes();
    void updateDownloadActions();

    std::optional<Project> project_;
    QLabel* titleLabel_;
    QTabWidget* tabs_;
    QPdfDocument* pdfDocument_;
    PdfDocumentView* pdfView_;
    QFileSystemModel* fileModel_;
    QTreeView* sourceTree_;
    QPlainTextEdit* notes_;
    QTimer* noteSaveTimer_;
    QToolButton* downloadLatexButton_;
    QToolButton* downloadGitHubButton_;
    bool sourceDownloadActive_{false};
    bool gitHubDownloadActive_{false};
    QPdfDocument* latexDocument_{nullptr};
    PdfDocumentView* latexView_{nullptr};
    QPlainTextEdit* sourceText_{nullptr};
};

} // namespace scholarvault::ui
