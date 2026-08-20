#pragma once

#include "scholarvault/zotero.hpp"

#include <QDialog>

#include <optional>

class QComboBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QStackedWidget;

namespace scholarvault::ui {

struct NewProjectRequest {
    enum class Source { Arxiv, Pdf, Zotero };
    Source source{Source::Arxiv};
    QString input;
    QString title;
    std::optional<ZoteroPaper> zoteroPaper;
};

class NewProjectDialog final : public QDialog {
    Q_OBJECT

public:
    explicit NewProjectDialog(QWidget* parent = nullptr);
    [[nodiscard]] NewProjectRequest request() const;

private:
    void choosePdf();
    void chooseZoteroDirectory();
    void chooseZoteroPaper();
    void validate();

    QComboBox* source_;
    QStackedWidget* sourcePages_;
    QLineEdit* arxivInput_;
    QLineEdit* pdfInput_;
    QLineEdit* zoteroDirectory_;
    QPushButton* zoteroPaperButton_;
    QLabel* zoteroSelection_;
    QLineEdit* titleInput_;
    QPushButton* acceptButton_;
    std::optional<ZoteroPaper> zoteroPaper_;
};

} // namespace scholarvault::ui
