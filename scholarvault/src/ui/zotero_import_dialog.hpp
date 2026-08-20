#pragma once

#include "scholarvault/zotero.hpp"

#include <QDialog>

#include <optional>
#include <vector>

class QLabel;
class QLineEdit;
class QPushButton;
class QTableWidget;

namespace scholarvault::ui {

class ZoteroCatalogLoader;

class ZoteroImportDialog final : public QDialog {
    Q_OBJECT

public:
    explicit ZoteroImportDialog(QString dataDirectory, QWidget* parent = nullptr,
                                quint16 apiPort = 23119);
    [[nodiscard]] std::optional<ZoteroPaper> selectedPaper() const;

private:
    void setPapers(std::vector<ZoteroPaper> papers, const QString& source);
    void filterRows(const QString& text);
    void updateSelection();
    void acceptSelection();

    QString dataDirectory_;
    ZoteroCatalogLoader* loader_;
    QLineEdit* search_;
    QTableWidget* table_;
    QLabel* status_;
    QPushButton* acceptButton_;
    std::vector<ZoteroPaper> papers_;
    std::optional<std::size_t> selectedIndex_;
};

} // namespace scholarvault::ui
