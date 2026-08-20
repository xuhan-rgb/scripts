#pragma once

#include <QAbstractScrollArea>
#include <QImage>
#include <QPdfSelection>

#include <optional>
#include <unordered_map>
#include <vector>

class QPdfDocument;
class QPdfPageRenderer;
class QTimer;

namespace scholarvault::ui {

class PdfDocumentView final : public QAbstractScrollArea {
    Q_OBJECT

public:
    enum class ZoomMode { Custom, FitToWidth };
    struct ViewState {
        ZoomMode zoomMode{ZoomMode::FitToWidth};
        qreal zoomFactor{1.0};
        int page{0};
        qreal pageOffset{0.0};
    };

    explicit PdfDocumentView(QWidget* parent = nullptr);

    void setDocument(QPdfDocument* document);
    [[nodiscard]] QPdfDocument* document() const { return document_; }
    [[nodiscard]] qreal zoomFactor() const { return zoomFactor_; }
    [[nodiscard]] ZoomMode zoomMode() const { return zoomMode_; }
    [[nodiscard]] QString selectedText() const;
    [[nodiscard]] ViewState viewState() const;
    void restoreViewState(const ViewState& state);

public slots:
    void setZoomFactor(qreal factor);
    void fitToWidth();
    void zoomIn();
    void zoomOut();
    void copySelection();

signals:
    void zoomFactorChanged(qreal factor);
    void pageRenderRequested(int page, QSize pixelSize);
    void viewStateChanged();

protected:
    void contextMenuEvent(QContextMenuEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void paintEvent(QPaintEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    void scrollContentsBy(int dx, int dy) override;
    void wheelEvent(QWheelEvent* event) override;

private:
    struct CachedPage {
        QSize pixelSize;
        QImage image;
    };
    struct PagePosition {
        int page{-1};
        QPointF point;
    };

    void clearRenderCache();
    void clearSelection();
    void layoutPages();
    void requestPage(int page, const QSize& pixelSize);
    void pruneCache(int firstVisiblePage, int lastVisiblePage);
    [[nodiscard]] QRectF pageViewportRect(int page) const;
    [[nodiscard]] std::optional<PagePosition> pagePositionAt(const QPointF& position) const;
    [[nodiscard]] QSize renderPixelSize(int page) const;

    QPdfDocument* document_{nullptr};
    QPdfPageRenderer* renderer_;
    QTimer* resizeTimer_;
    ZoomMode zoomMode_{ZoomMode::FitToWidth};
    qreal zoomFactor_{1.0};
    qreal maximumPageWidth_{0.0};
    qreal documentWidth_{0.0};
    qreal documentHeight_{0.0};
    std::vector<QRectF> pageRects_;
    std::unordered_map<int, CachedPage> cache_;
    std::unordered_map<int, QSize> pendingPages_;
    std::unordered_map<quint64, quint64> requestGenerations_;
    quint64 renderGeneration_{0};
    std::optional<QPdfSelection> selection_;
    std::optional<ViewState> pendingViewState_;
    int selectionPage_{-1};
    QPointF selectionStart_;
    bool selecting_{false};
};

} // namespace scholarvault::ui
