#include "ui/pdf_document_view.hpp"

#include <QAction>
#include <QContextMenuEvent>
#include <QKeyEvent>
#include <QMenu>
#include <QMouseEvent>
#include <QPainter>
#include <QPdfDocument>
#include <QPdfDocumentRenderOptions>
#include <QPdfPageRenderer>
#include <QResizeEvent>
#include <QScrollBar>
#include <QTimer>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>

namespace scholarvault::ui {
namespace {

constexpr int DocumentMargin = 16;
constexpr int PageSpacing = 16;
constexpr qreal MinimumZoom = 0.5;
constexpr qreal MaximumZoom = 5.0;
constexpr qreal ZoomStep = 1.2;

} // namespace

PdfDocumentView::PdfDocumentView(QWidget* parent)
    : QAbstractScrollArea(parent),
      renderer_(new QPdfPageRenderer(this)),
      resizeTimer_(new QTimer(this)) {
    setObjectName("pdfDocumentView");
    setFocusPolicy(Qt::StrongFocus);
    setMouseTracking(true);
    viewport()->setCursor(Qt::IBeamCursor);
    viewport()->setAutoFillBackground(false);
    setStyleSheet("QAbstractScrollArea { background: #d7d9dc; border: 0; }");
    renderer_->setRenderMode(QPdfPageRenderer::RenderMode::MultiThreaded);
    resizeTimer_->setSingleShot(true);
    resizeTimer_->setInterval(80);
    connect(resizeTimer_, &QTimer::timeout, this, [this] {
        clearRenderCache();
        layoutPages();
    });
    connect(verticalScrollBar(), &QScrollBar::valueChanged, this,
            [this] { emit viewStateChanged(); });
    connect(renderer_, &QPdfPageRenderer::pageRendered, this,
            [this](int page, QSize pixelSize, const QImage& image,
                   QPdfDocumentRenderOptions, quint64 requestId) {
                const auto generation = requestGenerations_.find(requestId);
                if (generation == requestGenerations_.end()) return;
                const bool currentGeneration = generation->second == renderGeneration_;
                requestGenerations_.erase(generation);
                if (!currentGeneration) return;
                const auto pending = pendingPages_.find(page);
                if (pending == pendingPages_.end() || pending->second != pixelSize) return;
                pendingPages_.erase(pending);
                cache_[page] = CachedPage{pixelSize, image};
                viewport()->update(pageViewportRect(page).toAlignedRect());
            });
}

void PdfDocumentView::setDocument(QPdfDocument* document) {
    if (document_ == document) return;
    if (document_ != nullptr) disconnect(document_, nullptr, this, nullptr);
    document_ = document;
    renderer_->setDocument(document);
    clearSelection();
    clearRenderCache();
    if (document_ != nullptr) {
        connect(document_, &QPdfDocument::statusChanged, this,
                [this](QPdfDocument::Status status) {
                    if (status != QPdfDocument::Status::Ready) clearRenderCache();
                    layoutPages();
                });
        connect(document_, &QPdfDocument::pageCountChanged, this,
                [this](int) { layoutPages(); });
    }
    layoutPages();
}

QString PdfDocumentView::selectedText() const {
    return selection_ ? selection_->text() : QString{};
}

PdfDocumentView::ViewState PdfDocumentView::viewState() const {
    ViewState state{zoomMode_, zoomFactor_, 0, 0.0};
    if (pageRects_.empty()) return state;
    const qreal position = verticalScrollBar()->value();
    for (int page = 0; page < static_cast<int>(pageRects_.size()); ++page) {
        if (position > pageRects_[page].bottom() && page + 1 <
            static_cast<int>(pageRects_.size())) {
            continue;
        }
        state.page = page;
        state.pageOffset = pageRects_[page].height() > 0
            ? std::clamp((position - pageRects_[page].top()) /
                             pageRects_[page].height(), 0.0, 1.0)
            : 0.0;
        break;
    }
    return state;
}

void PdfDocumentView::restoreViewState(const ViewState& state) {
    pendingViewState_ = state;
    zoomMode_ = state.zoomMode;
    zoomFactor_ = std::clamp(state.zoomFactor, MinimumZoom, MaximumZoom);
    clearSelection();
    clearRenderCache();
    layoutPages();
    emit zoomFactorChanged(zoomFactor_);
}

void PdfDocumentView::setZoomFactor(qreal factor) {
    const qreal bounded = std::clamp(factor, MinimumZoom, MaximumZoom);
    if (zoomMode_ == ZoomMode::Custom && qFuzzyCompare(zoomFactor_, bounded)) return;
    zoomMode_ = ZoomMode::Custom;
    zoomFactor_ = bounded;
    clearSelection();
    clearRenderCache();
    layoutPages();
    emit zoomFactorChanged(zoomFactor_);
    emit viewStateChanged();
}

void PdfDocumentView::fitToWidth() {
    zoomMode_ = ZoomMode::FitToWidth;
    clearSelection();
    clearRenderCache();
    layoutPages();
    emit viewStateChanged();
}

void PdfDocumentView::zoomIn() {
    setZoomFactor(zoomFactor_ * ZoomStep);
}

void PdfDocumentView::zoomOut() {
    setZoomFactor(zoomFactor_ / ZoomStep);
}

void PdfDocumentView::copySelection() {
    if (selection_ && selection_->isValid() && !selection_->text().isEmpty()) {
        selection_->copyToClipboard();
    }
}

void PdfDocumentView::contextMenuEvent(QContextMenuEvent* event) {
    QMenu menu(this);
    QAction* copy = menu.addAction(tr("复制所选文字"));
    copy->setShortcut(QKeySequence::Copy);
    copy->setEnabled(!selectedText().isEmpty());
    connect(copy, &QAction::triggered, this, &PdfDocumentView::copySelection);
    menu.exec(event->globalPos());
}

void PdfDocumentView::keyPressEvent(QKeyEvent* event) {
    if (event->matches(QKeySequence::Copy) && !selectedText().isEmpty()) {
        copySelection();
        event->accept();
        return;
    }
    if (event->modifiers().testFlag(Qt::ControlModifier) &&
        (event->key() == Qt::Key_Plus || event->key() == Qt::Key_Equal)) {
        zoomIn();
        event->accept();
        return;
    }
    if (event->modifiers().testFlag(Qt::ControlModifier) &&
        event->key() == Qt::Key_Minus) {
        zoomOut();
        event->accept();
        return;
    }
    if (event->modifiers().testFlag(Qt::ControlModifier) &&
        event->key() == Qt::Key_0) {
        fitToWidth();
        event->accept();
        return;
    }
    QAbstractScrollArea::keyPressEvent(event);
}

void PdfDocumentView::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        const auto position = pagePositionAt(event->position());
        if (position) {
            selectionPage_ = position->page;
            selectionStart_ = position->point;
            selection_.reset();
            selecting_ = true;
            viewport()->update();
            event->accept();
            return;
        }
    }
    QAbstractScrollArea::mousePressEvent(event);
}

void PdfDocumentView::mouseMoveEvent(QMouseEvent* event) {
    if (selecting_ && document_ != nullptr) {
        const auto position = pagePositionAt(event->position());
        if (position && position->page == selectionPage_) {
            QPdfSelection candidate = document_->getSelection(
                selectionPage_, selectionStart_, position->point);
            if (candidate.isValid()) selection_ = std::move(candidate);
            viewport()->update();
        }
        event->accept();
        return;
    }
    viewport()->setCursor(pagePositionAt(event->position())
                              ? Qt::IBeamCursor : Qt::ArrowCursor);
    QAbstractScrollArea::mouseMoveEvent(event);
}

void PdfDocumentView::mouseReleaseEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton && selecting_) {
        selecting_ = false;
        event->accept();
        return;
    }
    QAbstractScrollArea::mouseReleaseEvent(event);
}

void PdfDocumentView::paintEvent(QPaintEvent*) {
    QPainter painter(viewport());
    painter.fillRect(viewport()->rect(), QColor("#d7d9dc"));
    if (document_ == nullptr || pageRects_.empty()) return;

    int firstVisible = -1;
    int lastVisible = -1;
    for (int page = 0; page < static_cast<int>(pageRects_.size()); ++page) {
        const QRectF target = pageViewportRect(page);
        if (!target.intersects(viewport()->rect())) continue;
        if (firstVisible < 0) firstVisible = page;
        lastVisible = page;
        painter.fillRect(target, Qt::white);
        const QSize pixelSize = renderPixelSize(page);
        const auto cached = cache_.find(page);
        if (cached != cache_.end() && cached->second.pixelSize == pixelSize) {
            painter.setRenderHint(QPainter::SmoothPixmapTransform, true);
            painter.drawImage(target, cached->second.image);
        } else {
            requestPage(page, pixelSize);
        }
        painter.setPen(QColor("#aeb3ba"));
        painter.drawRect(target.adjusted(0, 0, -1, -1));
    }

    if (selection_ && selectionPage_ >= 0 &&
        selectionPage_ < static_cast<int>(pageRects_.size())) {
        const QRectF pageRect = pageViewportRect(selectionPage_);
        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor(37, 99, 235, 92));
        for (QPolygonF polygon : selection_->bounds()) {
            for (QPointF& point : polygon) {
                point = pageRect.topLeft() + point * zoomFactor_;
            }
            painter.drawPolygon(polygon);
        }
    }
    if (firstVisible >= 0) pruneCache(firstVisible, lastVisible);
}

void PdfDocumentView::resizeEvent(QResizeEvent* event) {
    QAbstractScrollArea::resizeEvent(event);
    if (zoomMode_ == ZoomMode::FitToWidth) {
        resizeTimer_->start();
    }
}

void PdfDocumentView::scrollContentsBy(int, int) {
    viewport()->update();
}

void PdfDocumentView::wheelEvent(QWheelEvent* event) {
    if (event->modifiers().testFlag(Qt::ControlModifier)) {
        const int delta = !event->pixelDelta().isNull()
            ? event->pixelDelta().y() : event->angleDelta().y();
        if (delta > 0) zoomIn();
        else if (delta < 0) zoomOut();
        event->accept();
        return;
    }

    const QPoint delta = !event->pixelDelta().isNull()
        ? event->pixelDelta() : event->angleDelta();
    if (delta.isNull()) {
        QAbstractScrollArea::wheelEvent(event);
        return;
    }
    if (event->modifiers().testFlag(Qt::ShiftModifier)) {
        horizontalScrollBar()->setValue(horizontalScrollBar()->value() - delta.y());
    } else if (std::abs(delta.y()) >= std::abs(delta.x())) {
        verticalScrollBar()->setValue(verticalScrollBar()->value() - delta.y());
    } else {
        horizontalScrollBar()->setValue(horizontalScrollBar()->value() - delta.x());
    }
    event->accept();
}

void PdfDocumentView::clearRenderCache() {
    ++renderGeneration_;
    cache_.clear();
    pendingPages_.clear();
    requestGenerations_.clear();
    viewport()->update();
}

void PdfDocumentView::clearSelection() {
    selection_.reset();
    selectionPage_ = -1;
    selecting_ = false;
}

void PdfDocumentView::layoutPages() {
    pageRects_.clear();
    maximumPageWidth_ = 0;
    documentWidth_ = 0;
    documentHeight_ = 0;
    if (document_ == nullptr || document_->status() != QPdfDocument::Status::Ready ||
        document_->pageCount() <= 0) {
        horizontalScrollBar()->setRange(0, 0);
        verticalScrollBar()->setRange(0, 0);
        viewport()->update();
        return;
    }

    for (int page = 0; page < document_->pageCount(); ++page) {
        maximumPageWidth_ = std::max(maximumPageWidth_,
                                     document_->pageSize(page).width());
    }
    if (zoomMode_ == ZoomMode::FitToWidth && maximumPageWidth_ > 0) {
        const qreal available = std::max(1, viewport()->width() - 2 * DocumentMargin);
        const qreal fitted = std::clamp(available / maximumPageWidth_,
                                        MinimumZoom, MaximumZoom);
        if (!qFuzzyCompare(zoomFactor_, fitted)) {
            zoomFactor_ = fitted;
            emit zoomFactorChanged(zoomFactor_);
        }
    }

    const qreal scaledMaximumWidth = maximumPageWidth_ * zoomFactor_;
    qreal top = DocumentMargin;
    for (int page = 0; page < document_->pageCount(); ++page) {
        const QSizeF pageSize = document_->pageSize(page) * zoomFactor_;
        const qreal left = DocumentMargin + (scaledMaximumWidth - pageSize.width()) / 2.0;
        pageRects_.push_back(QRectF(QPointF(left, top), pageSize));
        top += pageSize.height() + PageSpacing;
    }
    documentWidth_ = scaledMaximumWidth + 2 * DocumentMargin;
    documentHeight_ = top - PageSpacing + DocumentMargin;
    horizontalScrollBar()->setPageStep(viewport()->width());
    horizontalScrollBar()->setRange(
        0, std::max(0, static_cast<int>(std::ceil(documentWidth_ - viewport()->width()))));
    verticalScrollBar()->setPageStep(viewport()->height());
    verticalScrollBar()->setRange(
        0, std::max(0, static_cast<int>(std::ceil(documentHeight_ - viewport()->height()))));
    if (pendingViewState_) {
        const int page = std::clamp(pendingViewState_->page, 0,
                                    static_cast<int>(pageRects_.size()) - 1);
        const qreal offset = std::clamp(pendingViewState_->pageOffset, 0.0, 1.0);
        verticalScrollBar()->setValue(qRound(pageRects_[page].top() +
                                              pageRects_[page].height() * offset));
        pendingViewState_.reset();
    }
    viewport()->update();
}

void PdfDocumentView::requestPage(int page, const QSize& pixelSize) {
    if (pixelSize.isEmpty() || document_ == nullptr) return;
    const auto pending = pendingPages_.find(page);
    if (pending != pendingPages_.end() && pending->second == pixelSize) return;
    pendingPages_[page] = pixelSize;
    QPdfDocumentRenderOptions options;
    options.setRenderFlags(QPdf::RenderAnnotations | QPdf::RenderOptimizedForLcd);
    const quint64 requestId = renderer_->requestPage(page, pixelSize, options);
    requestGenerations_[requestId] = renderGeneration_;
    emit pageRenderRequested(page, pixelSize);
}

void PdfDocumentView::pruneCache(int firstVisiblePage, int lastVisiblePage) {
    const int minimum = std::max(0, firstVisiblePage - 2);
    const int maximum = lastVisiblePage + 2;
    for (auto iterator = cache_.begin(); iterator != cache_.end();) {
        if (iterator->first < minimum || iterator->first > maximum) {
            iterator = cache_.erase(iterator);
        } else {
            ++iterator;
        }
    }
}

QRectF PdfDocumentView::pageViewportRect(int page) const {
    if (page < 0 || page >= static_cast<int>(pageRects_.size())) return {};
    const qreal centeredOffset = documentWidth_ < viewport()->width()
        ? (viewport()->width() - documentWidth_) / 2.0 : 0.0;
    return pageRects_[page].translated(
        centeredOffset - horizontalScrollBar()->value(),
        -verticalScrollBar()->value());
}

std::optional<PdfDocumentView::PagePosition>
PdfDocumentView::pagePositionAt(const QPointF& position) const {
    if (document_ == nullptr) return std::nullopt;
    for (int page = 0; page < static_cast<int>(pageRects_.size()); ++page) {
        const QRectF rect = pageViewportRect(page);
        if (!rect.contains(position)) continue;
        const QPointF point = (position - rect.topLeft()) / zoomFactor_;
        return PagePosition{page, point};
    }
    return std::nullopt;
}

QSize PdfDocumentView::renderPixelSize(int page) const {
    if (page < 0 || page >= static_cast<int>(pageRects_.size())) return {};
    const qreal scale = std::max<qreal>(1.5, devicePixelRatioF());
    return QSize(std::max(1, qRound(pageRects_[page].width() * scale)),
                 std::max(1, qRound(pageRects_[page].height() * scale)));
}

} // namespace scholarvault::ui
