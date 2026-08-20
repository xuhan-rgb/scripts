#include "ui/pdf_document_view.hpp"

#include <QApplication>
#include <QClipboard>
#include <QFileInfo>
#include <QMouseEvent>
#include <QPageLayout>
#include <QPageSize>
#include <QPainter>
#include <QPdfDocument>
#include <QPdfWriter>
#include <QSignalSpy>
#include <QScrollBar>
#include <QTemporaryDir>
#include <QTest>
#include <QWheelEvent>

#include <algorithm>

using scholarvault::ui::PdfDocumentView;

class PdfDocumentViewTest final : public QObject {
    Q_OBJECT

private slots:
    void rendersAtDeviceResolutionAndCopiesSelectedText();
    void reloadingSameDocumentInvalidatesRenderedPages();
};

namespace {

void writeSinglePagePdf(const QString& path, const QString& text) {
    QPdfWriter writer(path);
    writer.setResolution(72);
    QPainter painter(&writer);
    painter.drawText(QPointF(72, 100), text);
}

} // namespace

void PdfDocumentViewTest::rendersAtDeviceResolutionAndCopiesSelectedText() {
    QTemporaryDir temporary;
    QVERIFY(temporary.isValid());
    const QString path = temporary.filePath("selectable.pdf");
    {
        QPdfWriter writer(path);
        writer.setResolution(72);
        writer.setPageSize(QPageSize(QPageSize::Letter));
        writer.setPageMargins(QMarginsF(0, 0, 0, 0), QPageLayout::Point);
        QPainter painter(&writer);
        painter.setFont(QFont("DejaVu Sans", 18));
        painter.drawText(QPointF(72, 100), "Selectable ScholarVault text");
    }
    QVERIFY(QFileInfo(path).isFile());

    QPdfDocument document;
    QCOMPARE(document.load(path), QPdfDocument::NoError);
    QCOMPARE(document.status(), QPdfDocument::Ready);
    const QPdfSelection allText = document.getAllText(0);
    QVERIFY(allText.text().contains("Selectable"));
    QVERIFY(!allText.bounds().isEmpty());
    const QPointF textStart = allText.bounds().front().boundingRect().center();
    const QPointF textEnd = allText.bounds().back().boundingRect().center();
    const QPdfSelection directSelection = document.getSelection(0, textStart, textEnd);
    QVERIFY2(directSelection.text().contains("Selectable"),
             qPrintable(directSelection.text()));
    PdfDocumentView view;
    view.resize(700, 600);
    view.setDocument(&document);
    view.setZoomFactor(1.0);
    QSignalSpy renderRequested(&view, &PdfDocumentView::pageRenderRequested);
    view.show();
    view.viewport()->update();
    QTRY_VERIFY_WITH_TIMEOUT(!renderRequested.isEmpty(), 5000);

    const QSize renderedPixels = renderRequested.constFirst().at(1).toSize();
    QVERIFY(renderedPixels.width() >= 1200);
    QVERIFY(renderedPixels.height() >= 1500);

    const qreal pageLeft = (view.viewport()->width() - document.pageSize(0).width()) / 2.0;
    const QPoint start = (QPointF(pageLeft, 16) + textStart).toPoint();
    const QPoint end = (QPointF(pageLeft, 16) + textEnd).toPoint();
    QTest::mousePress(view.viewport(), Qt::LeftButton, Qt::NoModifier, start);
    QMouseEvent move(QEvent::MouseMove, end, end, Qt::NoButton,
                     Qt::LeftButton, Qt::NoModifier);
    QApplication::sendEvent(view.viewport(), &move);
    QTest::mouseRelease(view.viewport(), Qt::LeftButton, Qt::NoModifier, end);
    QTRY_VERIFY_WITH_TIMEOUT(view.selectedText().contains("Selectable"), 3000);

    view.copySelection();
    QCOMPARE(QApplication::clipboard()->text(), view.selectedText());

    view.verticalScrollBar()->setValue(0);
    const QPointF wheelPosition = view.viewport()->rect().center();
    QWheelEvent wheel(wheelPosition, view.viewport()->mapToGlobal(wheelPosition.toPoint()),
                      {}, QPoint(0, -120), Qt::NoButton, Qt::NoModifier,
                      Qt::NoScrollPhase, false);
    QApplication::sendEvent(view.viewport(), &wheel);
    QCOMPARE(view.verticalScrollBar()->value(),
             std::min(120, view.verticalScrollBar()->maximum()));
}

void PdfDocumentViewTest::reloadingSameDocumentInvalidatesRenderedPages() {
    QTemporaryDir temporary;
    QVERIFY(temporary.isValid());
    const QString firstPath = temporary.filePath("first.pdf");
    const QString secondPath = temporary.filePath("second.pdf");
    writeSinglePagePdf(firstPath, "First document");
    writeSinglePagePdf(secondPath, "Second document");

    QPdfDocument document;
    QCOMPARE(document.load(firstPath), QPdfDocument::NoError);
    PdfDocumentView view;
    view.resize(700, 600);
    view.setDocument(&document);
    QSignalSpy renderRequested(&view, &PdfDocumentView::pageRenderRequested);
    view.show();
    QTRY_VERIFY_WITH_TIMEOUT(!renderRequested.isEmpty(), 5000);
    QTest::qWait(500);
    renderRequested.clear();

    document.close();
    QCOMPARE(document.load(secondPath), QPdfDocument::NoError);
    view.viewport()->update();
    QTRY_VERIFY_WITH_TIMEOUT(!renderRequested.isEmpty(), 5000);
}

QTEST_MAIN(PdfDocumentViewTest)

#include "pdf_document_view_test.moc"
