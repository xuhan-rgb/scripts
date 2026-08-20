#include "ui/assistant_panel.hpp"

#include <QApplication>
#include <QFile>
#include <QTemporaryDir>
#include <QTest>
#include <QTimer>

using scholarvault::ui::AssistantPanel;

class AssistantPanelTest final : public QObject {
    Q_OBJECT

private slots:
    void nativePixelSizeUsesDeviceScale();
    void codexSessionNameUsesStableProjectId();
    void hostResizeSchedulesEmbeddedWindowResize();
};

void AssistantPanelTest::codexSessionNameUsesStableProjectId() {
    QTemporaryDir temporary;
    QVERIFY(temporary.isValid());
    QFile file(temporary.filePath("project.json"));
    QVERIFY(file.open(QIODevice::WriteOnly));
    file.write("{\"id\":\"paper:123/unsafe\"}");
    file.close();
    QCOMPARE(AssistantPanel::codexSessionName(temporary.path().toStdString()),
             QString("scholarvault-paper-123-unsafe"));
}

void AssistantPanelTest::nativePixelSizeUsesDeviceScale() {
    QWidget widget;
    widget.resize(400, 320);
    widget.show();
    QTest::qWait(20);

    QCOMPARE(widget.devicePixelRatioF(), 1.25);
    QCOMPARE(AssistantPanel::nativePixelSize(&widget), QSize(500, 400));
}

void AssistantPanelTest::hostResizeSchedulesEmbeddedWindowResize() {
    AssistantPanel panel;
    panel.resize(600, 700);
    panel.show();
    QTest::qWait(100);

    auto* host = panel.findChild<QWidget*>("chatGptNativeHost");
    QVERIFY(host != nullptr);
    QTimer* resizeTimer = nullptr;
    for (auto* timer : panel.findChildren<QTimer*>()) {
        if (timer->isSingleShot() && timer->interval() == 60) {
            resizeTimer = timer;
            break;
        }
    }
    QVERIFY(resizeTimer != nullptr);
    resizeTimer->stop();
    host->resize(host->width() - 25, host->height() - 25);
    QApplication::processEvents();
    QVERIFY(resizeTimer->isActive());
}

QTEST_MAIN(AssistantPanelTest)

#include "assistant_panel_test.moc"
