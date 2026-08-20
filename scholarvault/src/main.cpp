#include "ui/main_window.hpp"

#include <QApplication>
#include <QCoreApplication>
#include <QFont>
#include <QIcon>
#include <QStyleFactory>
#include <QTimer>

int main(int argc, char* argv[]) {
    QApplication application(argc, argv);
    QCoreApplication::setOrganizationName("ScholarVault");
    QCoreApplication::setApplicationName("ScholarVault");
    QCoreApplication::setApplicationVersion("0.1.0");
    application.setStyle(QStyleFactory::create("Fusion"));
    application.setFont(QFont("Noto Sans CJK SC", 10));
    application.setWindowIcon(QIcon::fromTheme("scholarvault"));

    scholarvault::ui::MainWindow window;
    window.show();
    QTimer::singleShot(0, &window, [&window] {
        window.raise();
        window.activateWindow();
    });
    if (application.arguments().contains("--smoke-test")) {
        QTimer::singleShot(0, &application, &QCoreApplication::quit);
    }
    return application.exec();
}
