#include "qtui.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    qtui w;
    w.show();
    return a.exec();
}
