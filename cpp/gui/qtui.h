#ifndef QTUI_H
#define QTUI_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE
namespace Ui { class qtui; }
QT_END_NAMESPACE

class qtui : public QMainWindow
{
    Q_OBJECT

public:
    qtui(QWidget *parent = nullptr);
    ~qtui();

private slots:
    void on_actionExit_triggered();

private:
    Ui::qtui *ui;
};
#endif // QTUI_H
