#include "qtui.h"
#include "./ui_qtui.h"

qtui::qtui(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::qtui)
{
    ui->setupUi(this);
}

qtui::~qtui()
{
    delete ui;
}


void qtui::on_actionExit_triggered()
{
    QApplication::quit();
}

