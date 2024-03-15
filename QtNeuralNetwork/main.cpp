#include <QCoreApplication>
#include <QDebug>

#include <QList>

#include "QNeuralNetwork.h"

#define COUNT   8
double data[COUNT][4] = {
        { 0.0, 0.0, 0.0, 0.0 },
        { 0.0, 0.0, 1.0, 1.0 },
        { 0.0, 1.0, 0.0, 0.0 },
        { 0.0, 1.0, 1.0, 0.0 },
        { 1.0, 0.0, 0.0, 1.0 },
        { 1.0, 0.0, 1.0, 1.0 },
        { 1.0, 1.0, 0.0, 0.0 },
        { 1.0, 1.0, 1.0, 0.0 }
};

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    qDebug() << " Creating a neural network ";
    Neural::network nn(0.1);

    Neural::layer l_in(3);
    Neural::layer l_hidden(2);
    Neural::layer l_out(1);

    nn.addLayer(l_in);
    nn.addLayer(l_hidden);
    nn.addLayer(l_out);

    qDebug() << " Layers count   = " << nn.getLayersCount();
    qDebug() << " Input neurons  = " << nn.getLayer(0).neurons.size();
    qDebug() << " Hidden neurons = " <<  nn.getLayer(1).neurons.size();
    qDebug() << " Output neurons = " <<  nn.getLayer(2).neurons.size();

    nn.init();

    qDebug() << " Network is initialized";
    qDebug() << " Starting training of the network ";
    long i;
    for(int d = 0; d<COUNT; d++)    {
        i = 0;
        do {
            i++;
            nn.training({data[d][0],data[d][1],data[d][2]}, {data[d][3]}, 10000);
        } while(i < 10000);
    }
    qDebug() << " Training is finished";

    return a.exec();
}
