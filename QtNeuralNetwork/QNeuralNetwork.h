#ifndef QNEURALNETWORK_H
#define QNEURALNETWORK_H

#include <QList>
#include "math.h"

namespace Neural {

// ============================================================================================

double sigm(double value)   {
    return 1.0/(1.0 + pow(2.718, -value));
}

int random(int low, int high)   {
    return (qrand() % ((high + 1) - low) + low);
}

// ============================================================================================

struct neuron   {
    QList<double> weight;
    double value;
    double weight_delta;
    double error;
    double bias;

    neuron()    {
        weight.clear();
        value = 0.0;
        weight_delta = 0.0;
        error = 0.0;
        bias = 0.0;
    }
};

// ============================================================================================

struct layer    {
    QList<neuron> neurons;

    layer() { neurons.clear(); }
    layer(int num)  { for(int i = 0; i < num; i++)  { neuron n; neurons.push_back(n); } }
    layer(QList<neuron> list) { neurons = list; }

    void addNeuron(neuron n) { neurons.push_back(n); }
};

// ============================================================================================

class network   {
public:
    network() {}
    network(double rate) { m_rate = rate; }

    void addLayer(layer l) { m_layers.push_back(l); }
    int activate(QList<double>inputs);
    int training(QList<double>inputs, QList<double>outputs, long long iteration);

    void init();
    layer getResult() { if(m_layers.size() > 0) return m_layers.back(); else return layer(); }
    layer getLayer(int index)   {if(index < 0 || index >= m_layers.size()) return layer(); else return m_layers[index]; }
    int getLayersCount() { return m_layers.size(); }

private:
    double m_rate;
    QList<layer> m_layers;

public:


private:
    //layer getLayer(int num) {
    //    if(num >= 0 && num < m_layers.size()) return m_layers[num]; else return layer();
    //}

    void randomizeWeights();
    void correctWeights();
};

int network::activate(QList<double> inputs)
{
    if(m_layers.isEmpty())  {
        return -1;
    }
    if(inputs.size() != m_layers[0].neurons.size()) {
        return -2;
    }

    for(int i = 0; i < inputs.size(); i++)  {
        m_layers[0].neurons[i].value = inputs[i];
    }
    for(int L = 1; L < m_layers.size(); L++)    {
        for(int hidden = 0; hidden < m_layers[L].neurons.size(); hidden++)  {
            double value = 0.0;
            for(int in = 0; in < m_layers[L-1].neurons.size(); in++)    {
                value += m_layers[L-1].neurons[in].value * m_layers[L].neurons[hidden].weight[in];
            }
            m_layers[L].neurons[hidden].value = sigm(value);
        }
    }
    return 0;
}

int network::training(QList<double> inputs, QList<double> outputs, long long iteration)
{
    activate(inputs);
    for(int L = m_layers.size()-1; L > 0; L--)  {
        if(L == m_layers.size()-1)  {
            for(int n = 0; n < m_layers[L].neurons.size(); n++)   {
                m_layers[L].neurons[n].error = m_layers[L].neurons[n].value - outputs[n];
                m_layers[L].neurons[n].weight_delta = m_layers[L].neurons[n].error * (m_layers[L].neurons[n].value * (1.0 - m_layers[L].neurons[n].value));
                for(int w = 0; w < m_layers[L].neurons[n].weight.size(); w++)   {
                    m_layers[L].neurons[n].weight[w] = m_layers[L].neurons[n].weight[w] - m_layers[L-1].neurons[w].value * m_layers[L].neurons[n].weight_delta * m_rate;
                }
            }
        } else {
            for(int n = 0; n < m_layers[L].neurons.size(); n++)   {
                double D = 0.0;
                for(int out = 0; out <= m_layers[L+1].neurons.size(); out++)    {
                    D += m_layers[L+1].neurons[out].weight_delta * m_layers[L+1].neurons[out].weight[n];
                }
                m_layers[L].neurons[n].weight_delta = D * m_layers[L].neurons[n].value * (1.0 - m_layers[L].neurons[n].value);
                for(int w = 0; w < m_layers[L].neurons[n].weight.size(); w++)   {
                    m_layers[L].neurons[n].weight[w] = m_layers[L].neurons[n].weight[w] - m_layers[L-1].neurons[w].value * m_layers[L].neurons[n].weight_delta * m_rate;
                }
            }
        }
    }
    return 0;
}

void network::init()
{
    for(int L = 0; L < m_layers.size(); L++)    {
        for(int n = 0; n < m_layers[L].neurons.size(); n++) {
            m_layers[L].neurons[n].weight.clear();
        }
    }

    for(int L = 0; L < m_layers.size(); L++)    {
        for(int in = 0; in < m_layers[L].neurons.size(); in++) {
            if(L == 0)  {
                m_layers[L].neurons[in].weight.push_back(static_cast<double>(random(0,100))/100.0 - 0.5);
            } else {
                for(int w = 0; w < m_layers[L-1].neurons.size(); w++)   {
                    //m_layers[L-1].neurons[in].weight[w] = static_cast<double>(random(0,100))/100.0 - 0.5;
                    m_layers[L-1].neurons[in].weight.push_back(static_cast<double>(random(0,100))/100.0 - 0.5);
                }
            }
        }
    }
}

void network::randomizeWeights()
{
    for(int L = 1; L < m_layers.size(); L++)    {
        for(int n = 0; n < m_layers[L].neurons.size(); n++) {
            for(int w = 0; w < m_layers[L-1].neurons[n].weight.size(); w++) {
                m_layers[L].neurons[n].weight[w] = static_cast<double>(random(0,100))/100.0 - 0.5;
            }
        }
    }
}

void network::correctWeights()
{
    for(int L = m_layers.size() - 1; L > 0; L--)    {
        if(L == m_layers.size() - 1)    {
            for(int n = 0; n < m_layers[L].neurons.size(); n++) {
                m_layers[L].neurons[n].error = 1.0;
                int val = random(0,2);
                if(val == 1)    {
                    m_layers[L].neurons[n].error *= -1.0;
                }
                m_layers[L].neurons[n].weight_delta = m_layers[L].neurons[n].error * m_layers[L].neurons[n].value * (1.0 - m_layers[L].neurons[n].value);
                for(int w = 0; w < m_layers[L].neurons[n].weight.size(); w++)   {
                    m_layers[L].neurons[n].weight[w] = m_layers[L].neurons[n].weight[w] - m_layers[L-1].neurons[w].value * m_layers[L].neurons[n].weight_delta * m_rate;
                }
            }
        } else {
            for(int n = 0; n < m_layers[L].neurons.size(); n++)   {
                double D = 0.0;
                for(int out = 0; out <= m_layers[L+1].neurons.size(); out++)    {
                    D += m_layers[L+1].neurons[out].weight_delta * m_layers[L+1].neurons[out].weight[n];
                }
                m_layers[L].neurons[n].weight_delta = D * m_layers[L].neurons[n].value * (1.0 - m_layers[L].neurons[n].value);
                for(int w = 0; w < m_layers[L].neurons[n].weight.size(); w++)   {
                    m_layers[L].neurons[n].weight[w] = m_layers[L].neurons[n].weight[w] - m_layers[L-1].neurons[w].value * m_layers[L].neurons[n].weight_delta * m_rate;
                }
            }
        }
    }
}

// ============================================================================================










}



#endif // QNEURALNETWORK_H
