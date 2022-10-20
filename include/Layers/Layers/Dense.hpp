# pragma once

# include <iostream>

# include "../Abstract.hpp"

namespace NNCpp
{

namespace Layers
{

class Dense : public Layers::AbstractLayer
{
public:
    Dense(uint32 input_n, uint32 unit)
    {
        neuron = nc::multiply(nc::random::randN<double>({input_n, unit}), sqrt(2.0/unit));
        bias = nc::multiply(nc::random::randN<double>({1, unit}), sqrt(2.0/unit));
    }

    Data forward(const Data& x, bool train = false) override
    {
        if (train) m_lastPredicted = x;
        Data c = nc::dot(x, neuron);
        for (int idx=0; double& i : c)
        {
            i -= bias[idx%bias.size()];
            ++idx;
        }
        return c;
    }

    Data backward(const Data& x, const double lr) override
    {
        Data dx = nc::dot(x, neuron.transpose());
        
        d_neuron = nc::dot(m_lastPredicted.transpose(), x);

        d_bias = nc::sum(x, nc::Axis::ROW);

        neuron -= nc::multiply(d_neuron, lr);

        bias -= nc::multiply(d_bias, lr);

        return dx;
    }
private:
    nc::NdArray<double>neuron, bias;
    nc::NdArray<double>d_neuron, d_bias;
    Data m_lastPredicted;
};

};

};