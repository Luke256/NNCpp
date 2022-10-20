# pragma once

# include "../Abstract.hpp"
# include <iostream>

namespace NNCpp
{

namespace Activation
{

class Relu : public Layers::AbstractLayer
{
public:
    Relu()
    {
    }

    Data forward(const Data& x, bool train = false) override
    {
        if (train) m_lastPredicted = x;
        Data c = nc::copy(x);
        for (auto& i : c) i *= (i>0);
        return c;
    }

    Data backward(const Data& x, const double lr) override
    {
        Data m = m_lastPredicted;
        for (auto& i : m) i = (i>0);
        return x*m;
    }
private:
    Data m_lastPredicted;
};

};

};