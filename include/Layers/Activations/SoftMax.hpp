# pragma once

# include "../Abstract.hpp"
# include <iostream>

namespace NNCpp
{

namespace Activation
{

class SoftMax : public Layers::AbstractLayer
{
public:
    SoftMax()
    {
    }

    Data forward(const Data& x, bool train = false) override
    {
        double c = nc::max(x)[0];
        Data exp_x = nc::exp(x-c);
        Data sum_exp = nc::sum(exp_x, nc::Axis::COL);
        for(int i = 0; i < exp_x.shape().rows; ++i)
        {
            for (int j = 0; j < exp_x.shape().cols; ++j)
            {
                exp_x(i,j) /= sum_exp[i];
            }
        }
        if (train) m_lastRes = exp_x;
        return exp_x;
    }

    Data backward(const Data& x, const double lr) override
    {
        Data s = nc::sum(nc::multiply(m_lastRes, x), nc::Axis::COL);
        Data a = x;
        for (int i = 0; i < a.numRows(); ++i)
        {
            for (int j = 0; j < a.numCols(); ++j)
            {
                a(i,j) -= s[i];
            }
        }
        return m_lastRes*a;
    }
private:
    Data m_lastRes;
};

};

};