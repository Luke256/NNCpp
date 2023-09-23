#pragma once
#include "../Loss.hpp"

namespace NNCpp
{

namespace Loss
{

class MeanSqure: public Loss
{
private:
    Cmat::Matrix m_back;
public:
    double forward(const Cmat::Matrix& x, const Cmat::Matrix& t) override
    {
        auto err = (x-t);

        m_back = err;

        err *= err * 0.5;

        double res = err.sum() / err.m_height;

        return res;
    }

    Cmat::Matrix backward() override
    {
        return m_back;
    }
};

}

}