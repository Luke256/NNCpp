#pragma once
#include "lib/Cmat/Cmat.hpp"

namespace NNCpp
{

namespace Loss
{

class Loss
{
public:
    virtual double forward(const Cmat::Matrix& x, const Cmat::Matrix& t) = 0;
    virtual Cmat::Matrix backward() = 0;
};

}

}