# pragma once

# include "../../thirdparty/NumCpp/NumCpp.hpp"

# include "../Util.hpp"

namespace NNCpp
{

namespace Loss
{

class AbstractLoss
{
protected:
public:
    AbstractLoss(){}
    ~AbstractLoss(){}

    virtual Data forward(const Data& x, const Data& target) = 0;
    virtual Data backward() = 0;
private:
};

};

};