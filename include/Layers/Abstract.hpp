# pragma once

# include "../../thirdparty/NumCpp/NumCpp.hpp"

# include "../Util.hpp"

namespace NNCpp
{

namespace Layers
{

class AbstractLayer
{
protected:
public:
    AbstractLayer(){}
    ~AbstractLayer(){}

    virtual Data forward(const Data& x, bool train = false) = 0;
    virtual Data backward(const Data& x, const double lr) = 0;
private:
};

};

};