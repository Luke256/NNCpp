# pragma once

# include "../../thirdparty/NumCpp/NumCpp.hpp"

# include "../Util.hpp"

namespace NNCpp
{

namespace Optimizer
{

class AbstractOptimizer
{
protected:
public:
    AbstractOptimizer(){}
    ~AbstractOptimizer(){}

    virtual void optimize(std::vector<std::shared_ptr<Layers::AbstractLayer>>&layers, double lr) = 0;
private:
};

};

};