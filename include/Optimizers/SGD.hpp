#pragma once
#include "../Optimizer.hpp"

namespace NNCpp
{

namespace Optimizer
{

class SGD: public Optimizer
{
private:
    double m_lr;

public:
    SGD(InitData init, std::shared_ptr<Cmat::Matrix>target): Optimizer(target), m_lr(init.lr)
    {

    }

    std::string OptimizerName() override { return "SGD"; }

    void step(const Cmat::Matrix& grad) override
    {
        *getTarget() -= grad * m_lr;
    }
};

}

}