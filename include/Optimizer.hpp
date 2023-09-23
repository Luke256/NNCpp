#pragma once
#include <string>
#include <memory>
#include "lib/Cmat/Cmat.hpp"

namespace NNCpp
{

namespace Optimizer
{

struct InitData
{
    double lr = 0;
};

class Optimizer
{
private:
    std::shared_ptr<Cmat::Matrix>m_target;
public:
    Optimizer(std::shared_ptr<Cmat::Matrix>target): m_target(target) {}

    virtual std::string OptimizerName() = 0;
    virtual void step(const Cmat::Matrix& grad) = 0;
    void operator()(const Cmat::Matrix& grad)
    {
        step(grad);
    }
protected:
    std::shared_ptr<Cmat::Matrix>getTarget()
    {
        return m_target;
    }
};

}

}