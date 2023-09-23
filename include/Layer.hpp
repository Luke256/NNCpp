#pragma once
#include <memory>
#include <string>
#include <functional>
#include "lib/Cmat/Cmat.hpp"
#include "Optimizer.hpp"

namespace NNCpp
{

namespace Layers
{

struct InitData
{
    int unit;
};

class Layer
{

public:
virtual std::string LayerName() = 0;
virtual Cmat::Matrix forward(Cmat::Matrix x) = 0;
virtual Cmat::Matrix backward(Cmat::Matrix x) = 0;
virtual void compile(Optimizer::InitData init) = 0;
template<class Opt>
void compile(Optimizer::InitData init) {
    getOptimizer = [&](Optimizer::InitData init, std::shared_ptr<Cmat::Matrix>target)
    { 
        return std::make_shared<Opt>(init, target);
    };
    compile(init);
}

protected:
std::function<std::shared_ptr<Optimizer::Optimizer>(Optimizer::InitData, std::shared_ptr<Cmat::Matrix>)>getOptimizer;

};

}

}