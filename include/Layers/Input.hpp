#pragma once
#include "../Layer.hpp"
#include "../utils.hpp"

namespace NNCpp
{

namespace Layers
{

class Input: public Layer
{
private:

public:
    Input(int& input, int& output, InitData init)
    {
        input = output = init.unit;
    }

    std::string LayerName() override 
    {
        return "Input";
    }

    void compile([[maybe_unused]]Optimizer::InitData init) override {};

    Cmat::Matrix forward(Cmat::Matrix x) override
    {
        return x;
    }

    Cmat::Matrix backward(Cmat::Matrix x) override
    {
        return x;
    }
};

}

}