#pragma once
#include <math.h>
#include "../Layer.hpp"
#include "../utils.hpp"

namespace NNCpp
{

namespace Layers 
{

class SoftMax: public Layer
{
private:
    BS::thread_pool pool;
public:
    SoftMax([[maybe_unused]]int& input, [[maybe_unused]]int& output, [[maybe_unused]]InitData init) {}

    std::string LayerName() override { return "SoftMax"; }

    void compile([[maybe_unused]]Optimizer::InitData init) override {}

    Cmat::Matrix forward(Cmat::Matrix x) override
    {
        for (int i = 0; i < x.m_height; ++i)
        {
            pool.push_task([&](int i){
                double sum = 0;
                for (int j = 0; j < x.m_width; ++j) sum += std::exp(x[i][j]);
                for (int j = 0; j < x.m_width; ++j) {
                    if (std::exp(x[i][j]) / (sum+0.0001) > 1) {
                        std::cout << std::exp(x[i][j]) << " " << sum << std::endl;
                        assert(false);
                    }
                    x[i][j] = std::exp(x[i][j]) / (sum+0.0001);
                }
            }, i);
        }

        pool.wait_for_tasks();

        return x;
    }

    Cmat::Matrix backward(Cmat::Matrix x) override
    {
        return x;
    }
};

}

}