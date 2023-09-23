#pragma once
#include "../Layer.hpp"
#include "../utils.hpp"

namespace NNCpp
{

namespace Layers 
{

class ReLU: public Layer
{
private:
    BS::thread_pool pool;
    Cmat::Matrix m_last;
public:
    ReLU([[maybe_unused]]int& input, [[maybe_unused]]int& output, [[maybe_unused]]InitData init)
    {

    }

    std::string LayerName() override { return "ReLU"; }

    void compile([[maybe_unused]]Optimizer::InitData init) override {}

    Cmat::Matrix forward(Cmat::Matrix x) override
    {
        m_last = x;

        for (int i = 0; i < x.m_height; ++i)
        {
            pool.push_task([&](int i){
                for (int j = 0; j < x.m_width; ++j)
                {
                    x[i][j] = std::max(0.0, x[i][j]);
                }
            }, i);
        }

        pool.wait_for_tasks();

        return x;
    }

    Cmat::Matrix backward(Cmat::Matrix x) override
    {
        for (int i = 0; i < x.m_height; ++i)
        {
            pool.push_task([&](int i){
                for (int j = 0; j < x.m_width; ++j)
                {
                    x[i][j] = x[i][j] * (m_last[i][j] >= 0);
                }
            }, i);
        }

        pool.wait_for_tasks();

        return x;
    }
};

}

}