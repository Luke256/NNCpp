#pragma once
#include "../Layer.hpp"
#include "../utils.hpp"
#include "../Optimizer.hpp"

namespace NNCpp
{

namespace Layers
{

class Dense: public Layer
{
private:
    std::shared_ptr<Cmat::Matrix> m_neuron, m_bias;
    std::shared_ptr<Optimizer::Optimizer>m_opt_neuron, m_opt_bias;
    Cmat::Matrix m_last;
    int m_unit;
    size_t n_cpu = std::thread::hardware_concurrency();
    int m_input, m_output;
    XorShift rnd;
    BS::thread_pool pool;

public:
    Dense([[maybe_unused]]int& input, int& output, InitData init): m_unit(init.unit), pool(n_cpu)
    {
        m_neuron = std::make_shared<Cmat::Matrix>(output, init.unit);
        m_bias = std::make_shared<Cmat::Matrix>(1, init.unit);

        double sigma = sqrt(2.0 / m_unit);

        for (int j = 0; j < init.unit; ++j)
        {
            for (int i = 0; i < output; ++i)
            {
                (*m_neuron)[i][j] = rnd.normal() * sigma;
            }
            (*m_bias)[0][j] = rnd.normal() * sigma;
        }

        m_input = output;
        m_output = init.unit;
        output = init.unit;
    }

    void compile(Optimizer::InitData init) override
    {
        m_opt_neuron = getOptimizer(init, m_neuron);
        m_opt_bias = getOptimizer(init, m_bias);
    }

    std::string LayerName() override 
    {
        return "Dense";
    }

    Cmat::Matrix forward(Cmat::Matrix x) override
    {
        m_last = x;

        x = x.dot(*m_neuron);

        for (int i = 0; i < x.m_height; ++i)
        {
            pool.push_task([&](int i){
                for (int j = 0; j < m_output; ++j)
                {
                    x[i][j] += (*m_bias)[0][j];
                }
            }, i);
        }

        pool.wait_for_tasks();

        return x;
    }

    Cmat::Matrix backward(Cmat::Matrix x) override
    {
        Cmat::Matrix dx = x.dot(m_neuron->transpose());
        Cmat::Matrix dw = m_last.transpose().dot(x);
        Cmat::Matrix db(1, x[0].size());
        for (int i = 0; i < x.m_height; ++i)
        {
            pool.push_task([&](int i){
                for (int j = 0; j < x.m_width; ++j)
                {
                    db[0][j] += x[i][j];
                }
            }, i);
        }
        pool.wait_for_tasks();

        // dw.print();
        m_opt_neuron->step(dw);
        m_opt_bias->step(db);

        return dx;
    }
};

}

}