#pragma once
#include <vector>
#include <memory>
#include <assert.h>
#include "lib/Cmat/Cmat.hpp"
#include "Layer.hpp"
#include "utils.hpp"
#include "Optimizer.hpp"
#include "Loss.hpp"

namespace NNCpp
{

struct SequentialModel
{

    std::vector<std::shared_ptr<Layers::Layer>>m_layers;
    std::shared_ptr<Loss::Loss>m_loss;
    int m_input=0, m_output=0;
    bool compiled = false;

    SequentialModel() {}
    ~SequentialModel() {}

    template<class T>

    void addLayer(Layers::InitData init)
    {
        m_layers.push_back(std::make_shared<T>(m_input, m_output, init));
    }

    template<class Opt, class Loss>
    void compile(Optimizer::InitData init)
    {
        m_loss = std::make_shared<Loss>();
        compiled = true;

        for (auto& layer : m_layers)
        {
            layer->compile<Opt>(init);
        }
    }

    Cmat::Matrix forward(Cmat::Matrix x)
    {
        for (auto& layer : m_layers)
        {
            x = layer->forward(x);
        }
        return x;
    }

    // y: predict, t: answer
    double eval(const Cmat::Matrix& y, const Cmat::Matrix& t)
    {
        assert(compiled);
        return m_loss->forward(y, t);
    }

    Cmat::Matrix backward() 
    {
        assert(compiled);
        Cmat::Matrix x = m_loss->backward();

        for (int i = m_layers.size()-1; i >= 0; --i)
        {
            x = m_layers[i]->backward(x);
        }
        return x;
    }

};

}