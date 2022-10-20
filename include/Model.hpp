# pragma once

# include <vector>
# include <memory>
# include <iostream>
# include <algorithm>
# include <map>
# include <string>

# include "Util.hpp"
# include "Layers.hpp"
# include "Loss.hpp"

namespace NNCpp
{

class Model
{
public:
    Model():m_output(0),compiled(false)
    {

    }

    template<class LayerClass>
    void add(uint32 unit, uint32 input_n = 0)
    {
        if (input_n == 0) 
        {
            if (m_output == 0)
            {
                std::cerr << "Number of inputs unknown." << std::endl;
                return;
            }
            input_n = m_output;
        }
        else if (m_output != input_n and m_output != 0)
        {
            std::cerr << "The number of inputs specified differs from the number of outputs in the final layer. The number of outputs in the final layer applies." << std::endl;
            input_n = m_output;
        }

        m_Layers.push_back(std::make_shared<LayerClass>(input_n, unit));

        m_output = unit;
    }

    template<class ActivationClass>
    void add()
    {
        m_Layers.push_back(std::make_shared<ActivationClass>());
    }

    template<class LossClass>
    void compile()
    {
        m_Loss = std::make_shared<LossClass>();
        compiled = true;
    }

    std::map<std::string, std::vector<double>> fit(const Data&x, const Data& y, const uint32 batch_size, const uint32 epochs, const double lr)
    {
        std::map<std::string, std::vector<double>> history;

        if (not compiled)
        {
            std::cerr << "This model has not been compiled. Please compile before call fit()." << std::endl;
            return history;
        }

        const uint32 iteration_per_epoch = x.shape().rows / batch_size + !(!(x.shape().rows % batch_size));
        const uint32 iteration = epochs * iteration_per_epoch;
        for (uint32 step = 0; step < iteration; ++step)
        {
            const uint32 t = step % iteration_per_epoch;
            Data batch_x = x({int32(t*batch_size), int32(std::min((t+1)*batch_size, x.shape().rows))}, x.cSlice());
            Data batch_y = y({int32(t*batch_size), int32(std::min((t+1)*batch_size, y.shape().rows))}, y.cSlice());
            Data predicted = predict(batch_x);

            Data loss = m_Loss->forward(predicted, batch_y);
            history["loss"].push_back(nc::mean(loss)[0]);

            {
                Data dx = m_Loss->backward();
                for (auto layer = m_Layers.rbegin(); layer != m_Layers.rend(); ++layer)
                {
                    dx = (*layer)->backward(dx, lr);
                }
            }
        }
        return history;
    }

    Data predict(const Data& x)
    {
        Data res = x;
        for (auto& layer : m_Layers)
        {
            res = layer->forward(res, true);
        }
        return res;
    }
private:
    bool compiled;
    std::vector<std::shared_ptr<Layers::AbstractLayer>>m_Layers;
    std::shared_ptr<Loss::AbstractLoss>m_Loss;
    uint32 m_output;

    void backward()
    {

    }
};

};