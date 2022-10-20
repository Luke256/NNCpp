# pragma once

# include "Abstract.hpp"

namespace NNCpp
{

namespace Loss
{

class Abs : public AbstractLoss
{
public:
    Abs(){}

    Data forward(const Data& x, const Data& target) override
    {
        last_x = x, last_target = target;
        
        Data loss = nc::abs(x - target);
        loss = nc::sum(loss, nc::Axis::COL);

        uint32 batch_size = x.shape().rows;
        return nc::divide(loss, (double)batch_size);
    }

    Data backward() override
    {
        Data diff = last_x - last_target;
        for (auto& i : diff) i = (i>0?1:-1);
        return diff;
    }

private:
    Data last_x, last_target;
};

};

};