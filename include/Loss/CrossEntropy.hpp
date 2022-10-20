# pragma once

# include "Abstract.hpp"

namespace NNCpp
{

namespace Loss
{

class CrossEntropy : public AbstractLoss
{
public:
    CrossEntropy(){}

    Data forward(const Data& x, const Data& target) override
    {
        last_x = x, last_target = target;
        Data loss = -nc::sum(target*nc::log(x+1e-7), nc::Axis::COL);
        uint32 batch_size = x.shape().rows;
        return nc::divide(loss, (double)batch_size);
    }

    Data backward() override
    {
        Data diff = -nc::divide(last_target, (last_x+1e-7));
        return diff;
    }

private:
    Data last_x, last_target;
};

};

};