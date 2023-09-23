#pragma once
#include <math.h>
#include <cmath>
#include "../Loss.hpp"
#include "../utils.hpp"

namespace NNCpp
{

namespace Loss
{

class CrossEntropy: public Loss
{
private:
    Cmat::Matrix m_back;
    BS::thread_pool pool = BS::thread_pool(std::thread::hardware_concurrency());
public:
    // x: predict, t: answer
    double forward(const Cmat::Matrix& x, const Cmat::Matrix& t) override
    {
        auto err = x;
        
        for (int i = 0; i < err.m_height; ++i)
        {
            pool.push_task([&](int i){
                for (int j = 0; j < err.m_width; ++j)
                {
                    if (isnan(log(err[i][j]+0.0001)))
                    {
                        std::cout << err[i][j] << std::endl;
                        assert(false);
                    }
                    err[i][j] = log(err[i][j]+0.0001);
                }
            }, i);
        }
        pool.wait_for_tasks();

        err *= t;

        m_back = x-t;

        return err.sum() / err.m_height * -1;
    }

    Cmat::Matrix backward() override
    {
        return m_back;
    }
};

}

}