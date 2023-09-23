#include <iostream>
#include <vector>
#include "include/NNCpp.hpp"
#include "IrisDataset.hpp"

int main () {
    NNCpp::SequentialModel model;
    model.addLayer<NNCpp::Layers::Input>({4});
    model.addLayer<NNCpp::Layers::ReLU>({});
    model.addLayer<NNCpp::Layers::Dense>({128});
    model.addLayer<NNCpp::Layers::ReLU>({});
    model.addLayer<NNCpp::Layers::Dense>({3});
    model.addLayer<NNCpp::Layers::SoftMax>({});

    model.compile<NNCpp::Optimizer::SGD, NNCpp::Loss::CrossEntropy>({0.001});

    NNCpp::XorShift rnd;
    Cmat::Matrix x,y;

    for (auto& data : IRIS_DATASET)
    {
        x.push_back(data.data);
        std::vector<double>t(3);
        t[data.kind] = 1;
        y.push_back(t);
    }

    for (int step=0; step < 1000; ++step)
    {
        Cmat::Matrix bx, by;

        for (int i = 0; i < 50; ++i)
        {
            int idx = rnd() * IRIS_DATASET.size();
            bx.push_back(x[idx]);
            by.push_back(y[idx]);
        }

        auto t = model.forward(bx);
        auto loss = model.eval(t, by);
        model.backward();


        if ((step+1)%100 == 0) {
            int accepted = 0;
            t = model.forward(x);
            for (int i = 0; i < t.m_height; ++i)
            {
                int a = std::max_element(t[i].begin(), t[i].end()) - t[i].begin();
                int b = std::max_element(y[i].begin(), y[i].end()) - y[i].begin();
                accepted += (a==b);
            }

            std::cout << "step:" << step+1 << " loss:" << loss << " accuracy:" << (double)accepted / t.m_height * 100 << "%" << std::endl;
        }
    }

}