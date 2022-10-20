# NNCpp
C++用機械学習ライブラリ

# 導入方法
NNCpp-masterを適当なフォルダに配置

# 依存系
boost 1.80.0

# コード例

```C++
# include "NNCpp/NNCpp.hpp"

# include <iostream>

int main()
{
    NNCpp::Model model;

    model.add<NNCpp::Layers::Dense>(64, 2);
    model.add<NNCpp::Activation::Relu>();
    model.add<NNCpp::Layers::Dense>(2);
    model.add<NNCpp::Activation::SoftMax>();

    NNCpp::Data x = { {1, 1}, {0, 1}, {1, 0}, {0, 0} };
    NNCpp::Data y = { {1, 0}, {0, 1}, {0, 1}, {1, 0} };
    model.compile<NNCpp::Loss::CrossEntropy>();
    auto history = model.fit(x, y, 4, 10000, 0.01);

    for (auto i : history["loss"]) std::cout << i << " ";
    std::cout << std::endl;

    model.predict(x).print();
}

```
