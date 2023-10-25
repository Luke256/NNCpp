# NNCpp

簡易的なヘッダーオンリーのニューラルネットワークライブラリ

C++17以上あれば多分動く

# 利用方法
当リポジトリ内の`include`フォルダを適当な場所に置くと、`include/NNCpp.hpp`をincludeすることで利用できます。

## データ型
行列用のクラス`Cmat::Matrix`を実装しています。これは内部的には`std::vector<std::vector<double>>`です。

## モデル構築
現在は`NNCpp::SequentialModel`クラスを提供しています。これはKerasでの`keras.models.Sequential`に相当し、単純に層を重ねていくものです。
そのため、分岐の含むような複雑なモデルは自分で定義する必要があります。

レイヤーの追加は`NNCpp::SequentialModel::addLayer`で可能です。テンプレート引数にレイヤーの種類、引数にレイヤーの情報(`NNCpp::Layers::InitData`)を渡します。

使用時にはモデルのコンパイルが必要です。`NNCpp::SequentialModel::compile`にて損失関数、及び最適化関数を指定し、最適化関数に用いるパラメータ(`NNCpp::Optimizer::InitData`)を引数に渡します。

## レイヤー
便宜上、活性化関数もレイヤーとして実装されています。
### NNCpp::Layers::Input
入力層です。SequentialModelは最初の層がInputであることを想定しています。

### NNCpp::Layers::Dense
全結合層です。SequentialModelにこのレイヤーを追加する際は、出力数(`unit`)を指定します。

### NNCpp::Layers::ReLU
ReLU関数です。初期化時には特に引数はいりません。

### NNCpp::Layers::SoftMax
SoftMax関数です。
**学習時に使用する際は、損失関数に後述する交差エントロピー誤差を使用してください。**

## 損失関数
損失関数は交差エントロピー誤差と二乗誤差があります。

## 最適化関数
Adamは実装していません

### 確率的勾配降下法
SGDが`NNCpp::Optimizer::SGD`にて提供されています。初期化時には学習率を指定します。

## モデルの学習
`keras.models.Sequential.fit`のような自動で学習を行う関数はありません。
forward関数で推論し、evalで誤差を計算、backwardで誤差逆伝播を行う機能があります。

# コード例

```C++
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
```