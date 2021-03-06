#pragma once
#include "types.h"
class PCA {
public:
    PCA(unsigned int n_components);

    void fit(Matrix X);

    Eigen::MatrixXd transform(Matrix X);
private:
    unsigned int _alpha;

    Vector _autovalores;
    Matrix _autovectores;
};
