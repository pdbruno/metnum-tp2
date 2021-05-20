#pragma once
#include "types.h"
class PCA {
public:
    PCA(unsigned int n_components);

    void fit(Matrix X);

    Eigen::MatrixXd transform(Matrix X);
    std::pair<Vector, Matrix> debugeameEsta();
private:
    unsigned int _alpha;

    Matrix covarianza;

    Vector _autovalores;
    Matrix _autovectores;
};
