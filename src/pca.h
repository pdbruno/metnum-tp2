#pragma once
#include "types.h"
class PCA {
public:
    PCA(unsigned int n_components);

    void fit(Matrix X);

    Eigen::MatrixXd transform(Matrix X);
private:
    unsigned int _alpha;

    Matrix covarianza;

    std::pair<Vector, Matrix> _first_alpha_pairs;
};
Vector chorizear(Matrix X)
