#pragma once

#include "types.h"

class KNNClassifier {
public:
    KNNClassifier(unsigned int n_neighbors);

    void fit(Matrix X, Matrix y);

    Vector predict(Matrix X);

protected:
    unsigned int _k;
    virtual int _knn(Vector x);
    Matrix _X;
    Matrix _y;
};
void sorteadito(double distancias[], int clases[], int n, int k);
