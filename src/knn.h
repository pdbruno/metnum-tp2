#pragma once

#include "types.h"

class KNNClassifier {
public:
    KNNClassifier(unsigned int n_neighbors);

    void fit(Matrix X, Matrix y);

    Vector predict(Matrix X);

private:
    unsigned int _k;
    Matrix _X;
    Matrix _y;
protected:
    int _knn(Vector x);
};
void sorteadito(double distancias[], int clases[], int n, int k);
