#pragma once

#include "types.h"


class KNNClassifier {
public:
    KNNClassifier(unsigned int n_neighbors);

    void fit(Matrix X, Matrix y);

    Vector predict(Matrix X);

    unsigned int get_K();
    std::vector<int> get_Clases();
private:
    int _knn(Vector x);
    unsigned int _k;
    Matrix _X;
    Matrix _y;
    std::vector<int> _clases;
};
void sorteadito(std::vector<double>& distancias, std::vector<int>& clases);
void print_vector(vector<int> &v);
