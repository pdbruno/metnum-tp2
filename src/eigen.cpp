#include <algorithm>
#include <chrono>
#include <iostream>
#include "eigen.h"

using namespace std;


pair<double, Vector> power_iteration(const Matrix &X, unsigned num_iter, double eps) {
    Vector b = Vector::Random(X.cols());
    double eigenvalue;

    for (int i = 0; i < num_iter; i++) {
        b = X * b;
        b = b / b.norm();
    }

    double norm = b.norm();
    return make_pair(b.transpose() * A * b, b / norm);
}

pair<Vector, Matrix> get_first_eigenvalues(const Matrix &X, unsigned num, unsigned num_iter, double epsilon) {
    Matrix A(X);
    Vector eigvalues(num);
    Matrix eigvectors(A.rows(), num);

    /***********************
     * COMPLETAR CODIGO
     **********************/
    return make_pair(eigvalues, eigvectors);
}
