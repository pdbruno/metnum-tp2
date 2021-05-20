#include <algorithm>
#include <chrono>
#include <iostream>
#include "eigen.h"

using namespace std;


pair<double, Vector> power_iteration(const Matrix &X, unsigned num_iter, double eps) {
    Vector b = Vector::Random(X.cols());

    for (int i = 0; i < num_iter; i++) {
        b = X * b;
        b = b / b.norm();
    }
    // Falta criterio de paradas, experimentar con eso
    double eigval = b.transpose() * X * b ; //hay una soberbia perdida de precision
    return make_pair(eigval , b);
}

pair<Vector, Matrix> get_first_eigenvalues(const Matrix &X, unsigned num, unsigned num_iter, double epsilon) {
    Matrix A;
    A = X;
    Vector eigvalues(num);
    Matrix eigvectors(A.rows(), num);

    int a = 0;
    Vector v = Vector::Zero(A.rows());
    for (int i = 0; i < num; i++) {
        A = A - (a * v * v.transpose());
        tie(a, v) = power_iteration(A, num_iter, epsilon);
        eigvalues[i] = a;
        eigvectors.col(i) = v;
    }

    return make_pair(eigvalues, eigvectors);
}

