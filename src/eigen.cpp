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
    // Falta criterio de paradas, experimentar con eso
    double norm = b.norm();
    return make_pair(b.transpose() * X * b, b / norm);
}

pair<Vector, Matrix> get_first_eigenvalues(const Matrix &X, unsigned num, unsigned num_iter, double epsilon) {
    Matrix A(X);
    Vector eigvalues(num);
    Matrix eigvectors(A.rows(), num);

    int a = 0;
    Vector v;
    for ( int i = 0; i < num; i++) {
        tie(a, v) = power_iteration(A - (a * v * v.transpose()), num_iter, epsilon);
        eigvalues[i] = a;
        eigvectors.row(i) = v;
    }
    
    return make_pair(eigvalues, eigvectors);
}

