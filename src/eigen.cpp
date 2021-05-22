#include <algorithm>
#include <chrono>
#include <iostream>
#include "eigen.h"

using namespace std;


pair<double, Vector> power_iteration(const Matrix &X, unsigned num_iter, double eps) {
    Eigen::VectorXd b = Vector::Random(X.cols());
    bool very_close = false;

    for (int i = 0; i < num_iter && !very_close; i++) {
        Eigen::VectorXd new_b = X * b;
        new_b = new_b / new_b.norm();
        double cos_angle = new_b.transpose() * b;
        very_close = (1-eps) < cos_angle && cos_angle <= 1;
        b = new_b;
    }
    // Falta criterio de paradas, experimentar con eso
    double eigval = b.transpose() * X * b ; //hay una soberbia perdida de precision
    return make_pair(eigval , b);
}

pair<Vector, Matrix> get_first_eigenvalues(const Matrix &X, unsigned num, unsigned num_iter, double epsilon) {
    Matrix A;
    A = X;
    Eigen::VectorXd eigvalues(num);
    Eigen::MatrixXd eigvectors(A.rows(), num);

    double a = 0;
    Eigen::VectorXd v = Vector::Zero(A.rows());
    for (int i = 0; i < num; i++) {
        A = A - (a * v * v.transpose());
        tie(a, v) = power_iteration(A, num_iter, epsilon);
        eigvalues[i] = a;
        eigvectors.col(i) = v;
    }

    return make_pair(eigvalues, eigvectors);
}

