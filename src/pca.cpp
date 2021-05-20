#include <iostream>
#include "pca.h"
#include "eigen.h"

using namespace std;


PCA::PCA(unsigned int n_components) : _alpha(n_components) {}

void PCA::fit(Matrix X) {
  int n = X.rows();

  Vector mu = X.rowwise().sum();
  mu = mu / n;

  Matrix res(X.rows(), X.cols());

  double denominador = sqrt(n - 1);

  for (int i = 0; i < n; i++)
    res.row(i) = (X.row(i) - mu) / denominador;

  covarianza = (res.transpose()) * res;

  _first_alpha_pairs = get_first_eigenvalues(covarianza, _alpha, 5000, 1e-16);
}


MatrixXd PCA::transform(Matrix X) {
  
  Eigen::Map<Vector> chorizo(X.data(), X.size(), 1);

  Matrix eigenvectors = _first_alpha_pairs.second; // pertenece a R^(alfa x 784)
  
  return eigenvectors * chorizo;
}

