#include <iostream>
#include "pca.h"
#include "eigen.h"

using namespace std;


PCA::PCA(unsigned int n_components) : _alpha(n_components) {}

void PCA::fit(Matrix X) {
  int n = X.rows();

  Vector mu = X.rowwise().sum();
  mu = mu / n;

  Vector iesima(X.cols());

  Matrix res(X.rows(), X.cols());

  double denominador = sqrt(n - 1);

  for (int i = 0; i < n; i++)
    res.row(i) = (X.row(i) - mu) / denominador;

  covarianza = (res.transpose()) * res;

  _first_alpha_pairs = get_first_eigenvalues(covarianza, _alpha);
}


MatrixXd PCA::transform(Matrix X) {
  
  MatrixXd res(X.rows(), _alpha);

  Vector chorizo = chorizear(X);

  Matrix eigenvectors = _first_alpha_pairs.second; // pertenece a R^(alfa x 784) , luego X pertenece a R^(alfa x 784)
  
  res = eigenvectors * chorizo;

  return res;
}

Vector chorizear(Matrix X){
  Vector chorizo(784);
  for (size_t i = 0; i < 28; i++)
    for (size_t j = 0; j < 28; j++)
      chorizo[(28*i)+j] = X(i, j);

  return chorizo;
}


