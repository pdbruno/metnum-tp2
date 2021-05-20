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

  res = (X.rowwise() - mu.transpose()) / denominador;

  covarianza = res.transpose() * res;

  tie(_autovalores, _autovectores) = get_first_eigenvalues(covarianza, _alpha, 5000, 1e-16);
}

std::pair<Vector, Matrix> PCA::debugeameEsta(){
  return make_pair(_autovalores, _autovectores);
}
MatrixXd PCA::transform(Matrix X) { //X pertenece a R^(cantIm x cantCoord)

  return X * _autovectores; //deberia ser de R^(cantIm x alpha)
}

