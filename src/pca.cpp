#include <iostream>
#include "pca.h"
#include "eigen.h"

using namespace std;


PCA::PCA(unsigned int n_components) : _alpha(n_components) {}

void PCA::fit(Matrix X) {
  int n = X.rows(); //cantidad de muestras

  Vector mu = X.rowwise().sum();
  mu = mu / n; //promedio de las muestras

  Matrix res(X.rows(), X.cols());

  res = (X.rowwise() - mu.transpose()) / sqrt(n - 1); // en la iesima fila esta (x_i − μ)^t / sqrt(n − 1).

  covarianza = res.transpose() * res;//M = X^t . X

  tie(_autovalores, _autovectores) = get_first_eigenvalues(covarianza, _alpha, 5000, 1e-16);
}

std::pair<Vector, Matrix> PCA::debugeameEsta(){
  return make_pair(_autovalores, _autovectores);
}
MatrixXd PCA::transform(Matrix X) { //X pertenece a R^(cantIm x cantCoord)
  return X * _autovectores; //deberia ser de R^(cantIm x alpha)
}

