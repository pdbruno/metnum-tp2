#include <iostream>
#include "pca.h"
#include "eigen.h"

using namespace std;


PCA::PCA(unsigned int n_components) : _alpha(n_components) {}

void PCA::fit(Matrix X) {
  int n = X.rows(); //cantidad de muestras
  X = X.rowwise() - X.colwise().mean(); // en la iesima fila esta (x_i − μ)^t.
  Matrix covarianza = X.transpose() * X / (n - 1);

  tie(_autovalores, _autovectores) = get_first_eigenvalues(covarianza, _alpha, 5000, 1e-16);
}

MatrixXd PCA::transform(Matrix X) { //X pertenece a R^(cantIm x cantCoord)
  return X * _autovectores; //deberia ser de R^(cantIm x alpha)
}

