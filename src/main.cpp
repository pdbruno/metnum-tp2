//
// Created by pachi on 5/6/19.
//

#include <iostream>
#include "pca.h"
#include "eigen.h"

int main(int argc, char **argv) {

  std::cout << "Hola mundo!" << std::endl;

  Matrix id(3, 3);
  id << (float)1/3, 0, 0,
        0, (float)1/3, 0,
        0, 0, (float)1/3;

  std::cout << id << std::endl;


  pair<Eigen::VectorXd, Matrix> par = get_first_eigenvalues(id, 3);

  std::cout << par.first << std::endl;
  std::cout << par.second << std::endl;


  return 0;
}
