#include <algorithm>
//#include <chrono>
#include <iostream>
#include "knn.h"
using namespace std;

KNNClassifier::KNNClassifier(unsigned int n_neighbors):_k(n_neighbors)
{
}

void KNNClassifier::fit(Matrix X, Matrix y)//Fit the model using X as training data and y as target values
{
    _X = X;
    _y = y;
    
}
int KNNClassifier::_knn(Vector x) //ESTO SEGURAMENTE SE PUEDE HACER MAS EFICIENTE EN MUCHOS ASPECTOS!!!!! (por ejemplo hacer k pasos del slection sort, por ejemplo)
{ 
    std::vector<double> distancias(_X.rows());
    std::vector<int> clases(_X.rows());

    for(int i = 0; i < _X.rows() ; i++){
        Vector diferencia = _X.row(i) - x;
        clases.push_back(_y(i));
        distancias.push_back(diferencia.norm());
    }

    sorteadito(distancias, clases);

    distancias.resize(_k);
    clases.resize(_k);

    int digitos[10];
    for (size_t i = 0; i < _k; i++)
    {
        digitos[clases[i]]++;
    }

    int moda = 0;
    int modaApariciones = digitos[0];

    for (size_t i = 0; i < 10; i++)
    {
        if(digitos[i]>modaApariciones){
            modaApariciones = digitos[i];
            moda = i;
        }
    }
    
    return moda;
}

Vector KNNClassifier::predict(Matrix X)
{
    // Creamos vector columna a devolver
    auto ret = Vector(X.rows());

    for (int i = 0; i < X.rows(); ++i)
    {
        ret(i) = _knn(X.row(i));
    }

    return ret;
}


void sorteadito(std::vector<double>& distancias, std::vector<int>& clases)
{
  unsigned int i;  
  int keyClases, j;
  double key;
  for (i = 1; i < distancias.size(); i++)
  {
    key = distancias[i];
    keyClases = clases[i];
    j = i - 1;

    while (j >= 0 && distancias[j] > key)
    {
      distancias[j + 1] = distancias[j];
      clases[j + 1] = clases[j];
      j = j - 1;
    }
    distancias[j + 1] = key;
    clases[j + 1] = keyClases;
  }
}