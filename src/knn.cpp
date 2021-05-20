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
/*
int KNNClassifier::_knn(Vector x) //ESTO SEGURAMENTE SE PUEDE HACER MAS EFICIENTE EN MUCHOS ASPECTOS!!!!! (por ejemplo hacer k pasos del slection sort, por ejemplo)
{ 
    std::vector<double> distancias(_X.rows());
    std::vector<int> clases(_X.rows());

    for(int i = 0; i < _X.rows() ; i++){
        Vector diferencia = _X.row(i) - x.transpose();
        clases[i] = _y(i, 0);
        distancias[i] = diferencia.norm();
    }

    sorteadito(distancias, clases);

    std::array<int, 10> digitos = {};
    for (size_t i = 0; i < _k; i++)
    {
        digitos[clases[i]]++;
    }


    int moda = 0; //todo esto se puede obviar y sacar la moda directamente en el for anterior
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
*/
int KNNClassifier::_knn(Vector x) //ESTO SEGURAMENTE SE PUEDE HACER MAS EFICIENTE EN MUCHOS ASPECTOS!!!!! (por ejemplo hacer k pasos del slection sort, por ejemplo)
{ 
    Eigen::VectorXd distancias(_X.rows());
    Vector clases(_X.rows());
    clases << _y.col(0);
    MatrixXd diferencias = (_X).colwise() - x;
    distancias = diferencias.rowwise().norm();

    distancias.conservativeResize(distancias.rows(), distancias.cols() + 1);
    distancias.col(1) = clases;

    vector<int> stdClases(clases.data(), clases.data() + clases.rows() * clases.cols()); 
    vector<double> stdDistancias(distancias.data(), distancias.data() + distancias.rows() * distancias.cols());
    
    sorteadito(stdDistancias, stdClases);// ENCONTRAR UNA MANEAR DE VECTORIZAR ESTA FUNCION, es complicado
    
    std::array<int, 10> digitos = {};
    for (size_t i = 0; i < _k; i++)
    {
        digitos[stdClases[i]]++;
    }


    int moda = 0; //todo esto se puede obviar y sacar la moda directamente en el for anterior
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


void print_vector(int v[])
{
    for (uint i = 0; i < 10; i++)
    {
        //pybind11::print(v[i], '\n');
    }
}


Vector KNNClassifier::predict(Matrix X)
{
    // Creamos vector columna a devolver
    auto ret = Vector(X.rows());

    for (int i = 0; i < X.rows(); ++i)
    {
        ret[i] = _knn(X.row(i));
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
/* void swap(int *xp, int *yp) 
{ 
    int temp = *xp; 
    *xp = *yp; 
    *yp = temp; 
} 
  
void selectionSort(std::vector<double>& distancias, std::vector<int>& clases, int n, int k) 
{ 
    int i, j, min_idx; 
  
    // One by one move boundary of unsorted subarray 
    for (i = 0; i < n-1; i++) 
    { 
        // Find the minimum element in unsorted array 
        min_idx = i; 
        for (j = i+1; j < n; j++) 
        if (arr[j] < arr[min_idx]) 
            min_idx = j; 
  
        // Swap the found minimum element with the first element 
        swap(&arr[min_idx], &arr[i]); 
    } 
} */