#include "knn.h"
using namespace std;

KNNClassifier::KNNClassifier(unsigned int n_neighbors) :_k(n_neighbors) {
}

void KNNClassifier::fit(Matrix X, Matrix y)//Fit the model using X as training data and y as target values
{
    _X = X;
    _y = y;

}
int KNNClassifier::_knn(Vector x) //ESTO SEGURAMENTE SE PUEDE HACER MAS EFICIENTE EN MUCHOS ASPECTOS!!!!! (por ejemplo hacer k pasos del slection sort, por ejemplo)
{
    int n = _X.rows();
    double distancias[n] = {};
    int clases[n] = {};

    for (int i = 0; i < _X.rows(); i++) {
        Vector diferencia = _X.row(i) - x.transpose();
        clases[i] = _y(i, 0);
        distancias[i] = diferencia.squaredNorm();
    }

    sorteadito(distancias, clases, n, _k);

    int digitos[10] = {};
    int moda = 0;
    int modaApariciones = digitos[0];

    for (size_t i = 0; i < _k; i++) {
        int digito = clases[i];
        digitos[digito]++;
        if(digitos[digito] > modaApariciones){
            modaApariciones = digitos[digito];
            moda = digito;
        }
    }

    return moda;
}
/* int KNNClassifier::_knn(Vector x) //ESTO SEGURAMENTE SE PUEDE HACER MAS EFICIENTE EN MUCHOS ASPECTOS!!!!! (por ejemplo hacer k pasos del slection sort, por ejemplo)
{
    Eigen::VectorXd distancias(_X.rows());
    Vector clases(_X.rows());
    clases << _y.col(0);
    MatrixXd diferencias = (_X).rowwise() - x.transpose();
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
} */


Vector KNNClassifier::predict(Matrix X) {
    // Creamos vector columna a devolver
    auto ret = Vector(X.rows());

    for (int i = 0; i < X.rows(); ++i)     {
        ret[i] = _knn(X.row(i));
    }

    return ret;
}


void sorteadito(double distancias[], int clases[], int n, int k) {
    int i, j, min_idx;

    for (i = 0; i < k; i++) {
        min_idx = i;
        for (j = i + 1; j < n; j++) {
            if (distancias[j] < distancias[min_idx]) {
                min_idx = j;
            }
        }
        //swap1
        double aux1 = distancias[i];
        distancias[i] = distancias[min_idx];
        distancias[min_idx] = aux1;
        //swap2
        int aux2 = clases[i];
        clases[i] = clases[min_idx];
        clases[min_idx] = aux2;
    }
}