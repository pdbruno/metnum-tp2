#include "better_knn.h"
using namespace std;


BetterKNNClassifier::BetterKNNClassifier(unsigned int n_neighbors) : KNNClassifier(n_neighbors) {
}

int BetterKNNClassifier::_knn(Vector x) //ESTO SEGURAMENTE SE PUEDE HACER MAS EFICIENTE EN MUCHOS ASPECTOS!!!!! (por ejemplo hacer k pasos del slection sort, por ejemplo)
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

    double accum_digitos[10] = {};
    int moda = 0;
    double valor_moda = 0;

    for (size_t i = 0; i < _k; i++) {
        int digito = clases[i];
        accum_digitos[digito] += 1/distancias[i];
        if(accum_digitos[digito] > valor_moda){
            valor_moda = accum_digitos[digito];
            moda = digito;
        }
    }

    return moda;
}
