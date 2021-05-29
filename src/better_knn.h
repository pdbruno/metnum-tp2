#pragma once

#include "types.h"
#include "knn.h"

class BetterKNNClassifier : public KNNClassifier {
public:
    BetterKNNClassifier(unsigned int n_neighbors);
protected:
    int _knn(Vector x) override;
};
