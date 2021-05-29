#pragma once

#include "types.h"
#include "knn.h"

class BetterKNNClassifier : public KNNClassifier {
protected:
    int _knn(Vector x);
};
