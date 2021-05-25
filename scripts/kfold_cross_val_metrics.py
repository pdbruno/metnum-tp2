from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
import numpy as np
import metnum
from pathlib import Path
import random

best_ks = [2]
kfold = [2]

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
y = y.astype(int)
X = X.astype(int)

resultados = np.zeros((len(best_ks), len(kfold)))

for index_k, k in enumerate(best_ks):
    print(f'Current k: {k}')
    for index_K, split in enumerate(kfold):
        print(f'Current K: {split}')
        kf = KFold(n_splits=split, shuffle=True)
        accuracies_by_split = 0
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf_metnum = metnum.KNNClassifier(k)
            clf_metnum.fit(X_train, y_train)
            accuracies_by_split += accuracy_score(clf_metnum.predict(X_test), y_test)

        resultados[index_k, index_K]= accuracies_by_split / split

myfile = Path(f'kfold_cross_val_metrics/accuracies_by_k_by_K.npy')
myfile.touch(exist_ok=True)
with open(f'kfold_cross_val_metrics/accuracies_by_k_by_K.npy', 'wb') as f:
    np.save(f, resultados)


