import metnum
import pandas as pd
import numpy as np 
from sklearn.metrics import accuracy_score
from pathlib import Path
import time
from sklearn.datasets import fetch_openml

X_all, y_all = fetch_openml('mnist_784', version=1, return_X_y=True)

cant_imagenes = [100, 500, 1000, 5000, 10000, 20000, 40000, 70000]
best_ks = [3]

accuracies = []
durations = []
for k in best_ks:
    for cant in cant_imagenes:
        y = y_all.astype(int)[:cant]
        X = X_all.astype(int)[:cant]
        limit = int(0.8 * X.shape[0]) 
        if cant == 100:
            limit = int(0.2 * X.shape[0]) 
            X_val, y_val = X[:limit], y[:limit]
            X_train, y_train = X[limit:], y[limit:]
        else:
            X_train, y_train = X[:limit], y[:limit]
            X_val, y_val = X[limit:], y[limit:]

        assert len(X_train) == len(y_train)
        assert len(X_val) == len(y_val)

        print(f"Ahora tengo {len(X_train)} instancias de entrenamiento y {len(X_val)} de validación")

        start =  time.process_time()
        clf_metnum = metnum.KNNClassifier(k)
        clf_metnum.fit(X_train, y_train)
        clf_mentum_predicted = clf_metnum.predict(X_val)
        end = time.process_time()
        accuracies.append(accuracy_score(clf_mentum_predicted, y_val))
        durations.append(end - start)


    myfile = Path(f'acc_by_size_dataset_k/acc_{k}.npy')
    myfile.touch(exist_ok=True)
    with open(f'acc_by_size_dataset_k/acc_{k}.npy', 'wb') as f:
        np.save(f, accuracies)


    myfile = Path(f'acc_by_size_dataset_k/duration_{k}.npy')
    myfile.touch(exist_ok=True)
    with open(f'acc_by_size_dataset_k/duration_{k}.npy', 'wb') as f:
        np.save(f, durations)
    accuracies = []
    durations = []