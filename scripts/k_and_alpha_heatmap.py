from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import metnum
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
import time

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
y = y.astype(int)
X = X.astype(int)

indxs = np.random.choice(len(y), 70000, replace=False)
new_y = [ y[idx] for idx in indxs ]
new_X = [ X[idx] for idx in indxs ]
X, y = np.array(new_X), np.array(new_y)

limit = int(0.8 * X.shape[0]) 

X_train, y_train = X[:limit], y[:limit]
X_val, y_val = X[limit:], y[limit:]

assert len(X_train) == len(y_train)
assert len(X_val) == len(y_val)

print(f"Ahora tengo {len(X_train)} instancias de entrenamiento y {len(X_val)} de validación")

acc_heatmap = np.zeros((8, 13))
duration_heatmap = np.zeros((8, 13))
for i, alpha in enumerate([16,17,19,20,21,22,23,24]):
    print(alpha)
    for j, k in enumerate([1,2,3,4,5,6,7,8,9,10,100,200,1000]):
        print(k)
        start =  time.process_time()
        pca = metnum.PCA(alpha)
        pca.fit(X_train)
        X_train_PCA = pca.transform(X_train)

        X_val_PCA = pca.transform(X_val)

        clf_metnum = metnum.KNNClassifier(k)
        clf_metnum.fit(X_train_PCA, y_train)
        clf_mentum_predicted = clf_metnum.predict(X_val_PCA)
        end = time.process_time()
        time_performance = end - start
        acc = accuracy_score(clf_mentum_predicted, y_val)
        acc_heatmap[i, j] = acc
        duration_heatmap[i, j] = time_performance


myfile = Path('k_and_alpha_heatmap/acc_heatmap_matrix.npy')
myfile.touch(exist_ok=True)
with open('k_and_alpha_heatmap/acc_heatmap_matrix.npy', 'wb') as f:
    np.save(f, acc_heatmap)


myfile = Path('k_and_alpha_heatmap/duration_heatmap_matrix.npy')
myfile.touch(exist_ok=True)
with open('k_and_alpha_heatmap/duration_heatmap_matrix.npy', 'wb') as f:
    np.save(f, duration_heatmap)
