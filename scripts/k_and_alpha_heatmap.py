from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import metnum
import pandas as pd
from sklearn.metrics import accuracy_score
import time

df_train = pd.read_csv("../data/train.csv")

df_train = df_train[:5000]
X = df_train[df_train.columns[1:]].values
y = df_train["label"].values.reshape(-1, 1)


limit = int(0.8 * X.shape[0]) 

X_train, y_train = X[:limit], y[:limit]
X_val, y_val = X[limit:], y[limit:]

assert len(X_train) == len(y_train)
assert len(X_val) == len(y_val)

print(f"Ahora tengo {len(X_train)} instancias de entrenamiento y {len(X_val)} de validación")

acc_heatmap = np.zeros((31, 101))
duration_heatmap = np.zeros((31, 101))
for alpha in range(31):
    for k in range(101):
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
        acc_heatmap[alpha, k] = acc
        duration_heatmap[alpha, k] = time_performance

myfile = Path('k_and_alpha_heatmap/acc_heatmap_matrix.npy')
myfile.touch(exist_ok=True)
with open('k_and_alpha_heatmap/acc_heatmap_matrix.npy', 'wb') as f:
    np.save(f, acc_heatmap)


myfile = Path('k_and_alpha_heatmap/duration_heatmap_matrix.npy')
myfile.touch(exist_ok=True)
with open('k_and_alpha_heatmap/duration_heatmap_matrix.npy', 'wb') as f:
    np.save(f, duration_heatmap)
