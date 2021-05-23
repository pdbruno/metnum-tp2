from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import metnum
import pandas as pd
from sklearn.metrics import accuracy_score
import time

df_train = pd.read_csv("../data/train.csv")

df_train = df_train[:10000]
X = df_train[df_train.columns[1:]].values
y = df_train["label"].values.reshape(-1, 1)


limit = int(0.8 * X.shape[0]) 

X_train, y_train = X[:limit], y[:limit]
X_val, y_val = X[limit:], y[limit:]

assert len(X_train) == len(y_train)
assert len(X_val) == len(y_val)

print(f"Ahora tengo {len(X_train)} instancias de entrenamiento y {len(X_val)} de validación")

accuracies = []
durations = []
for k in range(130):
    start =  time.process_time()
    clf_metnum = metnum.KNNClassifier(k)
    clf_metnum.fit(X_train, y_train)
    clf_mentum_predicted = clf_metnum.predict(X_val)
    end = time.process_time()
    accuracies.append(accuracy_score(clf_mentum_predicted, y_val))
    durations.append(end - start)


myfile = Path('best_k_knn/acc_by_k.npy')
myfile.touch(exist_ok=True)
with open('best_k_knn/acc_by_k.npy', 'wb') as f:
    np.save(f, accuracies)


myfile = Path('best_k_knn/duration_by_k.npy')
myfile.touch(exist_ok=True)
with open('best_k_knn/duration_by_k.npy', 'wb') as f:
    np.save(f, durations)
