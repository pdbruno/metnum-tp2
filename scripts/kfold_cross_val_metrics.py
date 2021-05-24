from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
import metnum

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
y = y.astype(int)
X = X.astype(int)

limit = int(0.8 * X.shape[0]) 

X_train, y_train = X[:limit], y[:limit]
X_val, y_val = X[limit:], y[limit:]

assert len(X_train) == len(y_train)
assert len(X_val) == len(y_val)
print(f"Ahora tengo {len(X_train)} instancias de entrenamiento y {len(X_val)} de validación")

kf = KFold(n_splits=10)
kf.get_n_splits(X)
KFold(n_splits=10, random_state=None, shuffle=False)
acc = 0
for k in range(1, 101):
    for train_index, test_index in kf.split(X):
        clf_metnum = metnum.KNNClassifier(10)
        clf_metnum.fit(X_train, y_train)
        acc += accuracy_score(clf_metnum.predict(X_val), y_val)

print(acc / 10)

