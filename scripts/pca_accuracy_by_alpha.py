import metnum
import pandas as pd
import numpy as np 
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

for i in range(100):
    start =  time.process_time()
    pca = metnum.PCA(i)
    pca.fit(X_train)
    X_train_PCA = pca.transform(X_train)

    X_val_PCA = pca.transform(X_val)

    clf_metnum = metnum.KNNClassifier(18)
    clf_metnum.fit(X_train_PCA, y_train)
    acc = accuracy_score(clf_metnum.predict(X_val_PCA), y_val)
    end = time.process_time()
    time_performance = end - start
    print("Alpha: {}. Accuracy: {}. Duration: {}".format(i, acc, time_performance))