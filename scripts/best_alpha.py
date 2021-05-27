import metnum
import pandas as pd
import numpy as np 
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
import time
from pathlib import Path
from utils import get_MNIST

X_train, y_train, X_val, y_val = get_MNIST(0.8)

print(f"Ahora tengo {len(X_train)} instancias de entrenamiento y {len(X_val)} de validación")

accuracy_con_pca = []
performance = []
for i in range(50):
    start =  time.process_time()
    pca = metnum.PCA(i)
    pca.fit(X_train)
    X_train_PCA = pca.transform(X_train)

    X_val_PCA = pca.transform(X_val)

    clf_metnum = metnum.KNNClassifier(3)
    clf_metnum.fit(X_train_PCA, y_train)
    acc = accuracy_score(clf_metnum.predict(X_val_PCA), y_val)
    end = time.process_time()
    time_performance = end - start
    accuracy_con_pca.append(acc)
    performance.append(time_performance)

myfile = Path('best_alpha/acc_2.npy')
myfile.touch(exist_ok=True)
with open('best_alpha/acc_2.npy', 'wb') as f:
    np.save(f, accuracy_con_pca)

myfile = Path('best_alpha/performance.npy')
myfile.touch(exist_ok=True)
with open('best_alpha/performance.npy', 'wb') as f:
    np.save(f, performance)