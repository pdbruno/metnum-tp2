from sklearn.metrics import confusion_matrix, accuracy_score
import metnum
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

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

pca = metnum.PCA(20)
pca.fit(X_train)
X_train_PCA = pca.transform(X_train)

X_val_PCA = pca.transform(X_val)

clf_metnum = metnum.KNNClassifier(18)
clf_metnum.fit(X_train_PCA, y_train)


############
cm = confusion_matrix(clf_metnum.predict(X_val_PCA), y_val)

plt.imshow(cm)
plt.xticks(np.arange(0,10))
plt.yticks(np.arange(0,10));

plt.show()