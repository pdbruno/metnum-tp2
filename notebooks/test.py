from typing import List
import metnum
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import numpy as np

""" df_train = pd.read_csv("../data/train.csv")
df_train = df_train[:5000]
X = df_train[df_train.columns[1:]].values
y = df_train["label"].values.reshape(-1, 1)
limit = int(0.8 * X.shape[0]) 

X_train, y_train = X[:limit], y[:limit]
X_val, y_val = X[limit:], y[limit:] """

matriz_rand = np.random.rand(10, 10)

""" clf_metnum = metnum.PCA(4)
clf_sklearn = PCA(4)

clf_metnum.fit(matriz_rand)
clf_sklearn.fit(matriz_rand) """

autoval1, autovec1 = metnum.get_first_eigenvalues(matriz_rand*matriz_rand.T, 10, 10000)
autoval2, autovec2 = np.linalg.eig(matriz_rand*matriz_rand.T)
autoval2: List = autoval2.tolist()
autoval2.sort(key= lambda x: abs(x), reverse=True)
#print(autoval1 - autoval2[-5:])
print('diferencias')
print(abs(autoval1 - autoval2))
print('nuestros')
print(autoval1)
print('de ellos')
print(autoval2)
#print(np.flip(autoval2)[])
#print(par[0])|
#print(par[1])

#matriz_rand_2 = np.random.rand(50, 50)

#transform_metnum = clf_metnum.transform(matriz_rand_2)
#transform_sklearn = clf_sklearn.transform(matriz_rand_2)
#print(transform_metnum)
#print(transform_sklearn)
#acc = accuracy_score(transform_metnum, transform_sklearn)
#print("Accuracy: {}".format(acc))