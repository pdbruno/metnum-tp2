import metnum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


model = metnum.KNNClassifier(3)
nuestros = np.empty((10,784))
for k in range(10):
    arr = np.load("./digitos_nuestros/nuestro"+ str(k) + ".npy")
    nuestros[k] = arr
    print(arr.shape)
    plt.imshow(arr.reshape((28,28)), cmap="Greys")
    plt.show()

train = pd.read_csv("../data/train.csv")
X_train = train[train.columns[1:]]
y_train = train[train.columns[0]]
#for k in range(15):
 #   plt.imshow(X_train.iloc[k,:].values.reshape((28,28)), cmap="Greys")
  #  plt.show()
print(y_train)
model.fit(X_train,y_train)
res = model.predict(nuestros)
print(res)



