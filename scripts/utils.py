import numpy as np 
from sklearn.datasets import fetch_openml

def get_MNIST(train_limit, shuffle=True):
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    y = y.astype(int)
    X = X.astype(int)

    if shuffle:
        indxs = np.random.choice(len(y), 70000, replace=False)
        new_y = [ y[idx] for idx in indxs ]
        new_X = [ X[idx] for idx in indxs ]
        X, y = np.array(new_X), np.array(new_y)

    limit = int(train_limit * X.shape[0]) 

    X_train, y_train = X[:limit], y[:limit]
    X_val, y_val = X[limit:], y[limit:]

    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)

    return X_train, y_train, X_val, y_val

