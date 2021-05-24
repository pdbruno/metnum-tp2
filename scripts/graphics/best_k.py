from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=30)          # controls default text sizes
plt.rc('axes', titlesize=30)     # fontsize of the axes title
plt.rc('axes', labelsize=30)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=30)    # fontsize of the tick labels
plt.rc('ytick', labelsize=30)    # fontsize of the tick labels
plt.rc('legend', fontsize=30)    # legend fontsize
plt.rc('figure', titlesize=30)  # fontsize of the figure title

data = np.load('../best_k_knn/acc_by_k.npy')[1:]
plt.plot(range(1, 129), data[1:], '-o')
plt.xlabel('K', fontsize=40)
plt.ylabel('Accuracy', fontsize=40)
plt.title('Accuracy de kNN para cada K de 1 a 130', fontsize=40)
plt.show()