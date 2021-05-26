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

acc = np.load('../best_alpha/acc.npy')
performance = np.load('../best_alpha/performance.npy')

plt.plot(acc / performance, '-o')
plt.xlabel('Alpha', fontsize=40)
plt.ylabel('Accuracy / Performance(segundos) ', fontsize=40)
plt.title('Accuracy / Performance de kNN + PCA para k = 3 y alpha de 1 a 50', fontsize=40)
plt.show()

plt.plot(range(12,49), acc[12:] / performance[12:], '-o')
plt.xlabel('Alpha', fontsize=40)
plt.ylabel('Accuracy / Performance(segundos) ', fontsize=40)
plt.xticks([12,15,20,25,30,35,40,45])
plt.title('Accuracy / Performance de kNN + PCA para k = 3 y alpha de 1 a 50', fontsize=40)
plt.show()


# Create two subplots and unpack the output array immediately
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(range(1, 50), acc, '-o')
ax2.plot(range(1, 50), performance, '-o')
ax1.set_title('Accuracy y performance de kNN + PCA para k = 3 y alpha de 1 a 50')
ax2.set_xlabel('Alpha')
ax2.set_ylabel('Performance (segundos)')
ax1.set_ylabel('Accuracy')
plt.show()