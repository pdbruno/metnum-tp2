from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

""" for k in [3,  5,  4,  6, 2]:
    cant_imagenes = ['100', '500', '1000', '5000', '10000', '20000', '40000', '70000']
    acc_2 = np.load('../acc_by_size_dataset_k/acc_2.npy')
    duration_2 = np.load('../acc_by_size_dataset_k/duration_2.npy')

    # Create two subplots and unpack the output array immediately
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.bar(cant_imagenes, acc_2)
    ax2.bar(cant_imagenes, duration_2)
    ax2.set_yscale('log')
    ax1.set_ylim(0.8, 1)
    ax1.set_title('Accuracy y duration de kNN con k = 2 usando MNIST con distintos tamaños de dataset')
    ax2.set_xlabel('Cantidad de imagenes totales de MNIST')
    ax2.set_ylabel('Duration (segundos)')
    ax1.set_ylabel('Accuracy')
    plt.show() """

cant_imagenes = ['100', '500', '1000', '5000', '10000', '20000', '40000', '70000']
acc_3 = np.load('../acc_by_size_dataset_k/acc_3.npy')
duration_3 = np.load('../acc_by_size_dataset_k/duration_3.npy')
acc_20 = np.load('../acc_by_size_dataset_k/acc_20.npy')
duration_5 = np.load('../acc_by_size_dataset_k/duration_5.npy')
acc_30 = np.load('../acc_by_size_dataset_k/acc_30.npy')
duration_4 = np.load('../acc_by_size_dataset_k/duration_4.npy')
acc_50 = np.load('../acc_by_size_dataset_k/acc_50.npy')
duration_6 = np.load('../acc_by_size_dataset_k/duration_6.npy')
acc_100 = np.load('../acc_by_size_dataset_k/acc_100.npy')
duration_2 = np.load('../acc_by_size_dataset_k/duration_2.npy')

plt.rc('font', size=20)          # controls default text sizes
plt.rc('axes', titlesize=30)     # fontsize of the axes title
plt.rc('axes', labelsize=30)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=30)    # fontsize of the tick labels
plt.rc('ytick', labelsize=30)    # fontsize of the tick labels

x = np.arange(len(cant_imagenes))  # the label locations
width = 0.13
fig, ax = plt.subplots()
rects1 = ax.bar(x - 2 * width, duration_2, width, label='2')
rects2 = ax.bar(x - width, duration_3, width, label='3')
rects3 = ax.bar(x, duration_4, width, label='4')
rects4 = ax.bar(x + width, duration_5, width, label='5')
rects5 = ax.bar(x + 2 * width, duration_6, width, label='6')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Duration')
ax.set_xlabel('Tamaño de dataset')
ax.set_title('Duration para k = 2, 3, 4, 5, 6 para distinto tamaño de dataset')
ax.set_xticks(x)
# ax.set_ylim(0.6)
ax.set_yscale('log')
ax.set_xticklabels(cant_imagenes)
ax.legend()

fig.tight_layout()

plt.show()
