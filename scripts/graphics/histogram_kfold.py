import matplotlib.pyplot as plt
import numpy as np

best_ks = ['2','3','4','5','6']
accuracies = np.load('../kfold_cross_val_metrics/accuracies_by_k_by_K.npy')

plt.rc('font', size=20)          # controls default text sizes
plt.rc('axes', titlesize=30)     # fontsize of the axes title
plt.rc('axes', labelsize=30)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=30)    # fontsize of the tick labels
plt.rc('ytick', labelsize=30)    # fontsize of the tick labels
plt.rc('legend', fontsize=15)   

x = np.arange(len(best_ks))  # the label locations
width = 0.13
fig, ax = plt.subplots()
accuracies_t = accuracies.T
rects1 = ax.bar(x - 2 * width, [0.9445, 0.9455, 0.9455, 0.9445, 0.9455], width, label='Sin K-Fold')
rects2 = ax.bar(x - width, accuracies_t[0], width, label='5')
rects3 = ax.bar(x, accuracies_t[1], width, label='10')
rects4 = ax.bar(x + width, accuracies_t[2], width, label='15')
rects6 = ax.bar(x + 2 * width, accuracies_t[3], width, label='20')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracies')
ax.set_xlabel('Parametro k de kNN')
ax.set_title('Accuracies para k = 2, 3, 4, 5, 6 para distinto K de k-fold')
ax.set_xticks(x)
ax.set_ylim(0.7)
ax.set_xticklabels(best_ks)
ax.legend()

fig.tight_layout()

plt.show()
