import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

data_acc = np.load('../k_and_alpha_heatmap/acc_heatmap_matrix.npy')
data_dur = np.load('../k_and_alpha_heatmap/duration_heatmap_matrix.npy')
df = pd.DataFrame(data_acc, columns=[1,2,3,4,5,6,7,8,9,10,100,200,1000], index=[16,17,19,20,21,22,23,24])

ax = sns.heatmap(df, vmin=0.85, vmax=1, annot=True, fmt='.3f')
ax.set_xlabel("k")
ax.set_ylabel("alpha")
plt.title('Accuracy para distintas combinaciones de k y alpha')
plt.show()

df = pd.DataFrame(data_acc / data_dur, columns=[1,2,3,4,5,6,7,8,9,10,100,200,1000], index=[16,17,19,20,21,22,23,24])

ax = sns.heatmap(df, annot=True, fmt='.3f')
ax.set_xlabel("k")
ax.set_ylabel("alpha")
plt.title('Ratio Accuracy/Duration para distintas combinaciones de k y alpha')
plt.show()