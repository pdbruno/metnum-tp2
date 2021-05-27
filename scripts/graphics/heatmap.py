import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

data = np.load('../k_and_alpha_heatmap/acc_heatmap_matrix.npy')
print(data)
df = pd.DataFrame(data, columns=[1,2,3,4,5,6,7,8,9,10,100,200,1000], index=[16,17,19,20,21,22,23,24])

ax = sns.heatmap(df, vmin=0.85, vmax=1)
plt.show()