import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

data = np.load('../k_and_alpha_heatmap/acc_heatmap_matrix.npy')

df = pd.DataFrame(data[10:, 1:], index=range(10, 31), columns=range(1,101))

ax = sns.heatmap(df, vmin=0.8, vmax=1)
plt.show()