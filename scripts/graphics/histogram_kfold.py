import matplotlib.pyplot as plt
import numpy as np

best_ks = ['2','3','4','5','6']
Kfolds = ['Sin K-Fold', '5', '10', '15', '20']
accuracies = np.load('../kfold_cross_val_metrics/accuracies_by_k_by_K.npy')

# function to add value labels
def graficar(x,y, k):
    plt.clf()
    plt.ylim(0.8)
    plt.yticks([0.8, 0.85, 0.9, 0.95])
    plt.title(f'Accuracies para k = {k} para distinto K de k-fold')
    plt.bar(x, y)
    for i in range(len(x)):
        plt.text(i, y[i], format(y[i], '.3f') , ha = 'center')
    
    plt.savefig(f'../imagenes/acc-k{k}-Kfold', bbox_inches='tight', dpi=200)


plt.tight_layout()
plt.rc('font', size=20)          # controls default text sizes
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
plt.rc('legend', fontsize=15)   
plt.rc('font', size=15)          # controls default text sizes
plt.ylabel('Accuracies')
plt.xlabel('Parametro K de K-Fold')
accuracies_k_2 = accuracies[0].tolist()
accuracies_k_3= accuracies[1].tolist()
accuracies_k_4= accuracies[2].tolist()
accuracies_k_5= accuracies[3].tolist()
accuracies_k_6= accuracies[4].tolist()

accuracies_k_2.insert(0, 0.9445)
accuracies_k_3.insert(0, 0.9455)
accuracies_k_4.insert(0, 0.9455)
accuracies_k_5.insert(0, 0.9445)
accuracies_k_6.insert(0, 0.9455)

graficar(Kfolds, accuracies_k_2, 2)
graficar(Kfolds, accuracies_k_3, 3)
graficar(Kfolds, accuracies_k_4, 4)
graficar(Kfolds, accuracies_k_5, 5)
graficar(Kfolds, accuracies_k_6, 6)
