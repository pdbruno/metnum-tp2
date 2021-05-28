import matplotlib.pyplot as plt

cant_imagenes = [100, 500, 1000, 5000, 10000, 20000, 50000, 70000]
resultados = [0.6,
 0.8460000000000001,
 0.8799999999999999,
 0.9369999999999999,
 0.9468,
 0.9584499999999998,
 0.9681599999999999,
 0.9701142857142857]


plt.bar(['100', '500', '1000', '5000', '10000', '20000', '50000', '70000'], resultados)
plt.ylim(0.4)
plt.ylabel('Accuracy')
plt.xlabel('Tamaño dataset')
plt.title(f'Accuracies para k = 3 y alpha = 19 para distintos tamanios de dataset con 5-Fold')
for i in range(8):
    plt.text(i, resultados[i], format(resultados[i], '.3f') , ha = 'center')
plt.show()