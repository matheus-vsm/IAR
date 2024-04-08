#pip install scikit-learn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

print('Carregando Arquivo de Teste...')
arquivo = np.load('teste2.npy')
x = arquivo[0]
y = np.ravel(arquivo[1])

#Teste 2 - Problema 1
regr = MLPRegressor(hidden_layer_sizes = (2),
                    max_iter = 10000,
                    activation = 'relu', #{'identity', 'logistic', 'tanh', 'relu'},
                    solver = 'adam',
                    learning_rate = 'adaptive',
                    n_iter_no_change = 50)

# #Teste 2 - Problema 2
# regr = MLPRegressor(hidden_layer_sizes = (8),
#                     max_iter = 100000,
#                     activation = 'relu', #{'identity', 'logistic', 'tanh', 'relu'},
#                     solver = 'adam',
#                     learning_rate = 'adaptive',
#                     n_iter_no_change = 50)

# #Teste 2 - Problema 3
# regr = MLPRegressor(hidden_layer_sizes = (20, 2),
#                     max_iter = 100000,
#                     activation = 'relu', #{'identity', 'logistic', 'tanh', 'relu'},
#                     solver = 'adam',
#                     learning_rate = 'adaptive',
#                     n_iter_no_change = 50)

# #Teste 3 - Problema 1
# regr = MLPRegressor(hidden_layer_sizes = (8),
#                     max_iter = 10000000,
#                     activation = 'relu', #{'identity', 'logistic', 'tanh', 'relu'},
#                     solver = 'adam',
#                     learning_rate = 'adaptive',
#                     n_iter_no_change = 50)

# #Teste 3 - Problema 2
# regr = MLPRegressor(hidden_layer_sizes = (20, 10),
#                     max_iter = 10000000,
#                     activation = 'relu', #{'identity', 'logistic', 'tanh', 'relu'},
#                     solver = 'adam',
#                     learning_rate = 'adaptive',
#                     n_iter_no_change = 50)

# #Teste 3 - Problema 3
# regr = MLPRegressor(hidden_layer_sizes = (10, 9, 3),
#                     max_iter = 100000,
#                     activation = 'relu', #{'identity', 'logistic', 'tanh', 'relu'},
#                     solver = 'adam',
#                     learning_rate = 'adaptive',
#                     n_iter_no_change = 50)

# #Teste 4 - Problema 1
# regr = MLPRegressor(hidden_layer_sizes = (13),
#                     max_iter = 1000000,
#                     activation = 'logistic', #{'identity', 'logistic', 'tanh', 'relu'},
#                     solver = 'adam',
#                     learning_rate = 'adaptive',
#                     n_iter_no_change = 50)

# #Teste 4 - Problema 2
# regr = MLPRegressor(hidden_layer_sizes = (22, 8),
#                     max_iter = 100000,
#                     activation = 'tanh', #{'identity', 'logistic', 'tanh', 'relu'},
#                     solver = 'adam',
#                     learning_rate = 'adaptive',
#                     n_iter_no_change = 50)

# #Teste 4 - Problema 3
# regr = MLPRegressor(hidden_layer_sizes = (10, 4, 26),
#                     max_iter = 10000,
#                     activation = 'relu', #{'identity', 'logistic', 'tanh', 'relu'},
#                     solver = 'adam',
#                     learning_rate = 'adaptive',
#                     n_iter_no_change = 50)

# #Teste 5 - Problema 1
# regr = MLPRegressor(hidden_layer_sizes = (82),
#                     max_iter = 1000000000,
#                     activation = 'logistic', #{'identity', 'logistic', 'tanh', 'relu'},
#                     solver = 'adam',
#                     learning_rate = 'adaptive',
#                     n_iter_no_change = 50)

# #Teste 5 - Problema 2
# regr = MLPRegressor(hidden_layer_sizes = (32, 8),
#                     max_iter = 1000000000,
#                     activation = 'relu', #{'identity', 'logistic', 'tanh', 'relu'},
#                     solver = 'adam',
#                     learning_rate = 'adaptive',
#                     n_iter_no_change = 50)

# #Teste 5 - Problema 3
# regr = MLPRegressor(hidden_layer_sizes = (124, 64, 32, 48, 102),
#                     max_iter = 8000000000,
#                     activation = 'relu', #{'identity', 'logistic', 'tanh', 'relu'},
#                     solver = 'adam',
#                     learning_rate = 'adaptive',
#                     n_iter_no_change = 50)

print('Treinando RNA...')
regr = regr.fit(x, y)

print('Preditor')
y_est = regr.predict(x)

loss = regr.best_loss_
print(loss)

plt.figure(figsize = [14, 7])

#plot curso original
plt.subplot(1, 3, 1)
plt.plot(x, y)

#plot aprendizagem
plt.subplot(1, 3, 2)
plt.plot(regr.loss_curve_)

#plot regressor
plt.subplot(1, 3, 3)
plt.plot(x, y, linewidth = 1, color = 'yellow')
plt.plot(x, y_est, linewidth = 2)

plt.show()