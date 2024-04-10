#pip install scikit-learn
#python -m pip install -U matplotlib
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

# Load data
data = load_iris()
features = data.data
target = data.target

plt.figure(figsize=(16, 12))

# Scatter plot of the original features
plt.subplot(3, 2, 1)
plt.scatter(features[:, 0], features[:, 1], c=target, marker='o', cmap='viridis')
plt.title('Original Data')

# Train a MLP classifier on the original data
Classificador = MLPClassifier(hidden_layer_sizes = (100), alpha=3, max_iter=100)
Classificador.fit(features, target)
predicao = Classificador.predict(features)

# Plot the predictions on the original data
plt.subplot(3, 2, 3)
plt.scatter(features[:, 0], features[:, 1], c=predicao, marker='d', cmap='viridis', s=150, label='Prediction')
plt.scatter(features[:, 0], features[:, 1], c=target, marker='o', cmap='viridis', s=15, label='Actual')
plt.title('Predictions on Original Data')
plt.legend()

# Confusion matrix for the original data classifier
plt.subplot(3,2,5)
conf_matrix = confusion_matrix(target, predicao)
plt.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3, fignum=0)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center')
plt.title('Confusion Matrix: Original Data')

# PCA transformation
pca = PCA(n_components=2, whiten=True, svd_solver='randomized')
pca_features = pca.fit_transform(features)
print('Percentage of variance retained: {:.2f}%'.format(sum(pca.explained_variance_ratio_) * 100))

# Scatter plot of PCA-transformed features
plt.subplot(3,2,2)
plt.scatter(pca_features[:,0], pca_features[:, 1], c=target, marker='o', cmap='viridis')
plt.title('PCA Transformed Data')

# Train a MLP classifier on the PCA-transformed data
ClassificadorPCA = MLPClassifier(hidden_layer_sizes = (10), alpha=1, max_iter=1000)
ClassificadorPCA.fit(pca_features, target)
predicao_pca = ClassificadorPCA.predict(pca_features)

# Plot the predictions on the PCA-transformed data
plt.subplot(3,2,4)
plt.scatter(pca_features[:,0], pca_features[:,1], c=predicao_pca, marker='d', cmap='viridis', s=150, label='Prediction')
plt.scatter(pca_features[:,0], pca_features[:,1], c=target, marker='o', cmap='viridis', s=15, label='Actual')
plt.title('Predictions on PCA Data')
plt.legend()

# Confusion matrix for the PCA-transformed data classifier
plt.subplot(3, 2, 6)
conf_matrix_pca = confusion_matrix(target, predicao_pca)
plt.matshow(conf_matrix_pca, cmap=plt.cm.Blues, alpha=0.3, fignum=0)
for i in range(conf_matrix_pca.shape[0]):
    for j in range(conf_matrix_pca.shape[1]):
        plt.text(x=j, y=i, s=conf_matrix_pca[i, j], va='center', ha='center')
plt.title('Confusion Matrix: PCA Data')

loss = Classificador.best_loss_
print(loss)

plt.tight_layout()
plt.show()