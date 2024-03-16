import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
Y = iris.target



pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)


plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Y, cmap='viridis', edgecolor='k', s=50)
plt.title('PCA zbiór iris')
plt.xlabel('Komponent 1')
plt.ylabel('Komponent 2')
plt.colorbar(label='Gatunek')
plt.show()
