from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Załaduj zbiór danych
digits = load_digits()
data = digits.data
target = digits.target

# Funkcja do redukcji wymiarowości
def reduce_dimensions(data, n_components=2):
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(data)
    return transformed_data, pca

# Redukcja do 2 wymiarów
reduced_data, pca = reduce_dimensions(data, 2)

# Krzywa wariancji
pca_full = PCA().fit(data)
plt.figure(figsize=(8, 4))
plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
plt.xlabel('Numer składowej')
plt.ylabel('Skumulowana wariancja')
plt.title('Krzywa wariancji')
plt.show()

# Wizualizacja
plt.figure(figsize=(8, 6))
colormap = plt.cm.get_cmap('tab10', 10)
scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=target, cmap=colormap, lw=0, s=30, alpha=0.5)
plt.colorbar(scatter, ticks=range(10))
plt.xlabel('Pierwsza główna składowa')
plt.ylabel('Druga główna składowa')
plt.title('Wizualizacja zbioru digits po redukcji wymiarów')
plt.show()
