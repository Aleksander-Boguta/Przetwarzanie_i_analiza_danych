import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA

# Załaduj zbiór danych Iris
iris = datasets.load_iris()
data = iris.data
target = iris.target
target_names = iris.target_names

# Redukcja wymiarów do dwóch komponentów
pca = PCA(n_components=2)
data_reduced = pca.fit_transform(data)

# Wizualizacja danych
colors = ['red', 'green', 'darkblue']
s = 15 #rozmiar punktów

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(data_reduced[target == i, 0], data_reduced[target == i, 1], color=color, alpha=0.9, s=s, lw=1, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA IRIS dataset')
plt.show()