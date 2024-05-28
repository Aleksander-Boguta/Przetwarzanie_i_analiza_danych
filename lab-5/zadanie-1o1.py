import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from scipy.stats import mode
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

class knn:
    def __init__(self, n_neighbors=1, use_KDTree=False):
        self.n_neighbors = n_neighbors
        self.use_KDTree = use_KDTree
        self.X_train = None
        self.y_train = None
        self.tree = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        if self.use_KDTree:
            self.tree = KDTree(self.X_train)

    def predict(self, X):
        if self.use_KDTree:
            dist, ind = self.tree.query(X, k=self.n_neighbors)
            k_nearest_labels = self.y_train[ind]
            if self._is_regression():
                return np.mean(k_nearest_labels, axis=1)
            else:
                return mode(k_nearest_labels, axis=1).mode[0]
        else:
            predictions = [self._predict(x) for x in X]
            return np.array(predictions)

    def _predict(self, x):
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        k_indices = np.argsort(distances)[:self.n_neighbors]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        if self._is_regression():
            return np.mean(k_nearest_labels)
        else:
            return mode(k_nearest_labels).mode[0]

    def score(self, X, y):
        predictions = self.predict(X)
        if self._is_regression():
            return mean_squared_error(y, predictions)
        else:
            return accuracy_score(y, predictions)

    def _is_regression(self):
        return np.issubdtype(self.y_train.dtype, np.floating)

# Przykładowe użycie:
# Załadowanie danych uczących i testowych
# X_train, y_train, X_test, y_test = ... (tutaj załadować odpowiednie dane)

# Utworzenie instancji klasy knn
model = knn(n_neighbors=3, use_KDTree=True)

# Trenowanie modelu
model.fit(X_train, y_train)

# Predykcja na danych testowych
predictions = model.predict(X_test)

# Ocena modelu
model_score = model.score(X_test, y_test)

# Wypisanie wyniku oceny
print(f'Model score: {model_score}')