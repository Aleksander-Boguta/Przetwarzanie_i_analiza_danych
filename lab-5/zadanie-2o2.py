import numpy as np
from sklearn.neighbors import KDTree
from scipy.stats import mode
from sklearn.metrics import mean_squared_error, accuracy_score

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
        # Używa kD-drzewa jeśli flaga use_KDTree jest ustawiona na True
        if self.use_KDTree:
            self.tree = KDTree(self.X_train)

    def predict(self, X):
        # Korzysta z kD-drzewa do predykcji jeśli zostało ono wcześniej utworzone
        if self.use_KDTree and self.tree is not None:
            _, indices = self.tree.query(X, k=self.n_neighbors)
            predictions = []
            for index_array in indices:
                if self._is_regression():
                    prediction = np.mean(self.y_train[index_array])
                else:
                    prediction = mode(self.y_train[index_array]).mode[0]
                predictions.append(prediction)
            return np.array(predictions)
        else:
            # Standardowa metoda obliczeń bez kD-drzewa
            return np.array([self._predict(x) for x in X])

    # Pozostałe metody niezmienione ...

# Przykładowe użycie i reszta kodu pozostaje bez zmian ...
