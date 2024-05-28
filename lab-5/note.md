Zadania z metod klasyfikacji


   def fit(self, X, y):

        if self.is_regression:
            self.model = KNeighborsRegressor(n_neighbors=self.n_neighbors)
        else:
            self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        
        self.model.fit(X, y)
        
        if self.use_KDTree:
            self.tree = KDTree(X)        
       


    def predict(self, X):
        for X
            if self.use_KDTree:
            _, idx = self.KDTree.query(..., self.k)
        
        return y

    def score(self, X, y):
        

________________________________________________


wersja 1 kodu zadanie 1

from collections import Counter
import numpy as np
from scipy.stats import mode
from sklearn.metrics import mean_squared_error, accuracy_score

class KNearestNeighbors:
    def __init__(self, n_neighbors=1, use_KDTree=False):
        self.n_neighbors = n_neighbors
        self.use_KDTree = use_KDTree
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        # Przechowuje dane treningowe
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        # Wykonuje predykcję dla każdego punktu w zbiorze X
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        # Oblicza odległości euklidesowe od punktu x do wszystkich punktów uczących
        distances = [np.sqrt(np.sum((x_train - x) ** 2)) for x_train in self.X_train]
        # Znajduje indeksy k najbliższych sąsiadów
        k_indices = np.argsort(distances)[:self.n_neighbors]
        # Znajduje najczęstszą etykietę wśród sąsiadów
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        if self._is_regression():
            # Średnia wartość dla regresji
            prediction = np.mean(k_nearest_labels)
        else:
            # Najczęstsza etykieta dla klasyfikacji
            prediction = mode(k_nearest_labels)[0][0]
        return prediction

    def score(self, X, y):
        # Oblicza predykcję
        predictions = self.predict(X)
        if self._is_regression():
            # Średni błąd kwadratowy dla regresji
            return mean_squared_error(y, predictions)
        else:
            # Dokładność dla klasyfikacji
            return accuracy_score(y, predictions)

    def _is_regression(self):
        # Sprawdza czy zadanie jest regresją (jeśli y jest ciągłym typem danych)
        return np.issubdtype(self.y_train.dtype, np.floating)

# Przykładowe użycie:
# knn = KNearestNeighbors(n_neighbors=3)
# knn.fit(X_train, y_train)
# predictions = knn.predict(X_test)
# score = knn.score(X_test, y_test)


__________________________________________________________________

Trzeba ogarnąć chat i dać mu tylko tyle zadań ile treba bo robi na zapas

