import numpy as np
from sklearn import datasets
from sklearn.feature_selection import VarianceThreshold


iris = datasets.load_iris()
X = iris.data #<---tu są kolumny bez klasy, dane od początku są podzielone
Y = iris.target 

#Usuwanie atrybutów o niskiej wariancji

selector = VarianceThreshold(threshold=0.2)
X_threshold = selector.fit_transform(X)

kolumny = X.shape[1] - X_threshold.shape[1]

print("Liczba usuniętych kolumn rówan się:")
print(kolumny)


#Usunięto jedną kolumnę