import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn.datasets as datasets
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from scipy.sparse import distance

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

splits = 6

rs = sklearn.model_selestion.ShuffleSplit(n_splits=splits, test_size=.25, random_state=0)
for k in range(1, 6):
    score = 0
    for i, (train_index, test_index) in enumerate(rs.split(X)):


class knn:
    def __init__(self, n_neighbors = 1, use_KDTree = False):

            self.n_neighbors = n_neighbors
            self.use_KDTree = use_KDTree
            self.is_regression = is_regression
            self.model = None
            self.tree = None

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
            if self.use_KDTree and self.tree is not None:
                dist, ind = self.tree.query(X, k=self.n_neighbors)
                if self.is_regression:
                    # Predict by averaging the targets of nearest neighbors
                    return np.mean(y[ind], axis=1)
                else:
                    # Predict by majority vote among the nearest neighbors
                    return mode(y[ind], axis=1)[0].ravel()
            else:
                return self.model.predict(X)
        
        return y

    def score(self, X, y):
        

















