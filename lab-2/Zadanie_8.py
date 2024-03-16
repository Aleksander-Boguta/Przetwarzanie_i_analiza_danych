from sklearn import datasets
from sklearn.feature_selection import mutual_info_classif
import numpy as np

iris = datasets.load_iris()
X = iris.data
Y = iris.target

info = mutual_info_classif(X,Y)

atrybuty = np.argsort(info)[-3:]
print([iris.feature_names[index] for index in atrybuty])
print(info[atrybuty])

