import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier

from sklearn import metrics
from sklearn.model_selection import train_test_split

X, y = datasets.make_classification(n_samples=2000, n_features=2, n_informative=2, n_classes=4, n_clusters_per_class=1, n_redundant=0, n_repeated=0, random_state=15, class_sep=2.5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

clss = [OneVsOneClassifier( SVC(kernel='linear', probability=True)),
         OneVsRestClassifier( SVC(kernel='linear', probability=True)),
         OneVsOneClassifier( SVC(kernel='rbf', probability=True)),
         OneVsRestClassifier( SVC(kernel='rbf', probability=True)),
         OneVsOneClassifier( LogisticRegression()),
         OneVsRestClassifier( LogisticRegression()),
         OneVsOneClassifier( GaussianNB()),
         OneVsRestClassifier( GaussianNB())]

for cl in clss:
    clssname = cl.__class__.__name__ + "(" + cl.estimator.__class__.__name__ + ")"
    
    #cl.fit
    #d = predict
    
    acc = 
    sen =   #avrade = "weighted"
    prc = 
    f1 = 
    
    if cl.__class__.__name__ == "OneVsRestClassifier":
        #pr = cl.predict_prob
        #auc = ... ---- avreage = "weighted" , multi_class = "ovr"
        
        for i in range(pr.shape[1]):
            #y_test_perc_cls = (y_test == i)
            #rysowanie roc, pr[:, i]
            #plt.plot(xroc, yroc, label = f"cls = {i}"")
        plt.legend()
            
    errr = abs(y_test - d)
    errr = errr >= 1
    
    #wykres bledów
    #wykres decyzyjnych
    # plt.contourf(xx, yy, np.reshape(zz, (nc, nc)), [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]) #4.5 jest nadmiarowe
    
    