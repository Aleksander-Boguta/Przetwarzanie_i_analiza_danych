import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc

from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

X, y = datasets.make_classification(n_samples=2000, n_features=2, n_informative=2, n_classes=4, n_clusters_per_class=1, n_redundant=0, n_repeated=0, random_state=15, class_sep=2.5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

classifiers = [
    OneVsOneClassifier(SVC(kernel='linear', probability=True)),
    OneVsRestClassifier(SVC(kernel='linear', probability=True)),
    OneVsOneClassifier(SVC(kernel='rbf', probability=True)),
    OneVsRestClassifier(SVC(kernel='rbf', probability=True)),
    OneVsOneClassifier(LogisticRegression()),
    OneVsRestClassifier(LogisticRegression()),
    OneVsOneClassifier(GaussianNB()),
    OneVsRestClassifier(GaussianNB())
]

for clf in classifiers:
    clf_name = clf.__class__.__name__ + " (" + clf.estimator.__class__.__name__ + ")"
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test) if hasattr(clf, 'predict_proba') else None

    acc = metrics.accuracy_score(y_test, y_pred)
    sen = metrics.recall_score(y_test, y_pred, average="weighted")
    prc = metrics.precision_score(y_test, y_pred, average="weighted")
    f1 = metrics.f1_score(y_test, y_pred, average="weighted")

    print(f"{clf_name}: Accuracy={acc:.2f}, Recall={sen:.2f}, Precision={prc:.2f}, F1 Score={f1:.2f}")

    if clf.__class__.__name__ == "OneVsRestClassifier" and y_prob is not None:
        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(4):
            fpr[i], tpr[i], _ = roc_curve(y_test == i, y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')

        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.show()

    # Wizualizacja wyników klasyfikacji
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')
    plt.title(f'Decision Boundary for {clf_name}')
    plt.show()

    # Wizualizacja błędów
    errors = (y_test != y_pred).astype(int)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=errors, cmap='coolwarm', edgecolor='k', alpha=0.5)
    plt.title(f'Classification Errors for {clf_name}')
    plt.show()
