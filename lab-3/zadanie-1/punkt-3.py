import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#Własna implementacja PCA
def PCA_transform(X, dim):
    cov = np.cov(X.T)
    lam, p = np.linalg.eig(cov)
    ind = np.argsort(-lam)
    lam = lam[ind]
    p = p[:, ind]
    Xt = X @ p[:, :dim]
    explain_var = np.cumsum(lam) / np.sum(lam)
    return Xt, lam, p, explain_var

# Przygotowanie danych
np.random.seed(13)
X = np.dot(np.random.rand(200, 2), np.random.rand(2, 2))

# Wykonanie PCA
pca = PCA(n_components=2)
pca.fit(X)

# Rzut danych na pierwszą główną składową
X_pca = pca.transform(X)
X_projected = np.dot(X_pca[:, 0][:, np.newaxis], pca.components_[0][:, np.newaxis].T)
X_projected += pca.mean_  # Dodajemy średnią, aby przesunąć rzuty

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Wizualizacja oryginalnych danych jako zielonych punktów
axes[0].scatter(X[:, 0], X[:, 1], color='green', alpha=0.5)

# Wizualizacja rzutów na pierwszą składową jako czerwonych punktów
axes[0].scatter(X_projected[:, 0], X_projected[:, 1], color='red', alpha=0.5)

# Rysowanie wektorów własnych
for length, vector in zip(pca.explained_variance_, pca.components_.T):
    v = vector * 2 * np.sqrt(length)
    axes[0].arrow(pca.mean_[0], pca.mean_[1], v[0], v[1], width=0.01, head_width=0.02, head_length=0.02, fc='black', ec='black')
axes[0].set_title('sklearn PCA')
axes[0].grid(True)
axes[0].axis('equal')

# Wykonanie PCA_transform 
Xt, lam, p, explain_var = PCA_transform(X, 2)

# Rzut danych na pierwszą główną składową (PC1)
X_projected = np.dot(X, p[:, :1]) * p[:, :1].T

# Wizualizacja oryginalnych danych jako zielonych punktów
axes[1].scatter(X[:, 0], X[:, 1], color='green', alpha=0.5)
# Wizualizacja rzutów na PC1 jako czerwonych punktów
axes[1].scatter(X_projected[:, 0], X_projected[:, 1], color='red', alpha=0.5)
# Rysowanie wektorów własnych jako strzałek
for length, vector in zip(lam, p.T):
    v = vector * 2 * np.sqrt(length)
    axes[1].arrow(np.mean(X[:, 0]), np.mean(X[:, 1]), v[0], v[1], width=0.01, head_width=0.02, head_length=0.02, fc='black', ec='black')

axes[1].set_title('Własna funkcja PCA')
axes[1].grid(True)
axes[1].axis('equal')

plt.tight_layout()
plt.show()
