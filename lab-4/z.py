import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Generate data
X = np.concatenate((np.random.randn(200), np.random.randn(800) * 2 + 7)).reshape(-1, 1)

# Compute histogram
H, bins = np.histogram(X, bins=20)
binWidth = bins[1] - bins[0]
H_normalized = H / np.sum(H)

# Create a dense grid of values
t = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)

# Kernel Density Estimation
for ker in ["gaussian", "tophat", "linear", "cosine"]:
    kd = KernelDensity(kernel=ker, bandwidth=0.5)
    kd.fit(X)
    y = np.exp(kd.score_samples(t))

    plt.figure()
    plt.bar(bins[:-1] + binWidth / 2, H_normalized, width=binWidth)
    plt.plot(t, y, "r")
    plt.title("Kernel: " + ker)

plt.show()

# Gaussian Mixture Models
for n_components in range(2, 6):
    gmm = GaussianMixture(n_components=n_components, tol=1e-5)
    gmm.fit(X)
    logprob = gmm.score_samples(t)
    responsibilities = gmm.predict_proba(t)
    bic = gmm.bic(X)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(bins[:-1] + binWidth / 2, H_normalized, width=binWidth)
    plt.plot(t, np.exp(logprob), "r")
    plt.title(f"GMM: N = {n_components}")

    plt.subplot(1, 2, 2)
    plt.plot(t, responsibilities)
    plt.title(f"BIC = {bic:.2f}")
    plt.tight_layout()

plt.show()
# Load Iris dataset
iris = load_iris()
X = iris.data
Y = iris.target

# PCA for dimensionality reduction
pca = PCA(n_components=2)
Xt = pca.fit_transform(X)

# Apply GMM and visualize results
for n_components in range(2, 6):
    gmm = GaussianMixture(n_components=n_components, tol=1e-5)
    gmm.fit(Xt)
    bic = gmm.bic(Xt)
    d = gmm.predict(Xt)

    plt.figure(figsize=(10, 5))
    
    # Original PCA-reduced data with true labels
    plt.subplot(1, 2, 1)
    plt.scatter(Xt[:, 0], Xt[:, 1], c=Y)
    plt.title(f"PCA-reduced Iris with True Labels, N = {n_components}")

    # PCA-reduced data with GMM-predicted labels
    plt.subplot(1, 2, 2)
    plt.scatter(Xt[:, 0], Xt[:, 1], c=d)
    plt.title(f"GMM Predicted, BIC = {bic:.2f}")
    
    plt.figure()
    gs = GaussianMixture (n_components=comp, tol=1e-6)
    gs.fit(Xt)

    nc = 1000
    x1min = 
    x1max = 
    x2min = 
    x2max = 
    x1range = np.linspace(x1min, x1max, nc)
    x2range = np.linspace(x2min, x2max, nc)
    xx , yy =np.exp(gm.scores_samples(np.array([xx.ravel(), yy.ravel()]).resize(-1 , 1)))
    

    plt.figure()
    plt.contour(xx , yy , np.reshape(zz, (nc, nc)), np.arrange(0.1, 1.0, 0.1))
    plt.scatter(Xt[:,1])
    plt.tight_layout()
    plt.show()



#Zadanie 6

