import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture

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
