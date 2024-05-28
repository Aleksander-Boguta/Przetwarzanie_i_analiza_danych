import numpy as np
import numpy.random
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity as kd
from sklearn.mixture import GaussianMixture as gm
X = np.array([ np.concatenate( (np.random.randn(200),
np.random.randn(800) * 2 + 7)) ]).T

H , bins = np.histogram(X, bins = 20)
binWidth = bins[1] - bins[0];
H = H / np.sum(H)

t = np.array([np.linspace(np.min(X),np.max(X),1000)]).T #uzu




for ker in ["gaussian" , "tophat" , "linear" , "cosine"]:
# utworzony obiekt KernelDensity
     
    kd.fit(X[:,np.newaxis])

    y = np.exp(kd.score_samples())

    plt.figure()
    plt.bar(bins[:-1] + binWidth / 2, H, width= binWidth)
    plt.plot(t , y)  #dodaje rozkład do histogramu
    plt.title("Kernel: " + ker)

plt.show()