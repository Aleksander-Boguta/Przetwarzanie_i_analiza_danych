import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
X = np.array([ np.concatenate( (np.random.randn(200),
np.random.randn(800) * 2 + 7)) ]).T

H , bins = np.histogram(X, bins = 20)
binWidth = bins[1] - bins[0];
H = H / np.sum(H)

t = np.array([np.linspace(np.min(X),np.max(X),1000)]).T #uzu




for ker in ["gaussian" , "tophat" , "linear" , "cosine"]:
# utworzony obiekt KernelDensity
    kd = KernelDensity(kernel = ker, bandwidth=0.5) 
    kd.fit(X[:,np.newaxis])

    y = np.exp(kd.score_samples(t))

    plt.figure()
    plt.bar(bins[:-1] + binWidth / 2, H, width= binWidth)
    plt.plot(t , y , "r")  #dodaje rozkład do histogramu
    plt.title("Kernel: " + ker)

plt.show()


for ker in range(2 , 6):
    #Utworzony obiekt GaussianMixture
    gm.fit(X)
    pr = gm.predict_proba()
    y = np.exp(gm.score_samples())

    plt.figure()
    plt.subplot(1 , 2, 1)
    plt.bar(bins[:-1] + binWidth / 2, H, width= binWidth)
    plt.plot(t , y , "r")  #dodaje rozkład do histogramu
    plt.title("N = : " + comp)

    plt.subplot(1 ,2 , 1)
    plt.plot(t , pr)
    plt.title("BIC = " + ...)
plt.show()
