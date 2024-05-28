import numpy as np
import matplotlib.pyplot as plt

#ziarno generatora liczb losowych dodane po to żeby ułatwić sobie porównywanie
# wyników (zestaw wartości losowych będzie taki sam z każdym razem)

np.random.seed(13)


X = np.dot(np.random.rand(200, 2), np.random.rand(2, 2)) 



plt.figure()
plt.scatter(X[:, 0], X[:, 1])
plt.title("Wizualizacja 200 obiektów dwuwymiarowych")
plt.show()