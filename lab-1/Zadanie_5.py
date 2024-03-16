mport numpy as np
import numpy.random as nr

A = nr.randint(1,10,(3,3))
B = nr.randint(1,10,(3,3))


# 1. Oblicz sum¦, warto±¢ minimum, maksimum, odchylenie standardowe dla macierzy b.

A+B

A-B

A*B

A@B

#Alternatywnie

np.dot(A,B)

A/B

A**B

np.sum(A)

np.min(A)

np.max(A)

np.sto(A)



# 2. Oblicz ±redni¡ dla wierszy w macierzy b.

np.mean(A,1)

# 3. Oblicz ±redni¡ dla kolumn macierzy b.

np.mean(A,0)

