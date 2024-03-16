mport numpy as np
import numpy.random as nr


#1. Sprawdź składni funkcji sort i argsort.

A = nr.rand((5,5))

print(np.sort(A, axis=1))


print(np.flip(np.sort(A,axis=0),axis=))

np.sort(A,axis=0)[::-1,:]



#2.Utwórz macierz a = np.random.randn(5, 5), a nast¦pnie p osortuj wiersze rosn¡co. Posortujkolumny malej¡co.

a = np.random.randn(5, 5)

