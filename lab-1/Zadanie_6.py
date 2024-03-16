mport numpy as np
import numpy.random as nr




# 1. Utwórz wektor skªada j¡cy si¦ z 50 liczb.

A = np.arange(50)

# 2. Za p omo c¡ funkcji 'reshap e' utwórz macierz 10x5. Spróbuj dokona¢ tego samego za p omo c¡funkcji resize.


A = np.arange(50)
B = A.reshape((10,5))
print(B)

A = np.arange(50)
A.resize(10,5)
print(A)


# 3. Sprawd¹ do czego sªu»y komenda ravel.

print(B.ravel())

# 4. Stwórz dwie tablice o rozmiarach 5 i 4 i do da j je do siebie. Sprawd¹ do czego sªu»y funkcja'NEWAXIS' i wykorzysta j j¡

A = np.arange(5)
B = np.arange(4)

A[:,np.NEWAXIS] + B


