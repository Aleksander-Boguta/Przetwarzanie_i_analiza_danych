import numpy as np
import numpy.random as nr

# 1. Stwórz dwie macierze w przedziale 0-10 o rozmiarach 3x3 (a i b). Dodaj, pomnó», podziel,sp ot¦guj ich elementy przez siebie.

A = nr.randint(1,10,(3,3))
B = nr.randint(1,10,(3,3))
# 2. Sprawd¹ czy jakakolwiek element z macierzy a jest wi¦kszy lub równy 4.

np.any(A>4)


# 3. Sprawd¹ czy jakakolwiek element z macierzy a nale»y do przedziaªu [1, 4].

np.any((A>= 1) & (A<=4))

# 4. Zna jd¹ funkcj¦ w 'numpy' do obliczenia sumy gªównej przek¡tnej macierzy b.

np.trace(A)