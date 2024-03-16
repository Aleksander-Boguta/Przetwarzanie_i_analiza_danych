import numpy as np
import numpy.random as nr


# 1. Utwórz macierz: b = np.array([[1,2,3,4,5], [6,7,8,9,10]], dtyp e = np.int32)

b = np.array([[1,2,3,4,5], [6,7,8,9,10]], dtype = np.int32)

# 2. Za p omo c¡ p ola klasy 'ndim' sprawd¹ ile wymiarów ma tablica b.

print(b.ndim)

# 3. Za p omo c¡ p ola klasy 'shap e', sprawd¹ wymiary tablicy b.

print(b.shape)

# 4. Wybierz warto±ci 2 i 4 z tablicy b.

print(b[0,1])

# 5. Wybierz pierwszy wiersz tablicy b.

print(b[0,3])

# 6. Wybierz wszystkie wiersze z kolumny 1.

print(b[0,:])
print(b[:,0])

# 7. Wygeneruj macierz losow¡ o rozmiarze 20x7, zªo»on¡ liczb caªkowitych w przedziale 0-100.Wy±wietl wszystkie wiersze dla czterech pierwszych kolumn.

c = nr.randint(0,100,(20,7))
print(c[:,0:4])

c[c>4]=0