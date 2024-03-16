import numpy as np
import numpy.random as nr

#1. Za p omo c¡ funkcji random utwórz tablic¦ z 20 liczb losowych rozkªadu normalnego, za-okr¡ glonych do dwó ch miejsc p o przecinku.

Tab = nr.rand(nr.normal(size = 20),decimals=2)
print(Tab)

#2. Wygeneruj losowo 100 liczb caªkowitych w zakresie o d 1 do 1000

Tab2 = nr.randint(1,1000,100)

# 3. Za p omo c¡ funkcji 'zeros' i 'ones' wygeneruj dwie macierze o rozmiarze 3x2.

zeros = np.zeros((2,3))

ones = np.ones((2,3))

# 4. Utwórz macierz losow¡ zªo»on¡ z liczb caªkowitych o rozmiarze 5x5 i nada j jej typ 32bit.

TabZ = nr.randint(1,10, (5,5) , dtype = 'int32')

# 5. Wygeneruj tablic¦ zªo»on¡ z losowo wybranych liczb dziesi¦tnych o d 0 - 10:

M = nr.rand((5,5)) * 10

A = np.astype('int32')

B = nr.rand().astype('int32')

print(A)
print(B)

# (a) Zamie« warto±ci na 'integer' i wstaw w now¡ tablic¦.



# (b) Zna jd¹ funkcj¦ 'numpy', która zaokr¡ gli tablic¦ (a) do liczb caªkowitych. Zamie« jenast¦pnie na typ 'integer'.



# (c) Porówna j wyniki z a i b.