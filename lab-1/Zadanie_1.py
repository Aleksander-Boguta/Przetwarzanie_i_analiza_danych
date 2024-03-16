import numpy as np
import numpy.random as nr
#1. Utwórz tablice jednowymiarową zawierającą kolejne cyfry o d 1 do 7

Tab = np.array([1,2,3,4,5,6,7])

# Wersja2

Tab2 = np.arange(1,8)

#2. Utwórz tablice dwuwymiarow¡ o p ostaci:

Tablica_dwuwymiarowa = np.array([[1,2,3,4],[5,6,7,8]])

#3. Utwórz macierz transp onowan¡ na p o dstawie macierzy z pkt. 2.

Transp = Tablica_dwuwymiarowa.T
print(Transp)

#4. Za p omo c¡ p olecenia arange utwórz wektor zawiera j¡cy warto±ci o d 1 do 20 z krokiem 0.5.

Wektor = np.arange(1,20.5,0.5)

#5. Za p omo c¡ linspace utwórz wektor zawiera j¡cy 100 równomiernie rozªo»onych warto±ci zprzedziaªu [0, 5].

Wektor_linspace = np.linspace(0,5,100)