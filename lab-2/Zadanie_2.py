import numpy as np

#Utwórz nową macierz zawierającą znormalizowane dane za pomocą następującego wzoru

a = np.array([[1,5,-2,3,2,3,-1,-2,4,-3,0,1],[2,6,2,7,2,-6,-2,8,1,3,4,-9]]).T

max_x = np.max(np.abs(a), axis=0)

x_norm = a / max_x

#print(x_norm)

print(np.max(x_norm,axis = 0)) #wartości maksymalne w kolumnach
print(np.min(x_norm,axis = 0)) #wartości minimalne w kolumnach
print(np.mean(x_norm,axis = 0)) #wartości średnie dla kolumn
print(np.std(x_norm, axis=0)) #odchylenie standardowe dla kolumn
