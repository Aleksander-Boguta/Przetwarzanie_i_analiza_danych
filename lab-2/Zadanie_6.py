import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


a = np.array([
    ['red', 'yes'],
    ['red', 'yes'],
    ['green', 'no'],
    ['blue', 'no'],
    ['blue', 'no'],
    ['green', 'yes'],
    ['red', 'no'],
    ['red', 'no'],
    ['yellow', 'yes']
])




encoder = ColumnTransformer(
    transformers=[
        ('kolor', OneHotEncoder(sparse_output=False), [0]),  # Pełne kodowanie one-hot dla kolorów
        ('yes/no', OneHotEncoder(drop='if_binary'), [1])  # Binarne kodowanie dla yes/no
    ]
)

# Wywołanie enkodera na macierzy a
bin_a = encoder.fit_transform(a)

print(bin_a)

# Dla atrybutu 1 binaryzacja skutkuje powstaniem czterech atrybutów po jednym na każdy kolor
# Dla binaryzacji atrybutu 2, optymalna jest jedna kolumna (wartość yes->1 no->0)
