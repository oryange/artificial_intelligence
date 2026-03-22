# Importa a biblioteca NumPy e cria um "apelido" chamado 'np' para ela.
# NumPy é a principal biblioteca de Python para trabalhar com arrays e operações matemáticas.
# O "as np" é uma convenção amplamente usada — evita escrever "numpy" inteiro toda vez.
import numpy as np

# Cria uma variável chamada 'data' que armazena um array 2D (matriz) usando o NumPy.
# np.array() converte uma lista de listas do Python em um array NumPy otimizado.
# O resultado é uma matriz 3x3:
#   [[1, 2, 3],
#    [4, 5, 6],
#    [7, 8, 9]]
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Calcula a média aritmética de TODOS os elementos da matriz e armazena em 'mean'.
# np.mean() soma todos os valores e divide pela quantidade de elementos.
# Aqui: (1+2+3+4+5+6+7+8+9) / 9 = 45 / 9 = 5.0
mean = np.mean(data)

# Exibe o valor da média no terminal.
# print() é a função nativa do Python para imprimir valores na saída padrão.
# Saída esperada: 5.0
print(mean)