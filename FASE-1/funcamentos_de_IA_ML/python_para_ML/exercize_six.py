# Importa apenas a classe LinearRegression do módulo sklearn.linear_model
# "from X import Y" permite importar somente o que precisa, sem carregar a biblioteca toda
# scikit-learn (sklearn) é a principal biblioteca de Machine Learning em Python —
# contém algoritmos prontos de classificação, regressão, clustering, etc.
from sklearn.linear_model import LinearRegression

# Importa numpy com o apelido "np" (convenção universal)
# NumPy é a base de computação numérica em Python — fornece arrays multidimensionais
# e operações matemáticas otimizadas (muito mais rápidas que listas Python puras)
import numpy as np

# Cria a matriz de features X (variáveis de entrada do modelo)
# np.array() converte uma lista de listas em um array NumPy 2D (matriz)
# Resultado: matriz 4x2 — 4 amostras, cada uma com 2 features
# [ [1,1],   <- amostra 1: feature_1=1, feature_2=1
#   [1,2],   <- amostra 2: feature_1=1, feature_2=2
#   [2,2],   <- amostra 3: feature_1=2, feature_2=2
#   [2,3] ]  <- amostra 4: feature_1=2, feature_2=3
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])

# Cria o vetor y (variável alvo / rótulos que o modelo deve aprender a prever)
# np.dot() realiza o produto escalar (multiplicação matricial) entre X e o vetor [1, 2]
# Isso calcula: y = (feature_1 * 1) + (feature_2 * 2) + 3
# Resultados: [1*1 + 1*2 + 3, 1*1 + 2*2 + 3, 2*1 + 2*2 + 3, 2*1 + 3*2 + 3] = [6, 8, 9, 11]
# O objetivo do exercício é que o modelo descubra sozinho os coeficientes [1, 2] e o +3
y = np.dot(X, np.array([1, 2])) + 3

# Instancia o modelo de Regressão Linear e já o treina com os dados X e y
# LinearRegression() cria o modelo (sem parâmetros = usa defaults)
# .fit(X, y) é o treinamento: o algoritmo ajusta os coeficientes internos para
# encontrar a reta/plano que melhor descreve a relação entre X e y
# Analogia: é como calibrar uma função — o modelo aprende os pesos de cada feature
model = LinearRegression().fit(X, y)

# Exibe os coeficientes aprendidos pelo modelo para cada feature
# model.coef_ é um atributo que armazena os pesos encontrados pelo algoritmo
# O sufixo "_" em Python (por convenção do sklearn) indica atributos calculados após o .fit()
# Resultado esperado: [1. 2.] — o modelo recuperou exatamente os coeficientes que usamos
# para gerar y, provando que o aprendizado funcionou corretamente
print(model.coef_)