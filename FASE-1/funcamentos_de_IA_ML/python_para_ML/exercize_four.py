# Importa a biblioteca matplotlib e apelida o módulo pyplot de "plt"
# matplotlib é a principal biblioteca de visualização de dados em Python
# pyplot é o submódulo que fornece funções para criar gráficos (similar a uma API)
# O apelido "plt" é uma convenção amplamente adotada pela comunidade
import matplotlib.pyplot as plt

# Define uma lista com os valores do eixo X (eixo horizontal)
# Em Python, listas são criadas com colchetes [] — equivalente a um Array em Kotlin
x = [1, 2, 3, 4, 5]

# Define uma lista com os valores do eixo Y (eixo vertical)
# Estes são os números primos: cada valor de Y corresponde ao mesmo índice de X
# Ex: quando x=1, y=2 | quando x=3, y=5 | quando x=5, y=11
y = [2, 3, 5, 7, 11]

# Cria a linha do gráfico ligando os pares de pontos (x[i], y[i])
# plt.plot() recebe os dados e prepara o gráfico na memória, mas ainda não exibe
# Analogia Kotlin: seria como construir um objeto View sem ainda adicioná-lo à tela
plt.plot(x, y)

# Exibe o gráfico em uma janela interativa
# Sem esta chamada, o gráfico seria construído mas nunca renderizado
# Analogia Kotlin: equivale ao setContentView() ou ao momento em que a Activity fica visível
plt.show()