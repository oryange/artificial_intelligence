# Importa especificamente a função 'solve_ivp' de dentro do módulo scipy.integrate.
# 'from X import Y' é diferente de 'import X as np' — aqui você pega apenas uma função,
# sem precisar escrever 'scipy.integrate.solve_ivp(...)' toda vez.
# solve_ivp = "Solve Initial Value Problem" → resolve equações diferenciais ordinárias (EDO).
from scipy.integrate import solve_ivp

# Define uma função chamada 'dydt' que representa a equação diferencial dy/dt = -0.5 * y.
# Em Python, funções são definidas com 'def', seguido do nome e dos parâmetros entre parênteses.
# Parâmetros:
#   t → o tempo atual (variável independente)
#   y → o valor atual da função (variável dependente)
def dydt(t, y):
    # O corpo da função retorna o valor da derivada no instante t.
    # 'return' em Python funciona igual ao Kotlin — encerra a função e devolve o valor.
    # A equação -0.5 * y descreve um decaimento exponencial (ex: desintegração radioativa).
    return -0.5 * y

# Resolve numericamente a equação diferencial usando solve_ivp.
# Argumentos:
#   dydt     → a função que define a EDO (passada como referência, igual a lambdas/function refs em Kotlin)
#   [0, 10]  → intervalo de tempo: começa em t=0 e termina em t=10
#   [2]      → condição inicial: y(0) = 2 (valor de y no instante zero)
# O resultado é um objeto com vários atributos sobre a solução encontrada.
solution = solve_ivp(dydt, [0, 10], [2])

# Exibe os valores de y calculados ao longo do tempo.
# 'solution.y' é um array 2D com as soluções — acesso a atributo igual ao Kotlin (objeto.propriedade).
# Cada coluna corresponde a um instante de tempo; os valores mostram o decaimento de y=2 até próximo de 0.
print(solution.y)