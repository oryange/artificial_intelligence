# ============================================================
# Importação da biblioteca
# ============================================================
# "sklearn" é o scikit-learn, a biblioteca mais popular de
# Machine Learning em Python.
# "svm" é o submódulo de Support Vector Machines (Máquinas de
# Vetores de Suporte), uma família de algoritmos de classificação.
# "LinearSVC" (Linear Support Vector Classifier) é um classificador
# que tenta encontrar uma reta (ou hiperplano) que melhor separe
# as classes dos dados. "Linear" significa que a fronteira de
# decisão é uma linha reta (em 2D) ou um plano (em 3D+).
from sklearn.svm import LinearSVC

# ============================================================
# Criação dos dados de treino (amostras / exemplos conhecidos)
# ============================================================
# Cada variável é uma lista com 3 valores (chamados de
# "features" ou "características"). Cada valor é 0 ou 1,
# representando a presença (1) ou ausência (0) de uma
# propriedade do composto.
# Exemplo: composto1 = [1, 1, 1] → possui as 3 propriedades.
composto1 = [1, 1, 1]  # propriedade A=sim, B=sim, C=sim
composto2 = [0, 0, 0]  # propriedade A=não, B=não, C=não
composto3 = [1, 0, 1]  # propriedade A=sim, B=não, C=sim
composto4 = [0, 1, 0]  # propriedade A=não, B=sim, C=não
composto5 = [1, 1, 0]  # propriedade A=sim, B=sim, C=não
composto6 = [0, 0, 1]  # propriedade A=não, B=não, C=sim

# ============================================================
# Organização dos dados em listas para o modelo
# ============================================================
# "dados_treino" é uma lista de listas (uma "matriz").
# Cada lista interna é uma amostra; o conjunto todo forma a
# tabela que o modelo vai usar para aprender.
# Formato: [[amostra1], [amostra2], ..., [amostraN]]
dados_treino = [composto1, composto2, composto3, composto4, composto5, composto6]

# "rotulos_treino" contém a resposta correta (rótulo / label)
# de cada amostra, na mesma ordem.
# 'S' = Sim (classe positiva)  |  'N' = Não (classe negativa)
# O modelo vai aprender que composto1 → 'S', composto2 → 'N', etc.
rotulos_treino = ['S', 'N', 'S', 'N', 'S', 'S']

# ============================================================
# Criação e treinamento do modelo
# ============================================================
# LinearSVC() cria uma INSTÂNCIA do classificador.
# Neste ponto ele ainda não sabe nada; é como um aluno
# antes de estudar.
modelo = LinearSVC()

# .fit() é o método que TREINA o modelo.
# Ele recebe os dados (X) e os rótulos (y) e ajusta os
# parâmetros internos (pesos) para encontrar a melhor
# fronteira linear que separe 'S' de 'N'.
# Depois desta linha o modelo já "aprendeu" com os exemplos.
modelo.fit(dados_treino, rotulos_treino)

# ============================================================
# Criação dos dados de teste (amostras novas, nunca vistas)
# ============================================================
# Agora criamos compostos que o modelo NUNCA viu durante o
# treino. Queremos saber se ele consegue generalizar e
# classificar corretamente.
teste1 = [1, 0, 0]  # propriedade A=sim, B=não, C=não
teste2 = [0, 1, 1]  # propriedade A=não, B=sim, C=sim
teste3 = [1, 1, 1]  # propriedade A=sim, B=sim, C=sim

# Agrupamos os testes em uma lista de listas, mesmo formato
# usado no treino.
dados_teste = [teste1, teste2, teste3]

# ============================================================
# Previsão e exibição dos resultados
# ============================================================
# .predict() recebe as amostras de teste e retorna um array
# com a classe prevista para cada uma delas.
# Internamente o modelo aplica a regra que aprendeu no .fit()
# para decidir se cada amostra é 'S' ou 'N'.
previsoes = modelo.predict(dados_teste)

mapeamento_previsoes = {'S': 'Solúvel', 'N': 'Insolúvel'}
# print() exibe no terminal o resultado, por exemplo:
# ['S' 'S' 'S']  (as previsões podem variar conforme o treino)
print("Previsões do modelo para os compostos testados:", previsoes)
for i, prev in enumerate(previsoes):
    print(f'O composto {i+1} pode ser considerado {mapeamento_previsoes[prev]}')