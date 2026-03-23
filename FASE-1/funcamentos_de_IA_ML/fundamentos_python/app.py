# Importa o algoritmo LinearSVC (Support Vector Classifier Linear) para classificação
from sklearn.svm import LinearSVC

# Importa a função que calcula a acurácia (percentual de acertos) do modelo
from sklearn.metrics import accuracy_score

# -------------------------------------------------------
# DADOS DE TREINAMENTO
# Cada ação é representada por 3 características (features) binárias.
# Exemplo: [1, 0, 1] pode representar indicadores técnicos como
# (preço acima da média, volume baixo, tendência de alta)
# -------------------------------------------------------
acao1 = [1,0,1]
acao2 = [0,1,0]
acao3 = [1,1,1]
acao4 = [0,0,1]
acao5 = [1,1,0]
acao6 = [0,1,1]

# Lista com todos os exemplos de treinamento
dados_treino = [acao1, acao2, acao3, acao4, acao5, acao6]

# Rótulos (labels) correspondentes a cada exemplo de treino:
# 1 = preço vai subir | 0 = preço vai cair
rotulos_treino = [1, 1, 1, 0, 0, 0]

# -------------------------------------------------------
# CRIAÇÃO E TREINAMENTO DO MODELO
# -------------------------------------------------------

# Instancia o modelo LinearSVC (um classificador baseado em SVM linear)
modelo = LinearSVC()

# Treina o modelo: ele aprende o padrão que diferencia subida (1) de queda (0)
# com base nos dados e rótulos de treinamento
modelo.fit(dados_treino, rotulos_treino)

# -------------------------------------------------------
# DADOS DE TESTE
# Exemplos novos, que o modelo nunca viu durante o treinamento
# -------------------------------------------------------
teste1 = [1, 0, 0]
teste2 = [0, 1, 1]
teste3 = [1, 1, 0]

# Lista com os exemplos de teste
dados_teste = [teste1, teste2, teste3]

# Rótulos reais dos exemplos de teste (gabarito para avaliar o modelo)
rotulos_teste = [1, 0, 0]

# -------------------------------------------------------
# PREVISÃO E AVALIAÇÃO
# -------------------------------------------------------

# O modelo faz previsões para os dados de teste com base no que aprendeu
previsoes = modelo.predict(dados_teste)

# Exibe as previsões feitas pelo modelo (array com 0s e 1s)
print("Previsões:", previsoes)

# Compara as previsões com os rótulos reais e calcula a acurácia
# Acurácia = acertos / total de exemplos (ex: 1.0 = 100% de acerto)
print("Acurácia:", accuracy_score(rotulos_teste, previsoes))
