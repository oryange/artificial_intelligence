# =============================================================================
# PASSO 1: Importações — "Pegando as ferramentas"
# sklearn (scikit-learn) é uma biblioteca famosa de Machine Learning.
# Importamos 4 ferramentas específicas que vamos usar no código.
# =============================================================================
from sklearn.feature_extraction.text import CountVectorizer  # Converte textos em números
from sklearn.model_selection import train_test_split          # Divide dados em treino e teste
from sklearn.naive_bayes import MultinomialNB                 # Algoritmo que aprende a classificar
from sklearn.metrics import accuracy_score                    # Mede o percentual de acertos


# =============================================================================
# PASSO 2: Dados de Exemplo — "A experiência de vida da IA"
# A IA vai aprender a partir desses exemplos rotulados.
# Analogia: como mostrar fotos de gatos e cachorros para uma criança
# dizendo "isso é gato", "isso é cachorro" — ela aprende pelo exemplo.
# =============================================================================
textos = [
    "O novo lançamento da Apple",       # tecnologia
    "Resultado do jogo de ontem",       # esportes
    "Eleições presidenciais",           # política
    "Atualização no mundo da tecnologia",# tecnologia
    "Campeonato de futebol",            # esportes
    "Política internacional"            # política
]
# Rótulos (respostas corretas) para cada texto acima, na mesma ordem
categorias = ["tecnologia", "esportes", "política", "tecnologia", "esportes", "política"]


# =============================================================================
# PASSO 3: CountVectorizer — "Traduzindo palavras para números"
# Computadores não entendem palavras, só números.
# O CountVectorizer transforma cada frase em uma lista de contagens de palavras.
# Exemplo: "O novo lançamento" vira [1, 0, 1, 1, 0, ...] (1 se a palavra aparece, 0 se não)
# =============================================================================
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(textos)  # X é a matriz numérica que representa todos os textos


# =============================================================================
# PASSO 4: train_test_split — "Separando estudo de prova"
# Divide os dados em dois grupos:
#   - Treino (train): a IA aprende com esses dados
#   - Teste (test):  usamos para avaliar se ela aprendeu de verdade
# test_size=0.5 significa 50% para treino e 50% para teste.
# random_state=42 garante que a divisão seja sempre igual (reproduzível).
# Analogia: você estuda com alguns exercícios e faz a prova com outros que nunca viu.
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, categorias, test_size=0.5, random_state=42)


# =============================================================================
# PASSO 5: MultinomialNB — "O algoritmo que aprende"
# MultinomialNB é o Naive Bayes — algoritmo clássico para classificação de textos.
# Ele usa probabilidade para decidir a categoria de um texto.
# .fit() é o momento do aprendizado: a IA analisa os exemplos de treino e memoriza os padrões.
# =============================================================================
clf = MultinomialNB()
clf.fit(X_train, y_train)  # Aqui a IA "estuda" — aprende com os dados de treino


# =============================================================================
# PASSO 6: Predição e Avaliação — "Fazendo a prova"
# .predict() faz a IA tentar adivinhar a categoria dos textos de teste (que ela nunca viu).
# accuracy_score compara as respostas da IA com as respostas corretas
# e retorna o percentual de acertos (ex: 0.66 = 66% de acerto).
# Obs: com apenas 6 exemplos, a acurácia pode ser baixa. Em projetos reais
# usamos centenas ou milhares de exemplos para a IA aprender melhor.
# =============================================================================
y_pred = clf.predict(X_test)
print(f"Acurácia: {accuracy_score(y_test, y_pred)}")
