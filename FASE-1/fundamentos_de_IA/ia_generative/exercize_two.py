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
    "Política internacional",           # política
    "Debate entre candidatos ao governo", # política
    "Reforma na legislação tributária", # política
    "Inteligência artificial revoluciona o mercado", # tecnologia
    "Novo processador bate recorde de desempenho",   # tecnologia
    "Campeonato mundial de patins no gelo",          # esportes
    "Torneio de roller derby reúne atletas"          # esportes
]
# Rótulos (respostas corretas) para cada texto acima, na mesma ordem
categorias = ["tecnologia", "esportes", "política", "tecnologia", "esportes", "política", "política", "política", "tecnologia", "tecnologia", "esportes", "esportes"]


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
# CONCEITO: O que é Acurácia?
#
# Acurácia é a métrica mais simples para avaliar um modelo de classificação.
# Ela responde à pergunta: "De tudo que o modelo tentou classificar, quantas
# vezes ele acertou?"
#
# Fórmula:
#   Acurácia = Acertos / Total de exemplos testados
#
# Exemplos práticos:
#   - O modelo testou 10 textos e acertou 8  → Acurácia = 0.80 (80%)
#   - O modelo testou 10 textos e acertou 5  → Acurácia = 0.50 (50%)
#   - O modelo testou 10 textos e acertou 10 → Acurácia = 1.00 (100%)
#
# Como interpretar:
#   - 1.0 (100%) → Acertou tudo (perfeito, mas pode ser overfitting)
#   - 0.8 (80%)  → Acertou 8 em cada 10 (bom para muitos casos)
#   - 0.5 (50%)  → Acertou metade (ruim — equivale a chutar)
#   - 0.0 (0%)   → Errou tudo
#
# Limitação importante:
#   A acurácia sozinha pode enganar. Se 95% dos textos forem de tecnologia,
#   um modelo que sempre responde "tecnologia" terá 95% de acurácia — mas
#   não aprendeu nada de verdade. Por isso em projetos reais usamos outras
#   métricas como Precisão, Recall e F1-Score junto com a acurácia.
# =============================================================================


# =============================================================================
# PASSO 6: Predição e Avaliação — "Fazendo a prova"
# .predict() faz a IA tentar adivinhar a categoria dos textos de teste (que ela nunca viu).
# accuracy_score compara as respostas da IA com as respostas corretas
# e retorna o percentual de acertos (ex: 0.66 = 66% de acerto).
# Obs: com apenas 12 exemplos, a acurácia pode ser baixa. Em projetos reais
# usamos centenas ou milhares de exemplos para a IA aprender melhor.
# =============================================================================
y_pred = clf.predict(X_test)
print(f"Acurácia: {accuracy_score(y_test, y_pred)}")


# =============================================================================
# OBSERVAÇÃO: A acurácia diminuiu com o aumento de textos — isso é bom ou ruim?
#
# Depende do contexto, mas nesse caso específico é provavelmente normal e não
# preocupante. Veja o raciocínio:
#
# 1. O conjunto de teste ficou mais variado
#    Com test_size=0.5, metade dos 12 textos vai para teste (6 textos).
#    Antes, com 6 textos, apenas 3 iam para teste. Com mais exemplos no teste,
#    há mais chances de errar.
#
# 2. O dataset ainda é muito pequeno
#    Com apenas 12 frases, cada texto errado representa ~16% de queda na
#    acurácia. Qualquer pequena variação tem impacto enorme. Isso não reflete
#    o comportamento real do modelo.
#
# 3. As frases novas são mais "ambíguas"
#    Frases como "Reforma na legislação tributária" têm palavras que o modelo
#    nunca viu antes. Com poucos exemplos de treino, ele não consegue
#    generalizar bem.
#
# A regra geral em ML é o oposto disso:
#    Em projetos reais, mais dados = mais acurácia, porque o modelo aprende
#    mais padrões. O que vemos aqui é um artefato do dataset minúsculo,
#    não uma falha do algoritmo.
#
# Resumo:
#    - Acurácia cai com mais dados (dataset pequeno) → Normal, variação
#      estatística sem significado real.
#    - Acurácia cai com mais dados (dataset grande)  → Sinal de problema,
#      checar qualidade dos dados ou do modelo.
#
# Conclusão: o modelo precisa de muito mais exemplos (centenas por categoria)
# para a acurácia ser um número confiável. Com 12 frases, o número que aparece
# é quase aleatório.
# =============================================================================
