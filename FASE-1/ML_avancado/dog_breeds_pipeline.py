# =============================================================================
# PIPELINE SUPERVISIONADO COMPLETO - DOG BREEDS
# =============================================================================
#
# Objetivo: Classificar o porte do cachorro (pequeno, medio, grande)
# com base em features extraidas do dataset dog_breeds.csv
#
# Este script segue o pipeline da Aula 1 de ML Avancado:
#   1. Importar dados
#   2. Explorar dados
#   3. Engenharia de features
#   4. Criar variavel target
#   5. Analisar por grupo (target)
#   6. Visualizar (scatter plot)
#   7. Separar X (features) e y (target)
#   8. Dividir treino e teste
#   9. Escolher algoritmo e treinar (KNN)
#  10. Prever
#  11. Avaliar
# =============================================================================


# =============================================================================
# PASSO 1 - IMPORTAR PACOTES E DADOS
# =============================================================================

# numpy: operacoes matematicas e arrays
import numpy as np

# pandas: manipulacao de dados em tabelas (DataFrames)
import pandas as pd

# matplotlib e seaborn: graficos e visualizacoes
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn (scikit-learn): o canivete suico do ML
# - train_test_split: separa dados em treino e teste
# - KNeighborsClassifier: algoritmo KNN
# - accuracy_score: calcula a acuracia (% de acertos)
# - classification_report: relatorio completo com precisao, recall e f1
# - confusion_matrix: matriz de confusao (mostra onde o modelo acerta e erra)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# carrega o CSV para um DataFrame (tabela do pandas)
# pd.read_csv le o arquivo e transforma em uma estrutura de linhas e colunas
df = pd.read_csv("dog_breeds.csv")

# shape retorna (linhas, colunas) - dimensao do dataset
print("=" * 60)
print("PASSO 1 - DADOS CARREGADOS")
print("=" * 60)
print(f"Dimensao: {df.shape[0]} linhas x {df.shape[1]} colunas")


# =============================================================================
# PASSO 2 - EXPLORAR OS DADOS
# =============================================================================
# Antes de qualquer modelo, precisamos ENTENDER os dados.
# Numeros entregam informacao, graficos entregam INTUICAO.

print("\n" + "=" * 60)
print("PASSO 2 - EXPLORACAO DOS DADOS")
print("=" * 60)

# head() mostra as 5 primeiras linhas - uma "espiada" nos dados
print("\n--- Primeiras 5 linhas ---")
print(df.head())

# info() mostra o tipo de cada coluna e se tem valores nulos
# Importante: colunas que parecem numericas podem estar como texto!
# Ex: "Height (in)" aparece como "object" (texto) porque tem "21-24"
print("\n--- Tipos de dados e valores nulos ---")
print(df.info())

# describe() mostra estatisticas: media, desvio padrao, min, max, quartis
# So funciona com colunas numericas por padrao
print("\n--- Estatisticas descritivas ---")
print(df.describe())


# =============================================================================
# PASSO 3 - ENGENHARIA DE FEATURES
# =============================================================================
# O dataset tem colunas como "21-24" (texto). O modelo precisa de NUMEROS.
# Precisamos transformar esses textos em valores numericos utilizaveis.

print("\n" + "=" * 60)
print("PASSO 3 - ENGENHARIA DE FEATURES")
print("=" * 60)

# --- Extrair altura numerica ---
# "21-24" -> split pelo "-" -> ["21", "24"] -> converter pra float
# expand=True faz o split virar duas colunas separadas no DataFrame
df[["height_min", "height_max"]] = (
    df["Height (in)"].str.split("-", expand=True).astype(float)
)
# media entre min e max: (21 + 24) / 2 = 22.5
df["height_avg"] = (df["height_min"] + df["height_max"]) / 2

# --- Extrair longevidade numerica ---
# mesma logica: "10-12" -> media = 11.0
df[["longevity_min", "longevity_max"]] = (
    df["Longevity (yrs)"].str.split("-", expand=True).astype(float)
)
df["longevity_avg"] = (df["longevity_min"] + df["longevity_max"]) / 2

# --- Contar character traits ---
# "Loyal, friendly, intelligent" -> split pela "," -> len = 3
df["num_traits"] = df["Character Traits"].str.split(",").str.len()

# --- Contar health problems ---
# "Hip dysplasia, obesity, ear infections" -> split -> len = 3
df["num_health_problems"] = df["Common Health Problems"].str.split(",").str.len()

# conferir as features criadas
print("\nFeatures extraidas:")
print(
    df[["Breed", "height_avg", "longevity_avg", "num_traits", "num_health_problems"]]
    .head(10)
    .to_string()
)


# =============================================================================
# PASSO 4 - CRIAR A VARIAVEL TARGET (Y)
# =============================================================================
# A variavel target e a RESPOSTA que queremos que o modelo aprenda a prever.
# Aqui, vamos classificar o porte do cachorro com base na altura media:
#   - pequeno: altura < 13 polegadas
#   - medio:   altura entre 13 e 22 polegadas
#   - grande:  altura > 22 polegadas

print("\n" + "=" * 60)
print("PASSO 4 - CRIACAO DA VARIAVEL TARGET")
print("=" * 60)


def classificar_porte(altura):
    """Classifica o porte com base na altura media em polegadas."""
    if altura < 13:
        return "pequeno"
    elif altura <= 22:
        return "medio"
    else:
        return "grande"


# apply() aplica a funcao a cada valor da coluna, linha por linha
df["porte"] = df["height_avg"].apply(classificar_porte)

# value_counts() conta quantos de cada classe existem
# Isso e CRUCIAL - se uma classe tiver muito mais exemplos que outra,
# o modelo pode ficar enviesado (problema de desbalanceamento)
print("\nDistribuicao das classes (target):")
print(df["porte"].value_counts())
print(f"\nTotal: {len(df)} racas")


# =============================================================================
# PASSO 5 - ANALISAR POR GRUPO (TARGET)
# =============================================================================
# groupby() separa os dados por classe e calcula estatisticas para cada uma.
# Olhar so as estatisticas GERAIS pode mascarar diferencas entre classes!
# Sempre analise POR GRUPO.

print("\n" + "=" * 60)
print("PASSO 5 - ANALISE POR GRUPO")
print("=" * 60)

# agrupando por porte e vendo a media de cada feature
print("\nMedia das features por porte:")
print(
    df.groupby("porte")[
        ["height_avg", "longevity_avg", "num_traits", "num_health_problems"]
    ]
    .mean()
    .round(2)
    .to_string()
)


# =============================================================================
# PASSO 6 - VISUALIZACAO (SCATTER PLOT)
# =============================================================================
# Se voce consegue VER os grupos separados no grafico, e um bom indicativo
# de que o algoritmo de ML tambem vai conseguir aprender esses padroes.

print("\n" + "=" * 60)
print("PASSO 6 - VISUALIZACAO")
print("=" * 60)
print("Gerando graficos...")

# --- Grafico 1: Scatter plot altura vs longevidade ---
plt.figure(figsize=(10, 6))
colors = {"pequeno": "blue", "medio": "green", "grande": "red"}
for porte, grupo in df.groupby("porte"):
    plt.scatter(
        grupo["height_avg"],
        grupo["longevity_avg"],
        c=colors[porte],
        label=porte,
        alpha=0.6,
        edgecolors="k",
    )
plt.xlabel("Altura Media (polegadas)")
plt.ylabel("Longevidade Media (anos)")
plt.title("Altura vs Longevidade por Porte")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- Grafico 2: Distribuicao das classes ---
plt.figure(figsize=(6, 4))
df["porte"].value_counts().plot(kind="bar", color=["green", "red", "blue"])
plt.title("Distribuicao das classes (porte)")
plt.ylabel("Quantidade")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# =============================================================================
# PASSO 7 - SEPARAR FEATURES (X) E TARGET (Y)
# =============================================================================
# X = features (caracteristicas) -> o que o modelo usa pra aprender
# y = target (resposta/rotulo) -> o que o modelo tenta prever
#
# ATENCAO com a sintaxe:
#   X usa COLCHETES DUPLOS [[...]] porque sao MULTIPLAS colunas (DataFrame)
#   y usa COLCHETE SIMPLES [...] porque e UMA coluna (Series)

print("\n" + "=" * 60)
print("PASSO 7 - SEPARAR X (FEATURES) E Y (TARGET)")
print("=" * 60)

features = ["height_avg", "longevity_avg", "num_traits", "num_health_problems"]

X = df[features]       # colchetes duplos -> DataFrame com varias colunas
y = df["porte"]        # colchete simples -> Series com uma coluna

print(f"X (features): {X.shape} -> {len(X)} amostras, {len(features)} features")
print(f"y (target):   {y.shape} -> {len(y)} rotulos")
print(f"\nFeatures usadas: {features}")
print(f"Target: porte (pequeno, medio, grande)")


# =============================================================================
# PASSO 8 - DIVIDIR EM TREINO E TESTE
# =============================================================================
# Por que separar?
#   - TREINO (80%): o modelo APRENDE os padroes
#   - TESTE (20%):  avaliamos se o modelo GENERALIZA (funciona com dados novos)
#
# Se testar com os mesmos dados que treinou, e como estudar so as questoes
# da prova - nao prova que voce domina a materia.
#
# Parametros importantes:
#   test_size=0.2    -> 20% para teste, 80% para treino
#   stratify=y       -> mantem a proporcao das classes nos dois conjuntos
#   random_state=42  -> semente para reproducibilidade (mesmo resultado sempre)

print("\n" + "=" * 60)
print("PASSO 8 - DIVIDIR TREINO E TESTE")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% teste, 80% treino
    stratify=y,          # manter proporcao das classes
    random_state=42      # reproducibilidade
)

print(f"Treino: {len(X_train)} amostras")
print(f"Teste:  {len(X_test)} amostras")
print(f"\nDistribuicao no treino:\n{y_train.value_counts().to_string()}")
print(f"\nDistribuicao no teste:\n{y_test.value_counts().to_string()}")


# =============================================================================
# PASSO 9 - ESCOLHER ALGORITMO E TREINAR (KNN)
# =============================================================================
# KNN (K-Nearest Neighbors / K-Vizinhos Mais Proximos):
#   - Para classificar um novo dado, olha os K vizinhos mais proximos
#   - Classifica pela "maioria vence" entre esses vizinhos
#   - K e um HIPERPARAMETRO: voce define ANTES do treinamento
#
# O fit() recebe X_train E y_train porque o modelo precisa saber:
#   "para ESTAS features, a resposta correta e ESTA"

print("\n" + "=" * 60)
print("PASSO 9 - TREINAR O MODELO (KNN com K=5)")
print("=" * 60)

# instanciar o modelo: n_neighbors=5 significa que olha os 5 vizinhos mais proximos
modelo_knn = KNeighborsClassifier(n_neighbors=5)

# treinar: o modelo memoriza os dados de treino (KNN e um algoritmo "lazy")
modelo_knn.fit(X_train, y_train)

print("Modelo KNN treinado com sucesso!")


# =============================================================================
# PASSO 10 - PREVER
# =============================================================================
# predict() recebe dados NOVOS e retorna a classe predita pelo modelo.
# Ele olha os 5 vizinhos mais proximos do dado novo e vota na maioria.

print("\n" + "=" * 60)
print("PASSO 10 - PREDICOES")
print("=" * 60)

# gerar predicoes para TODA a base de teste
y_predito = modelo_knn.predict(X_test)

# comparar as primeiras 10 predicoes com o valor real
comparacao = pd.DataFrame({
    "Real": y_test.values[:10],
    "Predito": y_predito[:10],
    "Acertou?": y_test.values[:10] == y_predito[:10]
})
print("\nComparacao Real vs Predito (primeiras 10 amostras):")
print(comparacao.to_string(index=False))

# --- Testando com um dado novo (como na aula) ---
# Imagine um cachorro com: altura media 25in, longevidade 10 anos,
# 4 traits, 3 health problems
# Note os COLCHETES DUPLOS [[...]] - o predict espera um array 2D
novo_cachorro = [[25, 10, 4, 3]]
predicao = modelo_knn.predict(novo_cachorro)
print(f"\nPredicao para cachorro novo (altura=25, longevidade=10): {predicao[0]}")


# =============================================================================
# PASSO 11 - AVALIAR O MODELO
# =============================================================================
# Acuracia = acertos / total de predicoes
#
# ALERTA: se a acuracia for 100%, DESCONFIE!
# Possiveis causas: base muito simples, overfitting, data leakage, erro no pipeline
#
# Alem da acuracia, usamos:
#   - classification_report: mostra precisao, recall e f1-score POR CLASSE
#   - confusion_matrix: mostra onde o modelo acerta e onde erra

print("\n" + "=" * 60)
print("PASSO 11 - AVALIACAO DO MODELO")
print("=" * 60)

# acuracia geral
acuracia = accuracy_score(y_test, y_predito)
print(f"\nAcuracia: {acuracia:.2%}")

# relatorio detalhado por classe
# - precision: dos que o modelo disse ser X, quantos realmente eram X?
# - recall: dos que realmente eram X, quantos o modelo acertou?
# - f1-score: media harmonica entre precision e recall
print("\nRelatorio de Classificacao:")
print(classification_report(y_test, y_predito))

# --- Matriz de confusao ---
# Cada celula mostra quantas vezes o modelo predisse X quando o real era Y
# Diagonal principal = acertos, fora da diagonal = erros
labels = ["pequeno", "medio", "grande"]
cm = confusion_matrix(y_test, y_predito, labels=labels)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusao")
plt.tight_layout()
plt.show()


# =============================================================================
# EXTRA - TESTANDO DIFERENTES VALORES DE K
# =============================================================================
# K e um hiperparametro. Qual o melhor valor?
# Vamos testar K de 1 a 20 e ver qual da a melhor acuracia.
# (Na proxima aula voce vai ver tecnicas mais sofisticadas como GridSearch)

print("\n" + "=" * 60)
print("EXTRA - TESTANDO DIFERENTES VALORES DE K")
print("=" * 60)

k_values = range(1, 21)
acuracias = []

for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train, y_train)
    y_pred_temp = knn_temp.predict(X_test)
    acuracias.append(accuracy_score(y_test, y_pred_temp))

# grafico de acuracia por K
plt.figure(figsize=(10, 5))
plt.plot(k_values, acuracias, marker="o")
plt.xlabel("Valor de K")
plt.ylabel("Acuracia")
plt.title("Acuracia por valor de K")
plt.xticks(k_values)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

melhor_k = list(k_values)[np.argmax(acuracias)]
print(f"\nMelhor K: {melhor_k} (acuracia: {max(acuracias):.2%})")

print("\n" + "=" * 60)
print("PIPELINE COMPLETO FINALIZADO!")
print("=" * 60)
