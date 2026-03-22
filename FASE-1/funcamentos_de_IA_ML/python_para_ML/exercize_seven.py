# Importa o TensorFlow com o apelido "tf" (convenção universal)
# TensorFlow é o framework de deep learning do Google — permite construir e treinar
# redes neurais. O Keras (tf.keras) é sua API de alto nível, mais fácil de usar
import tensorflow as tf

# Importa numpy para geração e manipulação dos dados de entrada
import numpy as np

# --- DADOS ---

# Gera uma matriz X com 100 linhas e 3 colunas, valores aleatórios entre 0 e 1
# Representa 100 amostras, cada uma com 3 features (características de entrada)
# np.random.random((linhas, colunas)) — o argumento é uma tupla com as dimensões
X = np.random.random((100, 3))

# Gera o vetor y com 100 linhas e 1 coluna, valores aleatórios entre 0 e 1
# Representa o valor alvo (label) que o modelo deve aprender a prever para cada amostra
# Neste exercício os dados são puramente aleatórios — o objetivo é apenas ver o fluxo funcionar
y = np.random.random((100, 1))

# --- ARQUITETURA DA REDE NEURAL (Functional API do Keras) ---

# Define a camada de entrada da rede
# shape=(3,) significa que cada amostra tem 3 valores de entrada (3 features)
# A vírgula dentro da tupla é obrigatória em Python para indicar que é uma tupla de 1 elemento
inputs = tf.keras.Input(shape=(3,))

# Adiciona uma camada oculta (Dense = totalmente conectada) com 10 neurônios
# activation='relu' aplica a função ReLU: f(x) = max(0, x) — a mais usada em redes neurais
# O (inputs) no final conecta esta camada à camada de entrada (sintaxe da Functional API)
# Analogia: cada neurônio é como um nó que recebe todos os 3 inputs e gera 1 saída
x = tf.keras.layers.Dense(10, activation='relu')(inputs)

# Adiciona a camada de saída com 1 neurônio e sem função de ativação (regressão linear)
# Sem activation → saída contínua, adequada para tarefas de regressão (prever um número)
# Recebe as 10 saídas da camada anterior e produz 1 valor final
outputs = tf.keras.layers.Dense(1)(x)

# Constrói o modelo conectando formalmente a entrada à saída
# tf.keras.Model define o grafo computacional completo: inputs → camada oculta → outputs
# Agora o modelo sabe seu formato, mas ainda não sabe como aprender (isso vem no compile)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# --- COMPILAÇÃO ---

# Configura o processo de aprendizado do modelo
# optimizer='adam': algoritmo que ajusta os pesos durante o treinamento
#   Adam (Adaptive Moment Estimation) é o otimizador padrão — adapta a taxa de aprendizado
#   automaticamente para cada parâmetro, converge mais rápido que o SGD clássico
# loss='mean_squared_error': função de perda (loss function) — mede o erro do modelo
#   MSE = média dos quadrados da diferença entre valor previsto e valor real
#   Quanto menor o loss, melhor o modelo está prevendo
model.compile(optimizer='adam', loss='mean_squared_error')

# --- TREINAMENTO ---

# Treina o modelo com os dados X e y
# epochs=5: número de vezes que o modelo verá o dataset completo
#   Em cada epoch, o modelo ajusta seus pesos para reduzir o loss
#   5 epochs é pouco — em problemas reais usa-se dezenas ou centenas
# Durante a execução você verá o loss sendo impresso a cada epoch —
# idealmente ele deve diminuir a cada iteração
#
# O que é um epoch?
# É uma passagem completa pelo dataset inteiro. Com 100 amostras e epochs=5:
#   Epoch 1/5 → modelo vê as 100 amostras, ajusta os pesos, calcula o loss
#   Epoch 2/5 → modelo vê as mesmas 100 amostras novamente, loss tende a cair
#   ...e assim até o epoch 5
#
# Na prática o dataset é dividido em lotes menores (batches, default=32 no Keras):
#   dataset: 100 amostras | batch_size: 32
#   Epoch 1:
#     batch 1 → amostras  0-31  → ajusta pesos
#     batch 2 → amostras 32-63  → ajusta pesos
#     batch 3 → amostras 64-95  → ajusta pesos
#     batch 4 → amostras 96-99  → ajusta pesos
#     ✓ epoch concluída
#
# Quantos epochs usar?
#   Poucos → underfitting (modelo não aprendeu o suficiente)
#   Muitos → overfitting (modelo memoriza treino mas falha em dados novos)
#   Solução: usar EarlyStopping para interromper automaticamente quando parar de melhorar
model.fit(X, y, epochs=5)