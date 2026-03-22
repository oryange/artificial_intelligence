# NOTA: este exercício tem a mesma estrutura do exercize_eight.py
# Use-o para praticar e fixar o fluxo de classificação binária com Sequential API

# Importa o modelo Sequential — camadas empilhadas linearmente, uma após a outra
from keras.models import Sequential

# Importa a camada Dense (totalmente conectada)
# Cada neurônio recebe sinal de TODOS os neurônios da camada anterior
from keras.layers import Dense

# NumPy para geração dos dados de treino
import numpy as np

# --- DADOS ---

# 100 amostras x 8 features, valores aleatórios entre 0 e 1
X = np.random.random((100, 8))

# 100 rótulos binários (0 ou 1) — tarefa de classificação binária
# np.random.randint(2, ...) gera apenas 0 ou 1
y = np.random.randint(2, size=(100, 1))

# --- ARQUITETURA ---

# Container vazio que receberá as camadas em sequência
model = Sequential()

# Camada oculta 1: 12 neurônios, recebe 8 features (input_dim=8), ativação ReLU
# ReLU permite aprender padrões não-lineares: f(x) = max(0, x)
model.add(Dense(12, input_dim=8, activation='relu'))

# Camada oculta 2: 8 neurônios, ativação ReLU
# Comprime a representação de 12 → 8, refinando os padrões aprendidos
model.add(Dense(8, activation='relu'))

# Camada de saída: 1 neurônio, ativação sigmoid
# sigmoid → saída entre 0 e 1, interpretada como probabilidade da classe 1
# Regra: sigmoid na saída + binary_crossentropy no loss = classificação binária
model.add(Dense(1, activation='sigmoid'))

# --- COMPILAÇÃO ---

# loss='binary_crossentropy': loss correto para 2 classes com saída sigmoid
# optimizer='adam': ajusta os pesos automaticamente durante o treino
# metrics=['accuracy']: exibe % de acertos a cada epoch (mais intuitivo que o loss sozinho)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# --- TREINAMENTO ---

# epochs=150: 150 passagens completas pelo dataset
# batch_size=10: pesos ajustados a cada 10 amostras
#   → 100 amostras / 10 por batch = 10 atualizações por epoch
model.fit(X, y, epochs=150, batch_size=10)